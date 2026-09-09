from __future__ import annotations

import importlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import tempfile
import time
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from poc.bm25 import load_prepared_queries
from poc.config import ConfigError, ModelSpec
from poc.datasets import DatasetIntegrityError, file_facts
from poc.latency import summarize_latency_ms
from poc.manifest import canonical_sha256, read_json, write_json
from poc.model_runtime import (
    compare_embedding_vectors,
    create_bulk_backend,
    hash_model_snapshot,
    model_snapshot_directory,
    release_device_memory,
    verify_registered_model_snapshot,
)
from poc.provenance import (
    collect_code_revision,
    collect_manifest_provenance,
    verify_decision_provenance,
)
from poc.query_correctness import select_self_retrieval_ids, top1_self_retrieval
from poc.trec import identifier_sort_key

_BACKEND = "sentence_transformers_onnx"
_PRECISION = "int8_dynamic"
_EXECUTION_PROVIDER = "CPUExecutionProvider"
_QUANTIZATION_CONFIGS = {"arm64", "avx2", "avx512", "avx512_vnni"}
_RUNTIME_KEYS = {
    "backend",
    "batch_size",
    "document_drift_sample_size",
    "execution_mode",
    "precision",
    "execution_provider",
    "inter_op_num_threads",
    "intra_op_num_threads",
    "maximum_absolute_query_delta",
    "minimum_document_drift_cosine_similarity",
    "minimum_mean_query_cosine_similarity",
    "minimum_query_cosine_similarity",
    "minimum_self_retrieval_top1",
    "parity_sample_size",
    "quantization_config",
    "self_retrieval_sample_size",
}
_INT8_OPERATOR_TYPES = {
    "DynamicQuantizeLinear",
    "MatMulInteger",
    "QGemm",
    "QLinearMatMul",
    "QuantizeLinear",
}
_ARCTIC_QUERY_ADAPTER = "transformers_cls_mrl_normalized"
QUERY_LATENCY_REPLAY_RELATIVE_TOLERANCE = 0.25
QUERY_LATENCY_REPLAY_ABSOLUTE_TOLERANCE_MS = 0.1


@dataclass(frozen=True, slots=True)
class QueryRuntimeSpec:
    backend: str
    precision: str
    execution_provider: str
    quantization_config: str
    parity_sample_size: int
    minimum_query_cosine_similarity: float
    minimum_mean_query_cosine_similarity: float
    maximum_absolute_query_delta: float
    document_drift_sample_size: int
    minimum_document_drift_cosine_similarity: float
    self_retrieval_sample_size: int
    minimum_self_retrieval_top1: float
    intra_op_num_threads: int
    inter_op_num_threads: int
    execution_mode: str
    batch_size: int

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> QueryRuntimeSpec:
        missing = sorted(_RUNTIME_KEYS - values.keys())
        if missing:
            raise ConfigError(f"query runtime is missing keys: {', '.join(missing)}")
        unexpected = sorted(values.keys() - _RUNTIME_KEYS)
        if unexpected:
            raise ConfigError(f"query runtime has unexpected keys: {', '.join(unexpected)}")

        backend = _required_string(values, "backend")
        precision = _required_string(values, "precision")
        execution_provider = _required_string(values, "execution_provider")
        quantization_config = _required_string(values, "quantization_config")
        if backend != _BACKEND:
            raise ConfigError(f"query runtime backend must be {_BACKEND}")
        if precision != _PRECISION:
            raise ConfigError(f"query runtime precision must be {_PRECISION}")
        if execution_provider != _EXECUTION_PROVIDER:
            raise ConfigError(
                f"query runtime execution_provider must be {_EXECUTION_PROVIDER}"
            )
        if quantization_config not in _QUANTIZATION_CONFIGS:
            raise ConfigError(
                "query runtime quantization_config must be one of "
                + ", ".join(sorted(_QUANTIZATION_CONFIGS))
            )
        execution_mode = _required_string(values, "execution_mode")
        if execution_mode != "ORT_SEQUENTIAL":
            raise ConfigError("query runtime execution_mode must be ORT_SEQUENTIAL")
        batch_size = _required_positive_int(values, "batch_size")
        if batch_size != 1:
            raise ConfigError("query runtime batch_size must be 1 for production parity")
        return cls(
            backend=backend,
            precision=precision,
            execution_provider=execution_provider,
            quantization_config=quantization_config,
            parity_sample_size=_required_positive_int(values, "parity_sample_size"),
            minimum_query_cosine_similarity=_required_ratio(
                values, "minimum_query_cosine_similarity"
            ),
            minimum_mean_query_cosine_similarity=_required_ratio(
                values, "minimum_mean_query_cosine_similarity"
            ),
            maximum_absolute_query_delta=_required_positive_float(
                values, "maximum_absolute_query_delta"
            ),
            document_drift_sample_size=_required_positive_int(
                values, "document_drift_sample_size"
            ),
            minimum_document_drift_cosine_similarity=_required_ratio(
                values, "minimum_document_drift_cosine_similarity"
            ),
            self_retrieval_sample_size=_required_positive_int(
                values, "self_retrieval_sample_size"
            ),
            minimum_self_retrieval_top1=_required_ratio(
                values, "minimum_self_retrieval_top1"
            ),
            intra_op_num_threads=_required_positive_int(values, "intra_op_num_threads"),
            inter_op_num_threads=_required_positive_int(values, "inter_op_num_threads"),
            execution_mode=execution_mode,
            batch_size=batch_size,
        )


def load_query_runtime_spec(
    path: Path | str = Path("config/query_runtime.toml"),
) -> QueryRuntimeSpec:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    if raw.pop("schema_version", None) != 1:
        raise ConfigError(f"unsupported query runtime schema in {config_path}")
    return QueryRuntimeSpec.from_mapping(raw)


def onnx_artifact_directory(
    root: Path,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
) -> Path:
    return (
        root
        / "data/cache/models/onnx-int8"
        / model.name
        / model.revision
        / runtime.quantization_config
    )


def onnx_quantized_filename(runtime: QueryRuntimeSpec) -> str:
    return f"onnx/model_qint8_{runtime.quantization_config}.onnx"


def onnx_manifest_path(
    root: Path,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
) -> Path:
    return (
        root
        / "results/wands/query-runtime"
        / f"{model.name}.{runtime.quantization_config}.manifest.json"
    )


class OnnxInt8QueryBackend:
    def __init__(
        self,
        *,
        root: Path,
        model: ModelSpec,
        runtime: QueryRuntimeSpec,
    ) -> None:
        _prepare_transformers_module_cache(root)
        manifest = verify_onnx_int8_artifact(root=root, model=model, runtime=runtime)
        artifact_directory = onnx_artifact_directory(root, model, runtime)
        sentence_transformers = importlib.import_module("sentence_transformers")
        self._encoder: Any = sentence_transformers.SentenceTransformer(
            str(artifact_directory),
            backend="onnx",
            device="cpu",
            trust_remote_code=model.trust_remote_code,
            truncate_dim=model.dims,
            config_kwargs={
                "use_memory_efficient_attention": model.use_memory_efficient_attention
            },
            model_kwargs={
                "file_name": onnx_quantized_filename(runtime),
                "provider": runtime.execution_provider,
                "session_options": create_onnx_session_options(runtime),
            },
            local_files_only=True,
        )
        self._encoder.max_seq_length = model.max_seq_length
        dimensions = self._encoder.get_embedding_dimension()
        if dimensions != model.dims:
            raise DatasetIntegrityError(
                f"{model.name} ONNX backend reports {dimensions} dimensions, expected {model.dims}"
            )
        if self._encoder.get_backend() != "onnx":
            raise DatasetIntegrityError(f"{model.name} did not load through the ONNX backend")
        self._model = model
        self._runtime = runtime
        self.device = "cpu"
        self.model_artifact_sha256 = str(manifest["artifact_sha256"])
        self.source_model_artifact_sha256 = str(manifest["source_model_sha256"])
        self.runtime = (
            f"sentence-transformers/{importlib.metadata.version('sentence-transformers')} "
            f"optimum-onnx/{importlib.metadata.version('optimum-onnx')} "
            f"onnxruntime/{importlib.metadata.version('onnxruntime')} "
            f"dynamic_int8/{runtime.quantization_config} "
            f"provider/{runtime.execution_provider} "
            f"threads/{runtime.intra_op_num_threads}:{runtime.inter_op_num_threads} "
            f"execution_mode/{runtime.execution_mode}"
        )

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, self._model.dims), dtype=np.float32)
        assert_registered_query_batch_size(len(texts), self._runtime)
        values = self._encoder.encode(
            texts,
            batch_size=len(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self._model.normalize,
        )
        vectors = np.asarray(values, dtype=np.float32)
        if vectors.shape != (len(texts), self._model.dims) or not np.isfinite(vectors).all():
            raise DatasetIntegrityError(f"invalid int8 ONNX embeddings from {self._model.name}")
        return vectors


class SentenceTransformersFp32OnnxQueryBackend:
    def __init__(
        self,
        *,
        root: Path,
        model: ModelSpec,
        runtime: QueryRuntimeSpec,
    ) -> None:
        manifest = verify_onnx_int8_artifact(root=root, model=model, runtime=runtime)
        sentence_transformers = importlib.import_module("sentence_transformers")
        self._encoder: Any = sentence_transformers.SentenceTransformer(
            str(onnx_artifact_directory(root, model, runtime)),
            backend="onnx",
            device="cpu",
            trust_remote_code=model.trust_remote_code,
            truncate_dim=model.dims,
            config_kwargs={
                "use_memory_efficient_attention": model.use_memory_efficient_attention
            },
            model_kwargs={
                "file_name": "onnx/model.onnx",
                "provider": runtime.execution_provider,
                "session_options": create_onnx_session_options(runtime),
            },
            local_files_only=True,
        )
        self._encoder.max_seq_length = model.max_seq_length
        self._model = model
        self.device = "cpu"
        del manifest
        self.model_artifact_sha256 = file_facts(
            onnx_artifact_directory(root, model, runtime) / "onnx/model.onnx"
        ).sha256
        self.runtime = (
            f"sentence-transformers/{importlib.metadata.version('sentence-transformers')} "
            f"optimum-onnx/{importlib.metadata.version('optimum-onnx')} "
            f"onnxruntime/{importlib.metadata.version('onnxruntime')} fp32 "
            f"provider/{runtime.execution_provider} "
            f"threads/{runtime.intra_op_num_threads}:{runtime.inter_op_num_threads} "
            f"execution_mode/{runtime.execution_mode}"
        )

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, self._model.dims), dtype=np.float32)
        values = self._encoder.encode(
            texts,
            batch_size=len(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self._model.normalize,
        )
        vectors = np.asarray(values, dtype=np.float32)
        if vectors.shape != (len(texts), self._model.dims) or not np.isfinite(vectors).all():
            raise DatasetIntegrityError(
                f"invalid fp32 ONNX embeddings from {self._model.name}"
            )
        return vectors


class ArcticOnnxQueryBackend:
    def __init__(
        self,
        *,
        root: Path,
        model: ModelSpec,
        runtime: QueryRuntimeSpec,
        quantized: bool,
    ) -> None:
        _prepare_transformers_module_cache(root)
        manifest = verify_onnx_int8_artifact(root=root, model=model, runtime=runtime)
        transformers = importlib.import_module("transformers")
        source_snapshot = model_snapshot_directory(root, model)
        self._tokenizer: Any = transformers.AutoTokenizer.from_pretrained(
            str(source_snapshot),
            trust_remote_code=model.trust_remote_code,
            local_files_only=True,
        )
        filename = (
            onnx_quantized_filename(runtime) if quantized else "onnx/model.onnx"
        )
        onnxruntime = importlib.import_module("onnxruntime")
        self._session: Any = onnxruntime.InferenceSession(
            str(onnx_artifact_directory(root, model, runtime) / filename),
            sess_options=create_onnx_session_options(runtime),
            providers=[runtime.execution_provider],
        )
        self._model = model
        self._runtime = runtime
        self._enforce_batch_size = quantized
        self.device = "cpu"
        artifact_directory = onnx_artifact_directory(root, model, runtime)
        self.model_artifact_sha256 = (
            str(manifest["artifact_sha256"])
            if quantized
            else file_facts(artifact_directory / "onnx/model.onnx").sha256
        )
        self.source_model_artifact_sha256 = str(manifest["source_model_sha256"])
        precision = f"dynamic_int8/{runtime.quantization_config}" if quantized else "fp32"
        self.runtime = (
            f"transformers/{importlib.metadata.version('transformers')} "
            f"onnxruntime/{importlib.metadata.version('onnxruntime')} "
            f"adapter/{_ARCTIC_QUERY_ADAPTER} {precision} "
            f"provider/{runtime.execution_provider} "
            f"threads/{runtime.intra_op_num_threads}:{runtime.inter_op_num_threads} "
            f"execution_mode/{runtime.execution_mode}"
        )

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, self._model.dims), dtype=np.float32)
        if self._enforce_batch_size:
            assert_registered_query_batch_size(len(texts), self._runtime)
        tokens = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._model.max_seq_length,
            return_tensors="np",
        )
        inputs = {
            item.name: np.asarray(tokens[item.name], dtype=np.int64)
            for item in self._session.get_inputs()
        }
        vectors = np.asarray(self._session.run(None, inputs)[0], dtype=np.float32)
        if vectors.shape != (len(texts), self._model.dims) or not np.isfinite(vectors).all():
            raise DatasetIntegrityError(
                f"invalid Arctic ONNX embeddings from {self._model.name}"
            )
        return vectors


def assert_registered_query_batch_size(
    batch_size: int,
    runtime: QueryRuntimeSpec,
) -> None:
    if batch_size > runtime.batch_size:
        raise DatasetIntegrityError(
            f"int8 ONNX query batch {batch_size} exceeds registered size "
            f"{runtime.batch_size}"
        )


def create_int8_query_backend(
    *,
    root: Path,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
) -> Any:
    if query_adapter_name(model) == _ARCTIC_QUERY_ADAPTER:
        return ArcticOnnxQueryBackend(
            root=root,
            model=model,
            runtime=runtime,
            quantized=True,
        )
    return OnnxInt8QueryBackend(root=root, model=model, runtime=runtime)


def create_arctic_fp32_onnx_query_backend(
    *,
    root: Path,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
) -> ArcticOnnxQueryBackend:
    if query_adapter_name(model) != _ARCTIC_QUERY_ADAPTER:
        raise ValueError("the CLS MRL ONNX adapter is registered only for Arctic")
    return ArcticOnnxQueryBackend(
        root=root,
        model=model,
        runtime=runtime,
        quantized=False,
    )


def create_fp32_onnx_query_backend(
    *,
    root: Path,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
) -> Any:
    if query_adapter_name(model) == _ARCTIC_QUERY_ADAPTER:
        return create_arctic_fp32_onnx_query_backend(
            root=root,
            model=model,
            runtime=runtime,
        )
    return SentenceTransformersFp32OnnxQueryBackend(
        root=root,
        model=model,
        runtime=runtime,
    )


def query_adapter_name(model: ModelSpec) -> str:
    return (
        _ARCTIC_QUERY_ADAPTER
        if model.name == "arctic_embed_m_v2"
        else "sentence_transformers"
    )


def create_onnx_session_options(runtime: QueryRuntimeSpec) -> Any:
    onnxruntime = importlib.import_module("onnxruntime")
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = runtime.intra_op_num_threads
    options.inter_op_num_threads = runtime.inter_op_num_threads
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    return options


def build_onnx_int8_artifact(
    *,
    root: Path,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
    generation_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    benchmark_provenance = _capture_provenance(root, generation_provenance)
    _verify_registered_runtime_inputs(root=root, model=model, runtime=runtime)
    _prepare_transformers_module_cache(root)
    _quarantine_legacy_onnx_artifact(root=root, model=model, runtime=runtime)
    if query_adapter_name(model) == _ARCTIC_QUERY_ADAPTER:
        return _build_arctic_onnx_int8_artifact(
            root=root,
            model=model,
            runtime=runtime,
            benchmark_provenance=benchmark_provenance,
        )
    artifact_directory = onnx_artifact_directory(root, model, runtime)
    manifest_path = onnx_manifest_path(root, model, runtime)
    if artifact_directory.exists() or manifest_path.exists():
        return _refresh_existing_onnx_manifest_attestation(
            root=root,
            model=model,
            runtime=runtime,
            benchmark_provenance=benchmark_provenance,
        )

    source_snapshot = model_snapshot_directory(root, model)
    source_model_sha256 = verify_registered_model_snapshot(
        root=root,
        model=model,
        snapshot=source_snapshot,
    )
    _verify_document_model_source(root, model, source_model_sha256)
    artifact_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{model.name}-onnx-",
        dir=artifact_directory.parent,
    ) as temporary:
        temporary_root = Path(temporary)
        source_copy = temporary_root / "source"
        output_directory = temporary_root / "output"
        shutil.copytree(source_snapshot, source_copy)
        _prepare_onnx_source_copy(source_copy, model)
        sentence_transformers = importlib.import_module("sentence_transformers")
        encoder: Any = sentence_transformers.SentenceTransformer(
            str(source_copy),
            backend="onnx",
            device="cpu",
            trust_remote_code=model.trust_remote_code,
            truncate_dim=model.dims,
            config_kwargs={
                "use_memory_efficient_attention": model.use_memory_efficient_attention
            },
            model_kwargs={
                "export": True,
                "provider": runtime.execution_provider,
            },
            local_files_only=True,
        )
        encoder.max_seq_length = model.max_seq_length
        if encoder.get_embedding_dimension() != model.dims:
            raise DatasetIntegrityError(f"ONNX export dimensions differ for {model.name}")
        encoder.save_pretrained(str(output_directory))
        sentence_transformers.export_dynamic_quantized_onnx_model(
            model=encoder,
            quantization_config=runtime.quantization_config,
            model_name_or_path=str(output_directory),
            push_to_hub=False,
        )
        quantized_path = output_directory / onnx_quantized_filename(runtime)
        graph_facts = inspect_int8_onnx_graph(quantized_path)
        output_directory.replace(artifact_directory)

    if verify_registered_model_snapshot(
        root=root,
        model=model,
        snapshot=source_snapshot,
    ) != source_model_sha256:
        raise DatasetIntegrityError(f"ONNX export mutated source snapshot for {model.name}")
    quantized_path = artifact_directory / onnx_quantized_filename(runtime)
    fp32_path = artifact_directory / "onnx/model.onnx"
    fp32_facts = file_facts(fp32_path)
    quantized_facts = file_facts(quantized_path)
    completion_revision = asdict(collect_code_revision(root))
    provenance_valid = _recorded_provenance_valid(
        root,
        benchmark_provenance,
        completion_revision,
        require_current=True,
    )
    embedding_eligible = _embedding_manifest_eligible(root, model)
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "model": asdict(model),
        "runtime": runtime_artifact_values(runtime),
        "source_model_sha256": source_model_sha256,
        "artifact_directory": str(artifact_directory.relative_to(root)),
        "artifact_sha256": hash_model_snapshot(artifact_directory),
        "fp32_file": {
            "path": str(fp32_path.relative_to(root)),
            "sha256": fp32_facts.sha256,
            "bytes": fp32_facts.bytes,
        },
        "quantized_file": {
            "path": str(quantized_path.relative_to(root)),
            "sha256": quantized_facts.sha256,
            "bytes": quantized_facts.bytes,
        },
        "graph": graph_facts,
        "query_adapter": query_adapter_name(model),
        "export_packages": _export_package_versions(),
        "runtime_registry": _artifact(root, root / "config/query_runtime.toml"),
        "model_registry": _artifact(root, root / "config/models.toml"),
        "model_artifact_registry": _artifact(
            root, root / "config/model_artifacts.toml"
        ),
        "embedding_manifest": _artifact(
            root,
            root / f"results/wands/embeddings/{model.name}.manifest.json",
        ),
        "benchmark_provenance": benchmark_provenance,
        "completion_code_revision": completion_revision,
        "quality_evidence_eligible_for_decision": (
            provenance_valid and embedding_eligible
        ),
    }
    write_json(manifest_path, manifest)
    return verify_onnx_int8_artifact(root=root, model=model, runtime=runtime)


def _build_arctic_onnx_int8_artifact(
    *,
    root: Path,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
    benchmark_provenance: dict[str, Any],
) -> dict[str, Any]:
    artifact_directory = onnx_artifact_directory(root, model, runtime)
    manifest_path = onnx_manifest_path(root, model, runtime)
    if artifact_directory.exists() or manifest_path.exists():
        return _refresh_existing_onnx_manifest_attestation(
            root=root,
            model=model,
            runtime=runtime,
            benchmark_provenance=benchmark_provenance,
        )
    source_snapshot = model_snapshot_directory(root, model)
    source_model_sha256 = verify_registered_model_snapshot(
        root=root,
        model=model,
        snapshot=source_snapshot,
    )
    _verify_document_model_source(root, model, source_model_sha256)
    artifact_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{model.name}-onnx-",
        dir=artifact_directory.parent,
    ) as temporary:
        output_directory = Path(temporary) / "output"
        onnx_directory = output_directory / "onnx"
        onnx_directory.mkdir(parents=True)
        fp32_path = onnx_directory / "model.onnx"
        quantized_path = output_directory / onnx_quantized_filename(runtime)
        backend = create_bulk_backend(root=root, model=model, device="cpu")
        encoder, tokenizer, torch = backend.onnx_export_components()

        def initialize_wrapper(wrapper: Any) -> None:
            torch.nn.Module.__init__(wrapper)
            wrapper.encoder = encoder

        def forward_wrapper(
            wrapper: Any,
            input_ids: Any,
            attention_mask: Any,
        ) -> Any:
            hidden_state = wrapper.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                unpad_inputs=False,
                return_dict=False,
            )[0]
            return torch.nn.functional.normalize(
                hidden_state[:, 0, : model.dims],
                p=2,
                dim=1,
            )

        export_texts = [
            model.render_query("trail running shoe"),
            model.render_query("solid walnut writing desk with drawers"),
        ]
        tokens = tokenizer(
            export_texts,
            padding=True,
            truncation=True,
            max_length=model.max_seq_length,
            return_tensors="pt",
        )
        wrapper_class = type(
            "ArcticQueryEncoder",
            (torch.nn.Module,),
            {"__init__": initialize_wrapper, "forward": forward_wrapper},
        )
        wrapper = wrapper_class().eval()
        torch.onnx.export(
            wrapper,
            (tokens["input_ids"], tokens["attention_mask"]),
            str(fp32_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["sentence_embedding"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "sentence_embedding": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
        del wrapper
        del backend
        release_device_memory()
        quantization = importlib.import_module("onnxruntime.quantization")
        quantization.quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(quantized_path),
            weight_type=quantization.QuantType.QInt8,
            per_channel=False,
        )
        graph_facts = inspect_int8_onnx_graph(quantized_path)
        output_directory.replace(artifact_directory)

    if verify_registered_model_snapshot(
        root=root,
        model=model,
        snapshot=source_snapshot,
    ) != source_model_sha256:
        raise DatasetIntegrityError(f"ONNX export mutated source snapshot for {model.name}")
    fp32_path = artifact_directory / "onnx/model.onnx"
    quantized_path = artifact_directory / onnx_quantized_filename(runtime)
    fp32_facts = file_facts(fp32_path)
    quantized_facts = file_facts(quantized_path)
    completion_revision = asdict(collect_code_revision(root))
    provenance_valid = _recorded_provenance_valid(
        root,
        benchmark_provenance,
        completion_revision,
        require_current=True,
    )
    embedding_eligible = _embedding_manifest_eligible(root, model)
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "model": asdict(model),
        "runtime": runtime_artifact_values(runtime),
        "source_model_sha256": source_model_sha256,
        "artifact_directory": str(artifact_directory.relative_to(root)),
        "artifact_sha256": hash_model_snapshot(artifact_directory),
        "fp32_file": {
            "path": str(fp32_path.relative_to(root)),
            "sha256": fp32_facts.sha256,
            "bytes": fp32_facts.bytes,
        },
        "quantized_file": {
            "path": str(quantized_path.relative_to(root)),
            "sha256": quantized_facts.sha256,
            "bytes": quantized_facts.bytes,
        },
        "graph": graph_facts,
        "query_adapter": query_adapter_name(model),
        "export_packages": _export_package_versions(),
        "runtime_registry": _artifact(root, root / "config/query_runtime.toml"),
        "model_registry": _artifact(root, root / "config/models.toml"),
        "model_artifact_registry": _artifact(
            root, root / "config/model_artifacts.toml"
        ),
        "embedding_manifest": _artifact(
            root,
            root / f"results/wands/embeddings/{model.name}.manifest.json",
        ),
        "benchmark_provenance": benchmark_provenance,
        "completion_code_revision": completion_revision,
        "quality_evidence_eligible_for_decision": (
            provenance_valid and embedding_eligible
        ),
    }
    write_json(manifest_path, manifest)
    return verify_onnx_int8_artifact(root=root, model=model, runtime=runtime)


def _refresh_existing_onnx_manifest_attestation(
    *,
    root: Path,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
    benchmark_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    artifact_directory = onnx_artifact_directory(root, model, runtime)
    manifest_path = onnx_manifest_path(root, model, runtime)
    if not artifact_directory.is_dir() or not manifest_path.is_file():
        raise DatasetIntegrityError(f"ONNX int8 artifact is incomplete for {model.name}")
    manifest_value = read_json(manifest_path)
    if not isinstance(manifest_value, dict):
        raise DatasetIntegrityError(
            f"existing ONNX artifact manifest is invalid for {model.name}"
        )
    manifest = cast(dict[str, Any], manifest_value)
    if (
        manifest.get("schema_version") != 2
        or manifest.get("model") != asdict(model)
        or manifest.get("runtime") != runtime_artifact_values(runtime)
        or manifest.get("query_adapter") != query_adapter_name(model)
        or manifest.get("artifact_directory")
        != str(artifact_directory.relative_to(root))
        or manifest.get("export_packages") != _export_package_versions()
    ):
        raise DatasetIntegrityError(
            f"existing ONNX artifact registration differs for {model.name}"
        )
    source_model_sha256 = verify_registered_model_snapshot(root=root, model=model)
    if manifest.get("source_model_sha256") != source_model_sha256:
        raise DatasetIntegrityError(f"ONNX source model differs for {model.name}")
    _verify_document_model_source(root, model, source_model_sha256)
    if manifest.get("artifact_sha256") != hash_model_snapshot(artifact_directory):
        raise DatasetIntegrityError(f"ONNX artifact files differ for {model.name}")
    fp32_path = artifact_directory / "onnx/model.onnx"
    quantized_path = artifact_directory / onnx_quantized_filename(runtime)
    fp32_facts = file_facts(fp32_path)
    quantized_facts = file_facts(quantized_path)
    expected_fp32 = {
        "path": str(fp32_path.relative_to(root)),
        "sha256": fp32_facts.sha256,
        "bytes": fp32_facts.bytes,
    }
    expected_quantized = {
        "path": str(quantized_path.relative_to(root)),
        "sha256": quantized_facts.sha256,
        "bytes": quantized_facts.bytes,
    }
    if (
        manifest.get("fp32_file") != expected_fp32
        or manifest.get("quantized_file") != expected_quantized
        or manifest.get("graph") != inspect_int8_onnx_graph(quantized_path)
    ):
        raise DatasetIntegrityError(f"existing ONNX artifact files differ for {model.name}")
    completion_revision = asdict(collect_code_revision(root))
    provenance_valid = _recorded_provenance_valid(
        root,
        benchmark_provenance,
        completion_revision,
        require_current=True,
    )
    refreshed = dict(manifest)
    refreshed.update(
        {
            "runtime_registry": _artifact(
                root, root / "config/query_runtime.toml"
            ),
            "model_registry": _artifact(root, root / "config/models.toml"),
            "model_artifact_registry": _artifact(
                root, root / "config/model_artifacts.toml"
            ),
            "embedding_manifest": _artifact(
                root,
                root / f"results/wands/embeddings/{model.name}.manifest.json",
            ),
            "benchmark_provenance": _json_copy(benchmark_provenance),
            "completion_code_revision": completion_revision,
            "quality_evidence_eligible_for_decision": (
                provenance_valid and _embedding_manifest_eligible(root, model)
            ),
        }
    )
    write_json(manifest_path, refreshed)
    return verify_onnx_int8_artifact(root=root, model=model, runtime=runtime)


def _quarantine_legacy_onnx_artifact(
    *,
    root: Path,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
) -> None:
    artifact_directory = onnx_artifact_directory(root, model, runtime)
    manifest_path = onnx_manifest_path(root, model, runtime)
    if not artifact_directory.exists() and not manifest_path.exists():
        return
    if not artifact_directory.is_dir() or not manifest_path.is_file():
        raise DatasetIntegrityError(f"ONNX int8 artifact is incomplete for {model.name}")
    manifest = read_json(manifest_path)
    if isinstance(manifest, dict) and manifest.get("schema_version") == 2:
        return
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise DatasetIntegrityError(
            f"unsupported legacy ONNX manifest for {model.name}"
        )
    legacy_directory = artifact_directory.with_name(
        f"{artifact_directory.name}.legacy-schema1"
    )
    legacy_manifest = manifest_path.with_name(
        f"{manifest_path.stem}.legacy-schema1.json"
    )
    if legacy_directory.exists() or legacy_manifest.exists():
        raise DatasetIntegrityError(
            f"legacy ONNX quarantine already exists for {model.name}; review it before rebuilding"
        )
    artifact_directory.replace(legacy_directory)
    manifest_path.replace(legacy_manifest)


def verify_onnx_int8_artifact(
    *,
    root: Path,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
) -> dict[str, Any]:
    _verify_registered_runtime_inputs(root=root, model=model, runtime=runtime)
    artifact_directory = onnx_artifact_directory(root, model, runtime)
    manifest_path = onnx_manifest_path(root, model, runtime)
    if not artifact_directory.is_dir() or not manifest_path.is_file():
        raise DatasetIntegrityError(f"ONNX int8 artifact is incomplete for {model.name}")
    manifest = cast(dict[str, Any], read_json(manifest_path))
    if manifest.get("schema_version") != 2:
        raise DatasetIntegrityError(f"ONNX artifact schema differs for {model.name}")
    if manifest.get("model") != asdict(model):
        raise DatasetIntegrityError(f"ONNX artifact model differs for {model.name}")
    recorded_adapter = manifest.get("query_adapter")
    if recorded_adapter != query_adapter_name(model):
        raise DatasetIntegrityError(f"ONNX query adapter differs for {model.name}")
    if manifest.get("runtime") != runtime_artifact_values(runtime):
        raise DatasetIntegrityError(f"ONNX runtime config differs for {model.name}")
    if manifest.get("artifact_directory") != str(artifact_directory.relative_to(root)):
        raise DatasetIntegrityError(f"ONNX artifact path differs for {model.name}")
    if manifest.get("export_packages") != _export_package_versions():
        raise DatasetIntegrityError(f"ONNX export packages differ for {model.name}")
    source_model_sha256 = verify_registered_model_snapshot(
        root=root,
        model=model,
    )
    if manifest.get("source_model_sha256") != source_model_sha256:
        raise DatasetIntegrityError(f"ONNX source model differs for {model.name}")
    _verify_document_model_source(root, model, source_model_sha256)
    if manifest.get("runtime_registry") != _artifact(
        root, root / "config/query_runtime.toml"
    ):
        raise DatasetIntegrityError(f"ONNX runtime registry differs for {model.name}")
    if manifest.get("model_registry") != _artifact(root, root / "config/models.toml"):
        raise DatasetIntegrityError(f"ONNX model registry differs for {model.name}")
    if manifest.get("model_artifact_registry") != _artifact(
        root, root / "config/model_artifacts.toml"
    ):
        raise DatasetIntegrityError(
            f"ONNX model artifact registry differs for {model.name}"
        )
    embedding_manifest_path = (
        root / f"results/wands/embeddings/{model.name}.manifest.json"
    )
    if manifest.get("embedding_manifest") != _artifact(root, embedding_manifest_path):
        raise DatasetIntegrityError(f"ONNX embedding binding differs for {model.name}")
    if manifest.get("artifact_sha256") != hash_model_snapshot(artifact_directory):
        raise DatasetIntegrityError(f"ONNX artifact files differ for {model.name}")
    quantized_path = artifact_directory / onnx_quantized_filename(runtime)
    quantized_facts = file_facts(quantized_path)
    expected_quantized = cast(dict[str, Any], manifest.get("quantized_file"))
    if expected_quantized != {
        "path": str(quantized_path.relative_to(root)),
        "sha256": quantized_facts.sha256,
        "bytes": quantized_facts.bytes,
    }:
        raise DatasetIntegrityError(f"quantized ONNX file differs for {model.name}")
    if manifest.get("graph") != inspect_int8_onnx_graph(quantized_path):
        raise DatasetIntegrityError(f"quantized ONNX graph differs for {model.name}")
    fp32_path = artifact_directory / "onnx/model.onnx"
    fp32_facts = file_facts(fp32_path)
    if manifest.get("fp32_file") != {
        "path": str(fp32_path.relative_to(root)),
        "sha256": fp32_facts.sha256,
        "bytes": fp32_facts.bytes,
    }:
        raise DatasetIntegrityError(f"fp32 ONNX file differs for {model.name}")
    provenance_valid = _recorded_provenance_valid(
        root,
        manifest.get("benchmark_provenance"),
        manifest.get("completion_code_revision"),
    )
    expected_eligible = provenance_valid and _embedding_manifest_eligible(root, model)
    if manifest.get("quality_evidence_eligible_for_decision") is not expected_eligible:
        raise DatasetIntegrityError(f"ONNX artifact eligibility differs for {model.name}")
    return manifest


def inspect_int8_onnx_graph(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise DatasetIntegrityError(f"ONNX artifact does not exist: {path}")
    onnx = importlib.import_module("onnx")
    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model)
    operator_counts: dict[str, int] = {}
    for node in model.graph.node:
        operator_counts[node.op_type] = operator_counts.get(node.op_type, 0) + 1
    quantized_operators = {
        name: operator_counts[name]
        for name in sorted(_INT8_OPERATOR_TYPES)
        if name in operator_counts
    }
    if not quantized_operators:
        raise DatasetIntegrityError(f"ONNX graph has no int8 quantization operators: {path}")
    integer_initializers = sum(
        initializer.data_type in {onnx.TensorProto.INT8, onnx.TensorProto.UINT8}
        for initializer in model.graph.initializer
    )
    if integer_initializers == 0:
        raise DatasetIntegrityError(f"ONNX graph has no int8 weight initializers: {path}")
    return {
        "operator_counts": quantized_operators,
        "integer_initializers": integer_initializers,
        "opset_imports": {
            item.domain or "ai.onnx": item.version for item in model.opset_import
        },
    }


def assert_query_runtime_parity(
    comparison: dict[str, float],
    runtime: QueryRuntimeSpec,
) -> None:
    if comparison["minimum_cosine_similarity"] < runtime.minimum_query_cosine_similarity:
        raise DatasetIntegrityError(
            "int8 query minimum cosine parity failed: "
            f"{comparison['minimum_cosine_similarity']:.9f} < "
            f"{runtime.minimum_query_cosine_similarity:.9f}"
        )
    if comparison["mean_cosine_similarity"] < runtime.minimum_mean_query_cosine_similarity:
        raise DatasetIntegrityError(
            "int8 query mean cosine parity failed: "
            f"{comparison['mean_cosine_similarity']:.9f} < "
            f"{runtime.minimum_mean_query_cosine_similarity:.9f}"
        )
    if comparison["maximum_absolute_delta"] > runtime.maximum_absolute_query_delta:
        raise DatasetIntegrityError(
            "int8 query maximum absolute parity failed: "
            f"{comparison['maximum_absolute_delta']:.9f} > "
            f"{runtime.maximum_absolute_query_delta:.9f}"
        )


def assert_document_runtime_parity(
    comparison: dict[str, float],
    runtime: QueryRuntimeSpec,
) -> None:
    if (
        comparison["minimum_cosine_similarity"]
        < runtime.minimum_document_drift_cosine_similarity
    ):
        raise DatasetIntegrityError(
            "document runtime cosine parity failed: "
            f"{comparison['minimum_cosine_similarity']:.9f} < "
            f"{runtime.minimum_document_drift_cosine_similarity:.9f}"
        )


def derive_query_parity_status(
    artifact: dict[str, Any],
    runtime: QueryRuntimeSpec,
) -> tuple[str, str | None]:
    """Recompute the diagnostic parity outcome from recorded measurements."""
    query_comparison = artifact.get("query_comparison")
    document_comparison = artifact.get("document_runtime_comparison")
    if not isinstance(query_comparison, dict) or not isinstance(
        document_comparison, dict
    ):
        raise DatasetIntegrityError("query parity artifact is missing comparisons")
    try:
        normalized_query = _comparison_values(query_comparison, "query")
        normalized_document = _comparison_values(document_comparison, "document")
        assert_query_runtime_parity(normalized_query, runtime)
        assert_document_runtime_parity(normalized_document, runtime)
    except DatasetIntegrityError as error:
        return "failed", str(error)
    return "passed", None


def derive_self_retrieval_status(
    artifact: dict[str, Any],
    runtime: QueryRuntimeSpec,
) -> str:
    """Recompute self-retrieval status from counts and the registered threshold."""
    if artifact.get("sample_size") != runtime.self_retrieval_sample_size:
        raise DatasetIntegrityError("self-retrieval sample size differs from runtime")
    if artifact.get("minimum_top1_rate") != runtime.minimum_self_retrieval_top1:
        raise DatasetIntegrityError("self-retrieval threshold differs from runtime")
    result = artifact.get("int8")
    if not isinstance(result, dict):
        raise DatasetIntegrityError("self-retrieval artifact is missing int8 results")
    queries = result.get("queries")
    hits = result.get("top1_hits")
    rate = result.get("top1_rate")
    failures = result.get("failures")
    if (
        not isinstance(queries, int)
        or isinstance(queries, bool)
        or queries != runtime.self_retrieval_sample_size
        or not isinstance(hits, int)
        or isinstance(hits, bool)
        or hits < 0
        or hits > queries
        or not isinstance(rate, int | float)
        or isinstance(rate, bool)
        or not isinstance(failures, list)
        or len(failures) != queries - hits
    ):
        raise DatasetIntegrityError("self-retrieval result counts are inconsistent")
    expected_rate = hits / queries
    if float(rate) != expected_rate:
        raise DatasetIntegrityError("self-retrieval top1 rate is inconsistent")
    for failure in failures:
        if (
            not isinstance(failure, dict)
            or not isinstance(failure.get("expected"), str)
            or not isinstance(failure.get("actual"), str)
        ):
            raise DatasetIntegrityError("self-retrieval failure record is invalid")
    return (
        "passed"
        if expected_rate >= runtime.minimum_self_retrieval_top1
        else "failed"
    )


def _replay_query_parity_measurements(
    *,
    root: Path,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
) -> dict[str, Any]:
    from poc.embedding_cache import load_cached_vectors, load_embedding_inputs

    queries = _registered_wands_queries(root)
    query_ids = sorted(queries, key=identifier_sort_key)
    if len(query_ids) != runtime.parity_sample_size:
        raise DatasetIntegrityError(
            f"query parity registered sample differs for {model.name}"
        )
    query_texts = [model.render_query(queries[query_id]) for query_id in query_ids]
    fp32_backend: Any = None
    onnx_fp32_backend: Any = None
    int8_backend: Any = None
    try:
        fp32_backend = create_bulk_backend(root=root, model=model, device="cpu")
        onnx_fp32_backend = create_fp32_onnx_query_backend(
            root=root,
            model=model,
            runtime=runtime,
        )
        int8_backend = create_int8_query_backend(
            root=root,
            model=model,
            runtime=runtime,
        )
        fp32_queries = _encode_runtime_batches(
            fp32_backend,
            query_texts,
            dimensions=model.dims,
            batch_size=32,
        )
        onnx_fp32_queries = _encode_runtime_batches(
            onnx_fp32_backend,
            query_texts,
            dimensions=model.dims,
            batch_size=32,
        )
        int8_queries = _encode_runtime_batches(
            int8_backend,
            query_texts,
            dimensions=model.dims,
            batch_size=runtime.batch_size,
        )
        products_path = root / "data/prepared/wands/products.jsonl"
        document_inputs = load_embedding_inputs(products_path, model)
        _, cached_document_vectors = load_cached_vectors(
            root=root,
            products_path=products_path,
            model=model,
        )
        sample_size = runtime.document_drift_sample_size
        if len(document_inputs) < sample_size:
            raise DatasetIntegrityError(
                f"document parity registered sample differs for {model.name}"
            )
        current_documents = _encode_runtime_batches(
            fp32_backend,
            [item.text for item in document_inputs[:sample_size]],
            dimensions=model.dims,
            batch_size=32,
        )
        artifact_manifest = verify_onnx_int8_artifact(
            root=root,
            model=model,
            runtime=runtime,
        )
        return {
            "query_comparison": compare_embedding_vectors(
                fp32_queries,
                int8_queries,
            ),
            "torch_to_onnx_fp32_comparison": compare_embedding_vectors(
                fp32_queries,
                onnx_fp32_queries,
            ),
            "onnx_fp32_to_int8_comparison": compare_embedding_vectors(
                onnx_fp32_queries,
                int8_queries,
            ),
            "worst_query": _worst_query_replay(
                query_ids,
                fp32_queries,
                int8_queries,
            ),
            "document_runtime_comparison": compare_embedding_vectors(
                cached_document_vectors[:sample_size],
                current_documents,
            ),
            "fp32_query_runtime": fp32_backend.runtime,
            "int8_query_runtime": int8_backend.runtime,
            "source_model_sha256": int8_backend.source_model_artifact_sha256,
            "int8_artifact_sha256": int8_backend.model_artifact_sha256,
            "int8_export_packages": artifact_manifest["export_packages"],
        }
    finally:
        del int8_backend
        del onnx_fp32_backend
        del fp32_backend
        release_device_memory()


def _replay_self_retrieval_results(
    *,
    root: Path,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
) -> dict[str, Any]:
    from poc.embedding_cache import load_cached_vectors

    products_path = root / "data/prepared/wands/products.jsonl"
    titles = load_registered_product_titles(products_path)
    sample_ids = select_self_retrieval_ids(
        list(titles),
        sample_size=runtime.self_retrieval_sample_size,
    )
    document_ids, document_vectors = load_cached_vectors(
        root=root,
        products_path=products_path,
        model=model,
    )
    texts = [model.render_query(titles[document_id]) for document_id in sample_ids]
    fp32_backend: Any = None
    int8_backend: Any = None
    try:
        fp32_backend = create_bulk_backend(root=root, model=model, device="cpu")
        int8_backend = create_int8_query_backend(
            root=root,
            model=model,
            runtime=runtime,
        )
        fp32_vectors = _encode_runtime_batches(
            fp32_backend,
            texts,
            dimensions=model.dims,
            batch_size=32,
        )
        int8_vectors = _encode_runtime_batches(
            int8_backend,
            texts,
            dimensions=model.dims,
            batch_size=runtime.batch_size,
        )
        artifact_manifest_path = onnx_manifest_path(root, model, runtime)
        artifact_manifest = verify_onnx_int8_artifact(
            root=root,
            model=model,
            runtime=runtime,
        )
        return {
            "fp32": top1_self_retrieval(
                query_document_ids=sample_ids,
                query_vectors=fp32_vectors,
                document_ids=document_ids,
                document_vectors=document_vectors,
            ),
            "int8": top1_self_retrieval(
                query_document_ids=sample_ids,
                query_vectors=int8_vectors,
                document_ids=document_ids,
                document_vectors=document_vectors,
            ),
            "int8_runtime": int8_backend.runtime,
            "int8_artifact_sha256": artifact_manifest["artifact_sha256"],
            "int8_manifest_sha256": file_facts(artifact_manifest_path).sha256,
        }
    finally:
        del int8_backend
        del fp32_backend
        release_device_memory()


def _encode_runtime_batches(
    backend: Any,
    texts: list[str],
    *,
    dimensions: int,
    batch_size: int,
) -> NDArray[np.float32]:
    batches: list[NDArray[np.float32]] = []
    for offset in range(0, len(texts), batch_size):
        values = np.asarray(
            backend.encode(texts[offset : offset + batch_size]),
            dtype=np.float32,
        )
        expected_rows = len(texts[offset : offset + batch_size])
        if values.shape != (expected_rows, dimensions) or not np.isfinite(values).all():
            raise DatasetIntegrityError("query runtime replay returned invalid vectors")
        batches.append(values)
    if not batches:
        raise DatasetIntegrityError("query runtime replay sample is empty")
    return np.concatenate(batches)


def _worst_query_replay(
    query_ids: list[str],
    reference: NDArray[np.float32],
    candidate: NDArray[np.float32],
) -> dict[str, object]:
    similarities = np.sum(reference * candidate, axis=1) / (
        np.linalg.norm(reference, axis=1) * np.linalg.norm(candidate, axis=1)
    )
    index = int(np.argmin(similarities))
    return {
        "query_id": query_ids[index],
        "cosine_similarity": float(similarities[index]),
        "maximum_absolute_delta": float(
            np.max(np.abs(reference[index] - candidate[index]))
        ),
    }


def _verify_query_parity_live_replay(
    *,
    root: Path,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
    artifact: Mapping[str, Any],
) -> None:
    replay = _replay_query_parity_measurements(
        root=root,
        model=model,
        runtime=runtime,
    )
    for key, expected in replay.items():
        recorded = artifact.get(key)
        if isinstance(expected, dict) and all(
            isinstance(value, float) for value in expected.values()
        ):
            if not _float_mapping_matches(recorded, expected):
                raise DatasetIntegrityError(
                    f"query parity live replay differs for {model.name}: {key}"
                )
        elif recorded != expected:
            raise DatasetIntegrityError(
                f"query parity live replay differs for {model.name}: {key}"
            )
    for key in ("fp32_query_seconds", "int8_query_seconds"):
        value = artifact.get(key)
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise DatasetIntegrityError(
                f"query parity recorded timing differs for {model.name}: {key}"
            )


def _verify_self_retrieval_live_replay(
    *,
    root: Path,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
    artifact: Mapping[str, Any],
) -> None:
    replay = _replay_self_retrieval_results(
        root=root,
        model=model,
        runtime=runtime,
    )
    for key, expected in replay.items():
        if artifact.get(key) != expected:
            raise DatasetIntegrityError(
                f"self-retrieval live replay differs for {model.name}: {key}"
            )


def _float_mapping_matches(recorded: object, expected: Mapping[str, float]) -> bool:
    if not isinstance(recorded, Mapping) or set(recorded) != set(expected):
        return False
    return all(
        isinstance(recorded.get(key), int | float)
        and not isinstance(recorded.get(key), bool)
        and math.isclose(
            float(cast(int | float, recorded[key])),
            value,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        for key, value in expected.items()
    )


def verify_query_parity_artifact(
    *,
    root: Path,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
) -> dict[str, Any]:
    _verify_registered_runtime_inputs(root=root, model=model, runtime=runtime)
    common = _query_runtime_source_evidence(root, model, runtime)
    path = root / f"results/wands/query-runtime/{model.name}.parity.json"
    artifact = cast(dict[str, Any], read_json(path))
    if artifact.get("schema_version") != 2:
        raise DatasetIntegrityError(f"query parity schema differs for {model.name}")
    if artifact.get("model") != asdict(model) or artifact.get("runtime") != asdict(runtime):
        raise DatasetIntegrityError(f"query parity registration differs for {model.name}")
    query_ids = _registered_wands_query_ids(root)
    if (
        artifact.get("query_sample_size") != len(query_ids)
        or len(query_ids) != runtime.parity_sample_size
        or artifact.get("query_ids_sha256") != canonical_sha256(query_ids)
    ):
        raise DatasetIntegrityError(f"query parity sample differs for {model.name}")
    if artifact.get("document_drift_sample_size") != runtime.document_drift_sample_size:
        raise DatasetIntegrityError(f"document parity sample differs for {model.name}")
    if artifact.get("source_evidence") != common:
        raise DatasetIntegrityError(f"query parity sources differ for {model.name}")
    _verify_query_parity_live_replay(
        root=root,
        model=model,
        runtime=runtime,
        artifact=artifact,
    )
    status, error = derive_query_parity_status(artifact, runtime)
    if artifact.get("status") != status or artifact.get("error") != error:
        raise DatasetIntegrityError(f"query parity status is not derived for {model.name}")
    eligible = _query_artifact_provenance_eligible(root, artifact, common)
    if artifact.get("quality_evidence_eligible_for_decision") is not eligible:
        raise DatasetIntegrityError(f"query parity eligibility differs for {model.name}")
    return artifact


def verify_self_retrieval_artifact(
    *,
    root: Path,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
) -> dict[str, Any]:
    _verify_registered_runtime_inputs(root=root, model=model, runtime=runtime)
    common = _query_runtime_source_evidence(root, model, runtime)
    path = root / f"results/wands/query-runtime/{model.name}.self-retrieval.json"
    artifact = cast(dict[str, Any], read_json(path))
    if artifact.get("schema_version") != 2:
        raise DatasetIntegrityError(f"self-retrieval schema differs for {model.name}")
    if artifact.get("model") != asdict(model) or artifact.get("runtime") != asdict(runtime):
        raise DatasetIntegrityError(f"self-retrieval registration differs for {model.name}")
    titles = load_registered_product_titles(root / "data/prepared/wands/products.jsonl")
    sample_ids = select_self_retrieval_ids(
        list(titles),
        sample_size=runtime.self_retrieval_sample_size,
    )
    if artifact.get("sample_ids_sha256") != canonical_sha256(sample_ids):
        raise DatasetIntegrityError(f"self-retrieval sample differs for {model.name}")
    if artifact.get("source_evidence") != common:
        raise DatasetIntegrityError(f"self-retrieval sources differ for {model.name}")
    if (
        artifact.get("query_recipe")
        != "registered_query_template_applied_to_product_title"
        or artifact.get("document_recipe")
        != "registered_full_document_template"
        or artifact.get("sample_method") != "sha256_product_id"
    ):
        raise DatasetIntegrityError(
            f"self-retrieval recipe differs for {model.name}"
        )
    _verify_self_retrieval_live_replay(
        root=root,
        model=model,
        runtime=runtime,
        artifact=artifact,
    )
    _validate_self_retrieval_result(
        artifact.get("fp32"),
        expected_queries=runtime.self_retrieval_sample_size,
        label="fp32",
    )
    status = derive_self_retrieval_status(artifact, runtime)
    if artifact.get("status") != status:
        raise DatasetIntegrityError(f"self-retrieval status is not derived for {model.name}")
    eligible = _query_artifact_provenance_eligible(root, artifact, common)
    if artifact.get("quality_evidence_eligible_for_decision") is not eligible:
        raise DatasetIntegrityError(f"self-retrieval eligibility differs for {model.name}")
    return artifact


def finalize_query_runtime_artifact(
    *,
    root: Path,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
    measurements: Mapping[str, Any],
    benchmark_provenance: Mapping[str, Any],
    artifact_type: str,
) -> dict[str, Any]:
    if artifact_type not in {"parity", "self_retrieval"}:
        raise ValueError("unsupported query-runtime artifact type")
    common = _query_runtime_source_evidence(root, model, runtime)
    completion_revision = asdict(collect_code_revision(root))
    output = cast(dict[str, Any], json.loads(json.dumps(dict(measurements))))
    output.update(
        {
            "schema_version": 2,
            "model": asdict(model),
            "runtime": asdict(runtime),
            "artifact_type": artifact_type,
            "source_evidence": common,
            "benchmark_provenance": cast(
                dict[str, Any], json.loads(json.dumps(dict(benchmark_provenance)))
            ),
            "completion_code_revision": completion_revision,
        }
    )
    output["quality_evidence_eligible_for_decision"] = (
        _query_artifact_provenance_eligible(root, output, common)
    )
    return output


def finalize_query_encoding_latency_artifact(
    *,
    root: Path,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
    query_ids: list[str],
    fp32_measurement: Mapping[str, Any],
    int8_measurement: Mapping[str, Any],
    benchmark_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    environment_path = root / "results/environment/benchmark-profile.json"
    environment = cast(dict[str, Any], read_json(environment_path))
    completion_revision = asdict(collect_code_revision(root))
    provenance_valid = _recorded_provenance_valid(
        root,
        benchmark_provenance,
        completion_revision,
    )
    architecture_eligible = _latency_environment_eligible(environment)
    fresh_measurement_verified = True
    fp32_summary = fp32_measurement.get("summary")
    int8_summary = int8_measurement.get("summary")
    if not isinstance(fp32_summary, Mapping) or not isinstance(int8_summary, Mapping):
        raise DatasetIntegrityError("query latency summaries are invalid")
    fp32_p95 = float(fp32_summary["p95_ms"])
    int8_p95 = float(int8_summary["p95_ms"])
    return {
        "schema_version": 2,
        "model": asdict(model),
        "runtime": asdict(runtime),
        "measurement_scope": "query_encoding_only",
        "query_ids_sha256": canonical_sha256(query_ids),
        "query_count": len(query_ids),
        "query_source": "WANDS dev and test splits",
        "fresh_measurement_verification": {
            "method": "live_registered_runtime_p50_p95",
            "relative_tolerance": QUERY_LATENCY_REPLAY_RELATIVE_TOLERANCE,
            "absolute_tolerance_ms": QUERY_LATENCY_REPLAY_ABSOLUTE_TOLERANCE_MS,
        },
        "source_evidence": _query_runtime_source_evidence(root, model, runtime),
        "host_architecture": platform.machine(),
        "benchmark_environment": _artifact(root, environment_path),
        "environment_verification": environment,
        "benchmark_provenance": _json_copy(benchmark_provenance),
        "completion_code_revision": completion_revision,
        "latency_decision_eligible": (
            provenance_valid
            and architecture_eligible
            and fresh_measurement_verified
        ),
        "ineligibility_reason": (
            None
            if provenance_valid
            and architecture_eligible
            and fresh_measurement_verified
            else _query_latency_ineligibility_reason(
                provenance_valid=provenance_valid,
                architecture_eligible=architecture_eligible,
                fresh_measurement_verified=fresh_measurement_verified,
            )
        ),
        "fp32_onnx": _json_copy(fp32_measurement),
        "dynamic_int8_onnx": _json_copy(int8_measurement),
        "p95_speedup_ratio_fp32_over_int8": fp32_p95 / int8_p95,
    }


def verify_query_encoding_latency_artifact(
    *,
    root: Path,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
) -> dict[str, Any]:
    path = root / f"results/wands/query-runtime/{model.name}.batch1-latency.json"
    artifact = cast(dict[str, Any], read_json(path))
    if (
        artifact.get("schema_version") != 2
        or artifact.get("model") != asdict(model)
        or artifact.get("runtime") != asdict(runtime)
        or artifact.get("measurement_scope") != "query_encoding_only"
        or artifact.get("query_source") != "WANDS dev and test splits"
        or artifact.get("fresh_measurement_verification")
        != {
            "method": "live_registered_runtime_p50_p95",
            "relative_tolerance": QUERY_LATENCY_REPLAY_RELATIVE_TOLERANCE,
            "absolute_tolerance_ms": QUERY_LATENCY_REPLAY_ABSOLUTE_TOLERANCE_MS,
        }
    ):
        raise DatasetIntegrityError(f"query latency metadata differs for {model.name}")
    query_ids = _registered_wands_query_ids(root)
    if (
        artifact.get("query_count") != len(query_ids)
        or artifact.get("query_ids_sha256") != canonical_sha256(query_ids)
    ):
        raise DatasetIntegrityError(f"query latency sample differs for {model.name}")
    source_evidence = _query_runtime_source_evidence(root, model, runtime)
    if artifact.get("source_evidence") != source_evidence:
        raise DatasetIntegrityError(f"query latency sources differ for {model.name}")
    environment_path = root / "results/environment/benchmark-profile.json"
    environment = cast(dict[str, Any], read_json(environment_path))
    if (
        artifact.get("benchmark_environment") != _artifact(root, environment_path)
        or artifact.get("environment_verification") != environment
        or artifact.get("host_architecture") != platform.machine()
    ):
        raise DatasetIntegrityError(f"query latency environment differs for {model.name}")
    onnx = verify_onnx_int8_artifact(root=root, model=model, runtime=runtime)
    measurements = {
        "fp32_onnx": artifact.get("fp32_onnx"),
        "dynamic_int8_onnx": artifact.get("dynamic_int8_onnx"),
    }
    for name, value in measurements.items():
        if not isinstance(value, dict):
            raise DatasetIntegrityError(f"query latency {name} samples are missing")
        samples = value.get("samples_ms")
        if (
            value.get("warmup_calls") != 5
            or value.get("batch_size") != 1
            or value.get("distinct_queries") != len(query_ids)
            or not isinstance(value.get("runtime"), str)
            or not value.get("runtime")
            or not isinstance(samples, list)
            or len(samples) != len(query_ids)
            or any(
                not isinstance(sample, int | float)
                or isinstance(sample, bool)
                or float(sample) <= 0
                for sample in samples
            )
            or value.get("summary")
            != summarize_latency_ms([float(sample) for sample in samples])
        ):
            raise DatasetIntegrityError(f"query latency {name} does not reproduce")
    int8 = cast(dict[str, Any], measurements["dynamic_int8_onnx"])
    if int8.get("artifact_sha256") != onnx.get("artifact_sha256"):
        raise DatasetIntegrityError(f"query latency int8 artifact differs for {model.name}")
    fp32 = cast(dict[str, Any], measurements["fp32_onnx"])
    fp32_path = onnx_artifact_directory(root, model, runtime) / "onnx/model.onnx"
    if fp32.get("artifact_sha256") != file_facts(fp32_path).sha256:
        raise DatasetIntegrityError(f"query latency fp32 artifact differs for {model.name}")
    fp32_p95 = float(cast(dict[str, Any], fp32["summary"])["p95_ms"])
    int8_p95 = float(cast(dict[str, Any], int8["summary"])["p95_ms"])
    if artifact.get("p95_speedup_ratio_fp32_over_int8") != fp32_p95 / int8_p95:
        raise DatasetIntegrityError(f"query latency speedup differs for {model.name}")
    provenance_valid = _recorded_provenance_valid(
        root,
        artifact.get("benchmark_provenance"),
        artifact.get("completion_code_revision"),
    )
    architecture_eligible = _latency_environment_eligible(environment)
    fresh_measurement_verified = False
    if provenance_valid and architecture_eligible:
        try:
            fresh_measurement_verified = _verify_fresh_query_latency_measurement(
                root=root,
                model=model,
                runtime=runtime,
                query_ids=query_ids,
                artifact=artifact,
            )
        except (DatasetIntegrityError, ImportError, OSError, RuntimeError, ValueError):
            fresh_measurement_verified = False
    eligible = (
        provenance_valid
        and architecture_eligible
        and fresh_measurement_verified
    )
    reason = (
        None
        if eligible
        else _query_latency_ineligibility_reason(
            provenance_valid=provenance_valid,
            architecture_eligible=architecture_eligible,
            fresh_measurement_verified=fresh_measurement_verified,
        )
    )
    if (
        artifact.get("latency_decision_eligible") is not eligible
        or artifact.get("ineligibility_reason") != reason
    ):
        raise DatasetIntegrityError(f"query latency eligibility differs for {model.name}")
    return artifact


def _verify_fresh_query_latency_measurement(
    *,
    root: Path,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
    query_ids: list[str],
    artifact: Mapping[str, Any],
) -> bool:
    queries = _registered_wands_queries(root)
    if query_ids != sorted(queries, key=identifier_sort_key):
        return False
    texts = [model.render_query(queries[query_id]) for query_id in query_ids]
    recorded_fp32 = artifact.get("fp32_onnx")
    recorded_int8 = artifact.get("dynamic_int8_onnx")
    if not isinstance(recorded_fp32, Mapping) or not isinstance(
        recorded_int8, Mapping
    ):
        return False
    fp32_backend: Any = None
    int8_backend: Any = None
    try:
        fp32_backend = create_fp32_onnx_query_backend(
            root=root,
            model=model,
            runtime=runtime,
        )
        int8_backend = create_int8_query_backend(
            root=root,
            model=model,
            runtime=runtime,
        )
        for backend, recorded in (
            (fp32_backend, recorded_fp32),
            (int8_backend, recorded_int8),
        ):
            for _ in range(5):
                _validate_fresh_latency_vector(
                    backend.encode([texts[0]]),
                    dimensions=model.dims,
                )
            samples: list[float] = []
            for text in texts:
                started = time.perf_counter_ns()
                _validate_fresh_latency_vector(
                    backend.encode([text]),
                    dimensions=model.dims,
                )
                samples.append((time.perf_counter_ns() - started) / 1_000_000)
            if len(samples) != len(query_ids) or any(
                not math.isfinite(sample) or sample <= 0.0 for sample in samples
            ):
                return False
            fresh_summary = summarize_latency_ms(samples)
            recorded_summary = recorded.get("summary")
            if not isinstance(recorded_summary, Mapping) or any(
                not _latency_statistic_matches(
                    recorded_summary.get(key),
                    fresh_summary[key],
                )
                for key in ("p50_ms", "p95_ms")
            ):
                return False
        return bool(
            recorded_fp32.get("runtime") == fp32_backend.runtime
            and recorded_fp32.get("artifact_sha256")
            == fp32_backend.model_artifact_sha256
            and recorded_int8.get("runtime") == int8_backend.runtime
            and recorded_int8.get("artifact_sha256")
            == int8_backend.model_artifact_sha256
        )
    finally:
        del int8_backend
        del fp32_backend
        release_device_memory()


def _latency_statistic_matches(recorded: object, fresh: object) -> bool:
    return (
        isinstance(recorded, int | float)
        and not isinstance(recorded, bool)
        and isinstance(fresh, int | float)
        and not isinstance(fresh, bool)
        and math.isclose(
            float(recorded),
            float(fresh),
            rel_tol=QUERY_LATENCY_REPLAY_RELATIVE_TOLERANCE,
            abs_tol=QUERY_LATENCY_REPLAY_ABSOLUTE_TOLERANCE_MS,
        )
    )


def _validate_fresh_latency_vector(value: Any, *, dimensions: int) -> None:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (1, dimensions) or not np.isfinite(vector).all():
        raise DatasetIntegrityError(
            "fresh query latency measurement returned invalid vectors"
        )


def _query_latency_ineligibility_reason(
    *,
    provenance_valid: bool,
    architecture_eligible: bool,
    fresh_measurement_verified: bool,
) -> str:
    reasons: list[str] = []
    if not provenance_valid:
        reasons.append("query latency has no valid clean committed provenance")
    if not architecture_eligible:
        reasons.append("host architecture is not registered for latency decisions")
    if not fresh_measurement_verified:
        reasons.append("fresh registered-runtime latency measurement is not verified")
    return "; ".join(reasons)


def _latency_environment_eligible(environment: Mapping[str, Any]) -> bool:
    opensearch = environment.get("opensearch")
    return (
        environment.get("schema_version") == 2
        and environment.get("limits_verified") is True
        and environment.get("latency_architecture_eligible") is True
        and environment.get("architecture") == "x86_64"
        and isinstance(environment.get("container_id"), str)
        and len(str(environment["container_id"])) >= 12
        and isinstance(environment.get("image"), str)
        and bool(environment.get("image"))
        and isinstance(environment.get("image_id"), str)
        and bool(environment.get("image_id"))
        and isinstance(opensearch, Mapping)
        and opensearch.get("version") == "3.8.0"
    )


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(json.dumps(dict(value))))


def load_registered_product_titles(path: Path) -> dict[str, str]:
    titles: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        values = cast(dict[str, Any], json.loads(line))
        product_id = values.get("product_id")
        title = values.get("title")
        if (
            not isinstance(product_id, str)
            or not product_id
            or not isinstance(title, str)
            or not title
            or product_id in titles
        ):
            raise DatasetIntegrityError(
                f"invalid self-retrieval product at {path}:{line_number}"
            )
        titles[product_id] = title
    if not titles:
        raise DatasetIntegrityError("self-retrieval products are empty")
    return titles


def _registered_wands_query_ids(root: Path) -> list[str]:
    return sorted(_registered_wands_queries(root), key=identifier_sort_key)


def _registered_wands_queries(root: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    for split in ("dev", "test"):
        queries.update(
            load_prepared_queries(
                root / f"data/prepared/wands/queries.{split}.jsonl",
                expected_split=split,
            )
        )
    return queries


def _query_runtime_source_evidence(
    root: Path,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
) -> dict[str, object]:
    from poc.embedding_cache import verify_embedding_cache

    products_path = root / "data/prepared/wands/products.jsonl"
    verify_embedding_cache(root=root, products_path=products_path, model=model)
    onnx_manifest = verify_onnx_int8_artifact(root=root, model=model, runtime=runtime)
    embedding_manifest_path = (
        root / f"results/wands/embeddings/{model.name}.manifest.json"
    )
    onnx_path = onnx_manifest_path(root, model, runtime)
    paths = {
        "model_registry": root / "config/models.toml",
        "model_artifact_registry": root / "config/model_artifacts.toml",
        "runtime_registry": root / "config/query_runtime.toml",
        "prepared_manifest": root / "data/prepared/wands/manifest.json",
        "products": products_path,
        "queries_dev": root / "data/prepared/wands/queries.dev.jsonl",
        "queries_test": root / "data/prepared/wands/queries.test.jsonl",
        "embedding_manifest": embedding_manifest_path,
        "onnx_manifest": onnx_path,
    }
    evidence: dict[str, object] = {
        name: _artifact(root, path) for name, path in sorted(paths.items())
    }
    embedding_manifest = cast(dict[str, Any], read_json(embedding_manifest_path))
    evidence["embedding_eligible"] = embedding_manifest.get(
        "quality_evidence_eligible_for_decision"
    ) is True
    evidence["onnx_eligible"] = onnx_manifest.get(
        "quality_evidence_eligible_for_decision"
    ) is True
    return evidence


def _query_artifact_provenance_eligible(
    root: Path,
    artifact: Mapping[str, Any],
    source_evidence: Mapping[str, object],
) -> bool:
    return (
        _recorded_provenance_valid(
            root,
            artifact.get("benchmark_provenance"),
            artifact.get("completion_code_revision"),
        )
        and source_evidence.get("embedding_eligible") is True
        and source_evidence.get("onnx_eligible") is True
    )


def _validate_self_retrieval_result(
    value: object,
    *,
    expected_queries: int,
    label: str,
) -> None:
    if not isinstance(value, dict):
        raise DatasetIntegrityError(f"self-retrieval {label} result is missing")
    queries = value.get("queries")
    hits = value.get("top1_hits")
    rate = value.get("top1_rate")
    failures = value.get("failures")
    if (
        queries != expected_queries
        or not isinstance(hits, int)
        or isinstance(hits, bool)
        or hits < 0
        or hits > expected_queries
        or not isinstance(rate, int | float)
        or isinstance(rate, bool)
        or float(rate) != hits / expected_queries
        or not isinstance(failures, list)
        or len(failures) != expected_queries - hits
    ):
        raise DatasetIntegrityError(
            f"self-retrieval {label} result counts are inconsistent"
        )


def _comparison_values(
    comparison: dict[str, Any],
    label: str,
) -> dict[str, float]:
    keys = (
        "minimum_cosine_similarity",
        "mean_cosine_similarity",
        "maximum_absolute_delta",
    )
    values: dict[str, float] = {}
    for key in keys:
        value = comparison.get(key)
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise DatasetIntegrityError(
                f"{label} runtime comparison {key} is invalid"
            )
        values[key] = float(value)
    return values


def _verify_document_model_source(root: Path, model: ModelSpec, source_sha256: str) -> None:
    embedding_manifest_path = root / f"results/wands/embeddings/{model.name}.manifest.json"
    if not embedding_manifest_path.is_file():
        raise DatasetIntegrityError(f"document embedding manifest is missing for {model.name}")
    embedding_manifest = cast(dict[str, Any], read_json(embedding_manifest_path))
    if embedding_manifest.get("model_artifact_sha256") != source_sha256:
        raise DatasetIntegrityError(
            f"query and document model sources differ for {model.name}"
        )
    from poc.embedding_cache import verify_embedding_cache

    verify_embedding_cache(
        root=root,
        products_path=root / "data/prepared/wands/products.jsonl",
        model=model,
    )


def _embedding_manifest_eligible(root: Path, model: ModelSpec) -> bool:
    path = root / f"results/wands/embeddings/{model.name}.manifest.json"
    manifest = read_json(path)
    return (
        isinstance(manifest, dict)
        and manifest.get("quality_evidence_eligible_for_decision") is True
    )


def _prepare_transformers_module_cache(root: Path) -> None:
    cache_root = root / "data/cache/models"
    modules = cache_root / "modules"
    huggingface_home = cache_root / "huggingface-home"
    modules.mkdir(parents=True, exist_ok=True)
    huggingface_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_MODULES_CACHE", str(modules))
    os.environ.setdefault("HF_HOME", str(huggingface_home))


def _prepare_onnx_source_copy(source_copy: Path, model: ModelSpec) -> None:
    if not model.trust_remote_code:
        return
    config_path = source_copy / "config.json"
    config = cast(dict[str, Any], read_json(config_path))
    config["use_memory_efficient_attention"] = model.use_memory_efficient_attention
    config["unpad_inputs"] = False
    write_json(config_path, config)


def runtime_artifact_values(runtime: QueryRuntimeSpec) -> dict[str, object]:
    return cast(dict[str, object], asdict(runtime))


def _verify_registered_runtime_inputs(
    *,
    root: Path,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
) -> None:
    from poc.config import load_model_registry

    registry = load_model_registry(root / "config/models.toml")
    if registry.get(model.name) != model:
        raise DatasetIntegrityError(
            f"query runtime model differs from the registered model: {model.name}"
        )
    if load_query_runtime_spec(root / "config/query_runtime.toml") != runtime:
        raise DatasetIntegrityError("query runtime differs from registered config")


def _capture_provenance(
    root: Path,
    supplied: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if supplied is not None:
        return cast(dict[str, Any], json.loads(json.dumps(dict(supplied))))
    return collect_manifest_provenance(root)


def _recorded_provenance_valid(
    root: Path,
    provenance: object,
    completion_revision: object,
    *,
    require_current: bool = False,
) -> bool:
    if not isinstance(provenance, Mapping) or not isinstance(
        completion_revision, Mapping
    ):
        return False
    start_revision = provenance.get("code_revision")
    if not isinstance(start_revision, Mapping) or dict(start_revision) != dict(
        completion_revision
    ):
        return False
    return verify_decision_provenance(
        provenance,
        root=root,
        profile_path=root / "config/benchmark.toml",
        environment_path=root / "results/environment/benchmark-profile.json",
        require_current_code_revision=require_current,
    )


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }


def _export_package_versions() -> dict[str, str]:
    return {
        package: importlib.metadata.version(package)
        for package in (
            "onnx",
            "onnxruntime",
            "optimum",
            "optimum-onnx",
            "sentence-transformers",
            "transformers",
        )
    }


def _required_string(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"query runtime {key} must be a non-empty string")
    return value


def _required_positive_int(values: dict[str, Any], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"query runtime {key} must be a positive integer")
    return value


def _required_positive_float(values: dict[str, Any], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"query runtime {key} must be positive")
    return float(value)


def _required_ratio(values: dict[str, Any], key: str) -> float:
    value = _required_positive_float(values, key)
    if value > 1:
        raise ConfigError(f"query runtime {key} must be at most 1")
    return value
