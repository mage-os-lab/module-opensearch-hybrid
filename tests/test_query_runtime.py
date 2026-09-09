from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime  # type: ignore[import-untyped]
import pytest
from onnx import TensorProto, helper

import poc.query_runtime as query_runtime_module
from poc.config import ConfigError, ModelSpec, load_model_registry
from poc.datasets import DatasetIntegrityError
from poc.embedding_cache import EmbeddingInput
from poc.latency import summarize_latency_ms
from poc.manifest import canonical_sha256
from poc.provenance import CodeRevision
from poc.query_runtime import (
    QueryRuntimeSpec,
    assert_document_runtime_parity,
    assert_query_runtime_parity,
    assert_registered_query_batch_size,
    create_onnx_session_options,
    inspect_int8_onnx_graph,
    load_query_runtime_spec,
    onnx_artifact_directory,
    onnx_quantized_filename,
    query_adapter_name,
    verify_query_parity_artifact,
    verify_self_retrieval_artifact,
)

ROOT = Path(__file__).resolve().parents[1]


def valid_runtime() -> dict[str, object]:
    return {
        "backend": "sentence_transformers_onnx",
        "precision": "int8_dynamic",
        "execution_provider": "CPUExecutionProvider",
        "quantization_config": "arm64",
        "parity_sample_size": 480,
        "minimum_query_cosine_similarity": 0.99,
        "minimum_mean_query_cosine_similarity": 0.999,
        "maximum_absolute_query_delta": 0.05,
        "document_drift_sample_size": 256,
        "minimum_document_drift_cosine_similarity": 0.9999,
        "self_retrieval_sample_size": 1000,
        "minimum_self_retrieval_top1": 0.8,
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1,
        "execution_mode": "ORT_SEQUENTIAL",
        "batch_size": 1,
    }


def test_registered_query_runtime_is_explicit_int8_onnx_cpu() -> None:
    runtime = load_query_runtime_spec(ROOT / "config/query_runtime.toml")

    assert runtime.backend == "sentence_transformers_onnx"
    assert runtime.precision == "int8_dynamic"
    assert runtime.execution_provider == "CPUExecutionProvider"
    assert runtime.quantization_config == "arm64"
    assert runtime.parity_sample_size == 480
    assert runtime.minimum_query_cosine_similarity == 0.99
    assert runtime.minimum_mean_query_cosine_similarity == 0.999
    assert runtime.maximum_absolute_query_delta == 0.05
    assert runtime.document_drift_sample_size == 256
    assert runtime.minimum_document_drift_cosine_similarity == 0.9999
    assert runtime.self_retrieval_sample_size == 1000
    assert runtime.minimum_self_retrieval_top1 == 0.8
    assert runtime.intra_op_num_threads == 1
    assert runtime.inter_op_num_threads == 1
    assert runtime.execution_mode == "ORT_SEQUENTIAL"
    assert runtime.batch_size == 1


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("backend", "torch"),
        ("precision", "float32"),
        ("execution_provider", "CoreMLExecutionProvider"),
        ("quantization_config", "moving_default"),
    ],
)
def test_non_registered_query_runtime_is_rejected(key: str, value: str) -> None:
    values = valid_runtime()
    values[key] = value

    with pytest.raises(ConfigError):
        QueryRuntimeSpec.from_mapping(values)


def test_onnx_artifact_path_binds_model_revision_and_quantization_target() -> None:
    model = load_model_registry(ROOT / "config/models.toml")["granite_embedding_english_r2"]
    runtime = QueryRuntimeSpec.from_mapping(valid_runtime())

    assert onnx_artifact_directory(ROOT, model, runtime) == (
        ROOT
        / "data/cache/models/onnx-int8/granite_embedding_english_r2"
        / model.revision
        / "arm64"
    )
    assert onnx_quantized_filename(runtime) == "onnx/model_qint8_arm64.onnx"


def test_float_only_onnx_graph_is_rejected_as_int8_artifact(tmp_path: Path) -> None:
    left = helper.make_tensor_value_info("left", TensorProto.FLOAT, [1])
    right = helper.make_tensor_value_info("right", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [helper.make_node("Add", ["left", "right"], ["output"])],
        "float-only",
        [left, right],
        [output],
    )
    path = tmp_path / "model.onnx"
    onnx.save(helper.make_model(graph), path)

    with pytest.raises(DatasetIntegrityError, match="no int8 quantization operators"):
        inspect_int8_onnx_graph(path)


def test_registered_query_parity_thresholds_are_enforced() -> None:
    runtime = QueryRuntimeSpec.from_mapping(valid_runtime())

    assert_query_runtime_parity(
        {
            "minimum_cosine_similarity": 0.995,
            "mean_cosine_similarity": 0.9995,
            "maximum_absolute_delta": 0.02,
        },
        runtime,
    )
    with pytest.raises(DatasetIntegrityError, match="minimum cosine"):
        assert_query_runtime_parity(
            {
                "minimum_cosine_similarity": 0.989,
                "mean_cosine_similarity": 0.9995,
                "maximum_absolute_delta": 0.02,
            },
            runtime,
        )


def test_document_runtime_drift_threshold_is_enforced() -> None:
    runtime = QueryRuntimeSpec.from_mapping(valid_runtime())

    assert_document_runtime_parity(
        {
            "minimum_cosine_similarity": 0.99995,
            "mean_cosine_similarity": 1.0,
            "maximum_absolute_delta": 0.0001,
        },
        runtime,
    )
    with pytest.raises(DatasetIntegrityError, match="document runtime"):
        assert_document_runtime_parity(
            {
                "minimum_cosine_similarity": 0.9998,
                "mean_cosine_similarity": 0.9999,
                "maximum_absolute_delta": 0.0001,
            },
            runtime,
        )


def test_registered_onnx_session_is_single_threaded_and_sequential() -> None:
    runtime = QueryRuntimeSpec.from_mapping(valid_runtime())

    options = create_onnx_session_options(runtime)

    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 1
    assert options.execution_mode == onnxruntime.ExecutionMode.ORT_SEQUENTIAL


def test_registered_int8_query_runtime_rejects_non_production_batch_size() -> None:
    runtime = QueryRuntimeSpec.from_mapping(valid_runtime())

    assert_registered_query_batch_size(1, runtime)
    with pytest.raises(DatasetIntegrityError, match="exceeds registered size 1"):
        assert_registered_query_batch_size(2, runtime)


def test_arctic_uses_explicit_cls_mrl_onnx_adapter() -> None:
    registry = load_model_registry(ROOT / "config/models.toml")

    assert query_adapter_name(registry["gte_modernbert_base"]) == "sentence_transformers"
    assert (
        query_adapter_name(registry["arctic_embed_m_v2"])
        == "transformers_cls_mrl_normalized"
    )


def test_existing_onnx_artifact_refreshes_attestation_on_second_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _semantic_model()
    runtime = _semantic_runtime()
    artifact_directory = onnx_artifact_directory(tmp_path, model, runtime)
    onnx_directory = artifact_directory / "onnx"
    onnx_directory.mkdir(parents=True)
    fp32_path = onnx_directory / "model.onnx"
    quantized_path = artifact_directory / onnx_quantized_filename(runtime)
    fp32_path.write_bytes(b"fp32 graph")
    quantized_path.write_bytes(b"int8 graph")
    manifest_path = query_runtime_module.onnx_manifest_path(
        tmp_path,
        model,
        runtime,
    )
    manifest_path.parent.mkdir(parents=True)
    for relative_path in (
        "config/models.toml",
        "config/model_artifacts.toml",
        "config/query_runtime.toml",
    ):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative_path}\n")
    embedding_manifest_path = (
        tmp_path / f"results/wands/embeddings/{model.name}.manifest.json"
    )
    embedding_manifest_path.parent.mkdir(parents=True)
    embedding_manifest_path.write_text(
        json.dumps(
            {
                "cache_hits": 2,
                "quality_evidence_eligible_for_decision": True,
            }
        )
        + "\n"
    )
    source_sha256 = "b" * 64
    artifact_sha256 = "c" * 64
    graph = {"integer_initializers": 1, "operator_counts": {"MatMulInteger": 1}}
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "model": asdict(model),
                "runtime": query_runtime_module.runtime_artifact_values(runtime),
                "source_model_sha256": source_sha256,
                "artifact_directory": str(artifact_directory.relative_to(tmp_path)),
                "artifact_sha256": artifact_sha256,
                "fp32_file": query_runtime_module._artifact(tmp_path, fp32_path),
                "quantized_file": query_runtime_module._artifact(
                    tmp_path,
                    quantized_path,
                ),
                "graph": graph,
                "query_adapter": query_adapter_name(model),
                "export_packages": {"fixture": "1"},
                "runtime_registry": {"stale": True},
                "model_registry": {"stale": True},
                "model_artifact_registry": {"stale": True},
                "embedding_manifest": {"stale": True},
                "benchmark_provenance": {"first_run": True},
                "completion_code_revision": {"first_run": True},
                "quality_evidence_eligible_for_decision": False,
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        query_runtime_module,
        "_verify_registered_runtime_inputs",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        query_runtime_module,
        "_prepare_transformers_module_cache",
        lambda _root: None,
    )
    monkeypatch.setattr(
        query_runtime_module,
        "_export_package_versions",
        lambda: {"fixture": "1"},
    )
    monkeypatch.setattr(
        query_runtime_module,
        "verify_registered_model_snapshot",
        lambda **_kwargs: source_sha256,
    )
    monkeypatch.setattr(
        query_runtime_module,
        "_verify_document_model_source",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        query_runtime_module,
        "hash_model_snapshot",
        lambda _path: artifact_sha256,
    )
    monkeypatch.setattr(
        query_runtime_module,
        "inspect_int8_onnx_graph",
        lambda _path: graph,
    )
    completion_revision = CodeRevision(
        git_commit="d" * 40,
        source_tree_sha256="e" * 64,
        source_dirty=False,
    )
    monkeypatch.setattr(
        query_runtime_module,
        "collect_code_revision",
        lambda _root: completion_revision,
    )
    monkeypatch.setattr(
        query_runtime_module,
        "_recorded_provenance_valid",
        lambda *_args, **_kwargs: True,
    )

    second_run_provenance = {"second_run": True}
    result = query_runtime_module.build_onnx_int8_artifact(
        root=tmp_path,
        model=model,
        runtime=runtime,
        generation_provenance=second_run_provenance,
    )

    assert result["embedding_manifest"] == query_runtime_module._artifact(
        tmp_path,
        embedding_manifest_path,
    )
    assert result["model_artifact_registry"] == query_runtime_module._artifact(
        tmp_path,
        tmp_path / "config/model_artifacts.toml",
    )
    assert result["benchmark_provenance"] == second_run_provenance
    assert result["completion_code_revision"] == asdict(completion_revision)
    assert result["quality_evidence_eligible_for_decision"] is True

    refreshed_manifest = manifest_path.read_bytes()
    quantized_path.write_bytes(b"tampered int8 graph")
    with pytest.raises(DatasetIntegrityError, match="artifact files differ"):
        query_runtime_module.build_onnx_int8_artifact(
            root=tmp_path,
            model=model,
            runtime=runtime,
            generation_provenance={"third_run": True},
        )
    assert manifest_path.read_bytes() == refreshed_manifest


class _FakeQueryBackend:
    device = "cpu"
    runtime = "fake-runtime"
    model_artifact_sha256 = "a" * 64
    source_model_artifact_sha256 = "b" * 64

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            if "one" in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


def _semantic_model() -> ModelSpec:
    return ModelSpec.from_mapping(
        "test_model",
        {
            "hf_id": "example/test-model",
            "revision": "a" * 40,
            "dims": 2,
            "normalize": True,
            "max_seq_length": 128,
            "bulk_dtype": "float32",
            "use_memory_efficient_attention": False,
            "trust_remote_code": False,
            "query_prefix": "",
            "document_prefix": "",
            "query_template": "{query}",
            "document_template": "{title}",
            "license": "Apache-2.0",
            "decision_eligible": True,
            "contamination": "none_known",
        },
    )


def _semantic_runtime() -> QueryRuntimeSpec:
    values = valid_runtime()
    values["parity_sample_size"] = 2
    values["document_drift_sample_size"] = 2
    values["self_retrieval_sample_size"] = 2
    return QueryRuntimeSpec.from_mapping(values)


def _patch_semantic_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeQueryBackend()
    monkeypatch.setattr(
        query_runtime_module,
        "_verify_registered_runtime_inputs",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        query_runtime_module,
        "_query_runtime_source_evidence",
        lambda *_args, **_kwargs: {
            "embedding_eligible": True,
            "onnx_eligible": True,
        },
    )
    monkeypatch.setattr(
        query_runtime_module,
        "_query_artifact_provenance_eligible",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        query_runtime_module,
        "create_bulk_backend",
        lambda **_kwargs: backend,
    )
    monkeypatch.setattr(
        query_runtime_module,
        "create_fp32_onnx_query_backend",
        lambda **_kwargs: backend,
    )
    monkeypatch.setattr(
        query_runtime_module,
        "create_int8_query_backend",
        lambda **_kwargs: backend,
    )
    monkeypatch.setattr(
        query_runtime_module,
        "release_device_memory",
        lambda: None,
    )
    monkeypatch.setattr(
        query_runtime_module,
        "verify_onnx_int8_artifact",
        lambda **_kwargs: {
            "artifact_sha256": "a" * 64,
            "export_packages": {"fixture": "1"},
        },
    )


def test_query_parity_verifier_replays_registered_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _semantic_model()
    runtime = _semantic_runtime()
    _patch_semantic_runtime(monkeypatch)
    monkeypatch.setattr(
        query_runtime_module,
        "_registered_wands_queries",
        lambda _root: {"1": "one", "2": "two"},
    )
    inputs = [
        EmbeddingInput("p1", "one", "k1"),
        EmbeddingInput("p2", "two", "k2"),
    ]
    monkeypatch.setattr(
        "poc.embedding_cache.load_embedding_inputs",
        lambda *_args, **_kwargs: inputs,
    )
    monkeypatch.setattr(
        "poc.embedding_cache.load_cached_vectors",
        lambda **_kwargs: (
            ["p1", "p2"],
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ),
    )
    replay = query_runtime_module._replay_query_parity_measurements(
        root=tmp_path,
        model=model,
        runtime=runtime,
    )
    output = tmp_path / f"results/wands/query-runtime/{model.name}.parity.json"
    output.parent.mkdir(parents=True)
    artifact = {
        "schema_version": 2,
        "model": asdict(model),
        "runtime": asdict(runtime),
        "query_ids_sha256": canonical_sha256(["1", "2"]),
        "query_sample_size": 2,
        "document_drift_sample_size": 2,
        "source_evidence": {
            "embedding_eligible": True,
            "onnx_eligible": True,
        },
        **replay,
        "status": "passed",
        "error": None,
        "quality_evidence_eligible_for_decision": False,
    }
    artifact["query_comparison"] = {
        "minimum_cosine_similarity": 0.999,
        "mean_cosine_similarity": 0.9995,
        "maximum_absolute_delta": 0.001,
    }
    output.write_text(json.dumps(artifact) + "\n")

    with pytest.raises(DatasetIntegrityError, match="live replay"):
        verify_query_parity_artifact(root=tmp_path, model=model, runtime=runtime)


def test_query_latency_fresh_verification_runs_registered_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _semantic_model()
    runtime = _semantic_runtime()

    class CountingBackend(_FakeQueryBackend):
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, texts: list[str]) -> np.ndarray:
            self.calls += 1
            return super().encode(texts)

    fp32 = CountingBackend()
    int8 = CountingBackend()
    monkeypatch.setattr(
        query_runtime_module,
        "_registered_wands_queries",
        lambda _root: {"1": "one", "2": "two"},
    )
    monkeypatch.setattr(
        query_runtime_module,
        "create_fp32_onnx_query_backend",
        lambda **_kwargs: fp32,
    )
    monkeypatch.setattr(
        query_runtime_module,
        "create_int8_query_backend",
        lambda **_kwargs: int8,
    )
    monkeypatch.setattr(query_runtime_module, "release_device_memory", lambda: None)
    clock = iter(range(0, 8_000_001, 1_000_000))
    monkeypatch.setattr(
        time,
        "perf_counter_ns",
        lambda: next(clock),
    )
    summary = summarize_latency_ms([1.0, 1.0])
    artifact = {
        "fp32_onnx": {
            "runtime": fp32.runtime,
            "artifact_sha256": fp32.model_artifact_sha256,
            "summary": summary,
        },
        "dynamic_int8_onnx": {
            "runtime": int8.runtime,
            "artifact_sha256": int8.model_artifact_sha256,
            "summary": summary,
        },
    }

    assert query_runtime_module._verify_fresh_query_latency_measurement(
        root=tmp_path,
        model=model,
        runtime=runtime,
        query_ids=["1", "2"],
        artifact=artifact,
    )
    assert fp32.calls == 7
    assert int8.calls == 7

    int8_record = artifact["dynamic_int8_onnx"]
    assert isinstance(int8_record, dict)
    int8_record["summary"] = summarize_latency_ms([0.1, 0.1])
    clock = iter(range(0, 8_000_001, 1_000_000))
    monkeypatch.setattr(
        time,
        "perf_counter_ns",
        lambda: next(clock),
    )
    assert not query_runtime_module._verify_fresh_query_latency_measurement(
        root=tmp_path,
        model=model,
        runtime=runtime,
        query_ids=["1", "2"],
        artifact=artifact,
    )

    int8_record["summary"] = summary
    int8_record["artifact_sha256"] = "0" * 64
    clock = iter(range(0, 8_000_001, 1_000_000))
    monkeypatch.setattr(
        time,
        "perf_counter_ns",
        lambda: next(clock),
    )
    assert not query_runtime_module._verify_fresh_query_latency_measurement(
        root=tmp_path,
        model=model,
        runtime=runtime,
        query_ids=["1", "2"],
        artifact=artifact,
    )


def test_self_retrieval_verifier_replays_current_fp32_and_int8_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _semantic_model()
    runtime = _semantic_runtime()
    _patch_semantic_runtime(monkeypatch)
    monkeypatch.setattr(
        query_runtime_module,
        "load_registered_product_titles",
        lambda _path: {"p1": "one", "p2": "two"},
    )
    monkeypatch.setattr(
        query_runtime_module,
        "select_self_retrieval_ids",
        lambda _ids, *, sample_size: ["p1", "p2"][:sample_size],
    )
    monkeypatch.setattr(
        "poc.embedding_cache.load_cached_vectors",
        lambda **_kwargs: (
            ["p1", "p2"],
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ),
    )
    onnx_manifest = tmp_path / f"results/wands/query-runtime/{model.name}.arm64.manifest.json"
    onnx_manifest.parent.mkdir(parents=True)
    onnx_manifest.write_text("{}\n")
    monkeypatch.setattr(
        query_runtime_module,
        "onnx_manifest_path",
        lambda *_args, **_kwargs: onnx_manifest,
    )
    replay = query_runtime_module._replay_self_retrieval_results(
        root=tmp_path,
        model=model,
        runtime=runtime,
    )
    output = tmp_path / f"results/wands/query-runtime/{model.name}.self-retrieval.json"
    artifact = {
        "schema_version": 2,
        "model": asdict(model),
        "runtime": asdict(runtime),
        "query_recipe": "registered_query_template_applied_to_product_title",
        "document_recipe": "registered_full_document_template",
        "sample_method": "sha256_product_id",
        "sample_ids_sha256": canonical_sha256(["p1", "p2"]),
        "sample_size": 2,
        "minimum_top1_rate": runtime.minimum_self_retrieval_top1,
        "source_evidence": {
            "embedding_eligible": True,
            "onnx_eligible": True,
        },
        **replay,
        "status": "passed",
        "quality_evidence_eligible_for_decision": False,
    }
    artifact["int8"] = {
        "queries": 2,
        "top1_hits": 1,
        "top1_rate": 0.5,
        "failures": [{"expected": "p2", "actual": "p1"}],
    }
    output.write_text(json.dumps(artifact) + "\n")

    with pytest.raises(DatasetIntegrityError, match="live replay"):
        verify_self_retrieval_artifact(root=tmp_path, model=model, runtime=runtime)
