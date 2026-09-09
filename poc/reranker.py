from __future__ import annotations

import importlib
import importlib.metadata
import json
import math
import platform
import time
import tomllib
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from poc.bm25 import load_prepared_queries, verify_wands_bm25_selection
from poc.config import (
    WANDS_INT8_MODEL_NAMES,
    ConfigError,
    ModelSpec,
    load_model_registry,
)
from poc.datasets import DatasetIntegrityError, file_facts
from poc.evaluation import Evaluation, evaluate_run
from poc.evaluation_crosscheck import cross_check_run
from poc.experiments import BM25Experiments, load_bm25_experiments
from poc.hybrid import verify_wands_hybrid_summary
from poc.indexing import (
    iter_prepared_products,
    load_wands_index_config,
    wands_completion_provenance_evidence,
    wands_recorded_provenance_valid,
)
from poc.latency import summarize_latency_ms
from poc.manifest import canonical_sha256, read_json, write_json
from poc.neural_sparse import sparse_document_text
from poc.os_client import OpenSearchClient
from poc.provenance import (
    collect_manifest_provenance,
    load_benchmark_profile,
    require_registered_opensearch_client,
)
from poc.query_runtime import QueryRuntimeSpec, load_query_runtime_spec
from poc.trec import RunRecord, identifier_sort_key, read_run, write_run

RERANKER_REPLAY_SAMPLE_SIZE = 256
RERANKER_REPLAY_ABSOLUTE_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class RerankerSpec:
    repo_id: str
    revision: str
    license: str
    max_length: int
    arm64_onnx_file: str
    x86_64_onnx_file: str
    arm64_runtime_artifacts_sha256: str
    x86_64_runtime_artifacts_sha256: str
    candidate_depth: int
    result_depth: int
    batch_size: int
    source_arms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WandsRerankerEvidenceContext:
    client: OpenSearchClient
    index: str
    models: tuple[ModelSpec, ...]
    runtime: QueryRuntimeSpec
    experiments: BM25Experiments


def _reranker_spec_payload(spec: RerankerSpec) -> dict[str, object]:
    payload = cast(dict[str, object], asdict(spec))
    payload["source_arms"] = list(spec.source_arms)
    return payload


class RerankerBackend(Protocol):
    runtime: str
    artifact_sha256: str

    def score(self, pairs: list[tuple[str, str]]) -> NDArray[np.float32]: ...


@dataclass(frozen=True, slots=True)
class RerankerRunResult:
    run_path: Path
    manifest_path: Path
    metrics_path: Path
    evaluation: Evaluation

    def selection_entry(self, root: Path) -> dict[str, object]:
        return {
            "run": str(self.run_path.relative_to(root)),
            "manifest": str(self.manifest_path.relative_to(root)),
            "metrics_file": str(self.metrics_path.relative_to(root)),
            "metrics": self.evaluation.metrics,
        }


def load_reranker_spec(path: Path) -> RerankerSpec:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    model = raw.get("model")
    experiment = raw.get("experiment")
    if (
        raw.get("schema_version") != 1
        or not isinstance(model, dict)
        or not isinstance(experiment, dict)
    ):
        raise ConfigError("reranker model or experiment configuration is missing")
    source_arms = experiment.get("source_arms")
    if (
        not isinstance(source_arms, list)
        or not source_arms
        or not all(isinstance(value, str) and value for value in source_arms)
        or len(set(source_arms)) != len(source_arms)
    ):
        raise ConfigError("reranker source arms must be unique non-empty strings")
    spec = RerankerSpec(
        repo_id=_required_string(model, "repo_id"),
        revision=_required_string(model, "revision"),
        license=_required_string(model, "license"),
        max_length=_required_positive_int(model, "max_length"),
        arm64_onnx_file=_required_string(model, "arm64_onnx_file"),
        x86_64_onnx_file=_required_string(model, "x86_64_onnx_file"),
        arm64_runtime_artifacts_sha256=_required_sha256(
            model, "arm64_runtime_artifacts_sha256"
        ),
        x86_64_runtime_artifacts_sha256=_required_sha256(
            model, "x86_64_runtime_artifacts_sha256"
        ),
        candidate_depth=_required_positive_int(experiment, "candidate_depth"),
        result_depth=_required_positive_int(experiment, "result_depth"),
        batch_size=_required_positive_int(experiment, "batch_size"),
        source_arms=tuple(source_arms),
    )
    if len(spec.revision) != 40 or any(
        character not in "0123456789abcdef" for character in spec.revision
    ):
        raise ConfigError("reranker revision must be a full lowercase Git SHA")
    if spec.license != "apache-2.0":
        raise ConfigError("reranker must use the verified Apache-2.0 license")
    if spec.candidate_depth > spec.result_depth:
        raise ConfigError("reranker candidate depth must not exceed result depth")
    if spec.batch_size > spec.candidate_depth:
        raise ConfigError("reranker batch size must not exceed candidate depth")
    return spec


def load_wands_reranker_evidence_context(
    *,
    root: Path,
    client: OpenSearchClient,
) -> WandsRerankerEvidenceContext:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    index = load_wands_index_config(root / "config/indexes.toml")
    registry = load_model_registry(root / "config/models.toml")
    return WandsRerankerEvidenceContext(
        client=client,
        index=index.name,
        models=tuple(registry[name] for name in WANDS_INT8_MODEL_NAMES),
        runtime=load_query_runtime_spec(root / "config/query_runtime.toml"),
        experiments=load_bm25_experiments(root / "config/experiments.toml"),
    )


def select_onnx_file(spec: RerankerSpec, architecture: str | None = None) -> str:
    normalized = (architecture or platform.machine()).lower()
    if normalized in {"arm64", "aarch64"}:
        return spec.arm64_onnx_file
    if normalized in {"x86_64", "amd64"}:
        return spec.x86_64_onnx_file
    raise DatasetIntegrityError(f"reranker has no int8 artifact for {normalized}")


def select_runtime_artifacts_sha256(
    spec: RerankerSpec,
    architecture: str | None = None,
) -> str:
    normalized = (architecture or platform.machine()).lower()
    if normalized in {"arm64", "aarch64"}:
        return spec.arm64_runtime_artifacts_sha256
    if normalized in {"x86_64", "amd64"}:
        return spec.x86_64_runtime_artifacts_sha256
    raise DatasetIntegrityError(f"reranker has no int8 artifact for {normalized}")


class EttinOnnxBackend:
    def __init__(
        self,
        *,
        root: Path,
        spec: RerankerSpec,
        local_files_only: bool,
    ) -> None:
        huggingface_hub = importlib.import_module("huggingface_hub")
        onnxruntime = importlib.import_module("onnxruntime")
        safetensors_torch = importlib.import_module("safetensors.torch")
        tokenizers = importlib.import_module("tokenizers")
        torch = importlib.import_module("torch")
        onnx_file = select_onnx_file(spec)
        cache = root / "data/cache/models/reranker"
        cache.mkdir(parents=True, exist_ok=True)
        required_files = _reranker_required_files(spec)
        files = {
            name: Path(
                huggingface_hub.hf_hub_download(
                    spec.repo_id,
                    name,
                    revision=spec.revision,
                    cache_dir=str(cache),
                    local_files_only=local_files_only,
                )
            )
            for name in required_files
        }
        _, runtime_hash, _ = _reranker_runtime_artifacts(root, spec)
        if runtime_hash != select_runtime_artifacts_sha256(spec):
            raise DatasetIntegrityError(
                "reranker runtime artifacts differ from the registered official bytes"
            )
        pooling = cast(dict[str, Any], json.loads(files["1_Pooling/config.json"].read_text()))
        dense_one = cast(dict[str, Any], json.loads(files["2_Dense/config.json"].read_text()))
        layer_norm = cast(dict[str, Any], json.loads(files["3_LayerNorm/config.json"].read_text()))
        dense_two = cast(dict[str, Any], json.loads(files["4_Dense/config.json"].read_text()))
        dimensions = dense_one.get("in_features")
        if (
            pooling.get("pooling_mode") != "cls"
            or not isinstance(dimensions, int)
            or dense_one.get("out_features") != dimensions
            or layer_norm.get("dimension") != dimensions
            or dense_two.get("in_features") != dimensions
            or dense_two.get("out_features") != 1
            or dense_one.get("activation_function") != "torch.nn.modules.activation.GELU"
            or dense_two.get("activation_function") != "torch.nn.modules.linear.Identity"
        ):
            raise DatasetIntegrityError("reranker scoring-head configuration differs")
        tokenizer = tokenizers.Tokenizer.from_file(str(files["tokenizer.json"]))
        pad_id = tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            raise DatasetIntegrityError("reranker tokenizer has no padding token")
        tokenizer.enable_truncation(
            max_length=spec.max_length,
            strategy="longest_first",
        )
        tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        session = onnxruntime.InferenceSession(
            str(files[onnx_file]),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        if [value.name for value in session.get_inputs()] != [
            "input_ids",
            "attention_mask",
        ]:
            raise DatasetIntegrityError("reranker ONNX inputs differ")
        output = session.get_outputs()
        if len(output) != 1 or output[0].name != "last_hidden_state":
            raise DatasetIntegrityError("reranker ONNX output differs")
        dense_one_weights = safetensors_torch.load_file(
            str(files["2_Dense/model.safetensors"]), device="cpu"
        )
        norm_weights = safetensors_torch.load_file(
            str(files["3_LayerNorm/model.safetensors"]), device="cpu"
        )
        dense_two_weights = safetensors_torch.load_file(
            str(files["4_Dense/model.safetensors"]), device="cpu"
        )
        self._tokenizer: Any = tokenizer
        self._session: Any = session
        self._torch: Any = torch
        self._dense_one_weight: Any = dense_one_weights["linear.weight"]
        self._norm_weight: Any = norm_weights["norm.weight"]
        self._norm_bias: Any = norm_weights["norm.bias"]
        self._dense_two_weight: Any = dense_two_weights["linear.weight"]
        self._dense_two_bias: Any = dense_two_weights["linear.bias"]
        self._dimensions = dimensions
        self._batch_size = spec.batch_size
        self.onnx_path = files[onnx_file]
        self.artifact_sha256 = canonical_sha256(
            [
                {
                    "path": name,
                    "sha256": file_facts(files[name]).sha256,
                    "bytes": file_facts(files[name]).bytes,
                }
                for name in required_files
            ]
        )
        self.runtime_components = {
            name: {
                "sha256": file_facts(files[name]).sha256,
                "bytes": file_facts(files[name]).bytes,
            }
            for name in required_files
        }
        self.runtime = (
            f"tokenizers/{importlib.metadata.version('tokenizers')} "
            f"onnxruntime/{importlib.metadata.version('onnxruntime')} "
            f"torch/{importlib.metadata.version('torch')} "
            f"CPUExecutionProvider int8/{onnx_file} "
            "threads/1:1 execution_mode/ORT_SEQUENTIAL"
        )

    def score(self, pairs: list[tuple[str, str]]) -> NDArray[np.float32]:
        if not pairs:
            return np.empty((0,), dtype=np.float32)
        outputs: list[NDArray[np.float32]] = []
        for offset in range(0, len(pairs), self._batch_size):
            encodings = self._tokenizer.encode_batch(pairs[offset : offset + self._batch_size])
            input_ids = np.asarray([encoding.ids for encoding in encodings], dtype=np.int64)
            attention_mask = np.asarray(
                [encoding.attention_mask for encoding in encodings], dtype=np.int64
            )
            hidden_state = self._session.run(
                ["last_hidden_state"],
                {"input_ids": input_ids, "attention_mask": attention_mask},
            )[0]
            if hidden_state.shape[0] != len(encodings) or hidden_state.shape[2] != self._dimensions:
                raise DatasetIntegrityError("reranker ONNX hidden-state shape differs")
            with self._torch.no_grad():
                values = self._torch.from_numpy(hidden_state[:, 0, :])
                values = self._torch.nn.functional.linear(values, self._dense_one_weight)
                values = self._torch.nn.functional.gelu(values)
                values = self._torch.nn.functional.layer_norm(
                    values,
                    (self._dimensions,),
                    self._norm_weight,
                    self._norm_bias,
                    1e-5,
                )
                values = self._torch.nn.functional.linear(
                    values,
                    self._dense_two_weight,
                    self._dense_two_bias,
                )
            outputs.append(np.asarray(values.squeeze(-1).numpy(), dtype=np.float32))
        scores = np.concatenate(outputs)
        if scores.shape != (len(pairs),) or not np.isfinite(scores).all():
            raise DatasetIntegrityError("reranker returned invalid scores")
        return scores


def load_product_documents(path: Path) -> dict[str, str]:
    documents = {
        document_id: sparse_document_text(document)
        for document_id, document in iter_prepared_products(path)
    }
    if not documents or any(not text for text in documents.values()):
        raise DatasetIntegrityError("reranker product documents are incomplete")
    return documents


def rerank_records(
    source: list[RunRecord],
    *,
    queries: Mapping[str, str],
    documents: Mapping[str, str],
    backend: RerankerBackend,
    spec: RerankerSpec,
    tag: str,
) -> tuple[list[RunRecord], list[float]]:
    by_query = _validated_source_rankings(source, queries=queries, spec=spec)
    output: list[RunRecord] = []
    latency_samples: list[float] = []
    for query_id in sorted(by_query, key=identifier_sort_key):
        ranking = by_query[query_id]
        candidates = ranking[: spec.candidate_depth]
        try:
            pairs = [(queries[query_id], documents[item.document_id]) for item in candidates]
        except KeyError as error:
            raise DatasetIntegrityError("reranker source references an unknown product") from error
        started = time.perf_counter()
        scores = backend.score(pairs)
        latency_samples.append((time.perf_counter() - started) * 1000.0)
        rescored = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-float(item[1]), item[0].rank, item[0].document_id),
        )
        reordered = [item[0] for item in rescored] + ranking[spec.candidate_depth :]
        output.extend(
            RunRecord(
                query_id=query_id,
                document_id=item.document_id,
                rank=rank,
                score=float(spec.result_depth - rank + 1),
                tag=tag,
            )
            for rank, item in enumerate(reordered, start=1)
        )
    return output, latency_samples


class CachedRerankerBackend:
    runtime = "recorded_pair_scores"

    def __init__(self, scores: Mapping[tuple[str, str], float], artifact_sha256: str):
        self._scores = scores
        self.artifact_sha256 = artifact_sha256

    def score(self, pairs: list[tuple[str, str]]) -> NDArray[np.float32]:
        try:
            values = [self._scores[pair] for pair in pairs]
        except KeyError as error:
            raise DatasetIntegrityError("reranker pair-score cache is incomplete") from error
        return np.asarray(values, dtype=np.float32)


def write_pair_score_cache(
    path: Path,
    scores: Mapping[tuple[str, str], float],
) -> None:
    if not scores or any(not math.isfinite(score) for score in scores.values()):
        raise DatasetIntegrityError("reranker pair-score cache is empty or non-finite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for (query, document), score in sorted(scores.items()):
            handle.write(
                json.dumps(
                    {"query": query, "document": document, "score": score},
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
    temporary.replace(path)


def load_pair_score_cache(path: Path) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = cast(dict[str, Any], json.loads(line))
            pair = (str(record.get("query", "")), str(record.get("document", "")))
            score = record.get("score")
            if (
                not all(pair)
                or pair in scores
                or not isinstance(score, int | float)
                or isinstance(score, bool)
                or not math.isfinite(float(score))
            ):
                raise DatasetIntegrityError(f"invalid reranker score at {path}:{line_number}")
            scores[pair] = float(score)
    if not scores:
        raise DatasetIntegrityError("reranker pair-score cache is empty")
    return scores


def _verify_reranker_score_replay(
    *,
    root: Path,
    spec: RerankerSpec,
    scores: Mapping[tuple[str, str], float],
    runtime_components: Mapping[str, Mapping[str, int | str]],
    runtime_artifact_sha256: str,
    selected_onnx_artifact: Mapping[str, int | str],
    backend: RerankerBackend | None = None,
) -> dict[str, object]:
    if not scores:
        raise DatasetIntegrityError("reranker score replay has no cached pairs")
    ordered_pairs = sorted(scores)
    indexes = _spanning_indexes(len(ordered_pairs), RERANKER_REPLAY_SAMPLE_SIZE)
    sample_pairs = [ordered_pairs[index] for index in indexes]
    replay_backend = backend or EttinOnnxBackend(
        root=root,
        spec=spec,
        local_files_only=True,
    )
    runtime_value = getattr(replay_backend, "runtime", None)
    backend_components = getattr(replay_backend, "runtime_components", None)
    backend_hash = getattr(replay_backend, "artifact_sha256", None)
    onnx_path = getattr(replay_backend, "onnx_path", None)
    if (
        not isinstance(runtime_value, str)
        or not runtime_value
        or backend_components != dict(runtime_components)
        or backend_hash != runtime_artifact_sha256
        or not isinstance(onnx_path, Path)
        or {
            "sha256": file_facts(onnx_path).sha256,
            "bytes": file_facts(onnx_path).bytes,
        }
        != dict(selected_onnx_artifact)
    ):
        raise DatasetIntegrityError("reranker score replay backend identity differs")
    replayed_batches: list[NDArray[np.float32]] = []
    for offset in range(0, len(sample_pairs), spec.batch_size):
        batch = sample_pairs[offset : offset + spec.batch_size]
        values = np.asarray(replay_backend.score(batch), dtype=np.float32)
        if values.shape != (len(batch),) or not np.isfinite(values).all():
            raise DatasetIntegrityError("reranker score replay returned invalid scores")
        replayed_batches.append(values)
    replayed = np.concatenate(replayed_batches)
    expected = np.asarray([scores[pair] for pair in sample_pairs], dtype=np.float32)
    if not np.allclose(
        replayed,
        expected,
        rtol=0.0,
        atol=RERANKER_REPLAY_ABSOLUTE_TOLERANCE,
    ):
        maximum_delta = float(np.max(np.abs(replayed - expected)))
        raise DatasetIntegrityError(
            "reranker deterministic score replay differs; "
            f"maximum absolute delta {maximum_delta:.9g}"
        )
    return {
        "method": "fixed_evenly_spaced_sorted_pairs_including_endpoints",
        "sample_size": len(sample_pairs),
        "sample_pairs_sha256": canonical_sha256(sample_pairs),
        "batch_size": spec.batch_size,
        "relative_tolerance": 0.0,
        "absolute_tolerance": RERANKER_REPLAY_ABSOLUTE_TOLERANCE,
        "runtime": runtime_value,
        "runtime_artifact_sha256": runtime_artifact_sha256,
        "status": "passed",
    }


def prepare_wands_reranker(
    *,
    root: Path,
    spec: RerankerSpec,
    evidence_context: WandsRerankerEvidenceContext,
    local_files_only: bool,
) -> dict[str, object]:
    benchmark_provenance = collect_manifest_provenance(root)
    architecture = _normalize_architecture(
        str(cast(dict[str, Any], benchmark_provenance["generator_host"])["architecture"])
    )
    profile_path = root / "config/benchmark.toml"
    environment_path = root / "results/environment/benchmark-profile.json"
    benchmark_profile_artifact = _artifact(root, profile_path)
    benchmark_environment_artifact = _artifact(root, environment_path)
    profile = load_benchmark_profile(profile_path)
    environment = cast(dict[str, Any], read_json(environment_path))
    code_revision = cast(dict[str, Any], benchmark_provenance["code_revision"])
    verified_summaries = _verify_reranker_source_summaries(
        root=root,
        context=evidence_context,
    )
    source_paths = reranker_source_run_paths(
        root,
        verified_summaries=verified_summaries,
    )
    backend = EttinOnnxBackend(
        root=root,
        spec=spec,
        local_files_only=local_files_only,
    )
    queries = _queries_by_split(root)
    documents = load_product_documents(root / "data/prepared/wands/products.jsonl")
    pairs = _expected_reranker_pairs(
        source_paths=source_paths,
        queries=queries,
        documents=documents,
        spec=spec,
    )
    ordered_pairs = sorted(pairs)
    scores: dict[tuple[str, str], float] = {}
    started = time.perf_counter()
    for offset in range(0, len(ordered_pairs), spec.batch_size):
        batch = ordered_pairs[offset : offset + spec.batch_size]
        values = backend.score(batch)
        scores.update((pair, float(score)) for pair, score in zip(batch, values, strict=True))
    scoring_seconds = time.perf_counter() - started
    cache_path = root / "data/cache/reranker/wands-ettin-32m-pair-scores.jsonl"
    write_pair_score_cache(cache_path, scores)
    cached_scores = load_pair_score_cache(cache_path)
    runtime_components, runtime_hash, selected_onnx_artifact = _reranker_runtime_artifacts(
        root, spec
    )
    replay_evidence = _verify_reranker_score_replay(
        root=root,
        spec=spec,
        scores=cached_scores,
        runtime_components=runtime_components,
        runtime_artifact_sha256=runtime_hash,
        selected_onnx_artifact=selected_onnx_artifact,
        backend=backend,
    )
    model_cache_inputs_verified = (
        cached_scores == scores
        and backend.runtime_components == runtime_components
        and backend.artifact_sha256 == runtime_hash
        and {
            "sha256": file_facts(backend.onnx_path).sha256,
            "bytes": file_facts(backend.onnx_path).bytes,
        }
        == selected_onnx_artifact
        and replay_evidence.get("status") == "passed"
    )
    if not model_cache_inputs_verified:
        raise DatasetIntegrityError("reranker model or pair-score cache verification failed")
    latency_samples = _standalone_latency_samples(
        backend,
        source_path=source_paths["dense_minmax"]["test"],
        queries=queries["test"],
        documents=documents,
        spec=spec,
    )
    source_evidence = _reranker_source_evidence(
        root,
        source_paths,
        verified_summaries=verified_summaries,
    )
    preparation_provenance_valid, completion_revision = wands_completion_provenance_evidence(
        root, benchmark_provenance
    )
    latency_eligible, latency_reason = _reranker_latency_eligibility(
        profile_id=profile.profile_id,
        cpu_limit=profile.cpu_limit,
        memory_limit_bytes=profile.memory_limit_bytes,
        required_architecture=profile.latency_required_architecture,
        environment=environment,
        host_architecture=architecture,
        code_revision=code_revision,
        preparation_provenance_valid=preparation_provenance_valid,
    )
    quality_eligible, quality_reason = _reranker_quality_eligibility(
        source_evidence=source_evidence,
        source_arms=spec.source_arms,
        model_cache_inputs_verified=model_cache_inputs_verified,
        preparation_provenance_valid=preparation_provenance_valid,
    )
    manifest: dict[str, object] = {
        "schema_version": 2,
        "dataset": "WANDS",
        "benchmark_provenance": benchmark_provenance,
        "completion_code_revision": completion_revision,
        "benchmark_profile": benchmark_profile_artifact,
        "benchmark_environment": benchmark_environment_artifact,
        "model": _reranker_spec_payload(spec),
        "model_source": (f"https://huggingface.co/{spec.repo_id}/tree/{spec.revision}"),
        "selected_onnx_file": select_onnx_file(spec),
        "selected_onnx_artifact": selected_onnx_artifact,
        "runtime_artifact_sha256": runtime_hash,
        "runtime_components": runtime_components,
        "runtime": backend.runtime,
        "host_architecture": architecture,
        "resource_profile": {
            "profile_id": profile.profile_id,
            "cpu_limit": profile.cpu_limit,
            "memory_limit_bytes": profile.memory_limit_bytes,
            "limits_verified": environment.get("limits_verified") is True,
            "host_process_limits_verified": False,
            "required_architecture": profile.latency_required_architecture,
            "environment_architecture": _normalize_architecture(
                str(environment.get("architecture", ""))
            ),
        },
        "pair_score_cache": _artifact(root, cache_path),
        "pair_keys_sha256": canonical_sha256(ordered_pairs),
        "deterministic_score_replay": replay_evidence,
        "model_cache_inputs_verified": model_cache_inputs_verified,
        "preparation_provenance_valid": preparation_provenance_valid,
        "quality_evidence_eligible_for_decision": quality_eligible,
        "quality_ineligibility_reason": quality_reason,
        "unique_pairs": len(scores),
        "quality_scoring_seconds": scoring_seconds,
        "standalone_rerank50_latency": summarize_latency_ms(latency_samples),
        "standalone_latency_queries": len(latency_samples),
        "standalone_latency_decision_eligible": latency_eligible,
        "standalone_latency_ineligibility_reason": latency_reason,
        "source_runs": {
            arm: {split: _artifact(root, path) for split, path in splits.items()}
            for arm, splits in source_paths.items()
        },
        "source_evidence": source_evidence,
    }
    write_json(root / "results/wands/reranker/precompute-manifest.json", manifest)
    return manifest


def _verify_reranker_source_summaries(
    *,
    root: Path,
    context: WandsRerankerEvidenceContext,
) -> dict[str, dict[str, Any]]:
    bm25 = verify_wands_bm25_selection(
        context.client,
        root=root,
        index=context.index,
        experiments=context.experiments,
    )
    hybrid = verify_wands_hybrid_summary(
        context.client,
        root=root,
        index=context.index,
        models=context.models,
        runtime=context.runtime,
        experiments=context.experiments,
    )
    return {"bm25": bm25, "hybrid": hybrid}


def reranker_source_run_paths(
    root: Path,
    *,
    verified_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Path]]:
    bm25 = verified_summaries["bm25"]
    hybrid = verified_summaries["hybrid"]
    bm25_artifacts = cast(dict[str, Any], bm25["artifacts"])
    hybrid_artifacts = cast(dict[str, Any], hybrid["artifacts"])
    selected_weight = round(float(hybrid["selected_lexical_weight"]) * 100)
    return {
        "tuned_bm25": {
            "dev": root / str(cast(dict[str, Any], bm25_artifacts["tuned/dev"])["run"]),
            "test": root / str(cast(dict[str, Any], bm25_artifacts["tuned/test"])["run"]),
        },
        "dense_minmax": {
            "dev": root
            / str(cast(dict[str, Any], hybrid_artifacts[f"dev/lw{selected_weight:03d}"])["run"]),
            "test": root / str(cast(dict[str, Any], hybrid_artifacts["test/selected"])["run"]),
        },
    }


def _reranker_source_paths_from_preparation(
    root: Path,
    preparation: Mapping[str, Any],
) -> dict[str, dict[str, Path]]:
    source_runs_value = preparation.get("source_runs")
    if not isinstance(source_runs_value, Mapping):
        raise DatasetIntegrityError("reranker preparation source runs are missing")
    source_paths: dict[str, dict[str, Path]] = {}
    for arm, splits_value in source_runs_value.items():
        if not isinstance(arm, str) or not isinstance(splits_value, Mapping):
            raise DatasetIntegrityError("reranker preparation source runs are malformed")
        splits: dict[str, Path] = {}
        for split, artifact_value in splits_value.items():
            if not isinstance(split, str) or not isinstance(artifact_value, Mapping):
                raise DatasetIntegrityError("reranker preparation source run is malformed")
            relative_path = artifact_value.get("path")
            if not isinstance(relative_path, str):
                raise DatasetIntegrityError("reranker preparation source path is missing")
            path = root / relative_path
            if dict(artifact_value) != _artifact(root, path):
                raise DatasetIntegrityError(
                    f"reranker preparation source artifact differs: {arm}/{split}"
                )
            splits[split] = path
        source_paths[arm] = splits
    return source_paths


def _read_bound_reranker_source_summary(
    root: Path,
    preparation: Mapping[str, Any],
    *,
    arm: str,
    path: Path,
) -> dict[str, Any]:
    source_evidence = preparation.get("source_evidence")
    if not isinstance(source_evidence, Mapping):
        raise DatasetIntegrityError("reranker preparation source evidence is missing")
    arm_evidence = source_evidence.get(arm)
    if not isinstance(arm_evidence, Mapping):
        raise DatasetIntegrityError(f"reranker source evidence is missing: {arm}")
    test_evidence = arm_evidence.get("test")
    if not isinstance(test_evidence, Mapping):
        raise DatasetIntegrityError(f"reranker held-out source evidence is missing: {arm}")
    if test_evidence.get("summary") != _artifact(root, path):
        raise DatasetIntegrityError(f"reranker source summary artifact differs: {arm}")
    return cast(dict[str, Any], read_json(path))


def _reranker_source_evidence(
    root: Path,
    source_paths: Mapping[str, Mapping[str, Path]],
    *,
    verified_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, object]]]:
    bm25_path = root / "results/wands/bm25-selection.json"
    hybrid_path = root / "results/wands/hybrid-summary.json"
    bm25 = verified_summaries["bm25"]
    hybrid = verified_summaries["hybrid"]
    if read_json(bm25_path) != bm25 or read_json(hybrid_path) != hybrid:
        raise DatasetIntegrityError("reranker source summary changed after verification")
    selected_weight = round(float(hybrid["selected_lexical_weight"]) * 100)
    source_specs = {
        "tuned_bm25": {
            "summary_path": bm25_path,
            "summary": bm25,
            "summary_eligibility_key": "quality_evidence_eligible_for_decision",
            "semantic_verifier": "verify_wands_bm25_selection",
            "artifact_keys": {
                "dev": "tuned/dev",
                "test": "tuned/test",
            },
        },
        "dense_minmax": {
            "summary_path": hybrid_path,
            "summary": hybrid,
            "summary_eligibility_key": "eligible_for_decision",
            "semantic_verifier": "verify_wands_hybrid_summary",
            "artifact_keys": {
                "dev": f"dev/lw{selected_weight:03d}",
                "test": "test/selected",
            },
        },
    }
    evidence: dict[str, dict[str, dict[str, object]]] = {}
    for arm, splits in source_paths.items():
        if arm not in source_specs:
            raise DatasetIntegrityError(f"reranker source arm {arm} has no evidence policy")
        source_spec = source_specs[arm]
        summary = cast(dict[str, Any], source_spec["summary"])
        summary_path = cast(Path, source_spec["summary_path"])
        eligibility_key = cast(str, source_spec["summary_eligibility_key"])
        semantic_verifier = cast(str, source_spec["semantic_verifier"])
        artifact_keys = cast(dict[str, str], source_spec["artifact_keys"])
        artifacts = cast(dict[str, Any], summary["artifacts"])
        evidence[arm] = {}
        for split, run_path in splits.items():
            artifact_key = artifact_keys[split]
            entry = cast(dict[str, Any], artifacts[artifact_key])
            if root / str(entry["run"]) != run_path:
                raise DatasetIntegrityError(
                    f"reranker source summary points to another run: {arm}/{split}"
                )
            manifest_path = root / str(entry["manifest"])
            manifest = cast(dict[str, Any], read_json(manifest_path))
            run_file = manifest.get("run_file")
            if not isinstance(run_file, dict):
                raise DatasetIntegrityError(
                    f"reranker source manifest has no run binding: {arm}/{split}"
                )
            if run_file.get("path") != str(run_path.relative_to(root)):
                raise DatasetIntegrityError(
                    f"reranker source manifest points to another run: {arm}/{split}"
                )
            if run_file.get("sha256") != file_facts(run_path).sha256:
                raise DatasetIntegrityError(
                    f"reranker source run differs from its manifest: {arm}/{split}"
                )
            manifest_eligible = manifest.get("eligible_for_decision") is True
            summary_eligible = summary.get(eligibility_key) is True
            eligible = manifest_eligible and summary_eligible
            reasons = []
            if not manifest_eligible:
                reasons.append("source run manifest is not decision eligible")
            if not summary_eligible:
                reasons.append("source summary is not decision eligible")
            evidence[arm][split] = {
                "summary": _artifact(root, summary_path),
                "summary_artifact_key": artifact_key,
                "summary_eligibility_key": eligibility_key,
                "summary_eligible_for_decision": summary_eligible,
                "semantic_summary_verified": True,
                "semantic_verifier": semantic_verifier,
                "run": _artifact(root, run_path),
                "manifest": _artifact(root, manifest_path),
                "manifest_eligible_for_decision": manifest_eligible,
                "eligible_for_decision": eligible,
                "ineligibility_reasons": reasons,
            }
    return evidence


def _reranker_quality_eligibility(
    *,
    source_evidence: Mapping[str, Mapping[str, Mapping[str, object]]],
    source_arms: tuple[str, ...],
    model_cache_inputs_verified: bool,
    preparation_provenance_valid: bool,
) -> tuple[bool, str | None]:
    reasons: list[str] = []
    if not preparation_provenance_valid:
        reasons.append("reranker preparation did not retain one clean current source revision")
    if not model_cache_inputs_verified:
        reasons.append("reranker model or pair-score cache inputs are not verified")
    for arm in source_arms:
        arm_evidence = source_evidence.get(arm)
        test_evidence = arm_evidence.get("test") if arm_evidence is not None else None
        if test_evidence is None or test_evidence.get("eligible_for_decision") is not True:
            reasons.append(f"source evidence is not decision eligible: {arm}/test")
    return not reasons, "; ".join(reasons) or None


def _reranker_artifact_eligibility(
    preparation: Mapping[str, Any],
    *,
    source_arm: str,
    split: str,
) -> tuple[bool, str | None]:
    source_evidence = cast(
        dict[str, Any],
        cast(dict[str, Any], preparation["source_evidence"])[source_arm],
    )
    split_source_evidence = cast(dict[str, Any], source_evidence[split])
    model_cache_verified = preparation.get("model_cache_inputs_verified") is True
    preparation_provenance_valid = preparation.get("preparation_provenance_valid") is True
    eligible = (
        split == "test"
        and preparation_provenance_valid
        and model_cache_verified
        and split_source_evidence.get("eligible_for_decision") is True
    )
    if split != "test":
        return False, "development context result"
    if not preparation_provenance_valid:
        return False, ("reranker preparation did not retain one clean current source revision")
    if not model_cache_verified:
        return False, "reranker model or pair-score cache inputs are not verified"
    if split_source_evidence.get("eligible_for_decision") is not True:
        return False, f"source evidence is not decision eligible: {source_arm}/{split}"
    return eligible, None


def _reranker_latency_eligibility(
    *,
    profile_id: str,
    cpu_limit: int,
    memory_limit_bytes: int,
    required_architecture: str,
    environment: Mapping[str, Any],
    host_architecture: str,
    code_revision: Mapping[str, Any],
    preparation_provenance_valid: bool,
) -> tuple[bool, str | None]:
    architecture = _normalize_architecture(host_architecture)
    environment_architecture = _normalize_architecture(str(environment.get("architecture", "")))
    if environment.get("limits_verified") is not True:
        return False, "benchmark environment resource limits are not verified"
    if environment.get("profile_id") != profile_id:
        return False, "benchmark environment profile differs from the registered profile"
    if environment.get("cpu_limit") != cpu_limit:
        return False, "benchmark environment CPU limit differs from the registered profile"
    if environment.get("memory_limit_bytes") != memory_limit_bytes:
        return False, "benchmark environment memory limit differs from the registered profile"
    if environment_architecture != architecture:
        return False, "benchmark environment architecture differs from the generator host"
    expected_architecture_eligible = architecture == required_architecture
    if environment.get("latency_architecture_eligible") is not expected_architecture_eligible:
        return False, "benchmark environment architecture eligibility marker differs"
    if not expected_architecture_eligible:
        return False, f"registered latency decision architecture is {required_architecture}"
    if not isinstance(code_revision.get("git_commit"), str) or not code_revision.get("git_commit"):
        return False, "latency run did not start from a clean committed source revision"
    if code_revision.get("source_dirty") is not False:
        return False, "latency run did not start from a clean committed source revision"
    if not preparation_provenance_valid:
        return False, "reranker source revision changed before preparation completed"
    return (
        False,
        "reranker host-process resource limits are not independently verified",
    )


def run_wands_reranker_benchmark(
    *,
    root: Path,
    spec: RerankerSpec,
    evidence_context: WandsRerankerEvidenceContext,
) -> dict[str, object]:
    preparation = verify_wands_reranker_preparation(
        root=root,
        spec=spec,
        evidence_context=evidence_context,
    )
    cache_path = root / str(cast(dict[str, Any], preparation["pair_score_cache"])["path"])
    backend = CachedRerankerBackend(
        load_pair_score_cache(cache_path),
        str(preparation["runtime_artifact_sha256"]),
    )
    queries = _queries_by_split(root)
    documents = load_product_documents(root / "data/prepared/wands/products.jsonl")
    source_paths = _reranker_source_paths_from_preparation(root, preparation)
    artifacts: dict[str, object] = {}
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for arm in spec.source_arms:
        if arm not in source_paths:
            raise DatasetIntegrityError(f"reranker source arm {arm} is unavailable")
        metrics[arm] = {}
        for split in ("dev", "test"):
            tag = f"wands-reranker-ettin32-{arm}-{split}-int8"
            records, _ = rerank_records(
                read_run(source_paths[arm][split]),
                queries=queries[split],
                documents=documents,
                backend=backend,
                spec=spec,
                tag=tag,
            )
            result = write_reranker_bundle(
                root=root,
                spec=spec,
                preparation=preparation,
                source_arm=arm,
                source_run_path=source_paths[arm][split],
                query_path=(root / f"data/prepared/wands/queries.{split}.jsonl"),
                qrels_path=root / f"data/prepared/wands/qrels.{split}.trec",
                split=split,
                tag=tag,
                records=records,
            )
            artifacts[f"{arm}/{split}"] = result.selection_entry(root)
            metrics[arm][split] = result.evaluation.metrics
    dense = _read_bound_reranker_source_summary(
        root,
        preparation,
        arm="dense_minmax",
        path=root / "results/wands/hybrid-summary.json",
    )
    bm25 = _read_bound_reranker_source_summary(
        root,
        preparation,
        arm="tuned_bm25",
        path=root / "results/wands/bm25-selection.json",
    )
    dense_test = float(cast(dict[str, Any], dense["held_out_test_metrics"])["ndcg@10"])
    tuned_test = float(
        cast(dict[str, Any], cast(dict[str, Any], bm25["held_out_test_metrics"])["tuned"])[
            "ndcg@10"
        ]
    )
    quality_eligible = preparation["quality_evidence_eligible_for_decision"] is True
    quality_reason = cast(str | None, preparation["quality_ineligibility_reason"])
    summary: dict[str, object] = {
        "schema_version": 2,
        "dataset": "WANDS",
        "model": _reranker_spec_payload(spec),
        "method": "offline_int8_cross_encoder_rerank50",
        "quality_evidence_eligible_for_decision": quality_eligible,
        "quality_ineligibility_reason": quality_reason,
        "latency_evidence_eligible_for_decision": preparation[
            "standalone_latency_decision_eligible"
        ],
        "latency_ineligibility_reason": preparation["standalone_latency_ineligibility_reason"],
        "source_arm_metrics": {
            "tuned_bm25_test_ndcg@10": tuned_test,
            "dense_minmax_test_ndcg@10": dense_test,
        },
        "reranked_metrics": metrics,
        "bm25_reranker_minus_tuned_bm25_ndcg@10": (
            metrics["tuned_bm25"]["test"]["ndcg@10"] - tuned_test
        ),
        "bm25_reranker_minus_dense_minmax_ndcg@10": (
            metrics["tuned_bm25"]["test"]["ndcg@10"] - dense_test
        ),
        "dense_reranker_minus_dense_minmax_ndcg@10": (
            metrics["dense_minmax"]["test"]["ndcg@10"] - dense_test
        ),
        "preparation_manifest": _artifact(
            root, root / "results/wands/reranker/precompute-manifest.json"
        ),
        "source_evidence": preparation["source_evidence"],
        "artifacts": artifacts,
    }
    write_json(root / "results/wands/reranker/summary.json", summary)
    return summary


def write_reranker_bundle(
    *,
    root: Path,
    spec: RerankerSpec,
    preparation: Mapping[str, Any],
    source_arm: str,
    source_run_path: Path,
    query_path: Path,
    qrels_path: Path,
    split: str,
    tag: str,
    records: list[RunRecord],
) -> RerankerRunResult:
    run_path = root / f"runs/{tag}.trec"
    manifest_path = root / f"runs/{tag}.manifest.json"
    metrics_path = root / f"results/wands/reranker/{tag}.metrics.json"
    write_run(run_path, records)
    evaluation = evaluate_run(qrels_path, run_path)
    write_json(
        metrics_path,
        {
            "schema_version": 1,
            "tag": tag,
            "split": split,
            "metrics": evaluation.metrics,
            "per_query": evaluation.per_query,
            "evaluator_crosscheck": cross_check_run(qrels_path, run_path, root=root).to_dict(),
        },
    )
    preparation_path = root / "results/wands/reranker/precompute-manifest.json"
    preparation_artifact = _artifact(root, preparation_path)
    source_evidence = cast(
        dict[str, Any],
        cast(dict[str, Any], preparation["source_evidence"])[source_arm],
    )
    split_source_evidence = cast(dict[str, Any], source_evidence[split])
    recorded_source_manifest = cast(dict[str, Any], split_source_evidence["manifest"])
    source_manifest_path = root / str(recorded_source_manifest["path"])
    eligible, ineligibility_reason = _reranker_artifact_eligibility(
        preparation,
        source_arm=source_arm,
        split=split,
    )
    manifest: dict[str, object] = {
        "schema_version": 2,
        "dataset": "WANDS",
        "split": split,
        "tag": tag,
        "decision_scope": "held_out" if split == "test" else "context_only",
        "eligible_for_decision": eligible,
        "ineligibility_reason": ineligibility_reason,
        "declared_variable": "reranker",
        "method": "offline_int8_cross_encoder_rerank50",
        "source_arm": source_arm,
        "source_run": _artifact(root, source_run_path),
        "source_manifest": _artifact(root, source_manifest_path),
        "source_evidence": split_source_evidence,
        "preparation_manifest": preparation_artifact,
        "provenance_binding": "preparation_manifest_and_source_manifest",
        "model": _reranker_spec_payload(spec),
        "selected_onnx_artifact": preparation["selected_onnx_artifact"],
        "runtime_artifact_sha256": preparation["runtime_artifact_sha256"],
        "pair_score_cache": preparation["pair_score_cache"],
        "candidate_depth": spec.candidate_depth,
        "result_depth": spec.result_depth,
        "ranking_policy": (
            "reranker_score_desc_then_source_rank_then_document_id; "
            "source ranks after candidate depth preserved; source result depth is a "
            "maximum and per-query source cardinality is preserved"
        ),
        "query_file": _artifact(root, query_path),
        "qrels_file": _artifact(root, qrels_path),
        "run_file": _artifact(root, run_path),
    }
    write_json(manifest_path, manifest)
    return RerankerRunResult(run_path, manifest_path, metrics_path, evaluation)


def verify_wands_reranker_preparation(
    *,
    root: Path,
    spec: RerankerSpec,
    evidence_context: WandsRerankerEvidenceContext,
) -> dict[str, Any]:
    path = root / "results/wands/reranker/precompute-manifest.json"
    manifest = cast(dict[str, Any], read_json(path))
    if manifest.get("schema_version") != 2:
        raise DatasetIntegrityError("reranker preparation schema differs")
    if manifest.get("model") != _reranker_spec_payload(spec):
        raise DatasetIntegrityError("reranker preparation model differs")
    cache = cast(dict[str, Any], manifest["pair_score_cache"])
    cache_path = root / str(cache["path"])
    if cache != _artifact(root, cache_path):
        raise DatasetIntegrityError("reranker pair-score cache differs")
    scores = load_pair_score_cache(cache_path)
    if manifest.get("unique_pairs") != len(scores):
        raise DatasetIntegrityError("reranker pair-score count differs")
    verified_summaries = _verify_reranker_source_summaries(
        root=root,
        context=evidence_context,
    )
    source_paths = reranker_source_run_paths(
        root,
        verified_summaries=verified_summaries,
    )
    expected_sources = {
        arm: {split: _artifact(root, run) for split, run in splits.items()}
        for arm, splits in source_paths.items()
    }
    if manifest.get("source_runs") != expected_sources:
        raise DatasetIntegrityError("reranker source runs differ")
    source_evidence = _reranker_source_evidence(
        root,
        source_paths,
        verified_summaries=verified_summaries,
    )
    if manifest.get("source_evidence") != source_evidence:
        raise DatasetIntegrityError("reranker source evidence differs")
    queries = _queries_by_split(root)
    documents = load_product_documents(root / "data/prepared/wands/products.jsonl")
    expected_pairs = _expected_reranker_pairs(
        source_paths=source_paths,
        queries=queries,
        documents=documents,
        spec=spec,
    )
    if set(scores) != expected_pairs:
        raise DatasetIntegrityError("reranker pair-score cache keys differ")
    if manifest.get("pair_keys_sha256") != canonical_sha256(sorted(expected_pairs)):
        raise DatasetIntegrityError("reranker pair-score key binding differs")
    components, runtime_hash, selected = _reranker_runtime_artifacts(root, spec)
    if manifest.get("runtime_components") != components:
        raise DatasetIntegrityError("reranker runtime components differ")
    if manifest.get("runtime_artifact_sha256") != runtime_hash:
        raise DatasetIntegrityError("reranker runtime artifact hash differs")
    if manifest.get("selected_onnx_artifact") != selected:
        raise DatasetIntegrityError("reranker selected ONNX artifact differs")
    replay_evidence = _verify_reranker_score_replay(
        root=root,
        spec=spec,
        scores=scores,
        runtime_components=components,
        runtime_artifact_sha256=runtime_hash,
        selected_onnx_artifact=selected,
    )
    if manifest.get("deterministic_score_replay") != replay_evidence:
        raise DatasetIntegrityError("reranker deterministic score replay differs")
    if manifest.get("runtime") != replay_evidence["runtime"]:
        raise DatasetIntegrityError("reranker runtime differs")
    if manifest.get("model_cache_inputs_verified") is not True:
        raise DatasetIntegrityError("reranker model and cache verification differs")
    profile_path = root / "config/benchmark.toml"
    environment_path = root / "results/environment/benchmark-profile.json"
    if manifest.get("benchmark_profile") != _artifact(root, profile_path):
        raise DatasetIntegrityError("reranker benchmark profile artifact differs")
    if manifest.get("benchmark_environment") != _artifact(root, environment_path):
        raise DatasetIntegrityError("reranker benchmark environment artifact differs")
    provenance = manifest.get("benchmark_provenance")
    if not isinstance(provenance, dict):
        raise DatasetIntegrityError("reranker start-captured provenance is missing")
    provenance_profile = provenance.get("benchmark_profile")
    provenance_environment = provenance.get("benchmark_environment")
    generator_host = provenance.get("generator_host")
    code_revision = provenance.get("code_revision")
    profile_artifact = cast(dict[str, Any], manifest["benchmark_profile"])
    environment_artifact = cast(dict[str, Any], manifest["benchmark_environment"])
    if (
        not isinstance(provenance_profile, dict)
        or provenance.get("benchmark_profile_sha256") != profile_artifact["sha256"]
    ):
        raise DatasetIntegrityError("reranker captured benchmark profile differs")
    expected_provenance_environment = {
        "status": "verified_artifact_present",
        "path": environment_artifact["path"],
        "sha256": environment_artifact["sha256"],
    }
    if provenance_environment != expected_provenance_environment:
        raise DatasetIntegrityError("reranker captured benchmark environment differs")
    if not isinstance(generator_host, dict):
        raise DatasetIntegrityError("reranker captured generator host is missing")
    if not isinstance(code_revision, dict):
        raise DatasetIntegrityError("reranker captured code revision is missing")
    preparation_provenance_valid = wands_recorded_provenance_valid(
        root,
        provenance=provenance,
        completion_revision=manifest.get("completion_code_revision"),
    )
    if manifest.get("preparation_provenance_valid") is not preparation_provenance_valid:
        raise DatasetIntegrityError("reranker preparation provenance validity differs")
    quality_eligible, quality_reason = _reranker_quality_eligibility(
        source_evidence=source_evidence,
        source_arms=spec.source_arms,
        model_cache_inputs_verified=True,
        preparation_provenance_valid=preparation_provenance_valid,
    )
    if manifest.get("quality_evidence_eligible_for_decision") is not quality_eligible:
        raise DatasetIntegrityError("reranker quality eligibility differs")
    if manifest.get("quality_ineligibility_reason") != quality_reason:
        raise DatasetIntegrityError("reranker quality ineligibility reason differs")
    architecture = _normalize_architecture(str(generator_host.get("architecture", "")))
    if manifest.get("host_architecture") != architecture:
        raise DatasetIntegrityError("reranker captured host architecture differs")
    profile = load_benchmark_profile(profile_path)
    expected_provenance_profile = asdict(profile)
    expected_provenance_profile["compose_files"] = list(profile.compose_files)
    if provenance_profile != expected_provenance_profile:
        raise DatasetIntegrityError("reranker captured benchmark profile values differ")
    environment = cast(dict[str, Any], read_json(environment_path))
    latency_eligible, latency_reason = _reranker_latency_eligibility(
        profile_id=profile.profile_id,
        cpu_limit=profile.cpu_limit,
        memory_limit_bytes=profile.memory_limit_bytes,
        required_architecture=profile.latency_required_architecture,
        environment=environment,
        host_architecture=architecture,
        code_revision=code_revision,
        preparation_provenance_valid=preparation_provenance_valid,
    )
    expected_resource_profile = {
        "profile_id": profile.profile_id,
        "cpu_limit": profile.cpu_limit,
        "memory_limit_bytes": profile.memory_limit_bytes,
        "limits_verified": environment.get("limits_verified") is True,
        "host_process_limits_verified": False,
        "required_architecture": profile.latency_required_architecture,
        "environment_architecture": _normalize_architecture(
            str(environment.get("architecture", ""))
        ),
    }
    if manifest.get("resource_profile") != expected_resource_profile:
        raise DatasetIntegrityError("reranker latency resource profile differs")
    if manifest.get("standalone_latency_decision_eligible") is not latency_eligible:
        raise DatasetIntegrityError("reranker latency eligibility differs")
    if manifest.get("standalone_latency_ineligibility_reason") != latency_reason:
        raise DatasetIntegrityError("reranker latency ineligibility reason differs")
    latency = cast(dict[str, Any], manifest["standalone_rerank50_latency"])
    if latency.get("samples") != manifest.get("standalone_latency_queries"):
        raise DatasetIntegrityError("reranker latency sample count differs")
    if latency.get("samples") != 240:
        raise DatasetIntegrityError("reranker latency query count differs")
    return manifest


def verify_wands_reranker_benchmark(
    *,
    root: Path,
    spec: RerankerSpec,
    evidence_context: WandsRerankerEvidenceContext,
) -> dict[str, Any]:
    preparation_path = root / "results/wands/reranker/precompute-manifest.json"
    summary = cast(dict[str, Any], read_json(root / "results/wands/reranker/summary.json"))
    preparation_artifact = _artifact(root, preparation_path)
    if summary.get("preparation_manifest") != preparation_artifact:
        raise DatasetIntegrityError("reranker preparation manifest binding differs")
    if summary.get("schema_version") != 2:
        raise DatasetIntegrityError("reranker summary schema differs")
    if summary.get("model") != _reranker_spec_payload(spec):
        raise DatasetIntegrityError("reranker summary model differs")
    preparation = verify_wands_reranker_preparation(
        root=root,
        spec=spec,
        evidence_context=evidence_context,
    )
    cache_path = root / str(cast(dict[str, Any], preparation["pair_score_cache"])["path"])
    backend = CachedRerankerBackend(
        load_pair_score_cache(cache_path),
        str(preparation["runtime_artifact_sha256"]),
    )
    queries = _queries_by_split(root)
    documents = load_product_documents(root / "data/prepared/wands/products.jsonl")
    sources = _reranker_source_paths_from_preparation(root, preparation)
    artifacts = cast(dict[str, Any], summary["artifacts"])
    expected_artifact_keys = {
        f"{arm}/{split}" for arm in spec.source_arms for split in ("dev", "test")
    }
    if set(artifacts) != expected_artifact_keys:
        raise DatasetIntegrityError("reranker summary artifact set differs")
    recomputed_metrics: dict[str, dict[str, dict[str, float]]] = {}
    test_manifest_eligibility: list[bool] = []
    for arm in spec.source_arms:
        recomputed_metrics[arm] = {}
        for split in ("dev", "test"):
            entry = cast(dict[str, Any], artifacts[f"{arm}/{split}"])
            run_path = root / str(entry["run"])
            manifest_path = root / str(entry["manifest"])
            metrics_path = root / str(entry["metrics_file"])
            manifest = cast(dict[str, Any], read_json(manifest_path))
            if manifest.get("schema_version") != 2:
                raise DatasetIntegrityError("reranker run manifest schema differs")
            expected, _ = rerank_records(
                read_run(sources[arm][split]),
                queries=queries[split],
                documents=documents,
                backend=backend,
                spec=spec,
                tag=str(manifest["tag"]),
            )
            if read_run(run_path) != expected:
                raise DatasetIntegrityError("reranker run does not reproduce from cache")
            qrels_path = root / f"data/prepared/wands/qrels.{split}.trec"
            evaluation = evaluate_run(qrels_path, run_path)
            metrics = cast(dict[str, Any], read_json(metrics_path))
            if metrics.get("metrics") != evaluation.metrics:
                raise DatasetIntegrityError("reranker metrics do not reproduce")
            if entry.get("metrics") != evaluation.metrics:
                raise DatasetIntegrityError("reranker summary metrics differ")
            recomputed_metrics[arm][split] = evaluation.metrics
            if manifest.get("source_run") != _artifact(root, sources[arm][split]):
                raise DatasetIntegrityError("reranker source artifact differs")
            source_evidence = cast(
                dict[str, Any],
                cast(dict[str, Any], preparation["source_evidence"])[arm],
            )[split]
            if manifest.get("source_evidence") != source_evidence:
                raise DatasetIntegrityError("reranker run source evidence differs")
            expected_source_manifest = cast(dict[str, Any], source_evidence)["manifest"]
            if manifest.get("source_manifest") != expected_source_manifest:
                raise DatasetIntegrityError("reranker source manifest binding differs")
            if manifest.get("preparation_manifest") != preparation_artifact:
                raise DatasetIntegrityError("reranker run preparation binding differs")
            if manifest.get("provenance_binding") != ("preparation_manifest_and_source_manifest"):
                raise DatasetIntegrityError("reranker run provenance binding differs")
            if manifest.get("selected_onnx_artifact") != preparation.get("selected_onnx_artifact"):
                raise DatasetIntegrityError("reranker run ONNX artifact differs")
            if manifest.get("runtime_artifact_sha256") != preparation.get(
                "runtime_artifact_sha256"
            ):
                raise DatasetIntegrityError("reranker run runtime artifact differs")
            if manifest.get("pair_score_cache") != preparation.get("pair_score_cache"):
                raise DatasetIntegrityError("reranker run pair-score cache differs")
            if manifest.get("query_file") != _artifact(
                root, root / f"data/prepared/wands/queries.{split}.jsonl"
            ):
                raise DatasetIntegrityError("reranker query artifact differs")
            if manifest.get("qrels_file") != _artifact(root, qrels_path):
                raise DatasetIntegrityError("reranker qrels artifact differs")
            if manifest.get("run_file") != _artifact(root, run_path):
                raise DatasetIntegrityError("reranker output run artifact differs")
            eligible, reason = _reranker_artifact_eligibility(
                preparation,
                source_arm=arm,
                split=split,
            )
            if manifest.get("eligible_for_decision") is not eligible:
                raise DatasetIntegrityError("reranker run eligibility differs")
            if manifest.get("ineligibility_reason") != reason:
                raise DatasetIntegrityError("reranker run ineligibility reason differs")
            if split == "test":
                test_manifest_eligibility.append(eligible)
    dense = _read_bound_reranker_source_summary(
        root,
        preparation,
        arm="dense_minmax",
        path=root / "results/wands/hybrid-summary.json",
    )
    bm25 = _read_bound_reranker_source_summary(
        root,
        preparation,
        arm="tuned_bm25",
        path=root / "results/wands/bm25-selection.json",
    )
    dense_test = float(cast(dict[str, Any], dense["held_out_test_metrics"])["ndcg@10"])
    tuned_test = float(
        cast(dict[str, Any], cast(dict[str, Any], bm25["held_out_test_metrics"])["tuned"])[
            "ndcg@10"
        ]
    )
    _verify_reranker_summary_claims(
        summary=summary,
        preparation=preparation,
        source_arms=spec.source_arms,
        recomputed_metrics=recomputed_metrics,
        test_manifest_eligibility=test_manifest_eligibility,
        tuned_test=tuned_test,
        dense_test=dense_test,
    )
    return summary


def _verify_reranker_summary_claims(
    *,
    summary: Mapping[str, Any],
    preparation: Mapping[str, Any],
    source_arms: tuple[str, ...],
    recomputed_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    test_manifest_eligibility: list[bool],
    tuned_test: float,
    dense_test: float,
) -> None:
    if summary.get("reranked_metrics") != recomputed_metrics:
        raise DatasetIntegrityError("reranker summary metric families differ")
    if summary.get("source_evidence") != preparation.get("source_evidence"):
        raise DatasetIntegrityError("reranker summary source evidence differs")
    quality_eligible = (
        preparation.get("quality_evidence_eligible_for_decision") is True
        and len(test_manifest_eligibility) == len(source_arms)
        and all(test_manifest_eligibility)
    )
    quality_reason = cast(str | None, preparation.get("quality_ineligibility_reason"))
    if summary.get("quality_evidence_eligible_for_decision") is not quality_eligible:
        raise DatasetIntegrityError("reranker summary quality eligibility differs")
    if summary.get("quality_ineligibility_reason") != quality_reason:
        raise DatasetIntegrityError("reranker summary quality ineligibility reason differs")
    if summary.get("latency_evidence_eligible_for_decision") is not preparation.get(
        "standalone_latency_decision_eligible"
    ):
        raise DatasetIntegrityError("reranker summary latency eligibility differs")
    if summary.get("latency_ineligibility_reason") != preparation.get(
        "standalone_latency_ineligibility_reason"
    ):
        raise DatasetIntegrityError("reranker summary latency ineligibility reason differs")
    expected_source_metrics = {
        "tuned_bm25_test_ndcg@10": tuned_test,
        "dense_minmax_test_ndcg@10": dense_test,
    }
    if summary.get("source_arm_metrics") != expected_source_metrics:
        raise DatasetIntegrityError("reranker source metrics differ")
    expected_deltas = {
        "bm25_reranker_minus_tuned_bm25_ndcg@10": (
            recomputed_metrics["tuned_bm25"]["test"]["ndcg@10"] - tuned_test
        ),
        "bm25_reranker_minus_dense_minmax_ndcg@10": (
            recomputed_metrics["tuned_bm25"]["test"]["ndcg@10"] - dense_test
        ),
        "dense_reranker_minus_dense_minmax_ndcg@10": (
            recomputed_metrics["dense_minmax"]["test"]["ndcg@10"] - dense_test
        ),
    }
    for name, value in expected_deltas.items():
        if summary.get(name) != value:
            raise DatasetIntegrityError(f"reranker summary delta differs: {name}")


def _queries_by_split(root: Path) -> dict[str, dict[str, str]]:
    return {
        split: load_prepared_queries(
            root / f"data/prepared/wands/queries.{split}.jsonl",
            expected_split=split,
        )
        for split in ("dev", "test")
    }


def _records_by_query(records: list[RunRecord]) -> dict[str, list[RunRecord]]:
    by_query: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        by_query[record.query_id].append(record)
    return {
        query_id: sorted(ranking, key=lambda item: item.rank)
        for query_id, ranking in by_query.items()
    }


def _validated_source_rankings(
    records: list[RunRecord],
    *,
    queries: Mapping[str, str],
    spec: RerankerSpec,
) -> dict[str, list[RunRecord]]:
    by_query = _records_by_query(records)
    if set(by_query).difference(queries):
        raise DatasetIntegrityError("reranker source run contains an unknown query")
    for query_id, ranking in by_query.items():
        if len(ranking) > spec.result_depth:
            raise DatasetIntegrityError(
                f"reranker source query {query_id} exceeds registered depth "
                f"{spec.result_depth} with {len(ranking)} results"
            )
    return by_query


def _expected_reranker_pairs(
    *,
    source_paths: Mapping[str, Mapping[str, Path]],
    queries: Mapping[str, Mapping[str, str]],
    documents: Mapping[str, str],
    spec: RerankerSpec,
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for split_paths in source_paths.values():
        for split, path in split_paths.items():
            if split not in queries:
                raise DatasetIntegrityError("reranker source split is not registered")
            by_query = _validated_source_rankings(
                read_run(path),
                queries=queries[split],
                spec=spec,
            )
            for query_id, ranking in by_query.items():
                for record in ranking[: spec.candidate_depth]:
                    try:
                        document = documents[record.document_id]
                    except KeyError as error:
                        raise DatasetIntegrityError(
                            "reranker source references an unknown product"
                        ) from error
                    pairs.add((queries[split][query_id], document))
    if not pairs:
        raise DatasetIntegrityError("reranker source pairs are empty")
    return pairs


def _spanning_indexes(length: int, sample_size: int) -> list[int]:
    if length <= 0 or sample_size <= 0:
        raise ValueError("spanning sample dimensions must be positive")
    if length <= sample_size:
        return list(range(length))
    return [
        position * (length - 1) // (sample_size - 1)
        for position in range(sample_size)
    ]


def _standalone_latency_samples(
    backend: RerankerBackend,
    *,
    source_path: Path,
    queries: Mapping[str, str],
    documents: Mapping[str, str],
    spec: RerankerSpec,
) -> list[float]:
    by_query = _validated_source_rankings(
        read_run(source_path),
        queries=queries,
        spec=spec,
    )
    samples: list[float] = []
    for query_id in sorted(by_query, key=identifier_sort_key):
        ranking = by_query[query_id]
        pairs = [
            (queries[query_id], documents[record.document_id])
            for record in ranking[: spec.candidate_depth]
        ]
        started = time.perf_counter()
        backend.score(pairs)
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }


def _reranker_required_files(spec: RerankerSpec) -> tuple[str, ...]:
    return (
        select_onnx_file(spec),
        "tokenizer.json",
        "1_Pooling/config.json",
        "2_Dense/config.json",
        "2_Dense/model.safetensors",
        "3_LayerNorm/config.json",
        "3_LayerNorm/model.safetensors",
        "4_Dense/config.json",
        "4_Dense/model.safetensors",
    )


def _reranker_runtime_artifacts(
    root: Path,
    spec: RerankerSpec,
) -> tuple[dict[str, dict[str, int | str]], str, dict[str, int | str]]:
    snapshot = _reranker_snapshot_path(root, spec)
    components: dict[str, dict[str, int | str]] = {
        name: {
            "sha256": file_facts(snapshot / name).sha256,
            "bytes": file_facts(snapshot / name).bytes,
        }
        for name in _reranker_required_files(spec)
    }
    runtime_hash = canonical_sha256(
        [{"path": name, **components[name]} for name in _reranker_required_files(spec)]
    )
    if runtime_hash != select_runtime_artifacts_sha256(spec):
        raise DatasetIntegrityError(
            "reranker runtime artifacts differ from the registered official bytes"
        )
    return components, runtime_hash, components[select_onnx_file(spec)]


def _reranker_snapshot_path(root: Path, spec: RerankerSpec) -> Path:
    repository = f"models--{spec.repo_id.replace('/', '--')}"
    return root / "data/cache/models/reranker" / repository / "snapshots" / spec.revision


def _normalize_architecture(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "aarch64": "arm64",
        "arm64": "arm64",
        "amd64": "x86_64",
        "x86_64": "x86_64",
    }
    return aliases.get(normalized, normalized)


def _required_string(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"reranker {key} must be a non-empty string")
    return value


def _required_sha256(values: Mapping[str, Any], key: str) -> str:
    value = _required_string(values, key)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ConfigError(f"reranker {key} must be a lowercase SHA-256")
    return value


def _required_positive_int(values: Mapping[str, Any], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"reranker {key} must be a positive integer")
    return value
