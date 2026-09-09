from __future__ import annotations

import importlib.metadata
import json
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from poc.bm25 import load_prepared_queries, verify_wands_bm25_selection
from poc.config import WANDS_INT8_MODEL_NAMES, ModelSpec, load_model_registry
from poc.datasets import DatasetIntegrityError, file_facts, load_wands_config
from poc.dense import RESULT_SIZE, encode_queries, query_vectors_sha256
from poc.embedding_cache import EmbeddingBackend
from poc.evaluation import Evaluation, evaluate_run
from poc.evaluation_crosscheck import cross_check_run
from poc.experiments import BM25Experiments, BM25Profile, load_bm25_experiments
from poc.index_evidence import (
    content_addressed_pipeline_id,
    verify_live_search_pipeline,
)
from poc.indexing import (
    load_wands_index_config,
    vector_field_name,
    verify_wands_lexical_index,
    wands_completion_provenance_evidence,
    wands_recorded_provenance_valid,
)
from poc.int8_dense import verify_wands_int8_dense_summary
from poc.manifest import canonical_sha256, collect_index_facts, read_json, write_json
from poc.os_client import OpenSearchClient
from poc.provenance import collect_manifest_provenance, require_registered_opensearch_client
from poc.query_runtime import (
    QueryRuntimeSpec,
    create_int8_query_backend,
    load_query_runtime_spec,
)
from poc.search import build_exact_hybrid_request, build_normalization_pipeline
from poc.trec import RunRecord, identifier_sort_key, read_run, write_run

PAGINATION_DEPTH = 100
_EXPECTED_OS_VERSION = "3.8.0"
_ARTIFACT_ENTRY_KEYS = {
    "run",
    "manifest",
    "metrics_file",
    "metrics",
    "run_sha256",
    "manifest_sha256",
    "metrics_sha256",
}
_RUN_MANIFEST_KEYS = {
    "schema_version",
    "dataset",
    "method",
    "split",
    "tag",
    "decision_scope",
    "eligible_for_decision",
    "ineligibility_reason",
    "benchmark_provenance",
    "benchmark_provenance_valid",
    "completion_code_revision",
    "upstream_evidence",
    "upstream_artifact_bindings_valid_at_completion",
    "selected_model_eligible",
    "declared_variable",
    "opensearch_version",
    "index",
    "index_manifest_sha256",
    "index_evidence",
    "index_definition_sha256",
    "vector_field",
    "quality_query",
    "ranking_policy",
    "trec_score_source",
    "pagination_depth",
    "pipeline_id",
    "pipeline",
    "pipeline_sha256",
    "fusion",
    "fusion_sha256",
    "method_definition",
    "method_definition_sha256",
    "bm25_profile",
    "bm25_profile_sha256",
    "bm25_selection_sha256",
    "bm25_selection",
    "model",
    "model_sha256",
    "model_evidence",
    "document_encoder_runtime",
    "query_encoder_runtime",
    "query_encoder_device",
    "query_encoder_artifact_sha256",
    "query_runtime_spec",
    "query_encoding_seconds",
    "query_vectors_sha256",
    "query_template_sha256",
    "query_template_example",
    "query_file",
    "qrels_file",
    "run_file",
    "metrics_file",
    "evaluation_packages",
}
_METRICS_KEYS = {
    "schema_version",
    "dataset",
    "method",
    "tag",
    "split",
    "query_file",
    "qrels_file",
    "run_file",
    "metrics",
    "per_query",
    "evaluator_crosscheck",
}
_HYBRID_SUMMARY_KEYS = {
    "schema_version",
    "dataset",
    "method",
    "method_definition",
    "method_definition_sha256",
    "benchmark_provenance",
    "benchmark_provenance_valid",
    "completion_code_revision",
    "upstream_evidence",
    "upstream_artifact_bindings_valid_at_completion",
    "required_source_manifest_keys",
    "source_manifest_bindings",
    "source_manifest_bindings_valid",
    "model",
    "model_selection_metric",
    "model_selection_split",
    "model_selection_tie_break",
    "candidate_model_dev_metrics",
    "query_runtime",
    "query_runtime_spec",
    "eligible_for_decision",
    "ineligibility_reason",
    "quality_ineligibility_reasons",
    "bm25_profile",
    "bm25_profile_sha256",
    "selection_metric",
    "selection_split",
    "tie_break",
    "hybrid_weight_tuning_budget",
    "bm25_tuning_budget",
    "registered_lexical_weight_grid",
    "selected_lexical_weight",
    "candidate_dev_metrics",
    "held_out_test_metrics",
    "tuned_bm25_held_out_ndcg@10",
    "hybrid_minus_tuned_bm25_ndcg@10",
    "competent_integrator_bm25_held_out_ndcg@10",
    "hybrid_minus_competent_integrator_bm25_ndcg@10",
    "parity_status",
    "quality_guard",
    "self_retrieval_status",
    "artifacts",
}


@dataclass(frozen=True, slots=True)
class HybridRunResult:
    tag: str
    run_path: Path
    manifest_path: Path
    metrics_path: Path
    evaluation: Evaluation
    records: list[RunRecord]

    def selection_entry(self, root: Path) -> dict[str, object]:
        return {
            "run": str(self.run_path.relative_to(root)),
            "manifest": str(self.manifest_path.relative_to(root)),
            "metrics_file": str(self.metrics_path.relative_to(root)),
            "metrics": self.evaluation.metrics,
            "run_sha256": file_facts(self.run_path).sha256,
            "manifest_sha256": file_facts(self.manifest_path).sha256,
            "metrics_sha256": file_facts(self.metrics_path).sha256,
        }


def _json_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(json.dumps(value)))


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }


def hybrid_upstream_artifact_bindings_valid(
    root: Path,
    evidence: Mapping[str, Any],
) -> bool:
    root_resolved = root.resolve()
    found = 0

    def verify(value: object) -> bool:
        nonlocal found
        if isinstance(value, Mapping):
            if set(value) == {"path", "sha256", "bytes"}:
                found += 1
                relative = value.get("path")
                expected_sha256 = value.get("sha256")
                expected_bytes = value.get("bytes")
                if (
                    not isinstance(relative, str)
                    or not relative
                    or Path(relative).is_absolute()
                    or not isinstance(expected_sha256, str)
                    or not isinstance(expected_bytes, int)
                    or isinstance(expected_bytes, bool)
                ):
                    return False
                path = (root / relative).resolve()
                if not path.is_relative_to(root_resolved) or not path.is_file():
                    return False
                facts = file_facts(path)
                return facts.sha256 == expected_sha256 and facts.bytes == expected_bytes
            return all(verify(item) for item in value.values())
        if isinstance(value, list | tuple):
            return all(verify(item) for item in value)
        return True

    valid = verify(evidence)
    return valid and found > 0


def expected_hybrid_artifact_keys(experiments: BM25Experiments) -> set[str]:
    return {
        *(f"dev/lw{_weight_label(weight)}" for weight in experiments.lexical_weight_grid),
        "test/selected",
    }


def derive_hybrid_run_eligibility(
    *,
    decision_scope: str,
    benchmark_provenance_valid: object,
    upstream_evidence_eligible: object,
    selected_model_eligible: object,
) -> tuple[bool, str | None]:
    if decision_scope == "tuning":
        return False, "development tuning result"
    if decision_scope != "held_out":
        return False, "artifact is not a held-out decision result"
    reasons: list[str] = []
    if benchmark_provenance_valid is not True:
        reasons.append("run has no valid clean committed start provenance")
    if upstream_evidence_eligible is not True:
        reasons.append("hybrid upstream evidence is not decision eligible")
    if selected_model_eligible is not True:
        reasons.append("selected int8 dense model is not decision eligible")
    return not reasons, "; ".join(reasons) or None


def hybrid_upstream_evidence_eligible(
    *,
    index_eligible: object,
    bm25_eligible: object,
    int8_summary_eligible: object,
) -> bool:
    return (
        index_eligible is True
        and bm25_eligible is True
        and int8_summary_eligible is True
    )


def verify_registered_hybrid_inputs(
    *,
    root: Path,
    index: str,
    models: tuple[ModelSpec, ...],
    runtime: QueryRuntimeSpec,
    experiments: BM25Experiments,
) -> dict[str, Any]:
    registry = load_model_registry(root / "config/models.toml")
    expected_models = tuple(registry[name] for name in WANDS_INT8_MODEL_NAMES)
    if models != expected_models:
        raise DatasetIntegrityError(
            "hybrid model candidates differ from the registered WANDS int8 model order"
        )
    registered_runtime = load_query_runtime_spec(root / "config/query_runtime.toml")
    if runtime != registered_runtime:
        raise DatasetIntegrityError("hybrid query runtime differs from registered configuration")
    registered_experiments = load_bm25_experiments(root / "config/experiments.toml")
    if experiments != registered_experiments:
        raise DatasetIntegrityError("hybrid experiment configuration differs from registry")
    index_spec = load_wands_index_config(root / "config/indexes.toml")
    if index != index_spec.name:
        raise DatasetIntegrityError("hybrid index differs from registered WANDS index")
    return _json_dict(
        {
            "index": index_spec.name,
            "index_spec": asdict(index_spec),
            "models": [model.name for model in expected_models],
            "model_specs": [asdict(model) for model in expected_models],
            "query_runtime": asdict(registered_runtime),
            "experiments": asdict(registered_experiments),
            "config_artifacts": {
                "datasets": _artifact(root, root / "config/datasets.toml"),
                "indexes": _artifact(root, root / "config/indexes.toml"),
                "models": _artifact(root, root / "config/models.toml"),
                "query_runtime": _artifact(root, root / "config/query_runtime.toml"),
                "experiments": _artifact(root, root / "config/experiments.toml"),
            },
        }
    )


def collect_verified_hybrid_upstream_evidence(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    models: tuple[ModelSpec, ...],
    runtime: QueryRuntimeSpec,
    experiments: BM25Experiments,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    registered_inputs = verify_registered_hybrid_inputs(
        root=root,
        index=index,
        models=models,
        runtime=runtime,
        experiments=experiments,
    )
    dataset_spec = load_wands_config(root / "config/datasets.toml")
    index_spec = load_wands_index_config(root / "config/indexes.toml")
    index_verification = verify_wands_lexical_index(
        client,
        dataset_spec=dataset_spec,
        index_spec=index_spec,
        prepared_directory=root / "data/prepared/wands",
        manifest_path=root / "results/wands/index-manifest.json",
    )
    expected_model_names = [model.name for model in models]
    if index_verification.get("vector_models") != expected_model_names:
        raise DatasetIntegrityError(
            "live WANDS vector index does not contain exactly the registered int8 models"
        )
    bm25_selection = verify_wands_bm25_selection(
        client,
        root=root,
        index=index,
        experiments=experiments,
    )
    int8_summary = verify_wands_int8_dense_summary(
        client,
        root=root,
        index=index,
        models=models,
        runtime=runtime,
    )
    int8_models_value = int8_summary.get("models")
    if not isinstance(int8_models_value, dict) or set(int8_models_value) != set(
        expected_model_names
    ):
        raise DatasetIntegrityError("int8 dense summary model set differs from registry")
    int8_models = cast(dict[str, dict[str, Any]], int8_models_value)
    model_eligibility = {
        name: int8_models[name].get("eligible_for_decision") is True
        for name in expected_model_names
    }
    root_response = cast(dict[str, Any], client.request("GET", "/"))
    version_value = root_response.get("version")
    if not isinstance(version_value, dict):
        raise DatasetIntegrityError("OpenSearch root response has no version")
    version = str(version_value.get("number", ""))
    if version != _EXPECTED_OS_VERSION:
        raise DatasetIntegrityError(
            f"hybrid OpenSearch version must be {_EXPECTED_OS_VERSION}, found {version!r}"
        )
    index_manifest_path = root / "results/wands/index-manifest.json"
    index_manifest = cast(dict[str, Any], read_json(index_manifest_path))
    preparation = index_manifest.get("preparation_evidence")
    if not isinstance(preparation, dict):
        raise DatasetIntegrityError("WANDS vector index has no preparation evidence")
    bm25_source_artifacts = _summary_entry_source_artifacts(
        root,
        bm25_selection.get("artifacts"),
        label="BM25 selection",
    )
    int8_source_artifacts = _summary_entry_source_artifacts(
        root,
        int8_summary.get("artifacts"),
        label="int8 dense summary",
    )
    evidence = _json_dict(
        {
            "schema_version": 2,
            "registered_inputs": registered_inputs,
            "opensearch_version": version,
            "prepared_wands_manifest": _artifact(
                root, root / "data/prepared/wands/manifest.json"
            ),
            "preparation_evidence": preparation,
            "preparation_evidence_sha256": canonical_sha256(preparation),
            "vector_index_manifest": _artifact(root, index_manifest_path),
            "vector_index_verification": index_verification,
            "embedding_manifests": {
                model.name: _artifact(
                    root,
                    root / f"results/wands/embeddings/{model.name}.manifest.json",
                )
                for model in models
            },
            "bm25_selection": _artifact(
                root, root / "results/wands/bm25-selection.json"
            ),
            "bm25_source_artifacts": bm25_source_artifacts,
            "bm25_quality_eligible_for_decision": (
                bm25_selection.get("quality_evidence_eligible_for_decision") is True
            ),
            "int8_dense_summary": _artifact(
                root, root / "results/wands/int8-dense-summary.json"
            ),
            "int8_source_artifacts": int8_source_artifacts,
            "int8_runtime_evidence": {
                name: int8_models[name].get("runtime_evidence")
                for name in expected_model_names
            },
            "int8_summary_quality_eligible_for_decision": (
                int8_summary.get("quality_evidence_eligible_for_decision") is True
            ),
            "int8_model_eligibility": model_eligibility,
            "eligible_for_decision": hybrid_upstream_evidence_eligible(
                index_eligible=index_verification.get(
                    "quality_evidence_eligible_for_decision"
                ),
                bm25_eligible=bm25_selection.get(
                    "quality_evidence_eligible_for_decision"
                ),
                int8_summary_eligible=int8_summary.get(
                    "quality_evidence_eligible_for_decision"
                ),
            ),
        }
    )
    return evidence, bm25_selection, int8_summary


def verify_wands_index_after_decision_replay(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    expected_verification: object,
) -> None:
    dataset_spec = load_wands_config(root / "config/datasets.toml")
    index_spec = load_wands_index_config(root / "config/indexes.toml")
    if index_spec.name != index:
        raise DatasetIntegrityError("decision replay used an unregistered WANDS index")
    completion_verification = verify_wands_lexical_index(
        client,
        dataset_spec=dataset_spec,
        index_spec=index_spec,
        prepared_directory=root / "data/prepared/wands",
        manifest_path=root / "results/wands/index-manifest.json",
    )
    if completion_verification != expected_verification:
        raise DatasetIntegrityError("WANDS index changed during decision replay")


def _summary_entry_source_artifacts(
    root: Path,
    entries_value: object,
    *,
    label: str,
) -> dict[str, dict[str, dict[str, object]]]:
    if not isinstance(entries_value, Mapping) or not entries_value:
        raise DatasetIntegrityError(f"{label} artifacts are missing")
    artifacts: dict[str, dict[str, dict[str, object]]] = {}
    for key, entry_value in entries_value.items():
        if not isinstance(key, str) or not isinstance(entry_value, Mapping):
            raise DatasetIntegrityError(f"{label} artifact entry is invalid")
        entry_artifacts: dict[str, dict[str, object]] = {}
        for field in ("run", "manifest", "metrics_file"):
            relative = entry_value.get(field)
            if not isinstance(relative, str) or not relative:
                raise DatasetIntegrityError(f"{label} artifact {key} has no {field}")
            entry_artifacts[field] = _artifact(root, root / relative)
        artifacts[key] = entry_artifacts
    return artifacts


def selected_int8_runtime_fields(
    model_evidence: Mapping[str, Any],
) -> tuple[object, dict[str, object], object]:
    runtime_value = model_evidence.get("runtime_evidence")
    if not isinstance(runtime_value, Mapping):
        raise DatasetIntegrityError("selected int8 model has no runtime evidence")
    quality_guard = {
        "artifact": runtime_value.get("quality_guard_artifact"),
        "status": runtime_value.get("quality_guard_status"),
        "passed": runtime_value.get("quality_guard_passed"),
        "verification_error": runtime_value.get("quality_guard_verification_error"),
    }
    return (
        runtime_value.get("parity_status"),
        quality_guard,
        runtime_value.get("self_retrieval_status"),
    )


def hybrid_method_definition(
    *,
    model: ModelSpec,
    declared_variable: str,
) -> dict[str, object]:
    return {
        "quality_query": "native_hybrid_with_exact_knn_score_script",
        "ranking_policy": "opensearch_fused_score_order",
        "trec_score_source": "opensearch_rank_derived_100_to_1",
        "pagination_depth": PAGINATION_DEPTH,
        "vector_field": vector_field_name(model),
        "declared_variable": declared_variable,
    }


def hybrid_trec_score_for_rank(rank: int) -> float:
    """Persist fused engine order without depending on serialized float scores."""
    if not 1 <= rank <= RESULT_SIZE:
        raise ValueError(f"hybrid rank must be between 1 and {RESULT_SIZE}")
    return float(RESULT_SIZE - rank + 1)


def select_hybrid_weight(
    weights: Sequence[float],
    metrics: Mapping[float, Mapping[str, float]],
    *,
    metric: str,
) -> float:
    if not weights:
        raise ValueError("hybrid weight grid must not be empty")
    missing = [weight for weight in weights if weight not in metrics]
    if missing:
        raise ValueError(f"hybrid metrics missing registered weights: {missing}")
    return max(weights, key=lambda weight: metrics[weight][metric])


def select_hybrid_model(
    models: Sequence[str],
    metrics: Mapping[str, Mapping[str, float]],
    *,
    metric: str,
) -> str:
    if not models:
        raise ValueError("hybrid model candidates must not be empty")
    missing = [model for model in models if model not in metrics]
    if missing:
        raise ValueError(f"hybrid metrics missing candidate models: {missing}")
    return max(models, key=lambda model: metrics[model][metric])


def hybrid_pipeline_id(model: ModelSpec, lexical_weight: float) -> str:
    if not 0.0 <= lexical_weight <= 1.0:
        raise ValueError("hybrid lexical weight must be in [0, 1]")
    weight_basis_points = round(lexical_weight * 100)
    definition = build_normalization_pipeline(lexical_weight=lexical_weight)
    return content_addressed_pipeline_id(
        (
            f"opensearch-hybrid-wands-{model.name.replace('_', '-')}"
            f"-minmax-lw{weight_basis_points:03d}-v1"
        ),
        definition,
    )


def execute_exact_hybrid_run(
    client: OpenSearchClient,
    *,
    index: str,
    queries: Mapping[str, str],
    query_vectors: Mapping[str, NDArray[np.float32]],
    profile: BM25Profile,
    model: ModelSpec,
    lexical_weight: float,
    tag: str,
) -> list[RunRecord]:
    pipeline = hybrid_pipeline_id(model, lexical_weight)
    pipeline_definition = build_normalization_pipeline(lexical_weight=lexical_weight)
    return execute_exact_hybrid_pipeline_run(
        client,
        index=index,
        queries=queries,
        query_vectors=query_vectors,
        profile=profile,
        model=model,
        pipeline_id=pipeline,
        pipeline_definition=pipeline_definition,
        tag=tag,
    )


def execute_exact_hybrid_pipeline_run(
    client: OpenSearchClient,
    *,
    index: str,
    queries: Mapping[str, str],
    query_vectors: Mapping[str, NDArray[np.float32]],
    profile: BM25Profile,
    model: ModelSpec,
    pipeline_id: str,
    pipeline_definition: dict[str, Any],
    tag: str,
) -> list[RunRecord]:
    if set(queries) != set(query_vectors):
        raise ValueError("hybrid queries and query vectors differ")
    if not pipeline_id:
        raise ValueError("hybrid pipeline ID must not be blank")
    client.put_search_pipeline(pipeline_id, pipeline_definition)
    verify_live_search_pipeline(
        client,
        pipeline_id=pipeline_id,
        expected_definition=pipeline_definition,
    )
    records: list[RunRecord] = []
    vector_field = vector_field_name(model)
    for query_id in sorted(queries, key=identifier_sort_key):
        lexical_query = profile.query(queries[query_id])
        request = build_exact_hybrid_request(
            lexical_query,
            query_vectors[query_id].tolist(),
            vector_field=vector_field,
            size=RESULT_SIZE,
            pagination_depth=PAGINATION_DEPTH,
        )
        response = client.search(index, request, pipeline=pipeline_id)
        hits = response.get("hits", {}).get("hits", [])
        if not isinstance(hits, list) or not hits:
            raise ValueError(f"hybrid query {query_id} returned no hits")
        for rank, hit in enumerate(hits, start=1):
            if not isinstance(hit, dict) or hit.get("_score") is None:
                raise ValueError(f"hybrid result for {query_id} has no score")
            records.append(
                RunRecord(
                    query_id=query_id,
                    document_id=str(hit["_id"]),
                    rank=rank,
                    score=hybrid_trec_score_for_rank(rank),
                    tag=tag,
                )
            )
    verify_live_search_pipeline(
        client,
        pipeline_id=pipeline_id,
        expected_definition=pipeline_definition,
    )
    return records


def write_hybrid_bundle(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    model: ModelSpec,
    backend: EmbeddingBackend,
    profile: BM25Profile,
    lexical_weight: float,
    query_path: Path,
    qrels_path: Path,
    split: str,
    tag: str,
    records: list[RunRecord],
    query_vectors_hash: str,
    encoding_seconds: float,
    eligible_for_decision: bool,
    ineligibility_reason: str | None,
    decision_scope: str,
    benchmark_provenance: dict[str, Any] | None = None,
    upstream_evidence: dict[str, Any] | None = None,
    selected_model_eligible: bool | None = None,
) -> HybridRunResult:
    pipeline_definition = build_normalization_pipeline(lexical_weight=lexical_weight)
    return write_hybrid_pipeline_bundle(
        client,
        root=root,
        index=index,
        model=model,
        backend=backend,
        profile=profile,
        pipeline_id=hybrid_pipeline_id(model, lexical_weight),
        pipeline_definition=pipeline_definition,
        fusion={
            "normalization": "min_max",
            "combination": "arithmetic_mean",
            "lexical_weight": lexical_weight,
            "dense_weight": 1.0 - lexical_weight,
        },
        declared_variable=("lexical_weight" if decision_scope == "tuning" else "retrieval_arm"),
        metrics_directory="hybrid",
        query_path=query_path,
        qrels_path=qrels_path,
        split=split,
        tag=tag,
        records=records,
        query_vectors_hash=query_vectors_hash,
        encoding_seconds=encoding_seconds,
        eligible_for_decision=eligible_for_decision,
        ineligibility_reason=ineligibility_reason,
        decision_scope=decision_scope,
        benchmark_provenance=benchmark_provenance,
        upstream_evidence=upstream_evidence,
        selected_model_eligible=selected_model_eligible,
        method_name="native_hybrid_min_max_exact_dense",
    )


def write_hybrid_pipeline_bundle(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    model: ModelSpec,
    backend: EmbeddingBackend,
    profile: BM25Profile,
    pipeline_id: str,
    pipeline_definition: dict[str, Any],
    fusion: dict[str, object],
    declared_variable: str,
    metrics_directory: str,
    query_path: Path,
    qrels_path: Path,
    split: str,
    tag: str,
    records: list[RunRecord],
    query_vectors_hash: str,
    encoding_seconds: float,
    eligible_for_decision: bool,
    ineligibility_reason: str | None,
    decision_scope: str,
    index_manifest_path: Path | None = None,
    benchmark_provenance: dict[str, Any] | None = None,
    upstream_evidence: dict[str, Any] | None = None,
    selected_model_eligible: bool | None = None,
    method_name: str | None = None,
) -> HybridRunResult:
    del eligible_for_decision, ineligibility_reason
    if not metrics_directory or "/" in metrics_directory:
        raise ValueError("hybrid metrics directory must be one path segment")
    if benchmark_provenance is None:
        benchmark_provenance = _json_dict(collect_manifest_provenance(root))
    else:
        benchmark_provenance = _json_dict(benchmark_provenance)
    if upstream_evidence is None:
        upstream_evidence = {
            "schema_version": 2,
            "eligible_for_decision": False,
            "ineligibility_reason": "no verified hybrid upstream evidence supplied",
        }
    else:
        upstream_evidence = _json_dict(upstream_evidence)
    if selected_model_eligible is None:
        eligibility_map = upstream_evidence.get("int8_model_eligibility")
        selected_model_eligible = (
            isinstance(eligibility_map, dict)
            and eligibility_map.get(model.name) is True
        )
    if method_name is None:
        method_name = {
            "hybrid": "native_hybrid_min_max_exact_dense",
            "rrf": "native_hybrid_rrf_exact_dense",
        }.get(metrics_directory, f"native_hybrid_{metrics_directory}")
    run_path = root / f"runs/{tag}.trec"
    manifest_path = root / f"runs/{tag}.manifest.json"
    metrics_path = root / f"results/wands/{metrics_directory}/{tag}.metrics.json"
    write_run(run_path, records)
    evaluation = evaluate_run(qrels_path, run_path)
    crosscheck = cross_check_run(qrels_path, run_path, root=root)
    query_facts = file_facts(query_path)
    qrels_facts = file_facts(qrels_path)
    run_facts = file_facts(run_path)
    write_json(
        metrics_path,
        {
            "schema_version": 2,
            "dataset": "WANDS",
            "method": method_name,
            "tag": tag,
            "split": split,
            "query_file": {
                "path": str(query_path.relative_to(root)),
                "sha256": query_facts.sha256,
            },
            "qrels_file": {
                "path": str(qrels_path.relative_to(root)),
                "sha256": qrels_facts.sha256,
            },
            "run_file": {
                "path": str(run_path.relative_to(root)),
                "sha256": run_facts.sha256,
                "records": len(records),
            },
            "metrics": evaluation.metrics,
            "per_query": evaluation.per_query,
            "evaluator_crosscheck": crosscheck.to_dict(),
        },
    )

    if index_manifest_path is None:
        index_manifest_path = root / "results/wands/index-manifest.json"
    model_manifest_path = root / f"results/wands/embeddings/{model.name}.manifest.json"
    bm25_selection_path = root / "results/wands/bm25-selection.json"
    index_manifest = cast(dict[str, Any], read_json(index_manifest_path))
    model_manifest = cast(dict[str, Any], read_json(model_manifest_path))
    bm25_selection = cast(dict[str, Any], read_json(bm25_selection_path))
    root_response = cast(dict[str, Any], client.request("GET", "/"))
    version = str(cast(dict[str, Any], root_response["version"])["number"])
    query_facts = file_facts(query_path)
    qrels_facts = file_facts(qrels_path)
    run_facts = file_facts(run_path)
    profile_values = profile.to_dict()
    profile_sha256 = canonical_sha256(profile_values)
    if profile_sha256 != bm25_selection.get("selected_profile_sha256"):
        raise DatasetIntegrityError("hybrid lexical clause is not the frozen tuned BM25 profile")
    provenance_valid, completion_revision = wands_completion_provenance_evidence(
        root,
        benchmark_provenance,
    )
    upstream_bindings_valid = hybrid_upstream_artifact_bindings_valid(
        root,
        upstream_evidence,
    )
    derived_eligible, derived_reason = derive_hybrid_run_eligibility(
        decision_scope=decision_scope,
        benchmark_provenance_valid=provenance_valid,
        upstream_evidence_eligible=(
            upstream_evidence.get("eligible_for_decision") is True
            and upstream_bindings_valid
        ),
        selected_model_eligible=selected_model_eligible,
    )
    method_definition = hybrid_method_definition(
        model=model,
        declared_variable=declared_variable,
    )
    index_evidence = {
        "manifest": _artifact(root, index_manifest_path),
        "definition_sha256": index_manifest["index_definition_sha256"],
    }
    model_evidence = {
        "manifest": _artifact(root, model_manifest_path),
        "model_artifact_sha256": model_manifest["model_artifact_sha256"],
        "encoder_runtime": model_manifest["encoder_runtime"],
    }
    manifest: dict[str, object] = {
        "schema_version": 2,
        "dataset": "WANDS",
        "method": method_name,
        "split": split,
        "tag": tag,
        "decision_scope": decision_scope,
        "eligible_for_decision": derived_eligible,
        "ineligibility_reason": derived_reason,
        "benchmark_provenance": benchmark_provenance,
        "benchmark_provenance_valid": provenance_valid,
        "completion_code_revision": completion_revision,
        "upstream_evidence": upstream_evidence,
        "upstream_artifact_bindings_valid_at_completion": upstream_bindings_valid,
        "selected_model_eligible": selected_model_eligible,
        "declared_variable": declared_variable,
        "opensearch_version": version,
        "index": asdict(collect_index_facts(client, index)),
        "index_manifest_sha256": file_facts(index_manifest_path).sha256,
        "index_evidence": index_evidence,
        "index_definition_sha256": index_manifest["index_definition_sha256"],
        "vector_field": vector_field_name(model),
        "quality_query": "native_hybrid_with_exact_knn_score_script",
        "ranking_policy": "opensearch_fused_score_order",
        "trec_score_source": "opensearch_rank_derived_100_to_1",
        "pagination_depth": PAGINATION_DEPTH,
        "pipeline_id": pipeline_id,
        "pipeline": pipeline_definition,
        "pipeline_sha256": canonical_sha256(pipeline_definition),
        "fusion": fusion,
        "fusion_sha256": canonical_sha256(fusion),
        "method_definition": method_definition,
        "method_definition_sha256": canonical_sha256(method_definition),
        "bm25_profile": profile_values,
        "bm25_profile_sha256": profile_sha256,
        "bm25_selection_sha256": file_facts(bm25_selection_path).sha256,
        "bm25_selection": _artifact(root, bm25_selection_path),
        "model": asdict(model),
        "model_sha256": model_manifest["model_artifact_sha256"],
        "model_evidence": model_evidence,
        "document_encoder_runtime": model_manifest["encoder_runtime"],
        "query_encoder_runtime": backend.runtime,
        "query_encoder_device": backend.device,
        "query_encoder_artifact_sha256": backend.model_artifact_sha256,
        "query_runtime_spec": asdict(
            load_query_runtime_spec(root / "config/query_runtime.toml")
        ),
        "query_encoding_seconds": encoding_seconds,
        "query_vectors_sha256": query_vectors_hash,
        "query_template_sha256": canonical_sha256(
            {"prefix": model.query_prefix, "template": model.query_template}
        ),
        "query_template_example": model.render_query("example product query"),
        "query_file": {
            "path": str(query_path.relative_to(root)),
            "sha256": query_facts.sha256,
        },
        "qrels_file": {
            "path": str(qrels_path.relative_to(root)),
            "sha256": qrels_facts.sha256,
        },
        "run_file": {
            "path": str(run_path.relative_to(root)),
            "sha256": run_facts.sha256,
            "records": len(records),
        },
        "metrics_file": _artifact(root, metrics_path),
        "evaluation_packages": {
            "ranx": importlib.metadata.version("ranx"),
            "pytrec_eval": importlib.metadata.version("pytrec-eval-terrier"),
        },
    }
    write_json(manifest_path, manifest)
    return HybridRunResult(
        tag=tag,
        run_path=run_path,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        evaluation=evaluation,
        records=records,
    )


def _hybrid_source_manifest_bindings(
    *,
    root: Path,
    artifacts: Mapping[str, Mapping[str, Any]],
    expected_keys: set[str],
    benchmark_provenance: Mapping[str, Any],
    upstream_evidence: Mapping[str, Any],
    selected_model_eligible: bool,
) -> tuple[dict[str, dict[str, object]], bool, list[str]]:
    if set(artifacts) != expected_keys:
        raise DatasetIntegrityError("hybrid summary artifact set differs")
    bindings: dict[str, dict[str, object]] = {}
    reasons: list[str] = []
    for key in sorted(expected_keys):
        entry = artifacts[key]
        if set(entry) != _ARTIFACT_ENTRY_KEYS:
            reasons.append(f"hybrid artifact entry fields differ: {key}")
            continue
        manifest_path = root / str(entry["manifest"])
        manifest = cast(dict[str, Any], read_json(manifest_path))
        bindings[key] = _artifact(root, manifest_path)
        expected_split = "test" if key == "test/selected" else "dev"
        expected_scope = "held_out" if expected_split == "test" else "tuning"
        provenance_valid = wands_recorded_provenance_valid(
            root,
            provenance=manifest.get("benchmark_provenance"),
            completion_revision=manifest.get("completion_code_revision"),
        )
        upstream_bindings_valid = hybrid_upstream_artifact_bindings_valid(
            root,
            upstream_evidence,
        )
        expected_eligible, expected_reason = derive_hybrid_run_eligibility(
            decision_scope=expected_scope,
            benchmark_provenance_valid=provenance_valid,
            upstream_evidence_eligible=(
                upstream_evidence.get("eligible_for_decision") is True
                and upstream_bindings_valid
            ),
            selected_model_eligible=selected_model_eligible,
        )
        if not (
            manifest.get("schema_version") == 2
            and manifest.get("dataset") == "WANDS"
            and manifest.get("split") == expected_split
            and manifest.get("decision_scope") == expected_scope
            and manifest.get("benchmark_provenance") == benchmark_provenance
            and manifest.get("benchmark_provenance_valid") is provenance_valid
            and manifest.get("upstream_evidence") == upstream_evidence
            and manifest.get("upstream_artifact_bindings_valid_at_completion")
            is upstream_bindings_valid
            and manifest.get("selected_model_eligible") is selected_model_eligible
            and manifest.get("eligible_for_decision") is expected_eligible
            and manifest.get("ineligibility_reason") == expected_reason
            and entry.get("manifest_sha256") == file_facts(manifest_path).sha256
        ):
            reasons.append(f"hybrid source manifest binding differs: {key}")
    return bindings, not reasons, reasons


def _hybrid_summary_eligibility(
    *,
    benchmark_provenance_valid: object,
    upstream_evidence_eligible: object,
    selected_model_eligible: object,
    source_bindings_valid: object,
    held_out_artifact_eligible: object,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if benchmark_provenance_valid is not True:
        reasons.append("summary has no valid clean committed start provenance")
    if upstream_evidence_eligible is not True:
        reasons.append("hybrid upstream evidence is not decision eligible")
    if selected_model_eligible is not True:
        reasons.append("selected int8 dense model is not decision eligible")
    if source_bindings_valid is not True:
        reasons.append("hybrid source manifest bindings differ")
    if held_out_artifact_eligible is not True:
        reasons.append("selected held-out hybrid artifact is not decision eligible")
    return not reasons, reasons


def run_wands_hybrid_minmax_benchmark(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    models: tuple[ModelSpec, ...],
    runtime: QueryRuntimeSpec,
    experiments: BM25Experiments,
) -> dict[str, object]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    benchmark_provenance = _json_dict(collect_manifest_provenance(root))
    upstream_evidence, bm25_selection, int8_summary = (
        collect_verified_hybrid_upstream_evidence(
            client,
            root=root,
            index=index,
            models=models,
            runtime=runtime,
            experiments=experiments,
        )
    )
    model_metrics = int8_dev_model_metrics(root, models, summary=int8_summary)
    selected_model_name = select_hybrid_model(
        tuple(model.name for model in models),
        model_metrics,
        metric=experiments.primary_metric,
    )
    model = next(model for model in models if model.name == selected_model_name)
    int8_models = cast(dict[str, dict[str, Any]], int8_summary["models"])
    selected_int8_evidence = int8_models[model.name]
    selected_model_eligible = (
        selected_int8_evidence.get("eligible_for_decision") is True
    )
    prepared = root / "data/prepared/wands"
    query_paths = {split: prepared / f"queries.{split}.jsonl" for split in ("dev", "test")}
    qrels_paths = {split: prepared / f"qrels.{split}.trec" for split in ("dev", "test")}
    queries = {
        split: load_prepared_queries(query_paths[split], expected_split=split)
        for split in ("dev", "test")
    }
    all_queries = {**queries["dev"], **queries["test"]}
    profile = BM25Profile.from_mapping(cast(dict[str, Any], bm25_selection["selected_profile"]))
    backend = create_int8_query_backend(root=root, model=model, runtime=runtime)
    started = time.perf_counter()
    vectors = encode_queries(
        backend,
        model=model,
        queries=all_queries,
        batch_size=runtime.batch_size,
    )
    encoding_seconds = time.perf_counter() - started

    dev_results: dict[float, HybridRunResult] = {}
    artifacts: dict[str, dict[str, object]] = {}
    for lexical_weight in experiments.lexical_weight_grid:
        weight_label = _weight_label(lexical_weight)
        tag = (
            f"wands-hybrid-{model.name.replace('_', '-')}-minmax-"
            f"lw{weight_label}-dev-int8-onnx-{runtime.quantization_config}"
        )
        records = execute_exact_hybrid_run(
            client,
            index=index,
            queries=queries["dev"],
            query_vectors={query_id: vectors[query_id] for query_id in queries["dev"]},
            profile=profile,
            model=model,
            lexical_weight=lexical_weight,
            tag=tag,
        )
        result = write_hybrid_bundle(
            client,
            root=root,
            index=index,
            model=model,
            backend=backend,
            profile=profile,
            lexical_weight=lexical_weight,
            query_path=query_paths["dev"],
            qrels_path=qrels_paths["dev"],
            split="dev",
            tag=tag,
            records=records,
            query_vectors_hash=query_vectors_sha256(
                {query_id: vectors[query_id] for query_id in queries["dev"]}
            ),
            encoding_seconds=encoding_seconds,
            eligible_for_decision=False,
            ineligibility_reason="development tuning result",
            decision_scope="tuning",
            benchmark_provenance=benchmark_provenance,
            upstream_evidence=upstream_evidence,
            selected_model_eligible=selected_model_eligible,
        )
        dev_results[lexical_weight] = result
        artifacts[f"dev/lw{weight_label}"] = result.selection_entry(root)

    candidate_metrics = {
        weight: result.evaluation.metrics for weight, result in dev_results.items()
    }
    selected_weight = select_hybrid_weight(
        experiments.lexical_weight_grid,
        candidate_metrics,
        metric=experiments.primary_metric,
    )
    selected_label = _weight_label(selected_weight)
    test_tag = (
        f"wands-hybrid-{model.name.replace('_', '-')}-minmax-"
        f"lw{selected_label}-test-int8-onnx-{runtime.quantization_config}"
    )
    test_records = execute_exact_hybrid_run(
        client,
        index=index,
        queries=queries["test"],
        query_vectors={query_id: vectors[query_id] for query_id in queries["test"]},
        profile=profile,
        model=model,
        lexical_weight=selected_weight,
        tag=test_tag,
    )
    test_result = write_hybrid_bundle(
        client,
        root=root,
        index=index,
        model=model,
        backend=backend,
        profile=profile,
        lexical_weight=selected_weight,
        query_path=query_paths["test"],
        qrels_path=qrels_paths["test"],
        split="test",
        tag=test_tag,
        records=test_records,
        query_vectors_hash=query_vectors_sha256(
            {query_id: vectors[query_id] for query_id in queries["test"]}
        ),
        encoding_seconds=encoding_seconds,
        eligible_for_decision=selected_model_eligible,
        ineligibility_reason=cast(str | None, selected_int8_evidence.get("ineligibility_reason")),
        decision_scope="held_out",
        benchmark_provenance=benchmark_provenance,
        upstream_evidence=upstream_evidence,
        selected_model_eligible=selected_model_eligible,
    )
    artifacts["test/selected"] = test_result.selection_entry(root)
    bm25_test = cast(
        dict[str, Any],
        cast(dict[str, Any], bm25_selection["held_out_test_metrics"])["tuned"],
    )
    competent_test = cast(
        dict[str, Any],
        cast(dict[str, Any], bm25_selection["held_out_test_metrics"])[
            "competent_integrator"
        ],
    )
    held_out_delta = test_result.evaluation.metrics["ndcg@10"] - float(bm25_test["ndcg@10"])
    competent_delta = test_result.evaluation.metrics["ndcg@10"] - float(
        competent_test["ndcg@10"]
    )
    expected_keys = expected_hybrid_artifact_keys(experiments)
    source_bindings, source_bindings_valid, source_binding_reasons = (
        _hybrid_source_manifest_bindings(
            root=root,
            artifacts=artifacts,
            expected_keys=expected_keys,
            benchmark_provenance=benchmark_provenance,
            upstream_evidence=upstream_evidence,
            selected_model_eligible=selected_model_eligible,
        )
    )
    provenance_valid, completion_revision = wands_completion_provenance_evidence(
        root,
        benchmark_provenance,
    )
    held_out_manifest = cast(
        dict[str, Any],
        read_json(root / str(artifacts["test/selected"]["manifest"])),
    )
    upstream_bindings_valid = hybrid_upstream_artifact_bindings_valid(
        root,
        upstream_evidence,
    )
    eligible, eligibility_reasons = _hybrid_summary_eligibility(
        benchmark_provenance_valid=provenance_valid,
        upstream_evidence_eligible=(
            upstream_evidence.get("eligible_for_decision") is True
            and upstream_bindings_valid
        ),
        selected_model_eligible=selected_model_eligible,
        source_bindings_valid=source_bindings_valid,
        held_out_artifact_eligible=held_out_manifest.get("eligible_for_decision"),
    )
    selected_pipeline = build_normalization_pipeline(lexical_weight=selected_weight)
    selected_fusion = {
        "normalization": "min_max",
        "combination": "arithmetic_mean",
        "lexical_weight": selected_weight,
        "dense_weight": 1.0 - selected_weight,
    }
    method_definition = {
        "method": "native_hybrid_min_max_exact_dense",
        "quality_query": "native_hybrid_with_exact_knn_score_script",
        "selection_metric": experiments.primary_metric,
        "selection_split": "dev",
        "tie_break": "first_weight_in_registered_order",
        "pipeline_id": hybrid_pipeline_id(model, selected_weight),
        "pipeline_sha256": canonical_sha256(selected_pipeline),
        "fusion_sha256": canonical_sha256(selected_fusion),
    }
    parity_status, quality_guard, self_retrieval_status = selected_int8_runtime_fields(
        selected_int8_evidence
    )
    summary: dict[str, object] = {
        "schema_version": 2,
        "dataset": "WANDS",
        "method": "native_hybrid_min_max_exact_dense",
        "method_definition": method_definition,
        "method_definition_sha256": canonical_sha256(method_definition),
        "benchmark_provenance": benchmark_provenance,
        "benchmark_provenance_valid": provenance_valid,
        "completion_code_revision": completion_revision,
        "upstream_evidence": upstream_evidence,
        "upstream_artifact_bindings_valid_at_completion": upstream_bindings_valid,
        "required_source_manifest_keys": sorted(expected_keys),
        "source_manifest_bindings": source_bindings,
        "source_manifest_bindings_valid": source_bindings_valid,
        "model": model.name,
        "model_selection_metric": experiments.primary_metric,
        "model_selection_split": "dev",
        "model_selection_tie_break": "first_model_in_registered_order",
        "candidate_model_dev_metrics": [
            {"model": candidate.name, "metrics": model_metrics[candidate.name]}
            for candidate in models
        ],
        "query_runtime": "dynamic_int8_onnx_cpu_batch_1",
        "query_runtime_spec": asdict(runtime),
        "eligible_for_decision": eligible,
        "ineligibility_reason": "; ".join(eligibility_reasons) or None,
        "quality_ineligibility_reasons": eligibility_reasons + source_binding_reasons,
        "bm25_profile": profile.to_dict(),
        "bm25_profile_sha256": canonical_sha256(profile.to_dict()),
        "selection_metric": experiments.primary_metric,
        "selection_split": "dev",
        "tie_break": "first_weight_in_registered_order",
        "hybrid_weight_tuning_budget": len(experiments.lexical_weight_grid),
        "bm25_tuning_budget": bm25_selection["bm25_tuning_budget"],
        "registered_lexical_weight_grid": list(experiments.lexical_weight_grid),
        "selected_lexical_weight": selected_weight,
        "candidate_dev_metrics": [
            {"lexical_weight": weight, "metrics": candidate_metrics[weight]}
            for weight in experiments.lexical_weight_grid
        ],
        "held_out_test_metrics": test_result.evaluation.metrics,
        "tuned_bm25_held_out_ndcg@10": bm25_test["ndcg@10"],
        "hybrid_minus_tuned_bm25_ndcg@10": held_out_delta,
        "competent_integrator_bm25_held_out_ndcg@10": competent_test["ndcg@10"],
        "hybrid_minus_competent_integrator_bm25_ndcg@10": competent_delta,
        "parity_status": parity_status,
        "quality_guard": quality_guard,
        "self_retrieval_status": self_retrieval_status,
        "artifacts": artifacts,
    }
    persisted_summary = _json_dict(summary)
    write_json(root / "results/wands/hybrid-summary.json", persisted_summary)
    return cast(dict[str, object], persisted_summary)


def verify_wands_hybrid_summary(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    models: tuple[ModelSpec, ...],
    runtime: QueryRuntimeSpec,
    experiments: BM25Experiments,
) -> dict[str, Any]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    summary = cast(dict[str, Any], read_json(root / "results/wands/hybrid-summary.json"))
    if (
        set(summary) != _HYBRID_SUMMARY_KEYS
        or summary.get("schema_version") != 2
        or summary.get("dataset") != "WANDS"
        or summary.get("method") != "native_hybrid_min_max_exact_dense"
    ):
        raise DatasetIntegrityError("hybrid summary schema, fields, dataset, or method differs")
    upstream_evidence, bm25_selection, int8_summary = (
        collect_verified_hybrid_upstream_evidence(
            client,
            root=root,
            index=index,
            models=models,
            runtime=runtime,
            experiments=experiments,
        )
    )
    if summary.get("upstream_evidence") != upstream_evidence:
        raise DatasetIntegrityError("hybrid summary upstream evidence differs")
    upstream_bindings_valid = hybrid_upstream_artifact_bindings_valid(
        root,
        upstream_evidence,
    )
    if (
        summary.get("upstream_artifact_bindings_valid_at_completion")
        is not upstream_bindings_valid
    ):
        raise DatasetIntegrityError("hybrid summary upstream artifact bindings differ")
    provenance = summary.get("benchmark_provenance")
    provenance_valid = wands_recorded_provenance_valid(
        root,
        provenance=provenance,
        completion_revision=summary.get("completion_code_revision"),
    )
    if summary.get("benchmark_provenance_valid") is not provenance_valid:
        raise DatasetIntegrityError("hybrid summary provenance validity differs")
    if not isinstance(provenance, Mapping):
        raise DatasetIntegrityError("hybrid summary start provenance is missing")
    model_metrics = int8_dev_model_metrics(root, models, summary=int8_summary)
    selected_model_name = select_hybrid_model(
        tuple(model.name for model in models),
        model_metrics,
        metric=experiments.primary_metric,
    )
    model = next(model for model in models if model.name == selected_model_name)
    int8_models = cast(dict[str, dict[str, Any]], int8_summary["models"])
    selected_int8_evidence = int8_models[model.name]
    selected_model_eligible = (
        selected_int8_evidence.get("eligible_for_decision") is True
    )
    if summary.get("model") != selected_model_name:
        raise DatasetIntegrityError("hybrid summary model differs")
    expected_model_candidates = [
        {"model": candidate.name, "metrics": model_metrics[candidate.name]} for candidate in models
    ]
    if summary.get("candidate_model_dev_metrics") != expected_model_candidates:
        raise DatasetIntegrityError("hybrid model-selection metrics differ")
    expected_profile = BM25Profile.from_mapping(
        cast(dict[str, Any], bm25_selection["selected_profile"])
    )
    if (
        summary.get("model_selection_metric") != experiments.primary_metric
        or summary.get("model_selection_split") != "dev"
        or summary.get("model_selection_tie_break")
        != "first_model_in_registered_order"
        or summary.get("query_runtime") != "dynamic_int8_onnx_cpu_batch_1"
        or summary.get("query_runtime_spec") != _json_dict(asdict(runtime))
        or summary.get("bm25_profile") != _json_dict(expected_profile.to_dict())
        or summary.get("bm25_profile_sha256")
        != canonical_sha256(expected_profile.to_dict())
        or summary.get("selection_metric") != experiments.primary_metric
        or summary.get("selection_split") != "dev"
        or summary.get("tie_break") != "first_weight_in_registered_order"
    ):
        raise DatasetIntegrityError("hybrid registered method or selection metadata differs")
    if summary.get("registered_lexical_weight_grid") != list(experiments.lexical_weight_grid):
        raise DatasetIntegrityError("hybrid registered weight grid differs")
    if (
        summary.get("hybrid_weight_tuning_budget")
        != len(experiments.lexical_weight_grid)
        or summary.get("bm25_tuning_budget") != bm25_selection.get("bm25_tuning_budget")
        or summary.get("hybrid_weight_tuning_budget")
        != summary.get("bm25_tuning_budget")
    ):
        raise DatasetIntegrityError("hybrid and BM25 tuning budgets differ")

    artifacts_value = summary.get("artifacts")
    if not isinstance(artifacts_value, dict):
        raise DatasetIntegrityError("hybrid summary artifacts are missing")
    artifacts = cast(dict[str, dict[str, Any]], artifacts_value)
    expected_keys = expected_hybrid_artifact_keys(experiments)
    if set(artifacts) != expected_keys:
        raise DatasetIntegrityError("hybrid summary artifact set differs")
    candidate_metrics: dict[float, dict[str, float]] = {}
    backend = create_int8_query_backend(root=root, model=model, runtime=runtime)
    backend_artifact_sha256 = backend.model_artifact_sha256
    for weight in experiments.lexical_weight_grid:
        key = f"dev/lw{_weight_label(weight)}"
        entry = artifacts[key]
        expected_tag = (
            f"wands-hybrid-{model.name.replace('_', '-')}-minmax-"
            f"lw{_weight_label(weight)}-dev-int8-onnx-{runtime.quantization_config}"
        )
        manifest = verify_hybrid_artifact(
            client,
            root=root,
            index=index,
            model=model,
            backend_artifact_sha256=backend_artifact_sha256,
            entry=entry,
            expected_lexical_weight=weight,
            expected_split="dev",
            expected_decision_scope="tuning",
            expected_benchmark_provenance=provenance,
            expected_upstream_evidence=upstream_evidence,
            expected_selected_model_eligible=selected_model_eligible,
            expected_runtime=runtime,
            expected_query_encoder_runtime=backend.runtime,
            expected_query_encoder_device=backend.device,
        )
        if (
            manifest.get("tag") != expected_tag
            or entry.get("run") != f"runs/{expected_tag}.trec"
            or entry.get("manifest") != f"runs/{expected_tag}.manifest.json"
            or entry.get("metrics_file")
            != f"results/wands/hybrid/{expected_tag}.metrics.json"
        ):
            raise DatasetIntegrityError(f"hybrid canonical artifact paths differ: {key}")
        candidate_metrics[weight] = cast(dict[str, float], entry["metrics"])
    expected_weight = select_hybrid_weight(
        experiments.lexical_weight_grid,
        candidate_metrics,
        metric=experiments.primary_metric,
    )
    if summary.get("selected_lexical_weight") != expected_weight:
        raise DatasetIntegrityError("hybrid selected weight is not the dev winner")
    test_entry = artifacts["test/selected"]
    expected_test_tag = (
        f"wands-hybrid-{model.name.replace('_', '-')}-minmax-"
        f"lw{_weight_label(expected_weight)}-test-int8-onnx-{runtime.quantization_config}"
    )
    test_manifest = verify_hybrid_artifact(
        client,
        root=root,
        index=index,
        model=model,
        backend_artifact_sha256=backend_artifact_sha256,
        entry=test_entry,
        expected_lexical_weight=expected_weight,
        expected_split="test",
        expected_decision_scope="held_out",
        expected_benchmark_provenance=provenance,
        expected_upstream_evidence=upstream_evidence,
        expected_selected_model_eligible=selected_model_eligible,
        expected_runtime=runtime,
        expected_query_encoder_runtime=backend.runtime,
        expected_query_encoder_device=backend.device,
    )
    if (
        test_manifest.get("tag") != expected_test_tag
        or test_entry.get("run") != f"runs/{expected_test_tag}.trec"
        or test_entry.get("manifest") != f"runs/{expected_test_tag}.manifest.json"
        or test_entry.get("metrics_file")
        != f"results/wands/hybrid/{expected_test_tag}.metrics.json"
    ):
        raise DatasetIntegrityError("hybrid canonical held-out artifact paths differ")
    del backend
    expected_candidate_metrics = [
        {"lexical_weight": weight, "metrics": candidate_metrics[weight]}
        for weight in experiments.lexical_weight_grid
    ]
    if summary.get("candidate_dev_metrics") != expected_candidate_metrics:
        raise DatasetIntegrityError("hybrid candidate development metrics differ")
    held_out_metrics = cast(dict[str, float], test_entry["metrics"])
    if summary.get("held_out_test_metrics") != held_out_metrics:
        raise DatasetIntegrityError("hybrid held-out metrics differ from verified test run")

    source_bindings, source_bindings_valid, source_binding_reasons = (
        _hybrid_source_manifest_bindings(
            root=root,
            artifacts=artifacts,
            expected_keys=expected_keys,
            benchmark_provenance=provenance,
            upstream_evidence=upstream_evidence,
            selected_model_eligible=selected_model_eligible,
        )
    )
    if (
        summary.get("required_source_manifest_keys") != sorted(expected_keys)
        or summary.get("source_manifest_bindings") != source_bindings
        or summary.get("source_manifest_bindings_valid") is not source_bindings_valid
    ):
        raise DatasetIntegrityError("hybrid source manifest bindings differ")
    eligible, eligibility_reasons = _hybrid_summary_eligibility(
        benchmark_provenance_valid=provenance_valid,
        upstream_evidence_eligible=(
            upstream_evidence.get("eligible_for_decision") is True
            and upstream_bindings_valid
        ),
        selected_model_eligible=selected_model_eligible,
        source_bindings_valid=source_bindings_valid,
        held_out_artifact_eligible=test_manifest.get("eligible_for_decision"),
    )
    if (
        summary.get("eligible_for_decision") is not eligible
        or summary.get("ineligibility_reason")
        != ("; ".join(eligibility_reasons) or None)
        or summary.get("quality_ineligibility_reasons")
        != eligibility_reasons + source_binding_reasons
    ):
        raise DatasetIntegrityError("hybrid evidence eligibility differs")
    parity_status, quality_guard, self_retrieval_status = selected_int8_runtime_fields(
        selected_int8_evidence
    )
    if (
        summary.get("parity_status") != parity_status
        or summary.get("quality_guard") != quality_guard
        or summary.get("self_retrieval_status") != self_retrieval_status
    ):
        raise DatasetIntegrityError("hybrid selected int8 runtime evidence differs")
    bm25_ndcg = float(
        cast(
            dict[str, Any],
            cast(dict[str, Any], bm25_selection["held_out_test_metrics"])["tuned"],
        )["ndcg@10"]
    )
    competent_ndcg = float(
        cast(
            dict[str, Any],
            cast(dict[str, Any], bm25_selection["held_out_test_metrics"])[
                "competent_integrator"
            ],
        )["ndcg@10"]
    )
    hybrid_ndcg = float(held_out_metrics["ndcg@10"])
    if (
        summary.get("tuned_bm25_held_out_ndcg@10") != bm25_ndcg
        or summary.get("hybrid_minus_tuned_bm25_ndcg@10")
        != hybrid_ndcg - bm25_ndcg
    ):
        raise DatasetIntegrityError("hybrid held-out BM25 delta differs")
    if (
        summary.get("competent_integrator_bm25_held_out_ndcg@10")
        != competent_ndcg
        or summary.get("hybrid_minus_competent_integrator_bm25_ndcg@10")
        != hybrid_ndcg - competent_ndcg
    ):
        raise DatasetIntegrityError("hybrid competent-integrator BM25 delta differs")
    expected_pipeline = build_normalization_pipeline(lexical_weight=expected_weight)
    expected_fusion = {
        "normalization": "min_max",
        "combination": "arithmetic_mean",
        "lexical_weight": expected_weight,
        "dense_weight": 1.0 - expected_weight,
    }
    method_definition = {
        "method": "native_hybrid_min_max_exact_dense",
        "quality_query": "native_hybrid_with_exact_knn_score_script",
        "selection_metric": experiments.primary_metric,
        "selection_split": "dev",
        "tie_break": "first_weight_in_registered_order",
        "pipeline_id": hybrid_pipeline_id(model, expected_weight),
        "pipeline_sha256": canonical_sha256(expected_pipeline),
        "fusion_sha256": canonical_sha256(expected_fusion),
    }
    if (
        summary.get("method_definition") != method_definition
        or summary.get("method_definition_sha256")
        != canonical_sha256(method_definition)
    ):
        raise DatasetIntegrityError("hybrid summary method definition differs")
    _verify_hybrid_live_reruns(
        client,
        root=root,
        index=index,
        model=model,
        runtime=runtime,
        profile=expected_profile,
        experiments=experiments,
        summary=summary,
    )
    verify_wands_index_after_decision_replay(
        client,
        root=root,
        index=index,
        expected_verification=upstream_evidence.get(
            "vector_index_verification"
        ),
    )
    return summary


def _verify_hybrid_live_reruns(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
    profile: BM25Profile,
    experiments: BM25Experiments,
    summary: Mapping[str, Any],
) -> None:
    prepared = root / "data/prepared/wands"
    queries = {
        split: load_prepared_queries(
            prepared / f"queries.{split}.jsonl",
            expected_split=split,
        )
        for split in ("dev", "test")
    }
    all_queries = {**queries["dev"], **queries["test"]}
    backend = create_int8_query_backend(root=root, model=model, runtime=runtime)
    vectors = encode_queries(
        backend,
        model=model,
        queries=all_queries,
        batch_size=runtime.batch_size,
    )
    artifacts = cast(dict[str, dict[str, Any]], summary["artifacts"])
    arms = [
        (f"dev/lw{round(weight * 100):03d}", "dev", weight)
        for weight in experiments.lexical_weight_grid
    ]
    arms.append(
        (
            "test/selected",
            "test",
            float(summary["selected_lexical_weight"]),
        )
    )
    with tempfile.TemporaryDirectory(
        prefix="opensearch-hybrid-hybrid-verify-"
    ) as temporary:
        temporary_root = Path(temporary)
        for artifact_key, split, lexical_weight in arms:
            entry = artifacts[artifact_key]
            manifest = cast(
                dict[str, Any], read_json(root / str(entry["manifest"]))
            )
            if backend.model_artifact_sha256 != manifest.get(
                "query_encoder_artifact_sha256"
            ):
                raise DatasetIntegrityError(
                    f"hybrid rerun loaded a different artifact: {artifact_key}"
                )
            split_vectors = {
                query_id: vectors[query_id] for query_id in queries[split]
            }
            if query_vectors_sha256(split_vectors) != manifest.get(
                "query_vectors_sha256"
            ):
                raise DatasetIntegrityError(
                    f"hybrid rerun query vectors differ: {artifact_key}"
                )
            records = execute_exact_hybrid_run(
                client,
                index=index,
                queries=queries[split],
                query_vectors=split_vectors,
                profile=profile,
                model=model,
                lexical_weight=lexical_weight,
                tag=str(manifest["tag"]),
            )
            repeated = execute_exact_hybrid_run(
                client,
                index=index,
                queries=queries[split],
                query_vectors=split_vectors,
                profile=profile,
                model=model,
                lexical_weight=lexical_weight,
                tag=str(manifest["tag"]),
            )
            if records != repeated:
                raise DatasetIntegrityError(
                    f"identical hybrid requests are not deterministic: {artifact_key}"
                )
            repeated_path = (
                temporary_root / f"{artifact_key.replace('/', '-')}.trec"
            )
            write_run(repeated_path, records)
            if repeated_path.read_bytes() != (
                root / str(entry["run"])
            ).read_bytes():
                raise DatasetIntegrityError(
                    f"hybrid live rerun is not byte-identical: {artifact_key}"
                )


def verify_hybrid_artifact(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    model: ModelSpec,
    backend_artifact_sha256: str,
    entry: dict[str, Any],
    expected_lexical_weight: float | None = None,
    expected_split: str | None = None,
    expected_decision_scope: str | None = None,
    expected_benchmark_provenance: Mapping[str, Any] | None = None,
    expected_upstream_evidence: Mapping[str, Any] | None = None,
    expected_selected_model_eligible: bool | None = None,
    expected_runtime: QueryRuntimeSpec | None = None,
    expected_query_encoder_runtime: str | None = None,
    expected_query_encoder_device: str | None = None,
) -> dict[str, Any]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    manifest_path = root / str(entry["manifest"])
    manifest = cast(dict[str, Any], read_json(manifest_path))
    fusion_value = manifest.get("fusion")
    if not isinstance(fusion_value, dict):
        raise DatasetIntegrityError("hybrid run fusion definition is missing")
    lexical_weight_value = fusion_value.get("lexical_weight")
    if (
        not isinstance(lexical_weight_value, int | float)
        or isinstance(lexical_weight_value, bool)
    ):
        raise DatasetIntegrityError("hybrid lexical weight has the wrong type")
    lexical_weight = float(lexical_weight_value)
    if expected_lexical_weight is not None and lexical_weight != expected_lexical_weight:
        raise DatasetIntegrityError("hybrid lexical weight differs from registered artifact")
    expected_pipeline = build_normalization_pipeline(lexical_weight=lexical_weight)
    return verify_hybrid_pipeline_artifact(
        client,
        root=root,
        index=index,
        backend_artifact_sha256=backend_artifact_sha256,
        entry=entry,
        expected_pipeline_id=hybrid_pipeline_id(model, lexical_weight),
        expected_pipeline=expected_pipeline,
        expected_fusion={
            "normalization": "min_max",
            "combination": "arithmetic_mean",
            "lexical_weight": lexical_weight,
            "dense_weight": 1.0 - lexical_weight,
        },
        expected_model=model,
        expected_method_name="native_hybrid_min_max_exact_dense",
        expected_split=expected_split,
        expected_decision_scope=expected_decision_scope,
        expected_benchmark_provenance=expected_benchmark_provenance,
        expected_upstream_evidence=expected_upstream_evidence,
        expected_selected_model_eligible=expected_selected_model_eligible,
        expected_runtime=expected_runtime,
        expected_query_encoder_runtime=expected_query_encoder_runtime,
        expected_query_encoder_device=expected_query_encoder_device,
    )


def verify_hybrid_pipeline_artifact(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    backend_artifact_sha256: str,
    entry: dict[str, Any],
    expected_pipeline_id: str,
    expected_pipeline: dict[str, Any],
    expected_fusion: dict[str, object],
    expected_index_manifest_path: Path | None = None,
    expected_model: ModelSpec | None = None,
    expected_method_name: str | None = None,
    expected_split: str | None = None,
    expected_decision_scope: str | None = None,
    expected_benchmark_provenance: Mapping[str, Any] | None = None,
    expected_upstream_evidence: Mapping[str, Any] | None = None,
    expected_selected_model_eligible: bool | None = None,
    expected_runtime: QueryRuntimeSpec | None = None,
    expected_query_encoder_runtime: str | None = None,
    expected_query_encoder_device: str | None = None,
) -> dict[str, Any]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    run_path = root / str(entry["run"])
    manifest_path = root / str(entry["manifest"])
    metrics_path = root / str(entry["metrics_file"])
    manifest = cast(dict[str, Any], read_json(manifest_path))
    if (
        set(entry) != _ARTIFACT_ENTRY_KEYS
        or set(manifest) != _RUN_MANIFEST_KEYS
        or manifest.get("schema_version") != 2
        or manifest.get("dataset") != "WANDS"
    ):
        raise DatasetIntegrityError(f"hybrid run schema or artifact fields differ: {run_path}")
    tag = manifest.get("tag")
    if (
        not isinstance(tag, str)
        or not tag
        or run_path != root / f"runs/{tag}.trec"
        or manifest_path != root / f"runs/{tag}.manifest.json"
        or metrics_path.name != f"{tag}.metrics.json"
        or metrics_path.parent.parent != root / "results/wands"
    ):
        raise DatasetIntegrityError(f"hybrid artifact paths or tag differ: {run_path}")
    if expected_method_name is not None and manifest.get("method") != expected_method_name:
        raise DatasetIntegrityError(f"hybrid run method differs: {run_path}")
    if expected_split is not None and manifest.get("split") != expected_split:
        raise DatasetIntegrityError(f"hybrid run split differs: {run_path}")
    if (
        expected_decision_scope is not None
        and manifest.get("decision_scope") != expected_decision_scope
    ):
        raise DatasetIntegrityError(f"hybrid run decision scope differs: {run_path}")
    model_value = manifest.get("model")
    if not isinstance(model_value, dict) or not isinstance(model_value.get("name"), str):
        raise DatasetIntegrityError(f"hybrid run model metadata is missing: {run_path}")
    registry = load_model_registry(root / "config/models.toml")
    model_name = str(model_value["name"])
    if model_name not in registry:
        raise DatasetIntegrityError(f"hybrid run model is not registered: {run_path}")
    registered_model = registry[model_name]
    if expected_model is not None and registered_model != expected_model:
        raise DatasetIntegrityError(f"hybrid run selected model differs: {run_path}")
    if model_value != _json_dict(asdict(registered_model)):
        raise DatasetIntegrityError(f"hybrid run model configuration differs: {run_path}")
    method_definition = hybrid_method_definition(
        model=registered_model,
        declared_variable=str(manifest.get("declared_variable", "")),
    )
    if (
        manifest.get("method_definition") != method_definition
        or manifest.get("method_definition_sha256")
        != canonical_sha256(method_definition)
        or manifest.get("quality_query") != method_definition["quality_query"]
        or manifest.get("ranking_policy") != method_definition["ranking_policy"]
        or manifest.get("trec_score_source") != method_definition["trec_score_source"]
        or manifest.get("pagination_depth") != method_definition["pagination_depth"]
        or manifest.get("vector_field") != method_definition["vector_field"]
    ):
        raise DatasetIntegrityError(f"hybrid run method definition differs: {run_path}")
    if manifest.get("query_encoder_artifact_sha256") != backend_artifact_sha256:
        raise DatasetIntegrityError(f"hybrid query artifact differs: {run_path}")
    if (
        expected_query_encoder_runtime is not None
        and manifest.get("query_encoder_runtime") != expected_query_encoder_runtime
    ) or (
        expected_query_encoder_device is not None
        and manifest.get("query_encoder_device") != expected_query_encoder_device
    ):
        raise DatasetIntegrityError(f"hybrid query encoder runtime differs: {run_path}")
    root_response = cast(dict[str, Any], client.request("GET", "/"))
    version_value = root_response.get("version")
    live_version = (
        str(version_value.get("number", "")) if isinstance(version_value, dict) else ""
    )
    if live_version != _EXPECTED_OS_VERSION or manifest.get("opensearch_version") != live_version:
        raise DatasetIntegrityError(f"hybrid OpenSearch version differs: {run_path}")
    if asdict(collect_index_facts(client, index)) != manifest.get("index"):
        raise DatasetIntegrityError(f"hybrid run references a different index: {run_path}")
    if expected_index_manifest_path is None:
        expected_index_manifest_path = root / "results/wands/index-manifest.json"
    expected_index_manifest = cast(
        dict[str, Any], read_json(expected_index_manifest_path)
    )
    expected_index_artifact = _artifact(root, expected_index_manifest_path)
    index_evidence = {
        "manifest": expected_index_artifact,
        "definition_sha256": expected_index_manifest["index_definition_sha256"],
    }
    if (
        manifest.get("index_manifest_sha256") != expected_index_artifact["sha256"]
        or manifest.get("index_evidence") != index_evidence
    ):
        raise DatasetIntegrityError(f"hybrid index manifest differs: {run_path}")
    if manifest.get("index_definition_sha256") != expected_index_manifest.get(
        "index_definition_sha256"
    ):
        raise DatasetIntegrityError(f"hybrid index definition differs: {run_path}")
    run_file_value = manifest.get("run_file")
    if not isinstance(run_file_value, dict) or set(run_file_value) != {
        "path",
        "sha256",
        "records",
    }:
        raise DatasetIntegrityError(f"hybrid run inventory differs: {run_path}")
    run_records = read_run(run_path)
    if (
        run_file_value.get("path") != str(run_path.relative_to(root))
        or file_facts(run_path).sha256 != run_file_value.get("sha256")
        or run_file_value.get("records") != len(run_records)
        or any(record.tag != tag for record in run_records)
        or entry.get("run_sha256") != run_file_value.get("sha256")
    ):
        raise DatasetIntegrityError(f"hybrid run differs from manifest: {run_path}")
    if manifest.get("fusion") != expected_fusion:
        raise DatasetIntegrityError(f"hybrid fusion definition differs: {run_path}")
    if manifest.get("fusion_sha256") != canonical_sha256(expected_fusion):
        raise DatasetIntegrityError(f"hybrid fusion hash differs: {run_path}")
    if manifest.get("pipeline") != expected_pipeline:
        raise DatasetIntegrityError(f"hybrid pipeline definition differs: {run_path}")
    if manifest.get("pipeline_sha256") != canonical_sha256(expected_pipeline):
        raise DatasetIntegrityError(f"hybrid pipeline hash differs: {run_path}")
    if manifest.get("pipeline_id") != expected_pipeline_id:
        raise DatasetIntegrityError(f"hybrid pipeline ID differs: {run_path}")
    pipeline_id = str(manifest["pipeline_id"])
    try:
        verify_live_search_pipeline(
            client,
            pipeline_id=pipeline_id,
            expected_definition=expected_pipeline,
        )
    except DatasetIntegrityError as error:
        raise DatasetIntegrityError(
            f"live hybrid pipeline differs: {run_path}: {error}"
        ) from error
    bm25_selection = cast(dict[str, Any], read_json(root / "results/wands/bm25-selection.json"))
    bm25_selection_path = root / "results/wands/bm25-selection.json"
    if (
        manifest.get("bm25_profile") != bm25_selection.get("selected_profile")
        or manifest.get("bm25_profile_sha256")
        != bm25_selection.get("selected_profile_sha256")
        or manifest.get("bm25_selection") != _artifact(root, bm25_selection_path)
        or manifest.get("bm25_selection_sha256")
        != file_facts(bm25_selection_path).sha256
    ):
        raise DatasetIntegrityError(f"hybrid BM25 clause is not frozen: {run_path}")
    model_manifest_path = root / f"results/wands/embeddings/{registered_model.name}.manifest.json"
    model_manifest = cast(dict[str, Any], read_json(model_manifest_path))
    expected_model_evidence = {
        "manifest": _artifact(root, model_manifest_path),
        "model_artifact_sha256": model_manifest["model_artifact_sha256"],
        "encoder_runtime": model_manifest["encoder_runtime"],
    }
    if (
        manifest.get("model_evidence") != expected_model_evidence
        or manifest.get("model_sha256") != model_manifest.get("model_artifact_sha256")
        or manifest.get("document_encoder_runtime") != model_manifest.get("encoder_runtime")
    ):
        raise DatasetIntegrityError(f"hybrid document model evidence differs: {run_path}")
    registered_runtime = load_query_runtime_spec(root / "config/query_runtime.toml")
    if expected_runtime is not None and expected_runtime != registered_runtime:
        raise DatasetIntegrityError(f"hybrid expected runtime is not registered: {run_path}")
    if manifest.get("query_runtime_spec") != _json_dict(asdict(registered_runtime)):
        raise DatasetIntegrityError(f"hybrid query runtime configuration differs: {run_path}")
    provenance = manifest.get("benchmark_provenance")
    provenance_valid = wands_recorded_provenance_valid(
        root,
        provenance=provenance,
        completion_revision=manifest.get("completion_code_revision"),
    )
    if (
        manifest.get("benchmark_provenance_valid") is not provenance_valid
        or (
            expected_benchmark_provenance is not None
            and provenance != expected_benchmark_provenance
        )
    ):
        raise DatasetIntegrityError(f"hybrid run provenance differs: {run_path}")
    upstream_evidence = manifest.get("upstream_evidence")
    if not isinstance(upstream_evidence, Mapping) or (
        expected_upstream_evidence is not None
        and upstream_evidence != expected_upstream_evidence
    ):
        raise DatasetIntegrityError(f"hybrid run upstream evidence differs: {run_path}")
    upstream_bindings_valid = hybrid_upstream_artifact_bindings_valid(
        root,
        upstream_evidence,
    )
    if (
        manifest.get("upstream_artifact_bindings_valid_at_completion")
        is not upstream_bindings_valid
    ):
        raise DatasetIntegrityError(
            f"hybrid run upstream artifact bindings differ: {run_path}"
        )
    selected_model_eligible = manifest.get("selected_model_eligible")
    if (
        not isinstance(selected_model_eligible, bool)
        or (
            expected_selected_model_eligible is not None
            and selected_model_eligible is not expected_selected_model_eligible
        )
    ):
        raise DatasetIntegrityError(f"hybrid selected-model eligibility differs: {run_path}")
    decision_scope = str(manifest.get("decision_scope", ""))
    eligible, reason = derive_hybrid_run_eligibility(
        decision_scope=decision_scope,
        benchmark_provenance_valid=provenance_valid,
        upstream_evidence_eligible=(
            upstream_evidence.get("eligible_for_decision") is True
            and upstream_bindings_valid
        ),
        selected_model_eligible=selected_model_eligible,
    )
    if (
        manifest.get("eligible_for_decision") is not eligible
        or manifest.get("ineligibility_reason") != reason
    ):
        raise DatasetIntegrityError(f"hybrid run eligibility differs: {run_path}")
    for key in ("query_file", "qrels_file"):
        facts_value = manifest.get(key)
        if not isinstance(facts_value, dict) or set(facts_value) != {"path", "sha256"}:
            raise DatasetIntegrityError(f"hybrid {key} inventory differs: {run_path}")
        path = root / str(facts_value["path"])
        if file_facts(path).sha256 != facts_value.get("sha256"):
            raise DatasetIntegrityError(f"hybrid {key} differs: {run_path}")
    if expected_split is not None:
        expected_query_path = f"data/prepared/wands/queries.{expected_split}.jsonl"
        expected_qrels_path = f"data/prepared/wands/qrels.{expected_split}.trec"
        if (
            cast(dict[str, Any], manifest["query_file"]).get("path")
            != expected_query_path
            or cast(dict[str, Any], manifest["qrels_file"]).get("path")
            != expected_qrels_path
        ):
            raise DatasetIntegrityError(f"hybrid canonical split paths differ: {run_path}")
    qrels_path = root / str(cast(dict[str, Any], manifest["qrels_file"])["path"])
    evaluation = evaluate_run(qrels_path, run_path)
    metrics = cast(dict[str, Any], read_json(metrics_path))
    if (
        set(metrics) != _METRICS_KEYS
        or metrics.get("schema_version") != 2
        or metrics.get("dataset") != "WANDS"
        or metrics.get("method") != manifest.get("method")
        or metrics.get("tag") != tag
        or metrics.get("split") != manifest.get("split")
        or metrics.get("query_file") != manifest.get("query_file")
        or metrics.get("qrels_file") != manifest.get("qrels_file")
        or metrics.get("run_file") != manifest.get("run_file")
        or metrics.get("metrics") != evaluation.metrics
        or metrics.get("per_query") != evaluation.per_query
    ):
        raise DatasetIntegrityError(f"hybrid metrics do not reproduce: {run_path}")
    if entry.get("metrics") != evaluation.metrics:
        raise DatasetIntegrityError(f"hybrid summary metrics do not reproduce: {run_path}")
    if (
        metrics.get("evaluator_crosscheck")
        != cross_check_run(qrels_path, run_path, root=root).to_dict()
    ):
        raise DatasetIntegrityError(f"hybrid evaluator cross-check differs: {run_path}")
    if (
        manifest.get("metrics_file") != _artifact(root, metrics_path)
        or entry.get("metrics_sha256") != file_facts(metrics_path).sha256
        or entry.get("manifest_sha256") != file_facts(manifest_path).sha256
    ):
        raise DatasetIntegrityError(f"hybrid artifact hashes differ: {run_path}")
    expected_packages = {
        "ranx": importlib.metadata.version("ranx"),
        "pytrec_eval": importlib.metadata.version("pytrec-eval-terrier"),
    }
    if manifest.get("evaluation_packages") != expected_packages:
        raise DatasetIntegrityError(f"hybrid evaluation packages differ: {run_path}")
    if (
        manifest.get("query_template_sha256")
        != canonical_sha256(
            {
                "prefix": registered_model.query_prefix,
                "template": registered_model.query_template,
            }
        )
        or manifest.get("query_template_example")
        != registered_model.render_query("example product query")
        or not isinstance(manifest.get("query_vectors_sha256"), str)
        or len(str(manifest["query_vectors_sha256"])) != 64
        or not isinstance(manifest.get("query_encoding_seconds"), int | float)
        or isinstance(manifest.get("query_encoding_seconds"), bool)
        or float(cast(float, manifest["query_encoding_seconds"])) < 0.0
    ):
        raise DatasetIntegrityError(f"hybrid query encoding metadata differs: {run_path}")
    return manifest


def _weight_label(weight: float) -> str:
    return f"{round(weight * 100):03d}"


def int8_dev_model_metrics(
    root: Path,
    models: tuple[ModelSpec, ...],
    *,
    summary: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    if summary is None:
        summary = cast(
            dict[str, Any], read_json(root / "results/wands/int8-dense-summary.json")
        )
    if summary.get("schema_version") != 2 or summary.get("dataset") != "WANDS":
        raise DatasetIntegrityError("int8 dense summary schema or dataset differs")
    model_entries_value = summary.get("models")
    if not isinstance(model_entries_value, Mapping) or set(model_entries_value) != {
        model.name for model in models
    }:
        raise DatasetIntegrityError("int8 dense summary model set differs")
    model_entries = cast(Mapping[str, Mapping[str, Any]], model_entries_value)
    metrics: dict[str, dict[str, float]] = {}
    for model in models:
        entry = model_entries.get(model.name)
        if entry is None:
            raise DatasetIntegrityError(
                f"int8 dense summary is missing hybrid model candidate {model.name}"
            )
        splits_value = entry.get("splits")
        if not isinstance(splits_value, Mapping) or set(splits_value) != {"dev", "test"}:
            raise DatasetIntegrityError(
                f"int8 dense summary splits differ for {model.name}"
            )
        dev_value = splits_value.get("dev")
        if not isinstance(dev_value, Mapping) or not isinstance(
            dev_value.get("metrics"), Mapping
        ):
            raise DatasetIntegrityError(
                f"int8 dense summary has no dev metrics for {model.name}"
            )
        model_metrics = dict(cast(Mapping[str, float], dev_value["metrics"]))
        if any(
            not isinstance(name, str)
            or not isinstance(value, int | float)
            or isinstance(value, bool)
            for name, value in model_metrics.items()
        ):
            raise DatasetIntegrityError(
                f"int8 dense summary dev metrics are invalid for {model.name}"
            )
        metrics[model.name] = model_metrics
    return metrics
