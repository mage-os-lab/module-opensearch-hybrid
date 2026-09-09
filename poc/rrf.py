from __future__ import annotations

import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from poc.bm25 import load_prepared_queries
from poc.config import ModelSpec
from poc.datasets import DatasetIntegrityError
from poc.dense import encode_queries, query_vectors_sha256
from poc.embedding_cache import EmbeddingBackend
from poc.experiments import BM25Experiments, BM25Profile
from poc.hybrid import (
    HybridRunResult,
    _hybrid_source_manifest_bindings,
    _hybrid_summary_eligibility,
    _json_dict,
    collect_verified_hybrid_upstream_evidence,
    execute_exact_hybrid_pipeline_run,
    hybrid_upstream_artifact_bindings_valid,
    int8_dev_model_metrics,
    select_hybrid_model,
    selected_int8_runtime_fields,
    verify_hybrid_pipeline_artifact,
    verify_wands_index_after_decision_replay,
    write_hybrid_pipeline_bundle,
)
from poc.index_evidence import content_addressed_pipeline_id
from poc.indexing import wands_completion_provenance_evidence, wands_recorded_provenance_valid
from poc.manifest import canonical_sha256, read_json, write_json
from poc.os_client import OpenSearchClient
from poc.provenance import collect_manifest_provenance, require_registered_opensearch_client
from poc.query_runtime import QueryRuntimeSpec, create_int8_query_backend
from poc.search import build_rrf_pipeline
from poc.trec import RunRecord, write_run

_RRF_SUMMARY_KEYS = {
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
    "rrf_tuning_budget",
    "bm25_tuning_budget",
    "registered_rank_constants",
    "selected_rank_constant",
    "candidate_dev_metrics",
    "held_out_test_metrics",
    "tuned_bm25_held_out_ndcg@10",
    "rrf_minus_tuned_bm25_ndcg@10",
    "competent_integrator_bm25_held_out_ndcg@10",
    "rrf_minus_competent_integrator_bm25_ndcg@10",
    "parity_status",
    "quality_guard",
    "self_retrieval_status",
    "artifacts",
}


def select_rrf_rank_constant(
    rank_constants: Sequence[int],
    metrics: Mapping[int, Mapping[str, float]],
    *,
    metric: str,
) -> int:
    if not rank_constants:
        raise ValueError("RRF rank constant grid must not be empty")
    missing = [value for value in rank_constants if value not in metrics]
    if missing:
        raise ValueError(f"RRF metrics missing registered rank constants: {missing}")
    return max(rank_constants, key=lambda value: metrics[value][metric])


def expected_rrf_artifact_keys(experiments: BM25Experiments) -> set[str]:
    return {
        *(f"dev/k{_constant_label(value)}" for value in experiments.rrf_rank_constants),
        "test/selected",
    }


def rrf_pipeline_id(model: ModelSpec, rank_constant: int) -> str:
    definition = build_rrf_pipeline(rank_constant=rank_constant)
    return content_addressed_pipeline_id(
        (
            f"opensearch-hybrid-wands-{model.name.replace('_', '-')}"
            f"-rrf-k{rank_constant:03d}-v1"
        ),
        definition,
    )


def execute_exact_rrf_run(
    client: OpenSearchClient,
    *,
    index: str,
    queries: Mapping[str, str],
    query_vectors: Mapping[str, NDArray[np.float32]],
    profile: BM25Profile,
    model: ModelSpec,
    rank_constant: int,
    tag: str,
) -> list[RunRecord]:
    return execute_exact_hybrid_pipeline_run(
        client,
        index=index,
        queries=queries,
        query_vectors=query_vectors,
        profile=profile,
        model=model,
        pipeline_id=rrf_pipeline_id(model, rank_constant),
        pipeline_definition=build_rrf_pipeline(rank_constant=rank_constant),
        tag=tag,
    )


def rrf_fusion_definition(rank_constant: int) -> dict[str, object]:
    build_rrf_pipeline(rank_constant=rank_constant)
    return {
        "combination": "rrf",
        "rank_constant": rank_constant,
        "subquery_weights": "equal_opensearch_default",
    }


def write_rrf_bundle(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    model: ModelSpec,
    backend: EmbeddingBackend,
    profile: BM25Profile,
    rank_constant: int,
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
    return write_hybrid_pipeline_bundle(
        client,
        root=root,
        index=index,
        model=model,
        backend=backend,
        profile=profile,
        pipeline_id=rrf_pipeline_id(model, rank_constant),
        pipeline_definition=build_rrf_pipeline(rank_constant=rank_constant),
        fusion=rrf_fusion_definition(rank_constant),
        declared_variable=("rank_constant" if decision_scope == "tuning" else "retrieval_arm"),
        metrics_directory="rrf",
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
        method_name="native_hybrid_rrf_exact_dense",
    )


def run_wands_rrf_benchmark(
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

    dev_results: dict[int, HybridRunResult] = {}
    artifacts: dict[str, dict[str, object]] = {}
    for rank_constant in experiments.rrf_rank_constants:
        constant_label = _constant_label(rank_constant)
        tag = (
            f"wands-hybrid-{model.name.replace('_', '-')}-rrf-"
            f"k{constant_label}-dev-int8-onnx-{runtime.quantization_config}"
        )
        dev_vectors = {query_id: vectors[query_id] for query_id in queries["dev"]}
        records = execute_exact_rrf_run(
            client,
            index=index,
            queries=queries["dev"],
            query_vectors=dev_vectors,
            profile=profile,
            model=model,
            rank_constant=rank_constant,
            tag=tag,
        )
        result = write_rrf_bundle(
            client,
            root=root,
            index=index,
            model=model,
            backend=backend,
            profile=profile,
            rank_constant=rank_constant,
            query_path=query_paths["dev"],
            qrels_path=qrels_paths["dev"],
            split="dev",
            tag=tag,
            records=records,
            query_vectors_hash=query_vectors_sha256(dev_vectors),
            encoding_seconds=encoding_seconds,
            eligible_for_decision=False,
            ineligibility_reason="development tuning result",
            decision_scope="tuning",
            benchmark_provenance=benchmark_provenance,
            upstream_evidence=upstream_evidence,
            selected_model_eligible=selected_model_eligible,
        )
        dev_results[rank_constant] = result
        artifacts[f"dev/k{constant_label}"] = result.selection_entry(root)

    candidate_metrics = {
        rank_constant: result.evaluation.metrics for rank_constant, result in dev_results.items()
    }
    selected_rank_constant = select_rrf_rank_constant(
        experiments.rrf_rank_constants,
        candidate_metrics,
        metric=experiments.primary_metric,
    )
    selected_label = _constant_label(selected_rank_constant)
    test_tag = (
        f"wands-hybrid-{model.name.replace('_', '-')}-rrf-"
        f"k{selected_label}-test-int8-onnx-{runtime.quantization_config}"
    )
    test_vectors = {query_id: vectors[query_id] for query_id in queries["test"]}
    test_records = execute_exact_rrf_run(
        client,
        index=index,
        queries=queries["test"],
        query_vectors=test_vectors,
        profile=profile,
        model=model,
        rank_constant=selected_rank_constant,
        tag=test_tag,
    )
    test_result = write_rrf_bundle(
        client,
        root=root,
        index=index,
        model=model,
        backend=backend,
        profile=profile,
        rank_constant=selected_rank_constant,
        query_path=query_paths["test"],
        qrels_path=qrels_paths["test"],
        split="test",
        tag=test_tag,
        records=test_records,
        query_vectors_hash=query_vectors_sha256(test_vectors),
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
    expected_keys = expected_rrf_artifact_keys(experiments)
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
    selected_pipeline = build_rrf_pipeline(rank_constant=selected_rank_constant)
    selected_fusion = rrf_fusion_definition(selected_rank_constant)
    method_definition = {
        "method": "native_hybrid_rrf_exact_dense",
        "quality_query": "native_hybrid_with_exact_knn_score_script",
        "selection_metric": experiments.primary_metric,
        "selection_split": "dev",
        "tie_break": "first_rank_constant_in_registered_order",
        "pipeline_id": rrf_pipeline_id(model, selected_rank_constant),
        "pipeline_sha256": canonical_sha256(selected_pipeline),
        "fusion_sha256": canonical_sha256(selected_fusion),
    }
    parity_status, quality_guard, self_retrieval_status = selected_int8_runtime_fields(
        selected_int8_evidence
    )
    summary: dict[str, object] = {
        "schema_version": 2,
        "dataset": "WANDS",
        "method": "native_hybrid_rrf_exact_dense",
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
        "tie_break": "first_rank_constant_in_registered_order",
        "rrf_tuning_budget": len(experiments.rrf_rank_constants),
        "bm25_tuning_budget": bm25_selection["bm25_tuning_budget"],
        "registered_rank_constants": list(experiments.rrf_rank_constants),
        "selected_rank_constant": selected_rank_constant,
        "candidate_dev_metrics": [
            {"rank_constant": value, "metrics": candidate_metrics[value]}
            for value in experiments.rrf_rank_constants
        ],
        "held_out_test_metrics": test_result.evaluation.metrics,
        "tuned_bm25_held_out_ndcg@10": bm25_test["ndcg@10"],
        "rrf_minus_tuned_bm25_ndcg@10": held_out_delta,
        "competent_integrator_bm25_held_out_ndcg@10": competent_test["ndcg@10"],
        "rrf_minus_competent_integrator_bm25_ndcg@10": competent_delta,
        "parity_status": parity_status,
        "quality_guard": quality_guard,
        "self_retrieval_status": self_retrieval_status,
        "artifacts": artifacts,
    }
    persisted_summary = _json_dict(summary)
    write_json(root / "results/wands/rrf-summary.json", persisted_summary)
    return cast(dict[str, object], persisted_summary)


def verify_wands_rrf_summary(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    models: tuple[ModelSpec, ...],
    runtime: QueryRuntimeSpec,
    experiments: BM25Experiments,
) -> dict[str, Any]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    summary = cast(dict[str, Any], read_json(root / "results/wands/rrf-summary.json"))
    if (
        set(summary) != _RRF_SUMMARY_KEYS
        or summary.get("schema_version") != 2
        or summary.get("dataset") != "WANDS"
        or summary.get("method") != "native_hybrid_rrf_exact_dense"
    ):
        raise DatasetIntegrityError("RRF summary schema, fields, dataset, or method differs")
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
        raise DatasetIntegrityError("RRF summary upstream evidence differs")
    upstream_bindings_valid = hybrid_upstream_artifact_bindings_valid(
        root,
        upstream_evidence,
    )
    if (
        summary.get("upstream_artifact_bindings_valid_at_completion")
        is not upstream_bindings_valid
    ):
        raise DatasetIntegrityError("RRF summary upstream artifact bindings differ")
    provenance = summary.get("benchmark_provenance")
    provenance_valid = wands_recorded_provenance_valid(
        root,
        provenance=provenance,
        completion_revision=summary.get("completion_code_revision"),
    )
    if summary.get("benchmark_provenance_valid") is not provenance_valid:
        raise DatasetIntegrityError("RRF summary provenance validity differs")
    if not isinstance(provenance, Mapping):
        raise DatasetIntegrityError("RRF summary start provenance is missing")
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
        raise DatasetIntegrityError("RRF summary model differs")
    expected_model_candidates = [
        {"model": candidate.name, "metrics": model_metrics[candidate.name]} for candidate in models
    ]
    if summary.get("candidate_model_dev_metrics") != expected_model_candidates:
        raise DatasetIntegrityError("RRF model-selection metrics differ")
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
        or summary.get("tie_break") != "first_rank_constant_in_registered_order"
    ):
        raise DatasetIntegrityError("RRF registered method or selection metadata differs")
    if summary.get("registered_rank_constants") != list(experiments.rrf_rank_constants):
        raise DatasetIntegrityError("RRF registered rank constant grid differs")
    if (
        summary.get("rrf_tuning_budget") != len(experiments.rrf_rank_constants)
        or summary.get("bm25_tuning_budget") != bm25_selection.get("bm25_tuning_budget")
        or summary.get("rrf_tuning_budget") != summary.get("bm25_tuning_budget")
    ):
        raise DatasetIntegrityError("RRF and BM25 tuning budgets differ")

    artifacts_value = summary.get("artifacts")
    if not isinstance(artifacts_value, dict):
        raise DatasetIntegrityError("RRF summary artifacts are missing")
    artifacts = cast(dict[str, dict[str, Any]], artifacts_value)
    expected_keys = expected_rrf_artifact_keys(experiments)
    if set(artifacts) != expected_keys:
        raise DatasetIntegrityError("RRF summary artifact set differs")
    backend = create_int8_query_backend(root=root, model=model, runtime=runtime)
    backend_artifact_sha256 = backend.model_artifact_sha256
    candidate_metrics: dict[int, dict[str, float]] = {}
    for rank_constant in experiments.rrf_rank_constants:
        key = f"dev/k{_constant_label(rank_constant)}"
        entry = artifacts[key]
        expected_tag = (
            f"wands-hybrid-{model.name.replace('_', '-')}-rrf-"
            f"k{_constant_label(rank_constant)}-dev-int8-onnx-"
            f"{runtime.quantization_config}"
        )
        manifest = verify_rrf_artifact(
            client,
            root=root,
            index=index,
            model=model,
            backend_artifact_sha256=backend_artifact_sha256,
            rank_constant=rank_constant,
            entry=entry,
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
            != f"results/wands/rrf/{expected_tag}.metrics.json"
        ):
            raise DatasetIntegrityError(f"RRF canonical artifact paths differ: {key}")
        candidate_metrics[rank_constant] = cast(dict[str, float], entry["metrics"])
    expected_constant = select_rrf_rank_constant(
        experiments.rrf_rank_constants,
        candidate_metrics,
        metric=experiments.primary_metric,
    )
    if summary.get("selected_rank_constant") != expected_constant:
        raise DatasetIntegrityError("RRF selected rank constant is not the dev winner")
    test_entry = artifacts["test/selected"]
    expected_test_tag = (
        f"wands-hybrid-{model.name.replace('_', '-')}-rrf-"
        f"k{_constant_label(expected_constant)}-test-int8-onnx-"
        f"{runtime.quantization_config}"
    )
    test_manifest = verify_rrf_artifact(
        client,
        root=root,
        index=index,
        model=model,
        backend_artifact_sha256=backend_artifact_sha256,
        rank_constant=expected_constant,
        entry=test_entry,
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
        != f"results/wands/rrf/{expected_test_tag}.metrics.json"
    ):
        raise DatasetIntegrityError("RRF canonical held-out artifact paths differ")
    del backend
    expected_candidate_metrics = [
        {"rank_constant": value, "metrics": candidate_metrics[value]}
        for value in experiments.rrf_rank_constants
    ]
    if summary.get("candidate_dev_metrics") != expected_candidate_metrics:
        raise DatasetIntegrityError("RRF candidate development metrics differ")
    held_out_metrics = cast(dict[str, float], test_entry["metrics"])
    if summary.get("held_out_test_metrics") != held_out_metrics:
        raise DatasetIntegrityError("RRF held-out metrics differ from verified test run")

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
        raise DatasetIntegrityError("RRF source manifest bindings differ")
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
        raise DatasetIntegrityError("RRF evidence eligibility differs")
    parity_status, quality_guard, self_retrieval_status = selected_int8_runtime_fields(
        selected_int8_evidence
    )
    if (
        summary.get("parity_status") != parity_status
        or summary.get("quality_guard") != quality_guard
        or summary.get("self_retrieval_status") != self_retrieval_status
    ):
        raise DatasetIntegrityError("RRF selected int8 runtime evidence differs")
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
    rrf_ndcg = float(held_out_metrics["ndcg@10"])
    if (
        summary.get("tuned_bm25_held_out_ndcg@10") != bm25_ndcg
        or summary.get("rrf_minus_tuned_bm25_ndcg@10") != rrf_ndcg - bm25_ndcg
    ):
        raise DatasetIntegrityError("RRF held-out BM25 delta differs")
    if (
        summary.get("competent_integrator_bm25_held_out_ndcg@10")
        != competent_ndcg
        or summary.get("rrf_minus_competent_integrator_bm25_ndcg@10")
        != rrf_ndcg - competent_ndcg
    ):
        raise DatasetIntegrityError("RRF competent-integrator BM25 delta differs")
    selected_pipeline = build_rrf_pipeline(rank_constant=expected_constant)
    selected_fusion = rrf_fusion_definition(expected_constant)
    method_definition = {
        "method": "native_hybrid_rrf_exact_dense",
        "quality_query": "native_hybrid_with_exact_knn_score_script",
        "selection_metric": experiments.primary_metric,
        "selection_split": "dev",
        "tie_break": "first_rank_constant_in_registered_order",
        "pipeline_id": rrf_pipeline_id(model, expected_constant),
        "pipeline_sha256": canonical_sha256(selected_pipeline),
        "fusion_sha256": canonical_sha256(selected_fusion),
    }
    if (
        summary.get("method_definition") != method_definition
        or summary.get("method_definition_sha256")
        != canonical_sha256(method_definition)
    ):
        raise DatasetIntegrityError("RRF summary method definition differs")
    _verify_rrf_live_reruns(
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


def _verify_rrf_live_reruns(
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
        (f"dev/k{rank_constant:03d}", "dev", rank_constant)
        for rank_constant in experiments.rrf_rank_constants
    ]
    arms.append(
        (
            "test/selected",
            "test",
            int(summary["selected_rank_constant"]),
        )
    )
    with tempfile.TemporaryDirectory(
        prefix="opensearch-hybrid-rrf-verify-"
    ) as temporary:
        temporary_root = Path(temporary)
        for artifact_key, split, rank_constant in arms:
            entry = artifacts[artifact_key]
            manifest = cast(
                dict[str, Any], read_json(root / str(entry["manifest"]))
            )
            if backend.model_artifact_sha256 != manifest.get(
                "query_encoder_artifact_sha256"
            ):
                raise DatasetIntegrityError(
                    f"RRF rerun loaded a different artifact: {artifact_key}"
                )
            split_vectors = {
                query_id: vectors[query_id] for query_id in queries[split]
            }
            if query_vectors_sha256(split_vectors) != manifest.get(
                "query_vectors_sha256"
            ):
                raise DatasetIntegrityError(
                    f"RRF rerun query vectors differ: {artifact_key}"
                )
            records = execute_exact_rrf_run(
                client,
                index=index,
                queries=queries[split],
                query_vectors=split_vectors,
                profile=profile,
                model=model,
                rank_constant=rank_constant,
                tag=str(manifest["tag"]),
            )
            repeated = execute_exact_rrf_run(
                client,
                index=index,
                queries=queries[split],
                query_vectors=split_vectors,
                profile=profile,
                model=model,
                rank_constant=rank_constant,
                tag=str(manifest["tag"]),
            )
            if records != repeated:
                raise DatasetIntegrityError(
                    f"identical RRF requests are not deterministic: {artifact_key}"
                )
            repeated_path = (
                temporary_root / f"{artifact_key.replace('/', '-')}.trec"
            )
            write_run(repeated_path, records)
            if repeated_path.read_bytes() != (
                root / str(entry["run"])
            ).read_bytes():
                raise DatasetIntegrityError(
                    f"RRF live rerun is not byte-identical: {artifact_key}"
                )


def verify_rrf_artifact(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    model: ModelSpec,
    backend_artifact_sha256: str,
    rank_constant: int,
    entry: dict[str, Any],
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
    manifest = cast(dict[str, Any], read_json(root / str(entry["manifest"])))
    fusion_value = manifest.get("fusion")
    if not isinstance(fusion_value, dict):
        raise DatasetIntegrityError("RRF run fusion definition is missing")
    recorded_rank_constant = fusion_value.get("rank_constant")
    if (
        not isinstance(recorded_rank_constant, int)
        or isinstance(recorded_rank_constant, bool)
        or recorded_rank_constant != rank_constant
    ):
        raise DatasetIntegrityError("RRF rank constant differs from registered artifact")
    return verify_hybrid_pipeline_artifact(
        client,
        root=root,
        index=index,
        backend_artifact_sha256=backend_artifact_sha256,
        entry=entry,
        expected_pipeline_id=rrf_pipeline_id(model, rank_constant),
        expected_pipeline=build_rrf_pipeline(rank_constant=rank_constant),
        expected_fusion=rrf_fusion_definition(rank_constant),
        expected_model=model,
        expected_method_name="native_hybrid_rrf_exact_dense",
        expected_split=expected_split,
        expected_decision_scope=expected_decision_scope,
        expected_benchmark_provenance=expected_benchmark_provenance,
        expected_upstream_evidence=expected_upstream_evidence,
        expected_selected_model_eligible=expected_selected_model_eligible,
        expected_runtime=expected_runtime,
        expected_query_encoder_runtime=expected_query_encoder_runtime,
        expected_query_encoder_device=expected_query_encoder_device,
    )


def _constant_label(rank_constant: int) -> str:
    return f"{rank_constant:03d}"
