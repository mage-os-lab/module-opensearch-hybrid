from __future__ import annotations

import importlib
import importlib.metadata
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from poc.bm25 import verify_wands_bm25_selection
from poc.config import (
    WANDS_INT8_MODEL_NAMES,
    ModelSpec,
    load_model_registry,
)
from poc.datasets import DatasetIntegrityError, file_facts
from poc.evaluation import evaluate_run
from poc.evaluation_crosscheck import configure_evaluator_caches, run_values_for_evaluators
from poc.experiments import BM25Experiments, load_bm25_experiments
from poc.hybrid import verify_wands_hybrid_summary
from poc.indexing import (
    load_wands_index_config,
    wands_completion_provenance_evidence,
    wands_recorded_provenance_valid,
)
from poc.manifest import read_json, write_json
from poc.neural_sparse import (
    NeuralSparseSpec,
    load_neural_sparse_spec,
)
from poc.os_client import OpenSearchClient
from poc.provenance import collect_manifest_provenance, require_registered_opensearch_client
from poc.query_runtime import QueryRuntimeSpec, load_query_runtime_spec
from poc.rrf import verify_wands_rrf_summary
from poc.sparse_benchmark import verify_wands_sparse_summary
from poc.trec import read_qrels, read_run


def holm_adjusted_p_values(p_values: Mapping[str, float]) -> dict[str, float]:
    if not p_values:
        raise ValueError("Holm correction requires at least one p-value")
    if any(not 0.0 <= value <= 1.0 for value in p_values.values()):
        raise ValueError("p-values must be in [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running_maximum = 0.0
    count = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        candidate = min(1.0, p_value * (count - index))
        running_maximum = max(running_maximum, candidate)
        adjusted[name] = running_maximum
    return {name: adjusted[name] for name in p_values}


def compare_run_family(
    *,
    root: Path,
    qrels_path: Path,
    baseline_name: str,
    baseline_run_path: Path,
    candidate_run_paths: Mapping[str, Path],
    metric: str,
    alpha: float,
    minimum_delta: float,
    eligible_for_decision: bool,
    dataset: str = "WANDS",
    split: str = "test",
    exact_gain: int = 2,
) -> dict[str, object]:
    if not candidate_run_paths:
        raise ValueError("paired comparison requires at least one candidate")
    configure_evaluator_caches(root)
    ranx = importlib.import_module("ranx")
    qrels_values: dict[str, dict[str, int]] = {}
    for qrel in read_qrels(qrels_path):
        qrels_values.setdefault(qrel.query_id, {})[qrel.document_id] = qrel.relevance
    qrels = ranx.Qrels(qrels_values, name=f"{dataset}-{split}")
    run_paths = {baseline_name: baseline_run_path, **candidate_run_paths}
    runs = []
    for name, run_path in run_paths.items():
        run_values = run_values_for_evaluators(read_run(run_path))
        runs.append(ranx.Run(run_values, name=name))
    report = ranx.compare(
        qrels=qrels,
        runs=runs,
        metrics=[metric],
        stat_test="student",
        max_p=alpha,
        threads=1,
        make_comparable=True,
    )
    report_values = cast(dict[str, Any], report.to_dict())
    raw_p_values = {
        name: float(report_values[baseline_name]["comparisons"][name][metric])
        for name in candidate_run_paths
    }
    adjusted_p_values = holm_adjusted_p_values(raw_p_values)
    baseline_native = evaluate_run(
        qrels_path, baseline_run_path, exact_gain=exact_gain
    ).metrics[metric]
    comparisons: dict[str, object] = {}
    for name, run_path in candidate_run_paths.items():
        candidate_native = evaluate_run(
            qrels_path, run_path, exact_gain=exact_gain
        ).metrics[metric]
        raw_p = raw_p_values[name]
        adjusted_p = adjusted_p_values[name]
        delta = candidate_native - baseline_native
        significant = adjusted_p <= alpha
        meets_delta = delta >= minimum_delta
        supports_gate = significant and meets_delta
        comparisons[name] = {
            "baseline_score": baseline_native,
            "candidate_score": candidate_native,
            "paired_mean_delta": delta,
            "minimum_required_delta": minimum_delta,
            "meets_minimum_delta": meets_delta,
            "raw_p_value": raw_p,
            "holm_adjusted_p_value": adjusted_p,
            "alpha": alpha,
            "statistically_significant": significant,
            "supports_quality_significance_gate": supports_gate,
            "eligible_for_decision": eligible_for_decision,
            "interpretation": (
                "passes_quality_significance_gate"
                if supports_gate and eligible_for_decision
                else (
                    "provisional_only_ineligible_runtime"
                    if supports_gate
                    else "does_not_pass_quality_significance_gate"
                )
            ),
            "win_tie_loss": report_values[name]["win_tie_loss"][baseline_name][metric],
            "ranx_scores": {
                "baseline": float(report_values[baseline_name]["scores"][metric]),
                "candidate": float(report_values[name]["scores"][metric]),
            },
        }
    return {
        "schema_version": 1,
        "dataset": dataset,
        "split": split,
        "metric": metric,
        "test": "ranx_paired_student_t_test_two_sided",
        "multiple_testing_correction": "holm",
        "holm_family_size": len(candidate_run_paths),
        "alpha": alpha,
        "baseline": baseline_name,
        "eligible_for_decision": eligible_for_decision,
        "packages": {
            "ranx": importlib.metadata.version("ranx"),
            "scipy": importlib.metadata.version("scipy"),
        },
        "comparisons": comparisons,
    }


def derive_family_eligibility(
    *,
    benchmark_provenance_valid: object,
    source_eligibilities: tuple[object, ...],
) -> bool:
    return benchmark_provenance_valid is True and all(
        value is True for value in source_eligibilities
    )


def verify_registered_wands_evidence_inputs(
    *,
    root: Path,
    index: str,
    models: tuple[ModelSpec, ...],
    runtime: QueryRuntimeSpec,
    sparse: NeuralSparseSpec | None,
    experiments: BM25Experiments,
    metric: str,
    alpha: float,
    minimum_delta: float,
    minimum_judged_at_10: float | None,
) -> dict[str, Any]:
    registered_experiments = load_bm25_experiments(
        root / "config/experiments.toml"
    )
    if experiments != registered_experiments:
        raise DatasetIntegrityError(
            "WANDS evidence experiment configuration differs from registry"
        )
    registry = load_model_registry(root / "config/models.toml")
    registered_models = tuple(
        registry[name] for name in WANDS_INT8_MODEL_NAMES
    )
    if models != registered_models:
        raise DatasetIntegrityError(
            "WANDS evidence model candidates differ from registry"
        )
    registered_runtime = load_query_runtime_spec(
        root / "config/query_runtime.toml"
    )
    if runtime != registered_runtime:
        raise DatasetIntegrityError(
            "WANDS evidence query runtime differs from registry"
        )
    registered_index = load_wands_index_config(
        root / "config/indexes.toml"
    )
    if index != registered_index.name:
        raise DatasetIntegrityError("WANDS evidence index differs from registry")
    registered_sparse = None
    if sparse is not None:
        registered_sparse = load_neural_sparse_spec(
            root / "config/neural_sparse.toml"
        )
        if sparse != registered_sparse:
            raise DatasetIntegrityError(
                "WANDS evidence neural-sparse configuration differs from registry"
            )
    thresholds_match = (
        metric == registered_experiments.primary_metric
        and alpha == registered_experiments.alpha
        and minimum_delta == registered_experiments.minimum_paired_delta
        and (
            minimum_judged_at_10 is None
            or minimum_judged_at_10
            == registered_experiments.minimum_judged_at_10
        )
    )
    if not thresholds_match:
        raise DatasetIntegrityError(
            "WANDS evidence decision thresholds differ from registered values"
        )
    config_paths: dict[str, Path] = {
        "datasets": root / "config/datasets.toml",
        "indexes": root / "config/indexes.toml",
        "models": root / "config/models.toml",
        "query_runtime": root / "config/query_runtime.toml",
        "experiments": root / "config/experiments.toml",
    }
    if sparse is not None:
        config_paths["neural_sparse"] = root / "config/neural_sparse.toml"
    return _json_dict(
        {
            "index": registered_index.name,
            "models": [asdict(model) for model in registered_models],
            "query_runtime": asdict(registered_runtime),
            "neural_sparse": (
                asdict(registered_sparse)
                if registered_sparse is not None
                else None
            ),
            "experiments": asdict(registered_experiments),
            "decision_thresholds": {
                "metric": registered_experiments.primary_metric,
                "alpha": registered_experiments.alpha,
                "minimum_delta": (
                    registered_experiments.minimum_paired_delta
                ),
                "minimum_judged_at_10": minimum_judged_at_10,
            },
            "config_artifacts": {
                name: _artifact(root, path)
                for name, path in config_paths.items()
            },
        }
    )


def collect_verified_wands_source_summaries(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    models: tuple[ModelSpec, ...],
    runtime: QueryRuntimeSpec,
    sparse: NeuralSparseSpec | None,
    experiments: BM25Experiments,
) -> dict[str, dict[str, Any]]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    bm25 = verify_wands_bm25_selection(
        client,
        root=root,
        index=index,
        experiments=experiments,
    )
    hybrid = verify_wands_hybrid_summary(
        client,
        root=root,
        index=index,
        models=models,
        runtime=runtime,
        experiments=experiments,
    )
    rrf = verify_wands_rrf_summary(
        client,
        root=root,
        index=index,
        models=models,
        runtime=runtime,
        experiments=experiments,
    )
    summaries = {"bm25": bm25, "hybrid": hybrid, "rrf": rrf}
    if sparse is not None:
        summaries["sparse"] = verify_wands_sparse_summary(
            client,
            root=root,
            spec=sparse,
            experiments=experiments,
        )
    return summaries


def build_wands_source_evidence(
    *,
    root: Path,
    summaries: Mapping[str, Mapping[str, Any]],
    include_sparse: bool,
) -> dict[str, dict[str, object]]:
    evidence = {
        "bm25_tuned": _source_evidence(
            root,
            summary_path=root / "results/wands/bm25-selection.json",
            summary=summaries["bm25"],
            artifact_key="tuned/test",
            summary_eligibility_key="quality_evidence_eligible_for_decision",
            semantic_verifier="verify_wands_bm25_selection",
        ),
        "bm25_competent_integrator": _source_evidence(
            root,
            summary_path=root / "results/wands/bm25-selection.json",
            summary=summaries["bm25"],
            artifact_key="competent_integrator/test",
            summary_eligibility_key="quality_evidence_eligible_for_decision",
            semantic_verifier="verify_wands_bm25_selection",
        ),
        "dense_minmax": _source_evidence(
            root,
            summary_path=root / "results/wands/hybrid-summary.json",
            summary=summaries["hybrid"],
            artifact_key="test/selected",
            summary_eligibility_key="eligible_for_decision",
            semantic_verifier="verify_wands_hybrid_summary",
        ),
        "dense_rrf": _source_evidence(
            root,
            summary_path=root / "results/wands/rrf-summary.json",
            summary=summaries["rrf"],
            artifact_key="test/selected",
            summary_eligibility_key="eligible_for_decision",
            semantic_verifier="verify_wands_rrf_summary",
        ),
    }
    if include_sparse:
        sparse_summary = summaries.get("sparse")
        if sparse_summary is None:
            raise DatasetIntegrityError(
                "verified neural-sparse summary is required"
            )
        evidence["sparse_only"] = _source_evidence(
            root,
            summary_path=(
                root / "results/wands/neural-sparse/summary.json"
            ),
            summary=sparse_summary,
            artifact_key="sparse/test",
            summary_eligibility_key=(
                "quality_evidence_eligible_for_decision"
            ),
            semantic_verifier="verify_wands_sparse_summary",
        )
        evidence["sparse_minmax"] = _source_evidence(
            root,
            summary_path=(
                root / "results/wands/neural-sparse/summary.json"
            ),
            summary=sparse_summary,
            artifact_key="hybrid/test/selected",
            summary_eligibility_key=(
                "quality_evidence_eligible_for_decision"
            ),
            semantic_verifier="verify_wands_sparse_summary",
        )
    return evidence


def build_wands_hybrid_significance(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    models: tuple[ModelSpec, ...],
    runtime: QueryRuntimeSpec,
    experiments: BM25Experiments,
    metric: str,
    alpha: float,
    minimum_delta: float,
) -> dict[str, object]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    benchmark_provenance = _json_dict(collect_manifest_provenance(root))
    registered_inputs = verify_registered_wands_evidence_inputs(
        root=root,
        index=index,
        models=models,
        runtime=runtime,
        sparse=None,
        experiments=experiments,
        metric=metric,
        alpha=alpha,
        minimum_delta=minimum_delta,
        minimum_judged_at_10=None,
    )
    summaries = collect_verified_wands_source_summaries(
        client,
        root=root,
        index=index,
        models=models,
        runtime=runtime,
        sparse=None,
        experiments=experiments,
    )
    source_evidence = build_wands_source_evidence(
        root=root,
        summaries=summaries,
        include_sparse=False,
    )
    qrels_artifact = _artifact(
        root, root / "data/prepared/wands/qrels.test.trec"
    )
    comparisons = _compute_wands_hybrid_comparisons(
        root=root,
        summaries=summaries,
        metric=metric,
        alpha=alpha,
        minimum_delta=minimum_delta,
    )
    _verify_significance_inputs_unchanged(
        root=root,
        summaries=summaries,
        include_sparse=False,
        source_evidence=source_evidence,
        qrels_artifact=qrels_artifact,
    )
    provenance_valid, completion_revision = (
        wands_completion_provenance_evidence(root, benchmark_provenance)
    )
    result = _assemble_wands_hybrid_significance(
        comparisons=comparisons,
        source_evidence=source_evidence,
        qrels_artifact=qrels_artifact,
        registered_inputs=registered_inputs,
        benchmark_provenance=benchmark_provenance,
        benchmark_provenance_valid=provenance_valid,
        completion_revision=completion_revision,
    )
    write_json(root / "results/wands/hybrid-significance.json", result)
    return result


def hybrid_candidate_run_paths(
    root: Path,
    minmax: Mapping[str, Any],
    rrf: Mapping[str, Any],
) -> dict[str, Path]:
    minmax_entry = cast(dict[str, Any], cast(dict[str, Any], minmax["artifacts"])["test/selected"])
    rrf_entry = cast(dict[str, Any], cast(dict[str, Any], rrf["artifacts"])["test/selected"])
    weight_label = round(float(minmax["selected_lexical_weight"]) * 100)
    rank_constant = int(rrf["selected_rank_constant"])
    return {
        f"hybrid_{minmax['model']}_minmax_lw{weight_label:03d}": root / str(minmax_entry["run"]),
        f"hybrid_{rrf['model']}_rrf_k{rank_constant:03d}": root / str(rrf_entry["run"]),
    }


def sparse_candidate_run_paths(
    root: Path,
    sparse: Mapping[str, Any],
) -> dict[str, Path]:
    artifacts = cast(dict[str, Any], sparse["artifacts"])
    sparse_entry = cast(dict[str, Any], artifacts["sparse/test"])
    hybrid_entry = cast(dict[str, Any], artifacts["hybrid/test/selected"])
    weight_label = round(float(sparse["selected_lexical_weight"]) * 100)
    return {
        "neural_sparse_doc_only": root / str(sparse_entry["run"]),
        f"bm25_neural_sparse_minmax_lw{weight_label:03d}": root
        / str(hybrid_entry["run"]),
    }


def build_wands_sparse_significance(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    models: tuple[ModelSpec, ...],
    runtime: QueryRuntimeSpec,
    sparse: NeuralSparseSpec,
    experiments: BM25Experiments,
    metric: str,
    alpha: float,
    minimum_delta: float,
) -> dict[str, object]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    benchmark_provenance = _json_dict(collect_manifest_provenance(root))
    registered_inputs = verify_registered_wands_evidence_inputs(
        root=root,
        index=index,
        models=models,
        runtime=runtime,
        sparse=sparse,
        experiments=experiments,
        metric=metric,
        alpha=alpha,
        minimum_delta=minimum_delta,
        minimum_judged_at_10=None,
    )
    summaries = collect_verified_wands_source_summaries(
        client,
        root=root,
        index=index,
        models=models,
        runtime=runtime,
        sparse=sparse,
        experiments=experiments,
    )
    source_evidence = build_wands_source_evidence(
        root=root,
        summaries=summaries,
        include_sparse=True,
    )
    qrels_artifact = _artifact(
        root, root / "data/prepared/wands/qrels.test.trec"
    )
    comparisons = _compute_wands_sparse_comparisons(
        root=root,
        summaries=summaries,
        metric=metric,
        alpha=alpha,
        minimum_delta=minimum_delta,
    )
    _verify_significance_inputs_unchanged(
        root=root,
        summaries=summaries,
        include_sparse=True,
        source_evidence=source_evidence,
        qrels_artifact=qrels_artifact,
    )
    provenance_valid, completion_revision = (
        wands_completion_provenance_evidence(root, benchmark_provenance)
    )
    result = _assemble_wands_sparse_significance(
        comparisons=comparisons,
        source_evidence=source_evidence,
        qrels_artifact=qrels_artifact,
        registered_inputs=registered_inputs,
        benchmark_provenance=benchmark_provenance,
        benchmark_provenance_valid=provenance_valid,
        completion_revision=completion_revision,
    )
    write_json(root / "results/wands/neural-sparse/significance.json", result)
    return result


def _compute_wands_sparse_comparisons(
    *,
    root: Path,
    summaries: Mapping[str, Mapping[str, Any]],
    metric: str,
    alpha: float,
    minimum_delta: float,
) -> dict[str, dict[str, object]]:
    sparse = summaries["sparse"]
    bm25 = summaries["bm25"]
    tuned_entry = cast(
        dict[str, Any],
        cast(dict[str, Any], bm25["artifacts"])["tuned/test"],
    )
    competent_entry = cast(
        dict[str, Any],
        cast(dict[str, Any], bm25["artifacts"])["competent_integrator/test"],
    )
    candidate_paths = sparse_candidate_run_paths(root, sparse)
    primary = compare_run_family(
        root=root,
        qrels_path=root / "data/prepared/wands/qrels.test.trec",
        baseline_name="tuned_bm25",
        baseline_run_path=root / str(tuned_entry["run"]),
        candidate_run_paths=candidate_paths,
        metric=metric,
        alpha=alpha,
        minimum_delta=minimum_delta,
        eligible_for_decision=False,
    )
    challenge = compare_run_family(
        root=root,
        qrels_path=root / "data/prepared/wands/qrels.test.trec",
        baseline_name="competent_integrator_bm25",
        baseline_run_path=root / str(competent_entry["run"]),
        candidate_run_paths=candidate_paths,
        metric=metric,
        alpha=alpha,
        minimum_delta=minimum_delta,
        eligible_for_decision=False,
    )
    hybrid = summaries["hybrid"]
    rrf = summaries["rrf"]
    sparse_hybrid_entry = cast(
        dict[str, Any],
        cast(dict[str, Any], sparse["artifacts"])["hybrid/test/selected"],
    )
    dense_reference = compare_run_family(
        root=root,
        qrels_path=root / "data/prepared/wands/qrels.test.trec",
        baseline_name="bm25_neural_sparse",
        baseline_run_path=root / str(sparse_hybrid_entry["run"]),
        candidate_run_paths=hybrid_candidate_run_paths(root, hybrid, rrf),
        metric=metric,
        alpha=alpha,
        minimum_delta=minimum_delta,
        eligible_for_decision=False,
    )
    return {
        "primary": primary,
        "post_review_challenge": challenge,
        "dense_reference": dense_reference,
    }


def verify_wands_sparse_significance(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    models: tuple[ModelSpec, ...],
    runtime: QueryRuntimeSpec,
    sparse: NeuralSparseSpec,
    experiments: BM25Experiments,
    metric: str,
    alpha: float,
    minimum_delta: float,
) -> dict[str, Any]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    path = root / "results/wands/neural-sparse/significance.json"
    recorded = cast(dict[str, Any], read_json(path))
    registered_inputs = verify_registered_wands_evidence_inputs(
        root=root,
        index=index,
        models=models,
        runtime=runtime,
        sparse=sparse,
        experiments=experiments,
        metric=metric,
        alpha=alpha,
        minimum_delta=minimum_delta,
        minimum_judged_at_10=None,
    )
    summaries = collect_verified_wands_source_summaries(
        client,
        root=root,
        index=index,
        models=models,
        runtime=runtime,
        sparse=sparse,
        experiments=experiments,
    )
    source_evidence = build_wands_source_evidence(
        root=root,
        summaries=summaries,
        include_sparse=True,
    )
    qrels_artifact = _artifact(
        root, root / "data/prepared/wands/qrels.test.trec"
    )
    comparisons = _compute_wands_sparse_comparisons(
        root=root,
        summaries=summaries,
        metric=metric,
        alpha=alpha,
        minimum_delta=minimum_delta,
    )
    _verify_significance_inputs_unchanged(
        root=root,
        summaries=summaries,
        include_sparse=True,
        source_evidence=source_evidence,
        qrels_artifact=qrels_artifact,
    )
    benchmark_provenance, completion_revision, provenance_valid = (
        _recorded_significance_provenance(root, recorded)
    )
    expected = _assemble_wands_sparse_significance(
        comparisons=comparisons,
        source_evidence=source_evidence,
        qrels_artifact=qrels_artifact,
        registered_inputs=registered_inputs,
        benchmark_provenance=benchmark_provenance,
        benchmark_provenance_valid=provenance_valid,
        completion_revision=completion_revision,
    )
    if recorded != expected:
        raise DatasetIntegrityError(
            "WANDS sparse significance artifact does not reproduce"
        )
    return recorded


def _compute_wands_hybrid_comparisons(
    *,
    root: Path,
    summaries: Mapping[str, Mapping[str, Any]],
    metric: str,
    alpha: float,
    minimum_delta: float,
) -> dict[str, dict[str, object]]:
    hybrid = summaries["hybrid"]
    rrf = summaries["rrf"]
    bm25 = summaries["bm25"]
    bm25_entry = cast(dict[str, Any], cast(dict[str, Any], bm25["artifacts"])["tuned/test"])
    competent_entry = cast(
        dict[str, Any],
        cast(dict[str, Any], bm25["artifacts"])["competent_integrator/test"],
    )
    candidate_paths = hybrid_candidate_run_paths(root, hybrid, rrf)
    primary = compare_run_family(
        root=root,
        qrels_path=root / "data/prepared/wands/qrels.test.trec",
        baseline_name="tuned_bm25",
        baseline_run_path=root / str(bm25_entry["run"]),
        candidate_run_paths=candidate_paths,
        metric=metric,
        alpha=alpha,
        minimum_delta=minimum_delta,
        eligible_for_decision=False,
    )
    challenge = compare_run_family(
        root=root,
        qrels_path=root / "data/prepared/wands/qrels.test.trec",
        baseline_name="competent_integrator_bm25",
        baseline_run_path=root / str(competent_entry["run"]),
        candidate_run_paths=candidate_paths,
        metric=metric,
        alpha=alpha,
        minimum_delta=minimum_delta,
        eligible_for_decision=False,
    )
    return {"primary": primary, "post_review_challenge": challenge}


def verify_wands_hybrid_significance(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    models: tuple[ModelSpec, ...],
    runtime: QueryRuntimeSpec,
    experiments: BM25Experiments,
    metric: str,
    alpha: float,
    minimum_delta: float,
) -> dict[str, Any]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    path = root / "results/wands/hybrid-significance.json"
    recorded = cast(dict[str, Any], read_json(path))
    registered_inputs = verify_registered_wands_evidence_inputs(
        root=root,
        index=index,
        models=models,
        runtime=runtime,
        sparse=None,
        experiments=experiments,
        metric=metric,
        alpha=alpha,
        minimum_delta=minimum_delta,
        minimum_judged_at_10=None,
    )
    summaries = collect_verified_wands_source_summaries(
        client,
        root=root,
        index=index,
        models=models,
        runtime=runtime,
        sparse=None,
        experiments=experiments,
    )
    source_evidence = build_wands_source_evidence(
        root=root,
        summaries=summaries,
        include_sparse=False,
    )
    qrels_artifact = _artifact(
        root, root / "data/prepared/wands/qrels.test.trec"
    )
    comparisons = _compute_wands_hybrid_comparisons(
        root=root,
        summaries=summaries,
        metric=metric,
        alpha=alpha,
        minimum_delta=minimum_delta,
    )
    _verify_significance_inputs_unchanged(
        root=root,
        summaries=summaries,
        include_sparse=False,
        source_evidence=source_evidence,
        qrels_artifact=qrels_artifact,
    )
    benchmark_provenance, completion_revision, provenance_valid = (
        _recorded_significance_provenance(root, recorded)
    )
    expected = _assemble_wands_hybrid_significance(
        comparisons=comparisons,
        source_evidence=source_evidence,
        qrels_artifact=qrels_artifact,
        registered_inputs=registered_inputs,
        benchmark_provenance=benchmark_provenance,
        benchmark_provenance_valid=provenance_valid,
        completion_revision=completion_revision,
    )
    if recorded != expected:
        raise DatasetIntegrityError(
            "WANDS hybrid significance artifact does not reproduce"
        )
    return recorded


def _assemble_wands_hybrid_significance(
    *,
    comparisons: Mapping[str, Mapping[str, object]],
    source_evidence: Mapping[str, Mapping[str, object]],
    qrels_artifact: Mapping[str, object],
    registered_inputs: Mapping[str, Any],
    benchmark_provenance: Mapping[str, Any],
    benchmark_provenance_valid: bool,
    completion_revision: Mapping[str, Any],
) -> dict[str, object]:
    families = {
        "primary": ("bm25_tuned", "dense_minmax", "dense_rrf"),
        "post_review_challenge": (
            "bm25_competent_integrator",
            "dense_minmax",
            "dense_rrf",
        ),
    }
    family_evidence = {
        name: _family_evidence(
            source_evidence,
            required_sources=required_sources,
            benchmark_provenance_valid=benchmark_provenance_valid,
        )
        for name, required_sources in families.items()
    }
    primary = _json_dict(comparisons["primary"])
    primary["schema_version"] = 2
    apply_family_eligibility(
        primary,
        family_evidence["primary"]["eligible_for_decision"] is True,
    )
    challenge = _json_dict(comparisons["post_review_challenge"])
    challenge["schema_version"] = 2
    apply_family_eligibility(
        challenge,
        family_evidence["post_review_challenge"][
            "eligible_for_decision"
        ]
        is True,
    )
    primary.update(
        {
            "classification": "registered_held_out_paired_significance",
            "benchmark_provenance": _json_dict(benchmark_provenance),
            "benchmark_provenance_valid": benchmark_provenance_valid,
            "completion_code_revision": _json_dict(completion_revision),
            "registered_inputs": _json_dict(registered_inputs),
            "qrels_artifact": _json_dict(qrels_artifact),
            "source_evidence": _json_dict(source_evidence),
            "source_evidence_unchanged_at_completion": True,
            "family_eligibility": _json_dict(family_evidence),
            "post_review_challenge": challenge,
        }
    )
    return cast(dict[str, object], primary)


def _assemble_wands_sparse_significance(
    *,
    comparisons: Mapping[str, Mapping[str, object]],
    source_evidence: Mapping[str, Mapping[str, object]],
    qrels_artifact: Mapping[str, object],
    registered_inputs: Mapping[str, Any],
    benchmark_provenance: Mapping[str, Any],
    benchmark_provenance_valid: bool,
    completion_revision: Mapping[str, Any],
) -> dict[str, object]:
    families = {
        "primary": ("bm25_tuned", "sparse_only", "sparse_minmax"),
        "post_review_challenge": (
            "bm25_competent_integrator",
            "sparse_only",
            "sparse_minmax",
        ),
        "dense_reference": (
            "sparse_minmax",
            "dense_minmax",
            "dense_rrf",
        ),
    }
    family_evidence = {
        name: _family_evidence(
            source_evidence,
            required_sources=required_sources,
            benchmark_provenance_valid=benchmark_provenance_valid,
        )
        for name, required_sources in families.items()
    }
    primary = _json_dict(comparisons["primary"])
    primary["schema_version"] = 2
    apply_family_eligibility(
        primary,
        family_evidence["primary"]["eligible_for_decision"] is True,
    )
    challenge = _json_dict(comparisons["post_review_challenge"])
    challenge["schema_version"] = 2
    apply_family_eligibility(
        challenge,
        family_evidence["post_review_challenge"][
            "eligible_for_decision"
        ]
        is True,
    )
    dense_reference = _json_dict(comparisons["dense_reference"])
    dense_reference["schema_version"] = 2
    apply_family_eligibility(
        dense_reference,
        family_evidence["dense_reference"]["eligible_for_decision"]
        is True,
    )
    primary.update(
        {
            "classification": "registered_held_out_paired_significance",
            "benchmark_provenance": _json_dict(benchmark_provenance),
            "benchmark_provenance_valid": benchmark_provenance_valid,
            "completion_code_revision": _json_dict(completion_revision),
            "registered_inputs": _json_dict(registered_inputs),
            "qrels_artifact": _json_dict(qrels_artifact),
            "source_evidence": _json_dict(source_evidence),
            "source_evidence_unchanged_at_completion": True,
            "family_eligibility": _json_dict(family_evidence),
            "post_review_challenge": challenge,
            "dense_reference": dense_reference,
        }
    )
    return cast(dict[str, object], primary)


def _source_evidence(
    root: Path,
    *,
    summary_path: Path,
    summary: Mapping[str, Any],
    artifact_key: str,
    summary_eligibility_key: str,
    semantic_verifier: str,
) -> dict[str, object]:
    recorded_summary = read_json(summary_path)
    if not isinstance(recorded_summary, Mapping) or _json_dict(
        recorded_summary
    ) != _json_dict(summary):
        raise DatasetIntegrityError(
            f"verified source summary changed after verification: {semantic_verifier}"
        )
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise DatasetIntegrityError(
            f"verified source has no artifacts: {semantic_verifier}"
        )
    entry = artifacts.get(artifact_key)
    if not isinstance(entry, Mapping):
        raise DatasetIntegrityError(
            f"verified source artifact is missing: {artifact_key}"
        )
    files: dict[str, dict[str, object]] = {}
    for key in ("run", "manifest", "metrics_file"):
        relative = entry.get(key)
        if not isinstance(relative, str) or not relative:
            raise DatasetIntegrityError(
                f"verified source artifact has no {key}: {artifact_key}"
            )
        path = _bound_source_path(root, relative)
        files[key] = _artifact(root, path)
    manifest = read_json(root / str(entry["manifest"]))
    if not isinstance(manifest, Mapping):
        raise DatasetIntegrityError(
            f"verified source manifest is not an object: {artifact_key}"
        )
    summary_eligible = summary.get(summary_eligibility_key) is True
    manifest_eligible = manifest.get("eligible_for_decision") is True
    return {
        "semantic_verifier": semantic_verifier,
        "summary": _artifact(root, summary_path),
        "summary_artifact_key": artifact_key,
        "summary_eligibility_key": summary_eligibility_key,
        "summary_eligible_for_decision": summary_eligible,
        "artifact_entry": _json_dict(entry),
        "files": files,
        "manifest_eligible_for_decision": manifest_eligible,
        "eligible_for_decision": summary_eligible and manifest_eligible,
    }


def _verify_significance_inputs_unchanged(
    *,
    root: Path,
    summaries: Mapping[str, Mapping[str, Any]],
    include_sparse: bool,
    source_evidence: Mapping[str, Mapping[str, object]],
    qrels_artifact: Mapping[str, object],
) -> None:
    completion_source_evidence = build_wands_source_evidence(
        root=root,
        summaries=summaries,
        include_sparse=include_sparse,
    )
    completion_qrels = _artifact(
        root, root / "data/prepared/wands/qrels.test.trec"
    )
    if (
        completion_source_evidence != source_evidence
        or completion_qrels != qrels_artifact
    ):
        raise DatasetIntegrityError(
            "WANDS significance source evidence changed during computation"
        )


def _family_evidence(
    source_evidence: Mapping[str, Mapping[str, object]],
    *,
    required_sources: tuple[str, ...],
    benchmark_provenance_valid: bool,
) -> dict[str, object]:
    source_eligibilities = {
        name: source_evidence[name].get("eligible_for_decision") is True
        for name in required_sources
    }
    eligible = derive_family_eligibility(
        benchmark_provenance_valid=benchmark_provenance_valid,
        source_eligibilities=tuple(source_eligibilities.values()),
    )
    reasons: list[str] = []
    if not benchmark_provenance_valid:
        reasons.append(
            "significance artifact has no valid clean committed start provenance"
        )
    reasons.extend(
        f"source is not decision eligible: {name}"
        for name, source_eligible in source_eligibilities.items()
        if not source_eligible
    )
    return {
        "required_sources": list(required_sources),
        "source_eligibilities": source_eligibilities,
        "benchmark_provenance_valid": benchmark_provenance_valid,
        "eligible_for_decision": eligible,
        "ineligibility_reasons": reasons,
    }


def apply_family_eligibility(
    comparison: dict[str, Any],
    eligible: bool,
) -> None:
    comparison["eligible_for_decision"] = eligible
    comparisons = comparison.get("comparisons")
    if not isinstance(comparisons, dict):
        raise DatasetIntegrityError("paired significance comparisons are missing")
    for value in comparisons.values():
        if not isinstance(value, dict):
            raise DatasetIntegrityError(
                "paired significance comparison is not an object"
            )
        supports = value.get("supports_quality_significance_gate") is True
        value["eligible_for_decision"] = eligible
        value["interpretation"] = (
            "passes_quality_significance_gate"
            if supports and eligible
            else (
                "provisional_only_ineligible_evidence"
                if supports
                else "does_not_pass_quality_significance_gate"
            )
        )


def _recorded_significance_provenance(
    root: Path,
    recorded: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    provenance = recorded.get("benchmark_provenance")
    completion = recorded.get("completion_code_revision")
    if not isinstance(provenance, Mapping) or not isinstance(
        completion, Mapping
    ):
        raise DatasetIntegrityError(
            "WANDS significance provenance is missing"
        )
    provenance_dict = _json_dict(provenance)
    completion_dict = _json_dict(completion)
    valid = wands_recorded_provenance_valid(
        root,
        provenance=provenance_dict,
        completion_revision=completion_dict,
    )
    return provenance_dict, completion_dict, valid


def _bound_source_path(root: Path, relative: str) -> Path:
    root_resolved = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root_resolved) or not path.is_file():
        raise DatasetIntegrityError(
            f"source artifact path is missing or escapes root: {relative}"
        )
    return path


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }


def _json_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(json.dumps(value)))
