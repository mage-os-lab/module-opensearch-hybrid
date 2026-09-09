from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast

from poc.config import ModelSpec
from poc.datasets import DatasetIntegrityError, file_facts
from poc.evaluation import evaluate_run
from poc.experiments import BM25Experiments
from poc.indexing import (
    wands_completion_provenance_evidence,
    wands_recorded_provenance_valid,
)
from poc.manifest import read_json, write_json
from poc.neural_sparse import NeuralSparseSpec
from poc.os_client import OpenSearchClient
from poc.provenance import collect_manifest_provenance, require_registered_opensearch_client
from poc.query_runtime import QueryRuntimeSpec
from poc.significance import (
    apply_family_eligibility,
    build_wands_source_evidence,
    collect_verified_wands_source_summaries,
    compare_run_family,
    verify_registered_wands_evidence_inputs,
)
from poc.trec import QrelRecord, RunRecord, read_qrels, read_run, write_qrels, write_run

RelevanceMapping = Literal["primary", "strict_binary", "broad_binary"]


def map_qrels(
    qrels: list[QrelRecord],
    *,
    mapping: RelevanceMapping,
    maximum_gain: int,
) -> list[QrelRecord]:
    if maximum_gain <= 0 or maximum_gain not in {
        record.relevance for record in qrels
    }:
        raise ValueError("declared maximum gain is absent from qrels")
    if mapping == "primary":
        return list(qrels)
    if mapping == "strict_binary":
        return [
            QrelRecord(
                record.query_id,
                record.document_id,
                int(record.relevance == maximum_gain),
            )
            for record in qrels
        ]
    if mapping == "broad_binary":
        return [
            QrelRecord(
                record.query_id,
                record.document_id,
                int(record.relevance > 0),
            )
            for record in qrels
        ]
    raise ValueError(f"unsupported relevance mapping {mapping}")


def rerank_within_judged(
    qrels: list[QrelRecord],
    run: list[RunRecord],
) -> list[RunRecord]:
    judged = {(record.query_id, record.document_id) for record in qrels}
    next_rank: dict[str, int] = {}
    filtered: list[RunRecord] = []
    for record in sorted(run, key=lambda item: (item.query_id, item.rank)):
        if (record.query_id, record.document_id) not in judged:
            continue
        rank = next_rank.get(record.query_id, 1)
        filtered.append(
            RunRecord(
                query_id=record.query_id,
                document_id=record.document_id,
                rank=rank,
                score=record.score,
                tag=record.tag,
            )
        )
        next_rank[record.query_id] = rank + 1
    return filtered


def derive_robustness_decision_eligibility(
    *,
    benchmark_provenance_valid: object,
    source_family_eligible: object,
    judged_coverage_valid: object,
) -> bool:
    return (
        benchmark_provenance_valid is True
        and source_family_eligible is True
        and judged_coverage_valid is True
    )


def build_wands_robustness_report(
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
    minimum_judged_at_10: float,
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
        minimum_judged_at_10=minimum_judged_at_10,
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
    analysis = _compute_wands_robustness_analysis(
        root=root,
        summaries=summaries,
        metric=metric,
        alpha=alpha,
        minimum_delta=minimum_delta,
        minimum_judged_at_10=minimum_judged_at_10,
    )
    _verify_robustness_inputs_unchanged(
        root=root,
        summaries=summaries,
        source_evidence=source_evidence,
        qrels_artifact=qrels_artifact,
    )
    provenance_valid, completion_revision = (
        wands_completion_provenance_evidence(root, benchmark_provenance)
    )
    result = _assemble_wands_robustness_report(
        analysis=analysis,
        source_evidence=source_evidence,
        registered_inputs=registered_inputs,
        benchmark_provenance=benchmark_provenance,
        benchmark_provenance_valid=provenance_valid,
        completion_revision=completion_revision,
    )
    write_json(root / "results/wands/robustness.json", result)
    return result


def verify_wands_robustness_report(
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
    minimum_judged_at_10: float,
) -> dict[str, Any]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    path = root / "results/wands/robustness.json"
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
        minimum_judged_at_10=minimum_judged_at_10,
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
    analysis = _compute_wands_robustness_analysis(
        root=root,
        summaries=summaries,
        metric=metric,
        alpha=alpha,
        minimum_delta=minimum_delta,
        minimum_judged_at_10=minimum_judged_at_10,
    )
    _verify_robustness_inputs_unchanged(
        root=root,
        summaries=summaries,
        source_evidence=source_evidence,
        qrels_artifact=qrels_artifact,
    )
    benchmark_provenance, completion_revision, provenance_valid = (
        _recorded_robustness_provenance(root, recorded)
    )
    expected = _assemble_wands_robustness_report(
        analysis=analysis,
        source_evidence=source_evidence,
        registered_inputs=registered_inputs,
        benchmark_provenance=benchmark_provenance,
        benchmark_provenance_valid=provenance_valid,
        completion_revision=completion_revision,
    )
    if recorded != expected:
        raise DatasetIntegrityError("WANDS robustness artifact does not reproduce")
    return recorded


def _compute_wands_robustness_analysis(
    *,
    root: Path,
    summaries: Mapping[str, Mapping[str, Any]],
    metric: str,
    alpha: float,
    minimum_delta: float,
    minimum_judged_at_10: float,
) -> dict[str, object]:
    qrels_path = root / "data/prepared/wands/qrels.test.trec"
    qrels = read_qrels(qrels_path)
    maximum_gain = max(record.relevance for record in qrels)
    baseline_path, candidate_paths = _headline_paths(root, summaries)
    source_paths = {"tuned_bm25": baseline_path, **candidate_paths}
    full_metrics = {
        name: evaluate_run(qrels_path, path, exact_gain=maximum_gain).metrics
        for name, path in source_paths.items()
    }
    coverage_passes = {
        name: metrics["judged@10"] >= minimum_judged_at_10
        for name, metrics in full_metrics.items()
    }
    mappings: dict[str, object] = {}
    with TemporaryDirectory(prefix="opensearch-hybrid-robustness-") as temporary:
        temporary_root = Path(temporary)
        mapping_names: tuple[RelevanceMapping, ...] = (
            "primary",
            "strict_binary",
            "broad_binary",
        )
        for mapping in mapping_names:
            mapped_qrels = map_qrels(
                qrels,
                mapping=mapping,
                maximum_gain=maximum_gain,
            )
            mapped_qrels_path = temporary_root / f"qrels-{mapping}.trec"
            write_qrels(mapped_qrels_path, mapped_qrels)
            exact_gain = maximum_gain if mapping == "primary" else 1
            mapping_views: dict[str, object] = {}
            for view in ("full_corpus", "rerank_within_judged"):
                if view == "full_corpus":
                    view_paths = source_paths
                else:
                    view_paths = _write_judged_only_runs(
                        temporary_root / mapping,
                        qrels=qrels,
                        run_paths=source_paths,
                    )
                view_baseline = view_paths["tuned_bm25"]
                view_candidates = {
                    name: view_paths[name] for name in candidate_paths
                }
                comparison = compare_run_family(
                    root=root,
                    qrels_path=mapped_qrels_path,
                    baseline_name="tuned_bm25",
                    baseline_run_path=view_baseline,
                    candidate_run_paths=view_candidates,
                    metric=metric,
                    alpha=alpha,
                    minimum_delta=minimum_delta,
                    eligible_for_decision=False,
                    dataset="WANDS",
                    split=f"test-{mapping}-{view}",
                    exact_gain=exact_gain,
                )
                mapping_views[view] = {
                    "metrics": {
                        name: evaluate_run(
                            mapped_qrels_path,
                            path,
                            exact_gain=exact_gain,
                        ).metrics
                        for name, path in view_paths.items()
                    },
                    "significance": comparison,
                }
            mappings[mapping] = mapping_views
    gate_holds = _quality_conclusion_holds(mappings, candidate_paths)
    return {
        "schema_version": 2,
        "dataset": "WANDS",
        "split": "test",
        "classification": "registered_validity_analysis_with_post_review_mapping_concretization",
        "mapping_definitions": {
            "primary": "distributed WANDS gains 2, 1, and 0",
            "strict_binary": "only maximum-gain exact judgments are relevant",
            "broad_binary": "every positive judgment is relevant",
        },
        "pool_hole_views": {
            "full_corpus": (
                "original retrieval ranking with unjudged documents treated as gain zero"
            ),
            "rerank_within_judged": (
                "original ranking filtered to explicitly judged documents and "
                "reranked contiguously"
            ),
        },
        "minimum_judged_at_10": minimum_judged_at_10,
        "full_corpus_primary_metrics": full_metrics,
        "full_corpus_judged_at_10_passes": coverage_passes,
        "all_headline_runs_meet_judged_at_10_floor": all(coverage_passes.values()),
        "mappings": mappings,
        "headline_quality_conclusion_holds_across_all_mappings_and_views": gate_holds,
        "source_artifacts": {
            "qrels": _artifact(root, qrels_path),
            "runs": {
                name: _artifact(root, path) for name, path in source_paths.items()
            },
        },
    }


def _assemble_wands_robustness_report(
    *,
    analysis: Mapping[str, object],
    source_evidence: Mapping[str, Mapping[str, object]],
    registered_inputs: Mapping[str, Any],
    benchmark_provenance: Mapping[str, Any],
    benchmark_provenance_valid: bool,
    completion_revision: Mapping[str, Any],
) -> dict[str, object]:
    result = _json_dict(analysis)
    required_sources = (
        "bm25_tuned",
        "dense_minmax",
        "dense_rrf",
        "sparse_minmax",
    )
    source_family_eligible = all(
        source_evidence[name].get("eligible_for_decision") is True
        for name in required_sources
    )
    judged_coverage_valid = (
        result.get("all_headline_runs_meet_judged_at_10_floor") is True
    )
    eligible = derive_robustness_decision_eligibility(
        benchmark_provenance_valid=benchmark_provenance_valid,
        source_family_eligible=source_family_eligible,
        judged_coverage_valid=judged_coverage_valid,
    )
    mappings = result.get("mappings")
    if not isinstance(mappings, dict):
        raise DatasetIntegrityError("WANDS robustness mappings are missing")
    for mapping_views in mappings.values():
        if not isinstance(mapping_views, dict):
            raise DatasetIntegrityError(
                "WANDS robustness mapping views are invalid"
            )
        for view in mapping_views.values():
            if not isinstance(view, dict):
                raise DatasetIntegrityError(
                    "WANDS robustness view is invalid"
                )
            significance = view.get("significance")
            if not isinstance(significance, dict):
                raise DatasetIntegrityError(
                    "WANDS robustness significance result is missing"
                )
            significance["schema_version"] = 2
            apply_family_eligibility(significance, eligible)
            view["family_eligibility"] = {
                "required_sources": list(required_sources),
                "source_family_eligible": source_family_eligible,
                "judged_coverage_valid": judged_coverage_valid,
                "benchmark_provenance_valid": (
                    benchmark_provenance_valid
                ),
                "eligible_for_decision": eligible,
            }
    numerical_holds = (
        result.get(
            "headline_quality_conclusion_holds_across_all_mappings_and_views"
        )
        is True
    )
    reasons: list[str] = []
    if not benchmark_provenance_valid:
        reasons.append(
            "robustness artifact has no valid clean committed start provenance"
        )
    reasons.extend(
        f"source is not decision eligible: {name}"
        for name in required_sources
        if source_evidence[name].get("eligible_for_decision") is not True
    )
    if not judged_coverage_valid:
        reasons.append("one or more headline runs misses the judged@10 floor")
    result.update(
        {
            "benchmark_provenance": _json_dict(benchmark_provenance),
            "benchmark_provenance_valid": benchmark_provenance_valid,
            "completion_code_revision": _json_dict(completion_revision),
            "registered_inputs": _json_dict(registered_inputs),
            "source_evidence": _json_dict(source_evidence),
            "source_evidence_unchanged_at_completion": True,
            "source_family_eligible_for_decision": (
                source_family_eligible
            ),
            "judged_coverage_valid_for_decision": judged_coverage_valid,
            "eligible_for_decision": eligible,
            "decision_ineligibility_reasons": reasons,
            "numerical_robustness_conclusion_holds": numerical_holds,
            "decision_quality_conclusion_supported": (
                numerical_holds and eligible
            ),
            "decision_interpretation": (
                "supports_headline_quality_conclusion"
                if numerical_holds and eligible
                else (
                    "numerically_robust_but_ineligible_for_decision"
                    if numerical_holds
                    else "headline_quality_conclusion_not_robust"
                )
            ),
        }
    )
    return cast(dict[str, object], result)


def _recorded_robustness_provenance(
    root: Path,
    recorded: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    provenance = recorded.get("benchmark_provenance")
    completion = recorded.get("completion_code_revision")
    if not isinstance(provenance, Mapping) or not isinstance(
        completion, Mapping
    ):
        raise DatasetIntegrityError("WANDS robustness provenance is missing")
    provenance_dict = _json_dict(provenance)
    completion_dict = _json_dict(completion)
    valid = wands_recorded_provenance_valid(
        root,
        provenance=provenance_dict,
        completion_revision=completion_dict,
    )
    return provenance_dict, completion_dict, valid


def _verify_robustness_inputs_unchanged(
    *,
    root: Path,
    summaries: Mapping[str, Mapping[str, Any]],
    source_evidence: Mapping[str, Mapping[str, object]],
    qrels_artifact: Mapping[str, object],
) -> None:
    completion_source_evidence = build_wands_source_evidence(
        root=root,
        summaries=summaries,
        include_sparse=True,
    )
    completion_qrels = _artifact(
        root, root / "data/prepared/wands/qrels.test.trec"
    )
    if (
        completion_source_evidence != source_evidence
        or completion_qrels != qrels_artifact
    ):
        raise DatasetIntegrityError(
            "WANDS robustness source evidence changed during computation"
        )


def _headline_paths(
    root: Path,
    summaries: Mapping[str, Mapping[str, Any]],
) -> tuple[Path, dict[str, Path]]:
    bm25 = summaries["bm25"]
    hybrid = summaries["hybrid"]
    rrf = summaries["rrf"]
    sparse = summaries["sparse"]
    baseline = _run_path(root, bm25, "tuned/test")
    return baseline, {
        "dense_minmax": _run_path(root, hybrid, "test/selected"),
        "dense_rrf": _run_path(root, rrf, "test/selected"),
        "sparse_minmax": _run_path(root, sparse, "hybrid/test/selected"),
    }


def _run_path(root: Path, summary: Mapping[str, Any], key: str) -> Path:
    artifacts = cast(Mapping[str, Any], summary["artifacts"])
    entry = cast(Mapping[str, Any], artifacts[key])
    return root / str(entry["run"])


def _write_judged_only_runs(
    directory: Path,
    *,
    qrels: list[QrelRecord],
    run_paths: Mapping[str, Path],
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    output: dict[str, Path] = {}
    for name, source in run_paths.items():
        path = directory / f"{name}.trec"
        write_run(path, rerank_within_judged(qrels, read_run(source)))
        output[name] = path
    return output


def _quality_conclusion_holds(
    mappings: Mapping[str, object],
    candidates: Mapping[str, Path],
) -> bool:
    for mapping_views in mappings.values():
        for view in cast(Mapping[str, Any], mapping_views).values():
            significance = cast(Mapping[str, Any], view["significance"])
            comparisons = cast(Mapping[str, Any], significance["comparisons"])
            for candidate in candidates:
                if (
                    comparisons[candidate].get(
                        "supports_quality_significance_gate"
                    )
                    is not True
                ):
                    return False
    return True


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }


def _json_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(json.dumps(value)))
