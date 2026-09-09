from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from poc.bm25 import verify_wands_bm25_selection
from poc.config import ModelSpec, load_model_registry
from poc.datasets import DatasetIntegrityError, file_facts
from poc.hybrid import verify_wands_hybrid_summary
from poc.manifest import read_json, write_json
from poc.neural_sparse import NeuralSparseSpec, load_neural_sparse_spec
from poc.os_client import OpenSearchClient
from poc.provenance import require_registered_opensearch_client
from poc.reranker import (
    RerankerSpec,
    WandsRerankerEvidenceContext,
    load_reranker_spec,
    load_wands_reranker_evidence_context,
    verify_wands_reranker_benchmark,
)
from poc.significance import compare_run_family
from poc.sparse_benchmark import verify_wands_sparse_summary
from poc.three_way import ThreeWaySpec, load_three_way_spec
from poc.three_way_benchmark import verify_wands_three_way_benchmark


@dataclass(frozen=True, slots=True)
class WandsAlternativesEvidenceContext:
    reranker: WandsRerankerEvidenceContext
    sparse: NeuralSparseSpec
    three_way: ThreeWaySpec
    three_way_model: ModelSpec
    reranker_spec: RerankerSpec


def load_wands_alternatives_evidence_context(
    *,
    root: Path,
    client: OpenSearchClient,
) -> WandsAlternativesEvidenceContext:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    reranker = load_wands_reranker_evidence_context(root=root, client=client)
    three_way = load_three_way_spec(root / "config/three_way.toml")
    registry = load_model_registry(root / "config/models.toml")
    return WandsAlternativesEvidenceContext(
        reranker=reranker,
        sparse=load_neural_sparse_spec(root / "config/neural_sparse.toml"),
        three_way=three_way,
        three_way_model=registry[three_way.dense_model],
        reranker_spec=load_reranker_spec(root / "config/reranker.toml"),
    )


def build_wands_alternatives_significance(
    *,
    root: Path,
    evidence_context: WandsAlternativesEvidenceContext,
    alpha: float,
    minimum_delta: float,
) -> dict[str, object]:
    result = _compute_wands_alternatives_significance(
        root=root,
        evidence_context=evidence_context,
        alpha=alpha,
        minimum_delta=minimum_delta,
    )
    write_json(root / "results/wands/alternatives-significance.json", result)
    return result


def verify_wands_alternatives_significance(
    *,
    root: Path,
    evidence_context: WandsAlternativesEvidenceContext,
    alpha: float,
    minimum_delta: float,
) -> dict[str, Any]:
    path = root / "results/wands/alternatives-significance.json"
    recorded = cast(dict[str, Any], read_json(path))
    expected = _compute_wands_alternatives_significance(
        root=root,
        evidence_context=evidence_context,
        alpha=alpha,
        minimum_delta=minimum_delta,
    )
    if recorded != expected:
        raise ValueError("WANDS alternatives significance does not reproduce")
    return recorded


def _compute_wands_alternatives_significance(
    *,
    root: Path,
    evidence_context: WandsAlternativesEvidenceContext,
    alpha: float,
    minimum_delta: float,
) -> dict[str, object]:
    verified_summaries = _verify_alternative_source_summaries(
        root=root,
        context=evidence_context,
    )
    sources = _run_sources(root, verified_summaries=verified_summaries)
    paths = {
        name: root / str(cast(dict[str, Any], source["run"])["path"])
        for name, source in sources.items()
    }
    qrels_path = root / "data/prepared/wands/qrels.test.trec"
    headline_candidates = {
        name: paths[name]
        for name in (
            "dense_minmax",
            "sparse_minmax",
            "three_way",
            "bm25_reranker",
            "dense_reranker",
        )
    }
    return {
        "schema_version": 2,
        "dataset": "WANDS",
        "split": "test",
        "eligibility_policy": (
            "every exact source summary and provenance-bearing run manifest in a "
            "comparison family must be decision eligible"
        ),
        "headline_family": compare_run_family(
            root=root,
            qrels_path=qrels_path,
            baseline_name="tuned_bm25",
            baseline_run_path=paths["tuned_bm25"],
            candidate_run_paths=headline_candidates,
            metric="ndcg@10",
            alpha=alpha,
            minimum_delta=minimum_delta,
            eligible_for_decision=_family_eligible(
                sources,
                "tuned_bm25",
                *headline_candidates,
            ),
        ),
        "dense_upgrade_family": compare_run_family(
            root=root,
            qrels_path=qrels_path,
            baseline_name="dense_minmax",
            baseline_run_path=paths["dense_minmax"],
            candidate_run_paths={
                "three_way": paths["three_way"],
                "dense_reranker": paths["dense_reranker"],
            },
            metric="ndcg@10",
            alpha=alpha,
            minimum_delta=0.0,
            eligible_for_decision=_family_eligible(
                sources,
                "dense_minmax",
                "three_way",
                "dense_reranker",
            ),
        ),
        "dense_vs_sparse_alternative": compare_run_family(
            root=root,
            qrels_path=qrels_path,
            baseline_name="sparse_minmax",
            baseline_run_path=paths["sparse_minmax"],
            candidate_run_paths={"dense_minmax": paths["dense_minmax"]},
            metric="ndcg@10",
            alpha=alpha,
            minimum_delta=0.03,
            eligible_for_decision=_family_eligible(
                sources,
                "sparse_minmax",
                "dense_minmax",
            ),
        ),
        "dense_vs_bm25_reranker_alternative": compare_run_family(
            root=root,
            qrels_path=qrels_path,
            baseline_name="bm25_reranker",
            baseline_run_path=paths["bm25_reranker"],
            candidate_run_paths={"dense_minmax": paths["dense_minmax"]},
            metric="ndcg@10",
            alpha=alpha,
            minimum_delta=0.0,
            eligible_for_decision=_family_eligible(
                sources,
                "bm25_reranker",
                "dense_minmax",
            ),
        ),
        "run_paths": {name: str(path.relative_to(root)) for name, path in paths.items()},
        "source_evidence": sources,
    }


def _verify_alternative_source_summaries(
    *,
    root: Path,
    context: WandsAlternativesEvidenceContext,
) -> dict[str, dict[str, Any]]:
    reranker_context = context.reranker
    bm25 = verify_wands_bm25_selection(
        reranker_context.client,
        root=root,
        index=reranker_context.index,
        experiments=reranker_context.experiments,
    )
    dense = verify_wands_hybrid_summary(
        reranker_context.client,
        root=root,
        index=reranker_context.index,
        models=reranker_context.models,
        runtime=reranker_context.runtime,
        experiments=reranker_context.experiments,
    )
    sparse = verify_wands_sparse_summary(
        reranker_context.client,
        root=root,
        spec=context.sparse,
        experiments=reranker_context.experiments,
    )
    three_way = verify_wands_three_way_benchmark(
        reranker_context.client,
        root=root,
        spec=context.three_way,
        sparse=context.sparse,
        model=context.three_way_model,
        runtime=reranker_context.runtime,
        experiments=reranker_context.experiments,
    )
    reranker = verify_wands_reranker_benchmark(
        root=root,
        spec=context.reranker_spec,
        evidence_context=reranker_context,
    )
    return {
        "bm25": bm25,
        "dense": dense,
        "sparse": sparse,
        "three_way": three_way,
        "reranker": reranker,
    }


def _run_sources(
    root: Path,
    *,
    verified_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, object]]:
    summary_paths = {
        "bm25": root / "results/wands/bm25-selection.json",
        "dense": root / "results/wands/hybrid-summary.json",
        "sparse": root / "results/wands/neural-sparse/summary.json",
        "three_way": root / "results/wands/three-way/summary.json",
        "reranker": root / "results/wands/reranker/summary.json",
    }
    summaries: dict[str, Mapping[str, Any]] = {}
    for name, path in summary_paths.items():
        verified = verified_summaries[name]
        if read_json(path) != verified:
            raise DatasetIntegrityError(
                f"alternative source summary changed after verification: {name}"
            )
        summaries[name] = verified
    return {
        "tuned_bm25": _source_evidence(
            root,
            summary_path=summary_paths["bm25"],
            summary=summaries["bm25"],
            artifact_key="tuned/test",
            summary_eligibility_key="quality_evidence_eligible_for_decision",
            semantic_verifier="verify_wands_bm25_selection",
        ),
        "dense_minmax": _source_evidence(
            root,
            summary_path=summary_paths["dense"],
            summary=summaries["dense"],
            artifact_key="test/selected",
            summary_eligibility_key="eligible_for_decision",
            semantic_verifier="verify_wands_hybrid_summary",
        ),
        "sparse_minmax": _source_evidence(
            root,
            summary_path=summary_paths["sparse"],
            summary=summaries["sparse"],
            artifact_key="hybrid/test/selected",
            summary_eligibility_key="quality_evidence_eligible_for_decision",
            semantic_verifier="verify_wands_sparse_summary",
        ),
        "three_way": _source_evidence(
            root,
            summary_path=summary_paths["three_way"],
            summary=summaries["three_way"],
            artifact_key="test/selected",
            summary_eligibility_key="eligible_for_decision",
            semantic_verifier="verify_wands_three_way_benchmark",
        ),
        "bm25_reranker": _source_evidence(
            root,
            summary_path=summary_paths["reranker"],
            summary=summaries["reranker"],
            artifact_key="tuned_bm25/test",
            summary_eligibility_key="quality_evidence_eligible_for_decision",
            semantic_verifier="verify_wands_reranker_benchmark",
        ),
        "dense_reranker": _source_evidence(
            root,
            summary_path=summary_paths["reranker"],
            summary=summaries["reranker"],
            artifact_key="dense_minmax/test",
            summary_eligibility_key="quality_evidence_eligible_for_decision",
            semantic_verifier="verify_wands_reranker_benchmark",
        ),
    }


def _source_evidence(
    root: Path,
    *,
    summary_path: Path,
    summary: Mapping[str, Any],
    artifact_key: str,
    summary_eligibility_key: str,
    semantic_verifier: str,
) -> dict[str, object]:
    artifacts = cast(dict[str, Any], summary["artifacts"])
    entry = cast(dict[str, Any], artifacts[artifact_key])
    run_path = root / str(entry["run"])
    manifest_path = root / str(entry["manifest"])
    manifest = cast(dict[str, Any], read_json(manifest_path))
    run_file = manifest.get("run_file")
    if not isinstance(run_file, dict):
        raise DatasetIntegrityError(
            f"alternative source manifest has no run binding: {artifact_key}"
        )
    if run_file.get("path") != str(run_path.relative_to(root)):
        raise DatasetIntegrityError(
            f"alternative source manifest points to another run: {artifact_key}"
        )
    if run_file.get("sha256") != file_facts(run_path).sha256:
        raise DatasetIntegrityError(
            f"alternative source run differs from its manifest: {artifact_key}"
        )
    manifest_eligible = manifest.get("eligible_for_decision") is True
    summary_eligible = summary.get(summary_eligibility_key) is True
    eligible = manifest_eligible and summary_eligible
    reasons = []
    if not manifest_eligible:
        reasons.append("source run manifest is not decision eligible")
    if not summary_eligible:
        reasons.append("source summary is not decision eligible")
    return {
        "summary": _artifact(root, summary_path),
        "summary_artifact_key": artifact_key,
        "summary_eligibility_key": summary_eligibility_key,
        "summary_eligible_for_decision": summary_eligible,
        "semantic_summary_verified": True,
        "semantic_verifier": semantic_verifier,
        "run": _artifact(root, run_path),
        "manifest": _artifact(root, manifest_path),
        "manifest_eligible_for_decision": manifest_eligible,
        "eligible_for_decision": eligible,
        "ineligibility_reasons": reasons,
    }


def _family_eligible(
    sources: dict[str, dict[str, object]],
    *names: str,
) -> bool:
    return all(sources[name]["eligible_for_decision"] is True for name in names)


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }
