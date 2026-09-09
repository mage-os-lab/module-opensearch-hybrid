from __future__ import annotations

from pathlib import Path

from poc.protocol_amendments import (
    assess_query_runtime_quality,
    compare_ranking_stability,
    load_query_runtime_quality_guard,
)
from poc.trec import RunRecord

ROOT = Path(__file__).resolve().parents[1]


def _records(rankings: dict[str, list[str]], tag: str) -> list[RunRecord]:
    return [
        RunRecord(query_id=query_id, document_id=document_id, rank=rank, score=1.0, tag=tag)
        for query_id, documents in rankings.items()
        for rank, document_id in enumerate(documents, start=1)
    ]


def test_dated_query_runtime_amendment_is_explicitly_post_measurement() -> None:
    guard = load_query_runtime_quality_guard(ROOT / "config/protocol_amendments.toml")

    assert guard.amendment_id == "2026-08-21-query-runtime-quality-guard"
    assert guard.classification == "post_measurement_protocol_amendment"
    assert guard.maximum_ndcg_at_10_loss == 0.01
    assert guard.minimum_mean_top10_overlap == 0.80
    assert guard.minimum_p05_top10_overlap == 0.60
    assert guard.minimum_top1_agreement == 0.75


def test_ranking_stability_measures_overlap_and_top1() -> None:
    reference = _records(
        {"1": ["a", "b"], "2": ["c", "d"], "3": ["e", "f"]},
        "fp32",
    )
    candidate = _records(
        {"1": ["a", "b"], "2": ["d", "c"], "3": ["x", "e"]},
        "int8",
    )

    stability = compare_ranking_stability(reference, candidate, cutoff=2)

    assert stability.query_count == 3
    assert stability.mean_overlap == 5 / 6
    assert stability.p05_overlap == 0.55
    assert stability.top1_agreement == 1 / 3


def test_quality_guard_uses_relevance_and_ranking_not_vector_distance() -> None:
    guard = load_query_runtime_quality_guard(ROOT / "config/protocol_amendments.toml")
    stability = compare_ranking_stability(
        _records({"1": ["a", "b"], "2": ["c", "d"]}, "fp32"),
        _records({"1": ["a", "b"], "2": ["c", "d"]}, "int8"),
        cutoff=2,
    )

    passed = assess_query_runtime_quality(
        fp32_ndcg_at_10=0.70,
        candidate_ndcg_at_10=0.695,
        stability=stability,
        guard=guard,
    )
    failed = assess_query_runtime_quality(
        fp32_ndcg_at_10=0.70,
        candidate_ndcg_at_10=0.68,
        stability=stability,
        guard=guard,
    )

    assert passed["status"] == "passed"
    assert failed["status"] == "failed"
    assert failed["checks"]["ndcg_at_10_loss"] is False
