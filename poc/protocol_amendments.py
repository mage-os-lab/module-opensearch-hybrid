from __future__ import annotations

import tomllib
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from statistics import fmean
from typing import Any, cast

from poc.config import ConfigError
from poc.datasets import DatasetIntegrityError
from poc.trec import RunRecord


@dataclass(frozen=True, slots=True)
class QueryRuntimeQualityGuard:
    amendment_id: str
    effective_date: str
    classification: str
    supersedes_gate: str
    rationale: str
    maximum_ndcg_at_10_loss: float
    minimum_mean_top10_overlap: float
    minimum_p05_top10_overlap: float
    minimum_top1_agreement: float


@dataclass(frozen=True, slots=True)
class RankingStability:
    query_count: int
    cutoff: int
    mean_overlap: float
    p05_overlap: float
    top1_agreement: float


def load_query_runtime_quality_guard(path: Path) -> QueryRuntimeQualityGuard:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if raw.get("schema_version") != 1:
        raise ConfigError("unsupported protocol amendment schema")
    values = raw.get("query_runtime_quality_guard")
    if not isinstance(values, dict):
        raise ConfigError("query-runtime quality guard is missing")

    required_strings = (
        "amendment_id",
        "effective_date",
        "classification",
        "supersedes_gate",
        "rationale",
    )
    for key in required_strings:
        if not isinstance(values.get(key), str) or not values[key]:
            raise ConfigError(f"query-runtime quality guard {key} must be a string")
    if values["classification"] != "post_measurement_protocol_amendment":
        raise ConfigError("query-runtime quality guard must disclose post-measurement status")
    try:
        date.fromisoformat(cast(str, values["effective_date"]))
    except ValueError as exc:
        raise ConfigError("query-runtime quality guard date is invalid") from exc

    thresholds = (
        "maximum_ndcg_at_10_loss",
        "minimum_mean_top10_overlap",
        "minimum_p05_top10_overlap",
        "minimum_top1_agreement",
    )
    for key in thresholds:
        value = values.get(key)
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not 0 <= float(value) <= 1
        ):
            raise ConfigError(f"query-runtime quality guard {key} must be in [0, 1]")
    return QueryRuntimeQualityGuard(
        amendment_id=cast(str, values["amendment_id"]),
        effective_date=cast(str, values["effective_date"]),
        classification=cast(str, values["classification"]),
        supersedes_gate=cast(str, values["supersedes_gate"]),
        rationale=cast(str, values["rationale"]),
        maximum_ndcg_at_10_loss=float(values["maximum_ndcg_at_10_loss"]),
        minimum_mean_top10_overlap=float(values["minimum_mean_top10_overlap"]),
        minimum_p05_top10_overlap=float(values["minimum_p05_top10_overlap"]),
        minimum_top1_agreement=float(values["minimum_top1_agreement"]),
    )


def compare_ranking_stability(
    reference: list[RunRecord],
    candidate: list[RunRecord],
    *,
    cutoff: int = 10,
) -> RankingStability:
    if cutoff <= 0:
        raise ValueError("ranking-stability cutoff must be positive")
    reference_rankings = _rankings(reference, cutoff=cutoff)
    candidate_rankings = _rankings(candidate, cutoff=cutoff)
    if not reference_rankings or reference_rankings.keys() != candidate_rankings.keys():
        raise DatasetIntegrityError("ranking-stability runs contain different query sets")

    overlaps: list[float] = []
    top1_hits = 0
    for query_id in sorted(reference_rankings):
        left = reference_rankings[query_id]
        right = candidate_rankings[query_id]
        if len(left) < cutoff or len(right) < cutoff:
            raise DatasetIntegrityError(
                f"ranking-stability query {query_id} has fewer than {cutoff} results"
            )
        overlaps.append(len(set(left) & set(right)) / cutoff)
        top1_hits += left[0] == right[0]
    return RankingStability(
        query_count=len(overlaps),
        cutoff=cutoff,
        mean_overlap=fmean(overlaps),
        p05_overlap=_percentile(overlaps, 0.05),
        top1_agreement=top1_hits / len(overlaps),
    )


def assess_query_runtime_quality(
    *,
    fp32_ndcg_at_10: float,
    candidate_ndcg_at_10: float,
    stability: RankingStability,
    guard: QueryRuntimeQualityGuard,
) -> dict[str, Any]:
    ndcg_loss = fp32_ndcg_at_10 - candidate_ndcg_at_10
    checks = {
        "ndcg_at_10_loss": ndcg_loss <= guard.maximum_ndcg_at_10_loss,
        "mean_top10_overlap": stability.mean_overlap >= guard.minimum_mean_top10_overlap,
        "p05_top10_overlap": stability.p05_overlap >= guard.minimum_p05_top10_overlap,
        "top1_agreement": stability.top1_agreement >= guard.minimum_top1_agreement,
    }
    return {
        "schema_version": 1,
        "amendment": asdict(guard),
        "fp32_ndcg@10": fp32_ndcg_at_10,
        "candidate_ndcg@10": candidate_ndcg_at_10,
        "ndcg@10_loss": ndcg_loss,
        "ranking_stability": asdict(stability),
        "checks": checks,
        "status": "passed" if all(checks.values()) else "failed",
    }


def _rankings(records: list[RunRecord], *, cutoff: int) -> dict[str, list[str]]:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        if record.rank <= cutoff:
            grouped[record.query_id].append(record)
    rankings: dict[str, list[str]] = {}
    for query_id, values in grouped.items():
        ordered = sorted(values, key=lambda value: value.rank)
        if [value.rank for value in ordered] != list(range(1, len(ordered) + 1)):
            raise DatasetIntegrityError(
                f"ranking-stability query {query_id} has non-contiguous ranks"
            )
        documents = [value.document_id for value in ordered]
        if len(set(documents)) != len(documents):
            raise DatasetIntegrityError(
                f"ranking-stability query {query_id} repeats a document"
            )
        rankings[query_id] = documents
    return rankings


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
