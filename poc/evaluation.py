from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from poc.trec import QrelRecord, RunRecord, read_qrels, read_run


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    ndcg_at_10: float
    ndcg_at_50: float
    recall_at_100: float
    mrr_exact_at_10: float
    judged_at_10: float


@dataclass(frozen=True, slots=True)
class Evaluation:
    metrics: dict[str, float]
    per_query: dict[str, dict[str, float]]


def evaluate_run(
    qrels_path: Path,
    run_path: Path,
    *,
    exact_gain: int = 2,
) -> Evaluation:
    return evaluate_records(
        read_qrels(qrels_path),
        read_run(run_path),
        exact_gain=exact_gain,
    )


def evaluate_records(
    qrels: list[QrelRecord],
    run: list[RunRecord],
    *,
    exact_gain: int = 2,
) -> Evaluation:
    qrels_by_query: dict[str, dict[str, int]] = defaultdict(dict)
    for qrel in qrels:
        qrels_by_query[qrel.query_id][qrel.document_id] = qrel.relevance
    run_by_query: dict[str, list[str]] = defaultdict(list)
    for run_record in sorted(run, key=lambda item: (item.query_id, item.rank)):
        run_by_query[run_record.query_id].append(run_record.document_id)

    per_query_metrics: dict[str, QueryMetrics] = {}
    for query_id, judgments in qrels_by_query.items():
        documents = run_by_query.get(query_id, [])
        per_query_metrics[query_id] = QueryMetrics(
            ndcg_at_10=_ndcg(documents, judgments, 10),
            ndcg_at_50=_ndcg(documents, judgments, 50),
            recall_at_100=_recall(documents, judgments, 100),
            mrr_exact_at_10=_reciprocal_rank(
                documents, judgments, 10, exact_gain=exact_gain
            ),
            judged_at_10=_judged(documents, judgments, 10),
        )
    if not per_query_metrics:
        raise ValueError("cannot evaluate empty qrels")

    per_query = {
        query_id: _metric_dict(metrics) for query_id, metrics in sorted(per_query_metrics.items())
    }
    names = tuple(next(iter(per_query.values())))
    macro = {
        name: sum(metrics[name] for metrics in per_query.values()) / len(per_query)
        for name in names
    }
    return Evaluation(metrics=macro, per_query=per_query)


def _metric_dict(metrics: QueryMetrics) -> dict[str, float]:
    return {
        "ndcg@10": metrics.ndcg_at_10,
        "ndcg@50": metrics.ndcg_at_50,
        "recall@100": metrics.recall_at_100,
        "mrr_exact@10": metrics.mrr_exact_at_10,
        "judged@10": metrics.judged_at_10,
    }


def _ndcg(documents: list[str], judgments: dict[str, int], cutoff: int) -> float:
    gains = [judgments.get(document_id, 0) for document_id in documents[:cutoff]]
    dcg = _discounted_gain(gains)
    ideal = _discounted_gain(sorted(judgments.values(), reverse=True)[:cutoff])
    return dcg / ideal if ideal else 0.0


def _discounted_gain(relevances: list[int]) -> float:
    return float(
        sum(
            relevance / math.log2(rank + 1)
            for rank, relevance in enumerate(relevances, start=1)
        )
    )


def _recall(documents: list[str], judgments: dict[str, int], cutoff: int) -> float:
    relevant = {document_id for document_id, gain in judgments.items() if gain > 0}
    if not relevant:
        return 0.0
    retrieved = set(documents[:cutoff])
    return len(relevant & retrieved) / len(relevant)


def _reciprocal_rank(
    documents: list[str],
    judgments: dict[str, int],
    cutoff: int,
    *,
    exact_gain: int,
) -> float:
    for rank, document_id in enumerate(documents[:cutoff], start=1):
        if judgments.get(document_id) == exact_gain:
            return 1.0 / rank
    return 0.0


def _judged(documents: list[str], judgments: dict[str, int], cutoff: int) -> float:
    judged = sum(document_id in judgments for document_id in documents[:cutoff])
    return judged / cutoff
