from __future__ import annotations

import importlib
import math
from types import SimpleNamespace
from typing import Any

import pytest

from poc import evaluation_crosscheck
from poc.evaluation import evaluate_records
from poc.evaluation_crosscheck import maximum_pairwise_delta, run_values_for_evaluators
from poc.trec import QrelRecord, RunRecord


def test_product_search_metrics_use_registered_gain_and_relevance_rules() -> None:
    qrels = [
        QrelRecord("q1", "exact", 2),
        QrelRecord("q1", "partial", 1),
        QrelRecord("q1", "irrelevant", 0),
    ]
    run = [
        RunRecord("q1", "partial", 1, 3.0, "test"),
        RunRecord("q1", "exact", 2, 2.0, "test"),
        RunRecord("q1", "unjudged", 3, 1.0, "test"),
    ]
    evaluation = evaluate_records(qrels, run)
    # The pre-registered metric is Järvelin NDCG, whose gain is the graded
    # relevance value itself. Burges NDCG would use 2**relevance - 1 here.
    expected_ndcg = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))

    assert math.isclose(evaluation.metrics["ndcg@10"], expected_ndcg)
    assert math.isclose(evaluation.metrics["ndcg@50"], expected_ndcg)
    assert evaluation.metrics["recall@100"] == 1.0
    assert evaluation.metrics["mrr_exact@10"] == 0.5
    assert evaluation.metrics["judged@10"] == 0.2


def test_exact_match_gain_can_follow_dataset_scale() -> None:
    qrels = [QrelRecord("q1", "trec-exact", 3), QrelRecord("q1", "other", 2)]
    run = [
        RunRecord("q1", "other", 1, 2.0, "test"),
        RunRecord("q1", "trec-exact", 2, 1.0, "test"),
    ]

    evaluation = evaluate_records(qrels, run, exact_gain=3)

    assert evaluation.metrics["mrr_exact@10"] == 0.5


def test_evaluator_crosscheck_uses_worst_metric_across_every_pair() -> None:
    engines = {
        "native": {"ndcg@10": 0.5, "ndcg@50": 0.6, "recall@100": 0.7},
        "ranx": {"ndcg@10": 0.5001, "ndcg@50": 0.6002, "recall@100": 0.7003},
        "pytrec_eval": {"ndcg@10": 0.5002, "ndcg@50": 0.6004, "recall@100": 0.7006},
    }

    assert math.isclose(maximum_pairwise_delta(engines), 0.0006)


def test_ranx_crosscheck_aligns_zero_hit_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    def evaluate(*args: Any, make_comparable: bool = False, **kwargs: Any) -> dict[str, float]:
        assert make_comparable
        return {"ndcg@10": 0.5, "ndcg@50": 0.6, "recall@100": 0.7}

    fake_ranx = SimpleNamespace(
        Qrels=lambda value: value,
        Run=lambda value: value,
        evaluate=evaluate,
    )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_ranx,
    )

    metrics = evaluation_crosscheck._evaluate_with_ranx(
        {"q1": {"d1": 2}, "zero-hit": {"d2": 1}},
        {"q1": {"d1": 1.0}},
    )

    assert metrics["ndcg@10"] == 0.5


def test_pytrec_crosscheck_aligns_zero_hit_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEvaluator:
        def evaluate(
            self,
            run: dict[str, dict[str, float]],
        ) -> dict[str, dict[str, float]]:
            assert set(run) == {"q1", "zero-hit"}
            assert run["zero-hit"] == {}
            return {
                query_id: {"ndcg_cut_10": 0.5, "ndcg_cut_50": 0.6, "recall_100": 0.7}
                for query_id in run
            }

    fake_pytrec = SimpleNamespace(RelevanceEvaluator=lambda *args, **kwargs: FakeEvaluator())
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_pytrec,
    )

    metrics = evaluation_crosscheck._evaluate_with_pytrec_eval(
        {"q1": {"d1": 2}, "zero-hit": {"d2": 1}},
        {"q1": {"d1": 1.0}},
    )

    assert metrics["recall@100"] == 0.7


def test_external_evaluators_receive_trec_rank_order_when_raw_scores_tie() -> None:
    run = [
        RunRecord("q1", "z-first", 1, 1.0, "test"),
        RunRecord("q1", "a-second", 2, 1.0, "test"),
    ]

    values = run_values_for_evaluators(run)

    assert values["q1"]["z-first"] > values["q1"]["a-second"]
