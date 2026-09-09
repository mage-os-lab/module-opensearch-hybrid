from __future__ import annotations

import importlib
import importlib.metadata
import os
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, cast

from poc.datasets import DatasetIntegrityError
from poc.evaluation import evaluate_run
from poc.trec import RunRecord, read_qrels, read_run

CROSSCHECK_METRICS = ("ndcg@10", "ndcg@50", "recall@100")
DEFAULT_TOLERANCE = 0.001


@dataclass(frozen=True, slots=True)
class EvaluatorCrossCheck:
    engines: dict[str, dict[str, float]]
    versions: dict[str, str]
    maximum_absolute_delta: float
    tolerance: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def configure_evaluator_caches(root: Path) -> None:
    cache_root = root / "data/cache/evaluation"
    values = {
        "IR_DATASETS_HOME": cache_root / "ir-datasets",
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "NUMBA_CACHE_DIR": cache_root / "numba",
        "XDG_CACHE_HOME": cache_root / "xdg",
    }
    for name, path in values.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(path))


def cross_check_run(
    qrels_path: Path,
    run_path: Path,
    *,
    root: Path,
    tolerance: float = DEFAULT_TOLERANCE,
) -> EvaluatorCrossCheck:
    if tolerance < 0:
        raise ValueError("cross-check tolerance must be non-negative")
    configure_evaluator_caches(root)

    qrels = read_qrels(qrels_path)
    run = read_run(run_path)
    qrels_values: dict[str, dict[str, int]] = {}
    for qrel in qrels:
        qrels_values.setdefault(qrel.query_id, {})[qrel.document_id] = qrel.relevance
    run_values = run_values_for_evaluators(run)

    native = evaluate_run(qrels_path, run_path).metrics
    engines = {
        "native": {metric: native[metric] for metric in CROSSCHECK_METRICS},
        "ranx": _evaluate_with_ranx(qrels_values, run_values),
        "pytrec_eval": _evaluate_with_pytrec_eval(qrels_values, run_values),
    }
    maximum_delta = maximum_pairwise_delta(engines)
    if maximum_delta > tolerance:
        raise DatasetIntegrityError(
            f"evaluator disagreement for {run_path}: {maximum_delta:.9f} > {tolerance:.9f}"
        )
    return EvaluatorCrossCheck(
        engines=engines,
        versions={
            "ranx": importlib.metadata.version("ranx"),
            "pytrec_eval": importlib.metadata.version("pytrec-eval-terrier"),
        },
        maximum_absolute_delta=maximum_delta,
        tolerance=tolerance,
    )


def maximum_pairwise_delta(engines: dict[str, dict[str, float]]) -> float:
    if len(engines) < 2:
        raise ValueError("at least two evaluators are required")
    for name, metrics in engines.items():
        if set(metrics) != set(CROSSCHECK_METRICS):
            raise ValueError(f"{name} did not return the registered cross-check metrics")
    return max(
        abs(left[metric] - right[metric])
        for left, right in combinations(engines.values(), 2)
        for metric in CROSSCHECK_METRICS
    )


def run_values_for_evaluators(run: list[RunRecord]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for record in run:
        # ranx and pytrec_eval rank by score and ignore the TREC rank column. Rank-derived
        # scores ensure both evaluate the exact recorded order even when retrieval scores tie.
        values.setdefault(record.query_id, {})[record.document_id] = -float(record.rank)
    return values


def _evaluate_with_ranx(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
) -> dict[str, float]:
    ranx = importlib.import_module("ranx")
    qrels_object = ranx.Qrels(qrels)
    run_object = ranx.Run(run)
    result = cast(
        dict[str, Any],
        ranx.evaluate(
            qrels_object,
            run_object,
            list(CROSSCHECK_METRICS),
            threads=1,
            make_comparable=True,
        ),
    )
    return {metric: float(result[metric]) for metric in CROSSCHECK_METRICS}


def _evaluate_with_pytrec_eval(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
) -> dict[str, float]:
    pytrec_eval = importlib.import_module("pytrec_eval")
    measures = {"ndcg_cut.10", "ndcg_cut.50", "recall.100"}
    comparable_run = {query_id: run.get(query_id, {}) for query_id in qrels}
    by_query = cast(
        dict[str, dict[str, float]],
        pytrec_eval.RelevanceEvaluator(qrels, measures).evaluate(comparable_run),
    )
    if set(by_query) != set(qrels):
        raise DatasetIntegrityError("pytrec_eval did not return every qrels query")
    keys = {
        "ndcg@10": "ndcg_cut_10",
        "ndcg@50": "ndcg_cut_50",
        "recall@100": "recall_100",
    }
    return {
        metric: sum(values[key] for values in by_query.values()) / len(by_query)
        for metric, key in keys.items()
    }
