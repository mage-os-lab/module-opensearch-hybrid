from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from poc.evaluation_crosscheck import cross_check_run
from poc.manifest import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    selection = cast(dict[str, Any], read_json(ROOT / "results/wands/bm25-selection.json"))
    artifacts = cast(dict[str, dict[str, Any]], selection["artifacts"])
    checks: dict[str, object] = {}
    maximum_delta = 0.0
    canonical_runs = (
        "naive/dev",
        "naive/test",
        "magento/dev",
        "magento/test",
        "tuned/dev",
        "tuned/test",
    )
    for name in canonical_runs:
        entry = artifacts[name]
        manifest = cast(dict[str, Any], read_json(ROOT / str(entry["manifest"])))
        qrels_file = cast(dict[str, str], manifest["qrels_file"])
        check = cross_check_run(
            ROOT / qrels_file["path"],
            ROOT / str(entry["run"]),
            root=ROOT,
        )
        checks[name] = check.to_dict()
        maximum_delta = max(maximum_delta, check.maximum_absolute_delta)

    output = {
        "schema_version": 1,
        "dataset": "WANDS",
        "registered_metrics": ["ndcg@10", "ndcg@50", "recall@100"],
        "external_evaluator_ranking_source": "trec_rank_column_via_monotonic_scores",
        "checks": checks,
        "maximum_absolute_delta": maximum_delta,
        "tolerance": 0.001,
    }
    write_json(ROOT / "results/wands/bm25-evaluator-crosscheck.json", output)
    print(f"cross-checked {len(checks)} BM25 runs: maximum evaluator delta={maximum_delta:.9f}")


if __name__ == "__main__":
    main()
