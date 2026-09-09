from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from poc.experiments import load_bm25_experiments
from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url
from poc.trec_benchmark import run_trec_bm25_benchmark
from poc.trec_indexing import load_trec_index_config, trec_opensearch_url

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    experiments = load_bm25_experiments(ROOT / "config/experiments.toml")
    index = load_trec_index_config(ROOT / "config/indexes.toml")
    wands_url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_WANDS_OS_URL",
    )
    with (
        OpenSearchClient(trec_opensearch_url(), timeout=300) as client,
        OpenSearchClient(wands_url, timeout=300) as wands_client,
    ):
        summary = run_trec_bm25_benchmark(
            client,
            wands_client,
            root=ROOT,
            index=index.name,
            experiments=experiments,
        )
    metrics = cast(dict[str, dict[str, Any]], summary["held_out_metrics"])
    print(
        "TREC frozen BM25 NDCG@10: "
        f"tuned={float(metrics['tuned']['ndcg@10']):.6f}, "
        f"competent={float(metrics['competent_integrator']['ndcg@10']):.6f}"
    )


if __name__ == "__main__":
    main()
