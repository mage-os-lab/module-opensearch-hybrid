from __future__ import annotations

from pathlib import Path

from poc.experiments import load_bm25_experiments
from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url
from poc.trec_indexing import trec_opensearch_url
from poc.trec_sparse import load_trec_sparse_config
from poc.trec_sparse_benchmark import run_trec_sparse_benchmark

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    spec = load_trec_sparse_config(ROOT / "config/trec_neural_sparse.toml")
    experiments = load_bm25_experiments(ROOT / "config/experiments.toml")
    wands_url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_WANDS_OS_URL",
    )
    with (
        OpenSearchClient(trec_opensearch_url(), timeout=300) as client,
        OpenSearchClient(wands_url, timeout=300) as wands_client,
    ):
        summary = run_trec_sparse_benchmark(
            client,
            wands_client,
            root=ROOT,
            spec=spec,
            experiments=experiments,
        )
    sparse = summary["sparse_only_metrics"]
    hybrid = summary["hybrid_metrics"]
    print(f"TREC sparse completed: sparse={sparse}; hybrid={hybrid}")


if __name__ == "__main__":
    main()
