from __future__ import annotations

from pathlib import Path

from poc.experiments import load_bm25_experiments
from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url
from poc.trec_indexing import trec_opensearch_url
from poc.trec_sparse import load_trec_sparse_config
from poc.trec_sparse_benchmark import verify_trec_sparse_significance

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    experiments = load_bm25_experiments(ROOT / "config/experiments.toml")
    spec = load_trec_sparse_config(ROOT / "config/trec_neural_sparse.toml")
    wands_url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_WANDS_OS_URL",
    )
    with (
        OpenSearchClient(trec_opensearch_url(), timeout=300) as client,
        OpenSearchClient(wands_url, timeout=300) as wands_client,
    ):
        verify_trec_sparse_significance(
            client,
            wands_client,
            root=ROOT,
            spec=spec,
            metric=experiments.primary_metric,
            alpha=experiments.alpha,
            minimum_delta=experiments.minimum_paired_delta,
        )
    print("verified TREC sparse paired significance artifact")


if __name__ == "__main__":
    main()
