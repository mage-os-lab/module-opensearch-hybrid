from __future__ import annotations

from pathlib import Path

from poc.experiments import load_bm25_experiments
from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url
from poc.trec_benchmark import verify_trec_bm25_summary
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
        verify_trec_bm25_summary(
            client,
            wands_client,
            root=ROOT,
            index=index.name,
            experiments=experiments,
        )
    print("verified 3 frozen TREC BM25 artifacts; tuned rerun byte-identical")


if __name__ == "__main__":
    main()
