from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url
from poc.trec_indexing import trec_opensearch_url
from poc.trec_sparse import load_trec_sparse_config
from poc.trec_sparse_benchmark import verify_trec_sparse_summary

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    spec = load_trec_sparse_config(ROOT / "config/trec_neural_sparse.toml")
    wands_url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_WANDS_OS_URL",
    )
    with (
        OpenSearchClient(trec_opensearch_url(), timeout=300) as client,
        OpenSearchClient(wands_url, timeout=300) as wands_client,
    ):
        summary = verify_trec_sparse_summary(
            client,
            wands_client,
            root=ROOT,
            spec=spec,
        )
    metrics = cast(dict[str, Any], summary["hybrid_metrics"])
    print(
        "verified TREC sparse artifacts; sparse-only rerun byte-identical; "
        f"hybrid NDCG@10={metrics['ndcg@10']:.6f}"
    )


if __name__ == "__main__":
    main()
