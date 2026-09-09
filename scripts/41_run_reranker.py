from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url
from poc.reranker import (
    load_reranker_spec,
    load_wands_reranker_evidence_context,
    run_wands_reranker_benchmark,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    spec = load_reranker_spec(ROOT / "config/reranker.toml")
    url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_OS_URL",
    )
    with OpenSearchClient(url, timeout=300) as client:
        evidence_context = load_wands_reranker_evidence_context(
            root=ROOT,
            client=client,
        )
        summary = run_wands_reranker_benchmark(
            root=ROOT,
            spec=spec,
            evidence_context=evidence_context,
        )
    metrics = cast(dict[str, dict[str, dict[str, Any]]], summary["reranked_metrics"])
    print(
        "reranker held-out NDCG@10: "
        f"BM25={metrics['tuned_bm25']['test']['ndcg@10']:.6f}; "
        f"dense={metrics['dense_minmax']['test']['ndcg@10']:.6f}"
    )


if __name__ == "__main__":
    main()
