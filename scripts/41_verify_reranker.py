from __future__ import annotations

from pathlib import Path

from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url
from poc.reranker import (
    load_reranker_spec,
    load_wands_reranker_evidence_context,
    verify_wands_reranker_benchmark,
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
        summary = verify_wands_reranker_benchmark(
            root=ROOT,
            spec=spec,
            evidence_context=evidence_context,
        )
    print(
        "verified reranker benchmark: "
        f"dense delta={summary['dense_reranker_minus_dense_minmax_ndcg@10']:+.6f}"
    )


if __name__ == "__main__":
    main()
