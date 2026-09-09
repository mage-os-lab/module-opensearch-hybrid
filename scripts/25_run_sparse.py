from __future__ import annotations

from pathlib import Path
from typing import cast

from poc.experiments import load_bm25_experiments
from poc.neural_sparse import load_neural_sparse_spec
from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url
from poc.sparse_benchmark import run_wands_sparse_benchmark

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    spec = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")
    experiments = load_bm25_experiments(ROOT / "config/experiments.toml")
    url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_OS_URL",
    )
    with OpenSearchClient(url, timeout=120) as client:
        summary = run_wands_sparse_benchmark(
            client,
            root=ROOT,
            spec=spec,
            experiments=experiments,
        )
    sparse = cast(dict[str, float], summary["sparse_only_test_metrics"])
    hybrid = cast(dict[str, float], summary["hybrid_test_metrics"])
    print(
        f"doc-only sparse held-out NDCG@10={sparse['ndcg@10']:.6f}; "
        f"BM25+sparse={hybrid['ndcg@10']:.6f}; "
        f"delta_vs_tuned_bm25={summary['sparse_hybrid_minus_tuned_bm25_ndcg@10']:+.6f}; "
        f"quality_eligible={summary['quality_evidence_eligible_for_decision']}"
    )


if __name__ == "__main__":
    main()
