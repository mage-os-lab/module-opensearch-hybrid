from __future__ import annotations

from pathlib import Path

from poc.experiments import load_bm25_experiments
from poc.neural_sparse import load_neural_sparse_spec
from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url
from poc.sparse_benchmark import verify_wands_sparse_summary

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    spec = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")
    experiments = load_bm25_experiments(ROOT / "config/experiments.toml")
    url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_OS_URL",
    )
    with OpenSearchClient(url, timeout=300) as client:
        summary = verify_wands_sparse_summary(
            client,
            root=ROOT,
            spec=spec,
            experiments=experiments,
        )
    print(
        "verified 8 neural-sparse artifacts; all dev and test reruns byte-identical; "
        f"quality_eligible={summary['quality_evidence_eligible_for_decision']}"
    )


if __name__ == "__main__":
    main()
