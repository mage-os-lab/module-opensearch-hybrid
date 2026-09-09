from __future__ import annotations

from pathlib import Path

from poc.alternatives import (
    load_wands_alternatives_evidence_context,
    verify_wands_alternatives_significance,
)
from poc.experiments import load_bm25_experiments
from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    experiments = load_bm25_experiments(ROOT / "config/experiments.toml")
    url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_OS_URL",
    )
    with OpenSearchClient(url, timeout=300) as client:
        evidence_context = load_wands_alternatives_evidence_context(
            root=ROOT,
            client=client,
        )
        result = verify_wands_alternatives_significance(
            root=ROOT,
            evidence_context=evidence_context,
            alpha=experiments.alpha,
            minimum_delta=experiments.minimum_paired_delta,
        )
    print(
        "reproduced WANDS alternatives significance family of "
        f"{result['headline_family']['holm_family_size']}"
    )


if __name__ == "__main__":
    main()
