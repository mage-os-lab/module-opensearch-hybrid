from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from poc.alternatives import (
    build_wands_alternatives_significance,
    load_wands_alternatives_evidence_context,
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
        result = build_wands_alternatives_significance(
            root=ROOT,
            evidence_context=evidence_context,
            alpha=experiments.alpha,
            minimum_delta=experiments.minimum_paired_delta,
        )
    comparisons = cast(
        dict[str, Any], cast(dict[str, Any], result["headline_family"])["comparisons"]
    )
    print(
        "verified alternatives family: "
        + ", ".join(
            f"{name}={values['holm_adjusted_p_value']:.3g}" for name, values in comparisons.items()
        )
    )


if __name__ == "__main__":
    main()
