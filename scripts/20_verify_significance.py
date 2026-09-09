from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from poc.config import WANDS_INT8_MODEL_NAMES, load_model_registry
from poc.experiments import load_bm25_experiments
from poc.indexing import load_wands_index_config
from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url
from poc.query_runtime import load_query_runtime_spec
from poc.significance import verify_wands_hybrid_significance

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    index = load_wands_index_config(ROOT / "config/indexes.toml").name
    registry = load_model_registry(ROOT / "config/models.toml")
    models = tuple(registry[name] for name in WANDS_INT8_MODEL_NAMES)
    runtime = load_query_runtime_spec(ROOT / "config/query_runtime.toml")
    experiments = load_bm25_experiments(ROOT / "config/experiments.toml")
    url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_OS_URL",
    )
    with OpenSearchClient(url, timeout=300) as client:
        result = verify_wands_hybrid_significance(
            client,
            root=ROOT,
            index=index,
            models=models,
            runtime=runtime,
            experiments=experiments,
            metric=experiments.primary_metric,
            alpha=experiments.alpha,
            minimum_delta=experiments.minimum_paired_delta,
        )
    comparisons = cast(dict[str, dict[str, Any]], result["comparisons"])
    for name, comparison in comparisons.items():
        print(
            f"verified paired significance for {name}: "
            f"Holm p={float(comparison['holm_adjusted_p_value']):.9f}"
        )
    challenge = cast(dict[str, Any], result["post_review_challenge"])
    challenge_comparisons = cast(dict[str, dict[str, Any]], challenge["comparisons"])
    for name, comparison in challenge_comparisons.items():
        print(
            f"verified challenge-baseline significance for {name}: "
            f"Holm p={float(comparison['holm_adjusted_p_value']):.9f}"
        )


if __name__ == "__main__":
    main()
