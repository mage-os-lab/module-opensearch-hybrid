from __future__ import annotations

from pathlib import Path

from poc.config import WANDS_INT8_MODEL_NAMES, load_model_registry
from poc.experiments import load_bm25_experiments
from poc.indexing import load_wands_index_config
from poc.neural_sparse import load_neural_sparse_spec
from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url
from poc.query_runtime import load_query_runtime_spec
from poc.robustness import verify_wands_robustness_report

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    index = load_wands_index_config(ROOT / "config/indexes.toml").name
    registry = load_model_registry(ROOT / "config/models.toml")
    models = tuple(registry[name] for name in WANDS_INT8_MODEL_NAMES)
    runtime = load_query_runtime_spec(ROOT / "config/query_runtime.toml")
    sparse = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")
    experiments = load_bm25_experiments(ROOT / "config/experiments.toml")
    url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_OS_URL",
    )
    with OpenSearchClient(url, timeout=300) as client:
        result = verify_wands_robustness_report(
            client,
            root=ROOT,
            index=index,
            models=models,
            runtime=runtime,
            sparse=sparse,
            experiments=experiments,
            metric=experiments.primary_metric,
            alpha=experiments.alpha,
            minimum_delta=experiments.minimum_paired_delta,
            minimum_judged_at_10=experiments.minimum_judged_at_10,
        )
    print(
        "verified WANDS mapping and pool-hole robustness: "
        f"holds={result['headline_quality_conclusion_holds_across_all_mappings_and_views']}"
    )


if __name__ == "__main__":
    main()
