from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from poc.config import load_model_registry
from poc.experiments import load_bm25_experiments
from poc.neural_sparse import load_neural_sparse_spec
from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url
from poc.query_runtime import load_query_runtime_spec
from poc.three_way import load_three_way_spec
from poc.three_way_benchmark import run_wands_three_way_benchmark

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    spec = load_three_way_spec(ROOT / "config/three_way.toml")
    sparse = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")
    model = load_model_registry(ROOT / "config/models.toml")[spec.dense_model]
    runtime = load_query_runtime_spec(ROOT / "config/query_runtime.toml")
    experiments = load_bm25_experiments(ROOT / "config/experiments.toml")
    url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_OS_URL",
    )
    with OpenSearchClient(url, timeout=300) as client:
        summary = run_wands_three_way_benchmark(
            client,
            root=ROOT,
            spec=spec,
            sparse=sparse,
            model=model,
            runtime=runtime,
            experiments=experiments,
        )
    metrics = cast(dict[str, Any], summary["held_out_test_metrics"])
    print(
        f"selected three-way {summary['selected_candidate']}; "
        f"held-out NDCG@10={metrics['ndcg@10']:.6f}"
    )


if __name__ == "__main__":
    main()
