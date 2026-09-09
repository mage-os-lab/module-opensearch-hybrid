from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from poc.config import WANDS_INT8_MODEL_NAMES, load_model_registry
from poc.datasets import load_wands_config
from poc.indexing import load_wands_index_config, verify_wands_lexical_index
from poc.int8_dense import run_wands_int8_dense_benchmark
from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url
from poc.query_runtime import load_query_runtime_spec

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = WANDS_INT8_MODEL_NAMES


def main() -> None:
    dataset_spec = load_wands_config(ROOT / "config/datasets.toml")
    index_spec = load_wands_index_config(ROOT / "config/indexes.toml")
    registry = load_model_registry(ROOT / "config/models.toml")
    runtime = load_query_runtime_spec(ROOT / "config/query_runtime.toml")
    models = tuple(registry[name] for name in MODEL_NAMES)
    url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_OS_URL",
    )
    with OpenSearchClient(url, timeout=300) as client:
        verify_wands_lexical_index(
            client,
            dataset_spec=dataset_spec,
            index_spec=index_spec,
            prepared_directory=ROOT / "data/prepared/wands",
            manifest_path=ROOT / "results/wands/index-manifest.json",
        )
        summary = run_wands_int8_dense_benchmark(
            client,
            root=ROOT,
            index=index_spec.name,
            models=models,
            runtime=runtime,
        )
    results = cast(dict[str, dict[str, Any]], summary["models"])
    for name in MODEL_NAMES:
        test = cast(dict[str, Any], results[name]["splits"])["test"]
        metrics = cast(dict[str, float], test["metrics"])
        print(
            f"{name} int8 exact dense: NDCG@10={metrics['ndcg@10']:.6f}, "
            f"delta_vs_fp32={float(test['int8_minus_fp32_ndcg@10']):+.6f}, "
            f"eligible={results[name]['eligible_for_decision']}"
        )


if __name__ == "__main__":
    main()
