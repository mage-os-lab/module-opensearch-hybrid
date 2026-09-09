from __future__ import annotations

from pathlib import Path

from poc.config import WANDS_INT8_MODEL_NAMES, load_model_registry
from poc.datasets import load_wands_config
from poc.indexing import (
    load_wands_index_config,
    verify_wands_lexical_index,
)
from poc.int8_dense import verify_wands_int8_dense_summary
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
        verify_wands_int8_dense_summary(
            client,
            root=ROOT,
            index=index_spec.name,
            models=models,
            runtime=runtime,
        )
    print(
        f"verified {len(MODEL_NAMES) * 2} int8 dense artifacts; "
        "all registered arms reran byte-identically"
    )
if __name__ == "__main__":
    main()
