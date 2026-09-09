from __future__ import annotations

from pathlib import Path

from poc.config import load_model_registry
from poc.datasets import load_wands_config
from poc.neural_sparse import load_neural_sparse_spec
from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url
from poc.three_way import load_three_way_spec
from poc.three_way_indexing import verify_three_way_index

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    dataset = load_wands_config(ROOT / "config/datasets.toml")
    spec = load_three_way_spec(ROOT / "config/three_way.toml")
    sparse = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")
    model = load_model_registry(ROOT / "config/models.toml")[spec.dense_model]
    url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_OS_URL",
    )
    with OpenSearchClient(url, timeout=300) as client:
        manifest = verify_three_way_index(
            client,
            root=ROOT,
            dataset=dataset,
            spec=spec,
            sparse=sparse,
            model=model,
            products_path=ROOT / "data/prepared/wands/products.jsonl",
            sparse_embeddings_path=(
                ROOT / "data/cache/neural-sparse/wands-doc-v3-distill.jsonl"
            ),
            sparse_manifest_path=(
                ROOT / "results/wands/neural-sparse/precompute-manifest.json"
            ),
            manifest_path=ROOT / "results/wands/three-way/index-manifest.json",
        )
    index = manifest["index"]
    print(f"verified three-way WANDS index {index['uuid']}")


if __name__ == "__main__":
    main()
