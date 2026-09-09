from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from poc.datasets import load_wands_config
from poc.neural_sparse import load_neural_sparse_spec
from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url
from poc.sku_slice import load_sku_slice_spec
from poc.sku_slice_indexing import build_sku_slice_index

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    dataset = load_wands_config(ROOT / "config/datasets.toml")
    sparse = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")
    sku = load_sku_slice_spec(ROOT / "config/sku_slice.toml")
    url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_OS_URL",
    )
    with OpenSearchClient(url, timeout=300) as client:
        manifest = build_sku_slice_index(
            client,
            root=ROOT,
            dataset_spec=dataset,
            sparse_spec=sparse,
            sku_spec=sku,
            products_path=ROOT / "data/prepared/wands/products.jsonl",
            embeddings_path=(
                ROOT / "data/cache/neural-sparse/wands-doc-v3-distill.jsonl"
            ),
            preparation_path=ROOT / "results/wands/sku-slice/preparation.json",
            precompute_manifest_path=(
                ROOT / "results/wands/neural-sparse/precompute-manifest.json"
            ),
            manifest_path=ROOT / "results/wands/sku-slice/index-manifest.json",
        )
    print(
        f"indexed synthetic SKU slice source: {dataset.expected_products} products at "
        f"{float(cast(Any, manifest['bulk_documents_per_second'])):.1f} docs/s"
    )


if __name__ == "__main__":
    main()
