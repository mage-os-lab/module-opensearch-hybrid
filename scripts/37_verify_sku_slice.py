from __future__ import annotations

from pathlib import Path

from poc.neural_sparse import load_neural_sparse_spec
from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url
from poc.sku_slice import load_sku_slice_spec
from poc.sku_slice_benchmark import verify_sku_slice_summary

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sparse = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")
    sku = load_sku_slice_spec(ROOT / "config/sku_slice.toml")
    url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_OS_URL",
    )
    with OpenSearchClient(url, timeout=300) as client:
        verify_sku_slice_summary(
            client,
            root=ROOT,
            sparse_spec=sparse,
            sku_spec=sku,
        )
    print("verified synthetic SKU artifacts; all three reruns byte-identical")


if __name__ == "__main__":
    main()
