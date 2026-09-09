from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from poc.neural_sparse import load_neural_sparse_spec
from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url
from poc.sku_slice import load_sku_slice_spec
from poc.sku_slice_benchmark import run_sku_slice_benchmark

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sparse = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")
    sku = load_sku_slice_spec(ROOT / "config/sku_slice.toml")
    url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_OS_URL",
    )
    with OpenSearchClient(url, timeout=300) as client:
        summary = run_sku_slice_benchmark(
            client,
            root=ROOT,
            sparse_spec=sparse,
            sku_spec=sku,
        )
    overall = cast(
        dict[str, Any],
        cast(dict[str, Any], summary["no_regression_assessment"])["overall"],
    )
    print(
        f"synthetic SKU gate={summary['passes_registered_synthetic_gate']}; "
        f"delta={overall['paired_mean_delta']:.6f}; "
        f"W/T/L={overall['wins']}/{overall['ties']}/{overall['losses']}"
    )


if __name__ == "__main__":
    main()
