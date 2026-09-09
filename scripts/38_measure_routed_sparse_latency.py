from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from poc.datasets import load_wands_config
from poc.end_to_end_latency import (
    load_latency_protocol,
    run_routed_sparse_latency,
    write_latency_result,
)
from poc.experiments import BM25Profile
from poc.manifest import read_json
from poc.neural_sparse import load_neural_sparse_spec
from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url
from poc.sku_slice import load_sku_slice_spec

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    protocol = load_latency_protocol(ROOT / "config/experiments.toml")
    dataset = load_wands_config(ROOT / "config/datasets.toml")
    sparse = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")
    sku = load_sku_slice_spec(ROOT / "config/sku_slice.toml")
    selection = cast(
        dict[str, Any], read_json(ROOT / "results/wands/bm25-selection.json")
    )
    profile = BM25Profile.from_mapping(
        cast(dict[str, Any], selection["selected_profile"])
    )
    url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_OS_URL",
    )
    with OpenSearchClient(url, timeout=300) as client:
        client.wait_until_ready(expected_version="3.8.0")
        result = run_routed_sparse_latency(
            client,
            root=ROOT,
            index=sku.index_name,
            dataset_spec=dataset,
            profile=profile,
            sparse_spec=sparse,
            sku_spec=sku,
            protocol=protocol,
        )
    output = write_latency_result(ROOT, result)
    target = cast(
        dict[str, Any],
        cast(dict[str, Any], result["measurements"])[
            str(protocol.target_concurrency)
        ],
    )
    print(
        f"routed sparse concurrency-4 p95={target['p95_ms']:.3f} ms; "
        f"decision_eligible={result['latency_decision_eligible']}; output={output}"
    )


if __name__ == "__main__":
    main()
