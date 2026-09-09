from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from poc.experiments import BM25Profile
from poc.facet_semantics import run_facet_semantics
from poc.manifest import read_json
from poc.neural_sparse import load_neural_sparse_spec
from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url
from poc.sku_slice import load_sku_slice_spec

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
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
        result = run_facet_semantics(
            client,
            root=ROOT,
            index=sku.index_name,
            profile=profile,
            sparse=sparse,
        )
    examples = cast(list[dict[str, Any]], result["examples"])
    print(
        f"facet semantics: recorded {len(examples)} diagnostic examples; "
        "relevance and latency eligibility=false"
    )


if __name__ == "__main__":
    main()
