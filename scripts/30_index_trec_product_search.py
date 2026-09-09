from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

from poc.datasets import load_trec_product_search_config
from poc.experiments import load_bm25_experiments
from poc.os_client import OpenSearchClient
from poc.trec_indexing import (
    build_trec_lexical_index,
    load_trec_index_config,
    trec_opensearch_url,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    dataset = load_trec_product_search_config(ROOT / "config/datasets.toml")
    index = load_trec_index_config(ROOT / "config/indexes.toml")
    experiments = load_bm25_experiments(ROOT / "config/experiments.toml")
    expected_version = os.environ.get("OPENSEARCH_HYBRID_EXPECTED_OS_VERSION", "2.19.6")
    with OpenSearchClient(trec_opensearch_url(), timeout=300) as client:
        manifest = build_trec_lexical_index(
            client,
            dataset_spec=dataset,
            index_spec=index,
            corpus_path=ROOT / "data/raw/trec-product-search-2024/collection.trec.gz",
            preparation_path=ROOT / "results/trec-product-search/preparation.json",
            manifest_path=ROOT / "results/trec-product-search/index-manifest.json",
            expected_version=expected_version,
            maximum_orphan_rate=experiments.maximum_trec_orphan_rate,
        )
    print(
        f"indexed TREC Product Search on {expected_version}: "
        f"{dataset.expected_products} products at "
        f"{float(cast(Any, manifest['bulk_documents_per_second'])):.1f} docs/s"
    )


if __name__ == "__main__":
    main()
