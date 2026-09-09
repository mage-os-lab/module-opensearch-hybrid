from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from poc.datasets import load_trec_product_search_config
from poc.os_client import OpenSearchClient
from poc.trec_indexing import trec_opensearch_url
from poc.trec_sparse import load_trec_sparse_config
from poc.trec_sparse_indexing import verify_trec_sparse_index

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = ROOT / "data/cache/models/neural-sparse/doc-v2-distill"


def main() -> None:
    dataset = load_trec_product_search_config(ROOT / "config/datasets.toml")
    spec = load_trec_sparse_config(ROOT / "config/trec_neural_sparse.toml")
    with OpenSearchClient(trec_opensearch_url(), timeout=300) as client:
        manifest = verify_trec_sparse_index(
            client,
            root=ROOT,
            dataset_spec=dataset,
            sparse_spec=spec,
            corpus_path=(
                ROOT / "data/raw/trec-product-search-2024/collection.trec.gz"
            ),
            embeddings_path=(
                ROOT
                / "data/cache/neural-sparse/trec-product-search-doc-v2-distill-full.jsonl"
            ),
            precompute_manifest_path=(
                ROOT
                / "results/trec-product-search/neural-sparse/precompute-full-manifest.json"
            ),
            validation_manifest_path=(
                ROOT
                / "results/trec-product-search/neural-sparse/"
                "precompute-full-validation-manifest.json"
            ),
            package_path=(
                ROOT / "data/cache/models/neural-sparse/doc-v2-distill.zip"
            ),
            model_path=(
                MODEL_DIRECTORY
                / "opensearch-neural-sparse-encoding-doc-v2-distill.pt"
            ),
            tokenizer_path=MODEL_DIRECTORY / "tokenizer.json",
            manifest_path=(
                ROOT / "results/trec-product-search/neural-sparse/index-manifest.json"
            ),
        )
    index = cast(dict[str, Any], manifest["index"])
    print(
        f"verified TREC neural-sparse index {index['uuid']}: "
        f"{index['document_count']} products, one segment"
    )


if __name__ == "__main__":
    main()
