from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from poc.datasets import load_trec_product_search_config
from poc.trec_sparse import load_trec_sparse_config
from poc.trec_sparse_validation import validate_recovered_trec_sparse

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = ROOT / "data/cache/models/neural-sparse/doc-v2-distill"


def main() -> None:
    dataset = load_trec_product_search_config(ROOT / "config/datasets.toml")
    sparse = load_trec_sparse_config(ROOT / "config/trec_neural_sparse.toml")
    result = validate_recovered_trec_sparse(
        root=ROOT,
        dataset_spec=dataset,
        sparse_spec=sparse,
        corpus_path=(
            ROOT / "data/raw/trec-product-search-2024/collection.trec.gz"
        ),
        embeddings_path=(
            ROOT
            / "data/cache/neural-sparse/trec-product-search-doc-v2-distill-full.jsonl"
        ),
        completion_manifest_path=(
            ROOT
            / "results/trec-product-search/neural-sparse/precompute-full-manifest.json"
        ),
        package_path=(
            ROOT / "data/cache/models/neural-sparse/doc-v2-distill.zip"
        ),
        model_path=(
            MODEL_DIRECTORY
            / "opensearch-neural-sparse-encoding-doc-v2-distill.pt"
        ),
        tokenizer_path=MODEL_DIRECTORY / "tokenizer.json",
        validation_manifest_path=(
            ROOT
            / "results/trec-product-search/neural-sparse/"
            "precompute-full-validation-manifest.json"
        ),
    )
    sample = cast(dict[str, Any], result["sample_reencoding"])
    embeddings = cast(dict[str, Any], result["embeddings"])
    evidence_status = (
        "quality eligible"
        if result["quality_evidence_eligible_for_decision"] is True
        else "diagnostic only"
    )
    print(
        "validated TREC sparse stream: "
        f"{embeddings['records']} records, "
        f"{sample['compared_records']} exact sample re-encodings, "
        f"{evidence_status}"
    )


if __name__ == "__main__":
    main()
