from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from poc.datasets import (
    DatasetIntegrityError,
    load_trec_product_search_config,
)
from poc.index_evidence import INDEX_WRITE_BLOCK_EVIDENCE
from poc.trec_indexing import verify_trec_lexical_manifest_metadata
from poc.trec_sparse import load_trec_sparse_config
from poc.trec_sparse_indexing import verify_trec_sparse_manifest_metadata

ROOT = Path(__file__).resolve().parents[1]


def test_trec_lexical_manifest_metadata_rejects_identity_changes() -> None:
    dataset = load_trec_product_search_config(ROOT / "config/datasets.toml")
    manifest: dict[str, object] = {
        "schema_version": 2,
        "dataset": "TREC Product Search 2024",
        "source_revision": dataset.revision,
        "opensearch_version": "2.19.6",
        "source_corpus": {"records": dataset.expected_products},
        "refresh_interval": "1s",
        "vector_state": "fields_not_added",
        "index_write_block": INDEX_WRITE_BLOCK_EVIDENCE,
    }
    verify_trec_lexical_manifest_metadata(
        manifest,
        dataset_spec=dataset,
        expected_version="2.19.6",
    )

    for key, replacement in (
        ("schema_version", 1),
        ("dataset", "other"),
        ("source_revision", "0" * 40),
        ("opensearch_version", "3.8.0"),
        ("refresh_interval", "-1"),
        ("vector_state", "loaded"),
        ("index_write_block", {"setting": "index.blocks.write", "value": False}),
    ):
        changed = deepcopy(manifest)
        changed[key] = replacement
        with pytest.raises(DatasetIntegrityError, match="metadata"):
            verify_trec_lexical_manifest_metadata(
                changed,
                dataset_spec=dataset,
                expected_version="2.19.6",
            )

    changed = deepcopy(manifest)
    changed["source_corpus"] = {"records": dataset.expected_products - 1}
    with pytest.raises(DatasetIntegrityError, match="metadata"):
        verify_trec_lexical_manifest_metadata(
            changed,
            dataset_spec=dataset,
            expected_version="2.19.6",
        )


def test_trec_sparse_manifest_metadata_rejects_model_or_identity_changes() -> None:
    dataset = load_trec_product_search_config(ROOT / "config/datasets.toml")
    sparse = load_trec_sparse_config(ROOT / "config/trec_neural_sparse.toml")
    manifest: dict[str, object] = {
        "schema_version": 2,
        "dataset": "TREC Product Search 2024",
        "source_revision": dataset.revision,
        "opensearch_version": sparse.expected_opensearch_version,
        "source_corpus": {"records": dataset.expected_products},
        "document_model": {
            "name": sparse.document_model_name,
            "version": sparse.document_model_version,
            "format": sparse.document_model_format,
            "package_sha256": sparse.document_model_sha256,
            "package_bytes": sparse.document_model_bytes,
            "maximum_token_length": sparse.maximum_token_length,
            "maximum_value_ratio": sparse.maximum_value_ratio,
        },
        "query_tokenizer": {
            "name": sparse.query_tokenizer_name,
            "version": sparse.query_tokenizer_version,
            "format": sparse.query_tokenizer_format,
            "content_sha256": sparse.query_tokenizer_content_sha256,
            "content_bytes": sparse.query_tokenizer_content_bytes,
        },
        "bulk_request_size": sparse.bulk_request_size,
        "refresh_interval": "1s",
        "vector_state": "document_only_neural_sparse_rank_features",
        "index_write_block": INDEX_WRITE_BLOCK_EVIDENCE,
    }
    verify_trec_sparse_manifest_metadata(
        manifest,
        dataset_spec=dataset,
        sparse_spec=sparse,
    )

    for path, replacement in (
        (("schema_version",), 1),
        (("dataset",), "other"),
        (("source_revision",), "0" * 40),
        (("opensearch_version",), "3.8.0"),
        (("document_model", "package_sha256"), "0" * 64),
        (("query_tokenizer", "content_sha256"), "0" * 64),
        (("bulk_request_size",), sparse.bulk_request_size + 1),
        (("refresh_interval",), "-1"),
        (("vector_state",), "other"),
        (("index_write_block", "value"), False),
    ):
        changed = deepcopy(manifest)
        target = changed
        for key in path[:-1]:
            target = cast(dict[str, object], target[key])
        target[path[-1]] = replacement
        with pytest.raises(DatasetIntegrityError, match="metadata"):
            verify_trec_sparse_manifest_metadata(
                changed,
                dataset_spec=dataset,
                sparse_spec=sparse,
            )
