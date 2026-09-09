from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from poc.config import load_model_registry
from poc.indexing import (
    iter_prepared_products,
    iter_products_with_vectors,
    load_wands_index_config,
    wands_index_definition,
)

ROOT = Path(__file__).resolve().parents[1]


def test_wands_index_is_vector_ready_and_uses_registered_text_analysis() -> None:
    spec = load_wands_index_config(ROOT / "config/indexes.toml")
    definition = wands_index_definition(spec)
    index_settings = definition["settings"]["index"]
    properties = definition["mappings"]["properties"]
    assert index_settings["knn"] is True
    assert index_settings["refresh_interval"] == "-1"
    assert properties["title"]["analyzer"] == "product_text"
    assert properties["title"]["fields"]["magento"] == {
        "type": "text",
        "analyzer": "magento_default",
    }
    assert properties["title"]["fields"]["prefix"] == {
        "type": "text",
        "analyzer": "prefix_search",
    }
    analysis = definition["settings"]["analysis"]
    assert analysis["analyzer"]["magento_default"]["filter"] == [
        "lowercase",
        "keyword_repeat",
        "asciifolding",
        "default_stemmer",
        "unique_stem",
    ]
    assert analysis["filter"]["default_stemmer"] == {
        "type": "stemmer",
        "language": "english",
    }
    assert properties["product_id"] == {"type": "keyword"}
    assert not any(value.get("type") == "knn_vector" for value in properties.values())


def test_prepared_product_iterator_uses_product_id_as_document_id(tmp_path: Path) -> None:
    path = tmp_path / "products.jsonl"
    record = {"product_id": "42", "title": "chair"}
    path.write_text(json.dumps(record) + "\n")
    assert list(iter_prepared_products(path)) == [("42", record)]


def test_wands_vector_mapping_and_documents_use_one_field_per_model(tmp_path: Path) -> None:
    index_spec = load_wands_index_config(ROOT / "config/indexes.toml")
    model = load_model_registry(ROOT / "config/models.toml")["gte_modernbert_base"]
    definition = wands_index_definition(index_spec, vector_models=(model,))
    vector_field = definition["mappings"]["properties"]["embedding_gte_modernbert_base"]
    assert vector_field["type"] == "knn_vector"
    assert vector_field["dimension"] == 768
    assert vector_field["method"]["engine"] == "lucene"
    assert vector_field["method"]["space_type"] == "cosinesimil"

    path = tmp_path / "products.jsonl"
    record = {"product_id": "42", "title": "chair"}
    path.write_text(json.dumps(record) + "\n")
    vectors = {model.name: (["42"], np.ones((1, model.dims), dtype=np.float32))}
    documents = list(iter_products_with_vectors(path, vectors))
    assert documents[0][0] == "42"
    assert len(documents[0][1]["embedding_gte_modernbert_base"]) == 768
