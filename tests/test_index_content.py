from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any, cast

import pytest

from poc.datasets import DatasetIntegrityError
from poc.index_content import verify_live_index_content


class FakeContentClient:
    def __init__(self, documents: Iterable[tuple[str, Mapping[str, Any]]]) -> None:
        self.documents = {document_id: deepcopy(dict(source)) for document_id, source in documents}
        self.reorder_response = False

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
    ) -> object:
        if method == "GET" and path == "/fixture/_count":
            return {"count": len(self.documents)}
        if method == "POST" and path == "/fixture/_mget":
            assert json_body is not None
            ids = cast(list[str], json_body["ids"])
            docs = [
                {
                    "_id": document_id,
                    "found": document_id in self.documents,
                    "_source": deepcopy(self.documents.get(document_id)),
                }
                for document_id in ids
            ]
            if self.reorder_response:
                docs.reverse()
            return {"docs": docs}
        raise AssertionError((method, path))


def _expected_documents() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("1", {"product_id": "1", "title": "one", "vector": [1.0, 0.0]}),
        ("10", {"product_id": "10", "title": "ten", "vector": [0.5, 0.5]}),
        ("2", {"product_id": "2", "title": "two", "vector": [0.0, 1.0]}),
    ]


def test_live_index_content_digest_streams_exact_sources_in_registered_id_order() -> None:
    expected = _expected_documents()
    evidence = verify_live_index_content(
        cast(Any, FakeContentClient(expected)),
        index="fixture",
        expected_documents=iter(expected),
        expected_count=3,
        page_size=2,
        float32_fields=("vector",),
    )

    assert evidence == {
        "algorithm": "sha256-canonical-opensearch-source-v1",
        "document_order": "registered_exact_id_sequence",
        "page_size": 2,
        "documents": 3,
        "float32_fields": ["vector"],
        "sha256": "20a6d0041e78cf8af5add302dc6a95c698254ea6a8fc02b94ab4d65902a281e0",
    }


@pytest.mark.parametrize("field", ("title", "vector"))
def test_live_index_content_digest_rejects_source_or_vector_tamper(field: str) -> None:
    expected = _expected_documents()
    client = FakeContentClient(expected)
    if field == "title":
        client.documents["2"][field] = "altered"
    else:
        client.documents["2"][field] = [0.25, 0.75]

    with pytest.raises(DatasetIntegrityError, match="source differs for document 2"):
        verify_live_index_content(
            cast(Any, client),
            index="fixture",
            expected_documents=iter(expected),
            expected_count=3,
            page_size=2,
            float32_fields=("vector",),
        )


def test_live_index_content_digest_rejects_duplicate_expected_ids() -> None:
    expected = [
        _expected_documents()[0],
        _expected_documents()[0],
        _expected_documents()[2],
    ]
    with pytest.raises(DatasetIntegrityError, match="uniqueness"):
        verify_live_index_content(
            cast(Any, FakeContentClient(_expected_documents())),
            index="fixture",
            expected_documents=iter(expected),
            expected_count=3,
            page_size=2,
        )


def test_live_index_content_normalizes_vectors_to_mapped_float32_precision() -> None:
    expected = [("1", {"product_id": "1", "vector": [0.10000000149]})]
    client = FakeContentClient(expected)
    client.documents["1"]["vector"] = [0.1]

    evidence = verify_live_index_content(
        cast(Any, client),
        index="fixture",
        expected_documents=iter(expected),
        expected_count=1,
        float32_fields=("vector",),
    )

    assert evidence["documents"] == 1


def test_live_index_content_digest_rejects_missing_extra_or_reordered_live_docs() -> None:
    expected = _expected_documents()
    missing = FakeContentClient(expected[:-1])
    with pytest.raises(DatasetIntegrityError, match="count differs"):
        verify_live_index_content(
            cast(Any, missing),
            index="fixture",
            expected_documents=iter(expected),
            expected_count=3,
        )

    extra = FakeContentClient([*expected, ("11", {"product_id": "11"})])
    with pytest.raises(DatasetIntegrityError, match="count differs"):
        verify_live_index_content(
            cast(Any, extra),
            index="fixture",
            expected_documents=iter(expected),
            expected_count=3,
        )

    reordered = FakeContentClient(expected)
    reordered.reorder_response = True
    with pytest.raises(DatasetIntegrityError, match="response order differs"):
        verify_live_index_content(
            cast(Any, reordered),
            index="fixture",
            expected_documents=iter(expected),
            expected_count=3,
            page_size=2,
        )


def test_second_content_binding_rejects_post_scan_document_mutation() -> None:
    expected = _expected_documents()
    client = FakeContentClient(expected)
    first = verify_live_index_content(
        cast(Any, client),
        index="fixture",
        expected_documents=iter(expected),
        expected_count=3,
    )
    assert first["documents"] == 3

    client.documents["2"]["title"] = "mutated after decision replay began"
    with pytest.raises(DatasetIntegrityError, match="source differs for document 2"):
        verify_live_index_content(
            cast(Any, client),
            index="fixture",
            expected_documents=iter(expected),
            expected_count=3,
        )
