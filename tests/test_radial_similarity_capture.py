from __future__ import annotations

import json
import struct
from base64 import b64encode
from typing import Any

import pytest

from poc.radial_similarity_capture import capture_radial_similarity


class FakeResponse:
    def __init__(self, value: dict[str, Any]) -> None:
        self.value = value

    def raise_for_status(self) -> None:
        return

    def json(self) -> dict[str, Any]:
        return self.value


class FakeClient:
    def __init__(
        self,
        *,
        exact_total: int = 2,
        version: str = "3.8.0",
        missing_review_product: bool = False,
    ) -> None:
        self.exact_total = exact_total
        self.version = version
        self.missing_review_product = missing_review_product
        self.requests: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        if url == "/":
            return FakeResponse({"version": {"number": self.version}})
        if url == "/fixture-index/_mapping":
            return FakeResponse(
                {
                    "fixture-index": {
                        "mappings": {
                            "properties": {
                                "embedding": {
                                    "type": "knn_vector",
                                    "dimension": 2,
                                    "method": {
                                        "engine": "lucene",
                                        "space_type": "cosinesimil",
                                    },
                                }
                            }
                        }
                    }
                }
            )
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        request = kwargs.get("json")
        assert isinstance(request, dict)
        self.requests.append(request)
        if url == "/fixture-index/_count":
            return FakeResponse({"count": 3})
        if url == "/fixture-index/_mget":
            assert request == {
                "docs": [
                    {
                        "_id": str(product_id),
                        "_source": [
                            "entity_id",
                            "sku",
                            "title",
                            "product_class",
                            "category",
                            "brand",
                        ],
                    }
                    for product_id in (2, 3, 41)
                ]
            }
            return FakeResponse(
                {
                    "docs": [
                        {
                            "_id": "2",
                            "found": True,
                            "_source": {
                                "entity_id": 2,
                                "sku": "candidate-two",
                                "title": "Candidate Two",
                                "product_class": "simple",
                                "category": "Shoes",
                                "brand": "Fixture",
                            },
                        },
                        {
                            "_id": "3",
                            "found": True,
                            "_source": {
                                "entity_id": 3,
                                "sku": "candidate-three",
                                "title": "Candidate Three",
                                "product_class": "simple",
                                "category": "Shoes",
                                "brand": "Fixture",
                            },
                        },
                        {
                            "_id": "41",
                            "found": not self.missing_review_product,
                            "_source": {
                                "entity_id": 41,
                                "sku": "seed-forty-one",
                                "title": "Seed Forty One",
                                "product_class": "simple",
                                "category": "Shoes",
                                "brand": "Fixture",
                            },
                        },
                    ]
                }
            )
        if url != "/fixture-index/_search":
            raise AssertionError(f"unexpected POST {url}")
        if "docvalue_fields" in request:
            encoded = b64encode(struct.pack("<2f", 1.0, 0.0)).decode()
            return FakeResponse(
                {
                    "timed_out": False,
                    "_shards": {"failed": 0},
                    "hits": {"hits": [{"fields": {"embedding": [encoded]}}]},
                }
            )
        if "script_score" in request.get("query", {}):
            return FakeResponse(
                {
                    "timed_out": False,
                    "_shards": {"failed": 0},
                    "hits": {
                        "total": {"value": self.exact_total, "relation": "eq"},
                        "hits": [
                            {"_score": 0.95, "fields": {"entity_id": [2]}},
                            {"_score": 0.90, "fields": {"entity_id": [3]}},
                        ],
                    },
                }
            )
        return FakeResponse(
            {
                "timed_out": False,
                "_shards": {"failed": 0},
                "hits": {
                    "hits": [
                        {"_score": 0.95, "fields": {"entity_id": [2]}},
                        {"_score": 0.90, "fields": {"entity_id": [3]}},
                    ]
                },
            }
        )


def test_capture_derives_case_identity_and_collects_exact_and_concurrent_radial_evidence() -> None:
    client = FakeClient()

    capture = capture_radial_similarity(client, _specification())

    assert capture["identity"]["judgment_set_sha256"] != "a" * 64
    assert len(capture["identity"]["judgment_set_sha256"]) == 64
    assert capture["capture_provenance"] == {
        **capture["capture_provenance"],
        "opensearch_version": "3.8.0",
        "physical_index": "fixture-index",
        "generation_id": 7,
        "index_document_count": 3,
        "eligible_document_count": 3,
        "exact_method": "knn_score_script_cosinesimil",
        "radial_method": "lucene_hnsw_min_score",
        "latency_samples_per_case": 4,
    }
    assert capture["qualification"]["scope"] == "smoke"
    assert capture["qualification"]["decision_eligible"] is False
    assert capture["qualification"]["checks"]["registered_scope"] is False
    case = capture["thresholds"][0]["cases"][0]
    assert case["exact_ids"] == [2, 3]
    assert case["exact_total"] == 2
    assert case["radial_ids"] == [2, 3]
    assert len(case["radial_latency_ms"]) == 4
    assert all(value >= 0.0 for value in case["radial_latency_ms"])
    assert capture["review_catalog"] == [
        {
            "product_id": 2,
            "sku": "candidate-two",
            "title": "Candidate Two",
            "product_class": "simple",
            "category": "Shoes",
            "brand": "Fixture",
        },
        {
            "product_id": 3,
            "sku": "candidate-three",
            "title": "Candidate Three",
            "product_class": "simple",
            "category": "Shoes",
            "brand": "Fixture",
        },
        {
            "product_id": 41,
            "sku": "seed-forty-one",
            "title": "Seed Forty One",
            "product_class": "simple",
            "category": "Shoes",
            "brand": "Fixture",
        },
    ]
    encoded = json.dumps(capture)
    assert "query_value" not in encoded
    radial_requests = [request for request in client.requests if "knn" in request.get("query", {})]
    assert len(radial_requests) == 5
    radial = radial_requests[0]["query"]["knn"]["embedding"]
    assert radial["min_score"] == pytest.approx(0.9)
    assert "k" not in radial
    assert radial["filter"]["bool"]["must_not"] == [{"term": {"entity_id": 41}}]


def test_capture_refuses_truncated_exact_membership() -> None:
    with pytest.raises(ValueError, match="exact similarity membership capture is truncated"):
        capture_radial_similarity(FakeClient(exact_total=3), _specification())


def test_capture_requires_opensearch_38() -> None:
    with pytest.raises(ValueError, match="requires OpenSearch 3.8.x"):
        capture_radial_similarity(FakeClient(version="3.7.0"), _specification())


def test_capture_refuses_missing_review_catalog_product() -> None:
    with pytest.raises(ValueError, match="review catalog product is unavailable"):
        capture_radial_similarity(FakeClient(missing_review_product=True), _specification())


def test_registered_capture_stays_ineligible_below_fifty_thousand_documents() -> None:
    specification = _specification()
    specification["qualification_scope"] = "registered"

    capture = capture_radial_similarity(
        FakeClient(),
        specification,
        "http://127.0.0.1:9200",
    )

    assert capture["qualification"]["scope"] == "registered"
    assert capture["qualification"]["decision_eligible"] is False
    assert capture["qualification"]["checks"]["index_document_count"] is False
    assert capture["qualification"]["checks"]["loopback_endpoint"] is True


def test_capture_requires_declared_catalog_evidence_scope() -> None:
    specification = _specification()
    del specification["catalog_evidence_scope"]

    with pytest.raises(ValueError, match="catalog evidence scope is invalid"):
        capture_radial_similarity(FakeClient(), specification)


def test_holdout_capture_accepts_only_one_frozen_threshold() -> None:
    specification = _specification()
    specification["split"] = "holdout"
    specification["thresholds"] = [0.9, 0.92]

    with pytest.raises(ValueError, match="threshold candidates are invalid"):
        capture_radial_similarity(FakeClient(), specification)


def _specification() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "split": "development",
        "qualification_scope": "smoke",
        "catalog_evidence_scope": "synthetic_execution",
        "index": "fixture-index",
        "generation_id": 7,
        "storefront_limit": 12,
        "maximum_result_count": 20,
        "concurrency": 4,
        "latency_samples": 4,
        "exact_capture_limit": 1000,
        "identity": {
            "model_id": "fixture/model",
            "model_revision": "revision-1",
            "dimension": 2,
            "similarity": "cosine",
            "source_recipe_version": "recipe-1",
            "store_id": 2,
            "use_case": "substitute",
        },
        "thresholds": [0.9],
        "cases": [
            {
                "case_id": "head-shoe",
                "seed_product_id": 41,
                "slices": ["head", "simple"],
                "unsafe_ids": [99],
            }
        ],
    }
