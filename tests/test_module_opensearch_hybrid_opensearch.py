from __future__ import annotations

import base64
import json
import os
import struct
import uuid
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from poc.radial_similarity_calibration import evaluate_capture
from poc.radial_similarity_capture import HttpClient, capture_radial_similarity

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "etc/opensearch_hybrid_contract.json"
DIMENSION = 256


@pytest.mark.integration
def test_numeric_and_base64_contracts_execute_on_opensearch_38() -> None:
    if os.environ.get("RUN_OPENSEARCH_INTEGRATION") != "1":
        pytest.skip("set RUN_OPENSEARCH_INTEGRATION=1 for the local OpenSearch 3 fixture")
    endpoint = os.environ.get("OPENSEARCH_URL", "http://127.0.0.1:9201").rstrip("/")
    suffix = uuid.uuid4().hex[:12]
    numeric_index = f"mageos-hybrid-numeric-{suffix}"
    base64_index = f"mageos-hybrid-base64-{suffix}"
    pipeline = f"mageos-opensearch-hybrid-integration-{suffix}"
    contract = json.loads(CONTRACT.read_text())
    first_vector = [1.0] + [0.0] * (DIMENSION - 1)
    second_vector = [0.8, 0.6] + [0.0] * (DIMENSION - 2)
    vectors = {"1": first_vector, "2": second_vector}
    client = httpx.Client(base_url=endpoint, timeout=10.0)
    try:
        info = client.get("/")
        info.raise_for_status()
        assert info.json()["version"]["number"] == "3.8.0"

        response = client.put(
            f"/_search/pipeline/{pipeline}",
            json=contract["pipeline"],
        )
        response.raise_for_status()
        for index in (numeric_index, base64_index):
            response = client.put(
                f"/{index}",
                json={
                    "settings": contract["index_settings"],
                    "mappings": contract["mapping"],
                },
            )
            response.raise_for_status()

        for document_id, vector in vectors.items():
            response = client.put(
                f"/{numeric_index}/_doc/{document_id}",
                json=_document(document_id, vector),
            )
            response.raise_for_status()
        response = client.put(
            f"/{numeric_index}/_doc/3",
            json=_document("3", None),
        )
        response.raise_for_status()

        bulk_lines: list[str] = []
        for document_id, vector in vectors.items():
            bulk_lines.append(json.dumps({"index": {"_index": base64_index, "_id": document_id}}))
            encoded = base64.b64encode(struct.pack("<256f", *vector)).decode("ascii")
            bulk_lines.append(json.dumps(_document(document_id, encoded)))
        bulk_lines.append(json.dumps({"index": {"_index": base64_index, "_id": "3"}}))
        bulk_lines.append(json.dumps(_document("3", None)))
        response = client.post(
            "/_bulk",
            content="\n".join(bulk_lines) + "\n",
            headers={"content-type": "application/x-ndjson"},
        )
        response.raise_for_status()
        assert response.json()["errors"] is False

        malformed = client.post(
            "/_bulk",
            content=(
                json.dumps({"index": {"_index": base64_index, "_id": "malformed"}})
                + "\n"
                + json.dumps(_document("4", "not-valid-base64"))
                + "\n"
            ),
            headers={"content-type": "application/x-ndjson"},
        )
        malformed.raise_for_status()
        malformed_body = malformed.json()
        assert malformed_body["errors"] is True
        assert malformed_body["items"][0]["index"]["status"] == 400

        for index in (numeric_index, base64_index):
            client.post(f"/{index}/_refresh").raise_for_status()
            mapping = client.get(f"/{index}/_mapping").raise_for_status().json()[index]["mappings"]
            settings = (
                client.get(f"/{index}/_settings")
                .raise_for_status()
                .json()[index]["settings"]["index"]
            )
            assert mapping == contract["mapping"]
            assert settings["knn"] == "true"
            assert settings["knn.algo_param"]["ef_search"] == "800"
        installed_pipeline = (
            client.get(f"/_search/pipeline/{pipeline}").raise_for_status().json()[pipeline]
        )
        assert installed_pipeline == contract["pipeline"]

        expected_bytes = {
            document_id: struct.pack("<256f", *vector) for document_id, vector in vectors.items()
        }
        numeric_bytes = _retrieved_vector_bytes(client, numeric_index)
        base64_bytes = _retrieved_vector_bytes(client, base64_index)
        assert numeric_bytes == expected_bytes
        assert base64_bytes == expected_bytes

        queries = {
            "exact": {
                "size": 2,
                "_source": False,
                "query": {
                    "script_score": {
                        "query": {"term": {"embedding_eligible": True}},
                        "script": {
                            "lang": "knn",
                            "source": "knn_score",
                            "params": {
                                "field": "embedding",
                                "query_value": first_vector,
                                "space_type": "cosinesimil",
                            },
                        },
                    }
                },
            },
            "ann": {
                "size": 2,
                "_source": False,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": first_vector,
                            "k": 2,
                            "filter": {"term": {"embedding_eligible": True}},
                        }
                    }
                },
            },
        }
        for query in queries.values():
            numeric_results = _search_results(client, numeric_index, query)
            base64_results = _search_results(client, base64_index, query)
            numeric_hits = [document_id for document_id, _score in numeric_results]
            base64_hits = [document_id for document_id, _score in base64_results]
            assert numeric_hits == base64_hits
            assert numeric_hits == ["1", "2"]
            assert [score for _document_id, score in numeric_results] == pytest.approx(
                [score for _document_id, score in base64_results], abs=1e-6
            )

        radial_request = {
            "size": 5,
            "track_total_hits": False,
            "_source": False,
            "fields": ["entity_id"],
            "sort": [{"_score": "desc"}, {"entity_id": "asc"}],
            "query": {
                "knn": {
                    "embedding": {
                        "vector": first_vector,
                        "min_score": 0.85,
                        "filter": {
                            "bool": {
                                "filter": [
                                    {"term": {"store_id": 1}},
                                    {"term": {"generation_id": 1}},
                                    {"term": {"model_revision": "a" * 64}},
                                    {"term": {"embedding_eligible": True}},
                                    {"term": {"status": 1}},
                                    {"terms": {"visibility": [3, 4]}},
                                    {"term": {"is_salable": True}},
                                ],
                                "must_not": [{"term": {"entity_id": 1}}],
                            }
                        },
                    }
                }
            },
        }
        numeric_radial = _search_results(client, numeric_index, radial_request)
        base64_radial = _search_results(client, base64_index, radial_request)
        assert [document_id for document_id, _score in numeric_radial] == ["2"]
        assert [document_id for document_id, _score in base64_radial] == ["2"]
        assert [score for _document_id, score in numeric_radial] == pytest.approx(
            [score for _document_id, score in base64_radial], abs=1e-6
        )

        capture = capture_radial_similarity(
            cast(HttpClient, client),
            {
                "schema_version": 1,
                "split": "development",
                "qualification_scope": "smoke",
                "catalog_evidence_scope": "synthetic_execution",
                "index": base64_index,
                "generation_id": 1,
                "storefront_limit": 5,
                "maximum_result_count": 5,
                "concurrency": 4,
                "latency_samples": 4,
                "exact_capture_limit": 20,
                "identity": {
                    "model_id": "fixture/model",
                    "model_revision": "a" * 64,
                    "dimension": DIMENSION,
                    "similarity": "cosine",
                    "source_recipe_version": "fixture-v1",
                    "store_id": 1,
                    "use_case": "substitute",
                },
                "thresholds": [0.85],
                "cases": [
                    {
                        "case_id": "fixture-seed-1",
                        "seed_product_id": 1,
                        "slices": ["fixture", "simple"],
                        "unsafe_ids": [],
                    }
                ],
            },
        )
        captured_case = capture["thresholds"][0]["cases"][0]
        assert captured_case["exact_ids"] == [2]
        assert captured_case["radial_ids"] == [2]
        captured_evaluation = evaluate_capture(capture)["candidates"][0]
        assert captured_evaluation["mean_ann_recall_at_limit"] == pytest.approx(1.0)
        assert captured_evaluation["technical_eligible"] is False
        assert captured_evaluation["gates"]["registered_capture_scope"] is False
        assert captured_evaluation["eligible"] is False
        assert captured_evaluation["unsafe_hit_count"] == 0

        hybrid_request = {
            "size": 100,
            "track_total_hits": False,
            "_source": False,
            "query": {
                "hybrid": {
                    "pagination_depth": 100,
                    "queries": [
                        {
                            "bool": {
                                "should": [
                                    {
                                        "constant_score": {
                                            "filter": {"term": {"_id": "1"}},
                                            "boost": 8.0,
                                        }
                                    },
                                    {
                                        "constant_score": {
                                            "filter": {"term": {"_id": "2"}},
                                            "boost": 2.0,
                                        }
                                    },
                                ],
                                "minimum_should_match": 1,
                            }
                        },
                        {
                            "knn": {
                                "embedding": {
                                    "vector": first_vector,
                                    "k": 100,
                                    "filter": {"terms": {"_id": ["1", "2"]}},
                                }
                            }
                        },
                    ],
                }
            },
        }
        numeric_hybrid = _search_results(client, numeric_index, hybrid_request, pipeline)
        base64_hybrid = _search_results(client, base64_index, hybrid_request, pipeline)
        assert [document_id for document_id, _score in numeric_hybrid] == ["1", "2"]
        assert [document_id for document_id, _score in base64_hybrid] == ["1", "2"]
        assert [score for _document_id, score in numeric_hybrid] == pytest.approx(
            [score for _document_id, score in base64_hybrid], abs=1e-6
        )

        vectorless = client.get(f"/{base64_index}/_doc/3").raise_for_status().json()
        assert vectorless["_source"]["embedding_eligible"] is False
        assert "embedding" not in vectorless["_source"]
        assert client.get(f"/{base64_index}/_doc/malformed").status_code == 404
    finally:
        client.delete(f"/{numeric_index}")
        client.delete(f"/{base64_index}")
        client.delete(f"/_search/pipeline/{pipeline}")
        client.close()


def _document(document_id: str, vector: list[float] | str | None) -> dict[str, Any]:
    document: dict[str, Any] = {
        "entity_id": int(document_id),
        "store_id": 1,
        "generation_id": 1,
        "sku": f"shoe-{document_id}",
        "sku_normalized": f"shoe-{document_id}",
        "title": f"fixture shoe {document_id}",
        "title_normalized": f"fixture shoe {document_id}",
        "description": f"fixture shoe {document_id}",
        "product_class": "simple",
        "category": "shoes",
        "brand": "fixture",
        "features": "fixture",
        "visibility": 4,
        "status": 1,
        "price": 10.0,
        "stock_id": 1,
        "is_salable": True,
        "embedding_eligible": vector is not None,
        "source_hash": document_id * 64,
        "model_revision": "a" * 64,
    }
    if vector is not None:
        document["embedding"] = vector
    return document


def _retrieved_vector_bytes(client: httpx.Client, index: str) -> dict[str, bytes]:
    response = client.post(
        f"/{index}/_search",
        json={
            "size": 10,
            "_source": False,
            "docvalue_fields": ["embedding"],
            "sort": [{"entity_id": "asc"}],
            "query": {"term": {"embedding_eligible": True}},
        },
    )
    response.raise_for_status()
    return {
        hit["_id"]: base64.b64decode(hit["fields"]["embedding"][0], validate=True)
        for hit in response.json()["hits"]["hits"]
    }


def _search_results(
    client: httpx.Client,
    index: str,
    request: dict[str, Any],
    pipeline: str | None = None,
) -> list[tuple[str, float]]:
    response = client.post(
        f"/{index}/_search",
        params={"search_pipeline": pipeline} if pipeline is not None else None,
        json=request,
    )
    response.raise_for_status()
    return [(hit["_id"], float(hit["_score"])) for hit in response.json()["hits"]["hits"]]
