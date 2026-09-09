from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from poc.embeddings import DeterministicHashEncoder
from poc.manifest import build_run_manifest, write_json
from poc.os_client import OpenSearchClient
from poc.search import (
    build_hybrid_request,
    build_normalization_pipeline,
    build_standalone_bm25_request,
    build_standalone_knn_request,
    hybrid_lexical_clause,
)
from poc.trec import RunRecord, write_run

INDEX = "opensearch-hybrid-smoke-v1"
PIPELINE = "opensearch-hybrid-smoke-minmax-v1"
VECTOR_FIELD = "smoke_vector"
EXPECTED_VERSION = "3.8.0"
PAGINATION_DEPTH = 100


@dataclass(frozen=True, slots=True)
class SmokeQuery:
    query_id: str
    text: str
    expected_document_id: str


SMOKE_QUERIES = (
    SmokeQuery("smoke-001", "red waterproof trail running shoe", "sku-000"),
    SmokeQuery("smoke-002", "solid walnut writing desk with drawers", "sku-001"),
    SmokeQuery("smoke-003", "stainless steel insulated water bottle", "sku-002"),
)


def index_definition(dims: int) -> dict[str, Any]:
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "-1",
            }
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "sku": {"type": "keyword"},
                "title": {"type": "text"},
                "product_class": {"type": "text"},
                "category": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "brand": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "description": {"type": "text"},
                "features": {"type": "text"},
                "price": {"type": "scaled_float", "scaling_factor": 100},
                "in_stock": {"type": "boolean"},
                VECTOR_FIELD: {
                    "type": "knn_vector",
                    "dimension": dims,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                        "parameters": {"ef_construction": 100, "m": 16},
                    },
                },
            },
        },
    }


def smoke_products() -> list[dict[str, object]]:
    products: list[dict[str, object]] = [
        {
            "sku": "SKU-000",
            "title": "Red waterproof trail running shoe",
            "product_class": "footwear",
            "category": "Shoes / Running / Trail",
            "brand": "Northstar",
            "description": "A grippy waterproof shoe for wet technical trails.",
            "features": "color:red | waterproof:yes | terrain:trail",
            "price": 129.00,
            "in_stock": True,
        },
        {
            "sku": "SKU-001",
            "title": "Solid walnut writing desk with drawers",
            "product_class": "furniture",
            "category": "Office / Desks / Writing",
            "brand": "Workshop",
            "description": "A compact hardwood desk with two quiet drawers.",
            "features": "material:walnut | drawers:2 | finish:oil",
            "price": 799.00,
            "in_stock": True,
        },
        {
            "sku": "SKU-002",
            "title": "Stainless steel insulated water bottle",
            "product_class": "drinkware",
            "category": "Outdoors / Hydration / Bottles",
            "brand": "Ridgeline",
            "description": "Vacuum insulated bottle that keeps drinks cold all day.",
            "features": "material:stainless-steel | capacity:24oz | insulated:yes",
            "price": 34.00,
            "in_stock": True,
        },
    ]

    colors = ("blue", "green", "black", "white", "orange", "purple")
    materials = ("cotton", "aluminum", "oak", "ceramic", "nylon")
    product_types = ("lamp", "chair", "jacket", "planter", "backpack", "rug", "clock")
    for number in range(3, 100):
        color = colors[number % len(colors)]
        material = materials[number % len(materials)]
        product_type = product_types[number % len(product_types)]
        products.append(
            {
                "sku": f"SKU-{number:03d}",
                "title": f"{color.title()} {material} {product_type} model {number:03d}",
                "product_class": product_type,
                "category": f"Catalog / {product_type.title()}",
                "brand": f"Maker-{number % 11}",
                "description": f"Everyday {product_type} in {color} with a {material} finish.",
                "features": f"color:{color} | material:{material} | series:{number // 10}",
                "price": float(20 + number * 3),
                "in_stock": number % 9 != 0,
            }
        )
    if len(products) != 100:
        raise AssertionError("smoke corpus must contain exactly 100 products")
    return products


def run_smoke(client: OpenSearchClient, root: Path) -> dict[str, object]:
    client.wait_until_ready(expected_version=EXPECTED_VERSION)
    encoder = DeterministicHashEncoder(dims=32)
    pipeline_definition = build_normalization_pipeline(lexical_weight=0.5)

    client.delete_index(INDEX)
    client.create_index(INDEX, index_definition(encoder.dims))
    client.put_search_pipeline(PIPELINE, pipeline_definition)

    products = smoke_products()
    documents: list[tuple[str, dict[str, object]]] = []
    for number, product in enumerate(products):
        document_id = f"sku-{number:03d}"
        document = dict(product)
        document[VECTOR_FIELD] = encoder.encode(str(product["title"]))
        documents.append((document_id, document))
    client.bulk_index(INDEX, documents)
    client.refresh(INDEX)
    client.force_merge(INDEX)

    request_artifacts: dict[str, object] = {"pipeline": pipeline_definition, "queries": {}}
    response_artifacts: dict[str, object] = {"queries": {}}
    records: list[RunRecord] = []

    for smoke_query in SMOKE_QUERIES:
        vector = encoder.encode(smoke_query.text)
        bm25_request = build_standalone_bm25_request(smoke_query.text)
        bm25_request["_source"] = {"excludes": [VECTOR_FIELD]}
        dense_request = build_standalone_knn_request(vector, vector_field=VECTOR_FIELD, k=10)
        hybrid_request = build_hybrid_request(
            smoke_query.text,
            vector,
            vector_field=VECTOR_FIELD,
            k=10,
            pagination_depth=PAGINATION_DEPTH,
        )
        if bm25_request["query"] != hybrid_lexical_clause(hybrid_request):
            raise AssertionError("hybrid lexical clause differs from standalone BM25")

        bm25_response = client.search(INDEX, bm25_request)
        dense_response = client.search(INDEX, dense_request)
        hybrid_response = client.search(INDEX, hybrid_request, pipeline=PIPELINE)
        top_ids = {
            "bm25": _hit_ids(bm25_response),
            "dense": _hit_ids(dense_response),
            "hybrid": _hit_ids(hybrid_response),
        }
        for arm, ids in top_ids.items():
            if not ids or ids[0] != smoke_query.expected_document_id:
                raise AssertionError(
                    f"{arm} top hit for {smoke_query.query_id} was {ids[:1]}, "
                    f"expected {smoke_query.expected_document_id}"
                )

        cast(dict[str, object], request_artifacts["queries"])[smoke_query.query_id] = {
            "text": smoke_query.text,
            "bm25": bm25_request,
            "dense": dense_request,
            "hybrid": hybrid_request,
        }
        cast(dict[str, object], response_artifacts["queries"])[smoke_query.query_id] = {
            "expected_document_id": smoke_query.expected_document_id,
            "top_ids": top_ids,
            "hybrid_took_ms": hybrid_response.get("took"),
        }

        for rank, hit in enumerate(_hits(hybrid_response), start=1):
            records.append(
                RunRecord(
                    query_id=smoke_query.query_id,
                    document_id=str(hit["_id"]),
                    rank=rank,
                    score=float(hit["_score"]),
                    tag="smoke-hybrid",
                )
            )

    manifest = build_run_manifest(
        client,
        index=INDEX,
        pipeline_id=PIPELINE,
        pipeline_definition=pipeline_definition,
        model_sha256=encoder.fingerprint,
        pagination_depth=PAGINATION_DEPTH,
        encoder_runtime="deterministic_hash_smoke_only",
        eligible_for_decision=encoder.eligible_for_decision,
        declared_variable="day0_hybrid_round_trip",
        hnsw={
            "engine": "lucene",
            "space_type": "cosinesimil",
            "m": 16,
            "ef_construction": 100,
            "ef_search": "plugin_default_smoke_only",
        },
    )
    if manifest.index.document_count != 100:
        raise AssertionError(f"index contains {manifest.index.document_count} documents, not 100")
    if manifest.index.segment_count != 1 or manifest.index.deleted_document_count != 0:
        raise AssertionError("smoke index must have one primary segment and zero deleted documents")

    run_path = root / "runs/smoke-hybrid.trec"
    write_run(run_path, records)
    write_json(root / "runs/smoke-hybrid.manifest.json", manifest.to_dict())
    write_json(root / "results/smoke/query-dsl.json", request_artifacts)
    write_json(root / "results/smoke/responses.json", response_artifacts)
    return {
        "index": manifest.index.name,
        "index_uuid": manifest.index.uuid,
        "documents": manifest.index.document_count,
        "segments": manifest.index.segment_count,
        "queries": len(SMOKE_QUERIES),
        "run": str(run_path),
        "eligible_for_decision": False,
    }


def _hits(response: dict[str, Any]) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], cast(dict[str, Any], response["hits"])["hits"])


def _hit_ids(response: dict[str, Any]) -> list[str]:
    return [str(hit["_id"]) for hit in _hits(response)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 100-product OpenSearch hybrid smoke test")
    parser.add_argument(
        "--opensearch-url",
        default=os.environ.get("OPENSEARCH_HYBRID_OS_URL", "http://127.0.0.1:9201"),
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    with OpenSearchClient(args.opensearch_url) as client:
        summary = run_smoke(client, args.root.resolve())
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
