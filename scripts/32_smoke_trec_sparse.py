from __future__ import annotations

from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from poc.manifest import collect_index_facts, read_json, write_json
from poc.os_client import OpenSearchClient
from poc.trec_indexing import trec_opensearch_url
from poc.trec_sparse import (
    build_trec_sparse_index_definition,
    build_trec_sparse_query,
    iter_trec_precomputed_sparse_documents,
    load_trec_sparse_config,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/trec-product-search/neural-sparse/opensearch-2.19-smoke.json"
SMOKE_INDEX = "opensearch-hybrid-trec-ps-2024-sparse-smoke"
SMOKE_RECORDS = 1000


def main() -> None:
    spec = load_trec_sparse_config(ROOT / "config/trec_neural_sparse.toml")
    smoke_spec = replace(spec, index_name=SMOKE_INDEX)
    query_model = cast(
        dict[str, Any],
        read_json(
            ROOT / "results/trec-product-search/neural-sparse/query-model.json"
        ),
    )
    if query_model.get("status") != "passed":
        raise ValueError("TREC sparse query tokenizer evidence is not passed")
    model_id = str(query_model["model_id"])
    artifact: dict[str, object] = {
        "schema_version": 1,
        "started_at": datetime.now(UTC).isoformat(),
        "dataset": "TREC Product Search 2024",
        "scope": "opensearch_2.19_sparse_compatibility_smoke",
        "index": SMOKE_INDEX,
        "records": SMOKE_RECORDS,
        "query_tokenizer_model_id": model_id,
        "status": "failed",
        "index_removed_after_smoke": False,
    }
    with OpenSearchClient(trec_opensearch_url(), timeout=300) as client:
        version = client.wait_until_ready(
            expected_version=spec.expected_opensearch_version
        )
        client.delete_index(SMOKE_INDEX)
        try:
            client.create_index(
                SMOKE_INDEX, build_trec_sparse_index_definition(smoke_spec)
            )
            client.bulk_index(
                SMOKE_INDEX,
                iter_trec_precomputed_sparse_documents(
                    ROOT / "data/raw/trec-product-search-2024/collection.trec.gz",
                    ROOT
                    / "data/cache/neural-sparse/trec-product-search-doc-v2-distill-preflight.jsonl",
                    smoke_spec,
                    maximum_records=SMOKE_RECORDS,
                ),
                batch_size=spec.bulk_request_size,
            )
            client.refresh(SMOKE_INDEX)
            facts = collect_index_facts(client, SMOKE_INDEX)
            if facts.document_count != SMOKE_RECORDS:
                raise ValueError("TREC sparse smoke indexed the wrong document count")
            response = client.search(
                SMOKE_INDEX,
                {
                    "size": 10,
                    "_source": ["product_id", "title"],
                    "query": build_trec_sparse_query(
                        smoke_spec, "running shoe", model_id=model_id
                    ),
                },
            )
            hits = cast(
                list[dict[str, Any]],
                cast(dict[str, Any], response["hits"])["hits"],
            )
            if not hits or any(hit.get("_score") is None for hit in hits):
                raise ValueError("TREC sparse smoke query returned no scored hits")
            artifact.update(
                {
                    "completed_at": datetime.now(UTC).isoformat(),
                    "opensearch_version": version,
                    "status": "passed",
                    "live_index": asdict(facts),
                    "query": "running shoe",
                    "returned_hits": len(hits),
                    "top_hits": [
                        {
                            "id": str(hit["_id"]),
                            "score": float(hit["_score"]),
                            "source": hit.get("_source"),
                        }
                        for hit in hits
                    ],
                }
            )
        finally:
            client.delete_index(SMOKE_INDEX)
            artifact["index_removed_after_smoke"] = True
            write_json(OUTPUT, artifact)
    print(
        f"OpenSearch {version} TREC sparse smoke passed: "
        f"{artifact['returned_hits']} scored hits"
    )


if __name__ == "__main__":
    main()
