from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

from poc.manifest import canonical_sha256, collect_index_facts, read_json
from poc.os_client import OpenSearchClient
from poc.smoke import EXPECTED_VERSION, INDEX, PIPELINE, SMOKE_QUERIES
from poc.trec import read_run

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = cast(dict[str, Any], read_json(ROOT / "runs/smoke-hybrid.manifest.json"))
    responses = cast(dict[str, Any], read_json(ROOT / "results/smoke/responses.json"))
    if manifest.get("eligible_for_decision") is not False:
        raise AssertionError("smoke manifest must be excluded from decision evidence")
    if manifest.get("encoder_runtime") != "deterministic_hash_smoke_only":
        raise AssertionError("unexpected smoke encoder runtime")

    url = os.environ.get("OPENSEARCH_HYBRID_OS_URL", "http://127.0.0.1:9201")
    with OpenSearchClient(url) as client:
        client.wait_until_ready(expected_version=EXPECTED_VERSION)
        facts = collect_index_facts(client, INDEX)
        pipeline_response = cast(
            dict[str, Any], client.request("GET", f"/_search/pipeline/{PIPELINE}")
        )

    manifest_index = cast(dict[str, Any], manifest["index"])
    if facts.uuid != manifest_index["uuid"]:
        raise AssertionError("live index UUID differs from the run manifest")
    if facts.document_count != 100:
        raise AssertionError(f"live smoke index has {facts.document_count} documents")
    if facts.segment_count != 1 or facts.deleted_document_count != 0:
        raise AssertionError("live smoke index is not force-merged and clean")

    live_pipeline = pipeline_response[PIPELINE]
    if canonical_sha256(live_pipeline) != manifest["pipeline_sha256"]:
        raise AssertionError("live pipeline differs from the run manifest")

    query_results = cast(dict[str, Any], responses["queries"])
    for query in SMOKE_QUERIES:
        result = cast(dict[str, Any], query_results[query.query_id])
        top_ids = cast(dict[str, list[str]], result["top_ids"])
        for arm in ("bm25", "dense", "hybrid"):
            if top_ids[arm][0] != query.expected_document_id:
                raise AssertionError(f"stored {arm} result failed for {query.query_id}")

    records = read_run(ROOT / "runs/smoke-hybrid.trec")
    query_ids = {record.query_id for record in records}
    if query_ids != {query.query_id for query in SMOKE_QUERIES}:
        raise AssertionError("TREC run does not contain exactly the smoke queries")
    if len(records) != len(SMOKE_QUERIES) * 10:
        raise AssertionError("TREC run does not contain ten hits per smoke query")

    print(
        f"verified: OpenSearch {EXPECTED_VERSION}, index {facts.uuid}, "
        f"{facts.document_count} docs, {facts.segment_count} segment, {len(records)} run rows"
    )


if __name__ == "__main__":
    main()
