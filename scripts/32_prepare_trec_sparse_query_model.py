from __future__ import annotations

import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from poc.manifest import write_json
from poc.os_client import OpenSearchClient
from poc.provenance import (
    collect_code_revision,
    collect_manifest_provenance,
    verify_decision_provenance,
)
from poc.trec_indexing import trec_opensearch_url
from poc.trec_sparse import (
    extract_sparse_prediction,
    load_trec_sparse_config,
    select_registered_model_identity,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/trec-product-search/neural-sparse/query-model.json"


def main() -> None:
    benchmark_provenance = collect_manifest_provenance(
        ROOT,
        profile_path=ROOT / "config/benchmark-2.19.toml",
        environment_path=(
            ROOT / "results/environment/benchmark-profile-opensearch-2.19.json"
        ),
    )
    spec = load_trec_sparse_config(ROOT / "config/trec_neural_sparse.toml")
    url = trec_opensearch_url()
    tasks: list[dict[str, Any]] = []
    with OpenSearchClient(url, timeout=300) as client:
        version = client.wait_until_ready(
            expected_version=spec.expected_opensearch_version
        )
        client.request(
            "PUT",
            "/_cluster/settings",
            json_body={
                "persistent": {
                    "plugins.ml_commons.only_run_on_ml_node": "false",
                    "plugins.ml_commons.native_memory_threshold": "99",
                    "plugins.ml_commons.jvm_heap_memory_threshold": "99",
                }
            },
        )
        search = cast(
            dict[str, Any],
            client.request(
                "POST",
                "/_plugins/_ml/models/_search",
                json_body={"size": 1000, "query": {"match_all": {}}},
            ),
        )
        hits = cast(
            list[dict[str, Any]], cast(dict[str, Any], search["hits"])["hits"]
        )
        selected = select_registered_model_identity(
            hits,
            name=spec.query_tokenizer_name,
            version=spec.query_tokenizer_version,
        )
        if selected is None:
            response = cast(
                dict[str, Any],
                client.request(
                    "POST",
                    "/_plugins/_ml/models/_register",
                    json_body={
                        "name": spec.query_tokenizer_name,
                        "version": spec.query_tokenizer_version,
                        "model_format": spec.query_tokenizer_format,
                    },
                ),
            )
            registration = wait_for_ml_task(client, str(response["task_id"]))
            tasks.append(registration)
            model_id = str(registration["model_id"])
        else:
            model_id = str(selected["_id"])
        live_model = cast(
            dict[str, Any],
            client.request("GET", f"/_plugins/_ml/models/{model_id}"),
        )
        if live_model.get("model_state") != "DEPLOYED":
            response = cast(
                dict[str, Any],
                client.request("POST", f"/_plugins/_ml/models/{model_id}/_deploy"),
            )
            tasks.append(wait_for_ml_task(client, str(response["task_id"])))
            live_model = cast(
                dict[str, Any],
                client.request("GET", f"/_plugins/_ml/models/{model_id}"),
            )
        if live_model.get("model_state") != "DEPLOYED":
            raise RuntimeError("TREC sparse query tokenizer is not deployed")
        if (
            live_model.get("name") != spec.query_tokenizer_name
            or live_model.get("model_format") != spec.query_tokenizer_format
            or live_model.get("algorithm") != "SPARSE_TOKENIZE"
            or live_model.get("model_content_hash_value")
            != spec.query_tokenizer_content_sha256
            or live_model.get("model_content_size_in_bytes")
            != spec.query_tokenizer_content_bytes
        ):
            raise RuntimeError("TREC sparse query tokenizer content differs")
        prediction_response = cast(
            dict[str, Any],
            client.request(
                "POST",
                f"/_plugins/_ml/_predict/sparse_encoding/{model_id}",
                json_body={"text_docs": ["running shoe"]},
            ),
        )
        prediction = extract_sparse_prediction(prediction_response)
    completion_code_revision = asdict(collect_code_revision(ROOT))
    provenance_valid = (
        completion_code_revision == benchmark_provenance.get("code_revision")
        and verify_decision_provenance(
            benchmark_provenance,
            root=ROOT,
            profile_path=ROOT / "config/benchmark-2.19.toml",
            environment_path=(
                ROOT
                / "results/environment/benchmark-profile-opensearch-2.19.json"
            ),
            require_current_code_revision=True,
        )
    )
    artifact = {
        "schema_version": 2,
        "dataset": "TREC Product Search 2024",
        "prepared_at": datetime.now(UTC).isoformat(),
        "status": "passed",
        "quality_evidence_eligible_for_decision": provenance_valid,
        "benchmark_provenance_valid": provenance_valid,
        "benchmark_provenance": benchmark_provenance,
        "completion_code_revision": completion_code_revision,
        "opensearch_url": url,
        "opensearch_version": version,
        "mode": "document_only_custom_query_tokenizer",
        "ml_commons_memory_thresholds": {
            "native_percent": 99,
            "jvm_heap_percent": 99,
            "scope": "isolated_resource_capped_benchmark_cluster",
        },
        "model_id": model_id,
        "registered_spec": {
            "name": spec.query_tokenizer_name,
            "version": spec.query_tokenizer_version,
            "format": spec.query_tokenizer_format,
        },
        "registered_model": live_model,
        "task_history": tasks,
        "probe": {
            "text": "running shoe",
            "token_count": len(prediction),
            "tokens": prediction,
        },
    }
    write_json(OUTPUT, artifact)
    print(
        f"deployed TREC sparse tokenizer {model_id}; "
        f"probe returned {len(prediction)} tokens"
    )


def wait_for_ml_task(
    client: OpenSearchClient,
    task_id: str,
    *,
    timeout_seconds: float = 1800,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        task = cast(
            dict[str, Any], client.request("GET", f"/_plugins/_ml/tasks/{task_id}")
        )
        state = str(task.get("state", ""))
        if state == "COMPLETED":
            return task
        if state in {"FAILED", "COMPLETED_WITH_ERROR"}:
            raise RuntimeError(f"ML Commons task {task_id} failed: {task}")
        time.sleep(2)
    raise TimeoutError(f"ML Commons task {task_id} timed out")


if __name__ == "__main__":
    main()
