from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from poc.manifest import write_json
from poc.neural_sparse import load_neural_sparse_spec, select_registered_model
from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/wands/neural-sparse/model-deployment.json"


def _wait_for_task(
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
    raise TimeoutError(f"ML Commons task {task_id} did not complete within {timeout_seconds}s")


def main() -> None:
    spec = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")
    url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_OS_URL",
    )
    task_history: list[dict[str, Any]] = []
    with OpenSearchClient(url, timeout=120) as client:
        client.request(
            "PUT",
            "/_cluster/settings",
            json_body={
                "persistent": {
                    "plugins.ml_commons.only_run_on_ml_node": "false",
                    "plugins.ml_commons.native_memory_threshold": "99",
                }
            },
        )
        search = cast(
            dict[str, Any],
            client.request(
                "POST",
                "/_plugins/_ml/models/_search",
                json_body={"size": 100, "query": {"match_all": {}}},
            ),
        )
        hits = cast(list[dict[str, Any]], cast(dict[str, Any], search["hits"])["hits"])
        selected = select_registered_model(hits, spec)
        if selected is None:
            response = cast(
                dict[str, Any],
                client.request(
                    "POST",
                    "/_plugins/_ml/models/_register",
                    json_body={
                        "name": spec.model_name,
                        "version": spec.model_version,
                        "model_format": spec.model_format,
                    },
                ),
            )
            registration = _wait_for_task(client, str(response["task_id"]))
            task_history.append(registration)
            model_id = str(registration["model_id"])
        else:
            model_id = str(selected["_id"])

        model = cast(
            dict[str, Any],
            client.request("GET", f"/_plugins/_ml/models/{model_id}"),
        )
        if model.get("model_state") != "DEPLOYED":
            response = cast(
                dict[str, Any],
                client.request("POST", f"/_plugins/_ml/models/{model_id}/_deploy"),
            )
            deployment = _wait_for_task(client, str(response["task_id"]))
            task_history.append(deployment)
            model = cast(
                dict[str, Any],
                client.request("GET", f"/_plugins/_ml/models/{model_id}"),
            )
        if model.get("model_state") != "DEPLOYED":
            raise RuntimeError(f"neural-sparse model is not deployed: {model}")

    artifact = {
        "schema_version": 1,
        "prepared_at": datetime.now(UTC).isoformat(),
        "opensearch_url": url,
        "model_id": model_id,
        "registered_model": model,
        "task_history": task_history,
        "registered_spec": {
            "name": spec.model_name,
            "version": spec.model_version,
            "format": spec.model_format,
            "query_analyzer": spec.query_analyzer,
        },
    }
    write_json(OUTPUT, artifact)
    print(f"deployed neural-sparse model {spec.model_name} as {model_id}")


if __name__ == "__main__":
    main()
