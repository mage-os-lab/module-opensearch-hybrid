from __future__ import annotations

import argparse
import json
import os
import platform
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from poc.config import load_model_registry
from poc.manifest import read_json, write_json
from poc.ml_commons import registration_body_for_version
from poc.os_client import OpenSearchClient
from poc.query_runtime import create_int8_query_backend, load_query_runtime_spec

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_MANIFEST = ROOT / "results/wands/ml-commons/arctic-package.json"
MODEL_NAME = "arctic_embed_m_v2"
PACKAGE_SERVER_SERVICE = "ml-commons-package-server"
PACKAGE_SERVER_PORT = 8080
PACKAGE_SERVER_ORIGIN = f"http://{PACKAGE_SERVER_SERVICE}:{PACKAGE_SERVER_PORT}"


def validate_package_server_binding(
    package: dict[str, Any],
    requested_opensearch_version: str,
) -> str:
    if package.get("schema_version") != 2:
        raise ValueError("ML Commons package manifest schema version must be 2")
    if package.get("opensearch_target") != requested_opensearch_version:
        raise ValueError(
            "ML Commons package manifest OpenSearch target differs from the requested version"
        )
    if package.get("eligible_for_decision") is not False:
        raise ValueError("ML Commons package manifest must remain diagnostic")
    registration = cast(dict[str, Any], package.get("registration_body", {}))
    binding = cast(dict[str, Any], package.get("package_server", {}))
    if (
        binding.get("service") != PACKAGE_SERVER_SERVICE
        or binding.get("port") != PACKAGE_SERVER_PORT
        or binding.get("network_scope") != "compose_internal"
    ):
        raise ValueError("ML Commons package server binding is not Compose-internal")
    url = registration.get("url")
    if url != binding.get("url"):
        raise ValueError("ML Commons registration URL differs from its package server binding")
    if not isinstance(url, str) or not url.startswith(f"{PACKAGE_SERVER_ORIGIN}/"):
        raise ValueError("ML Commons registration URL is not on the package server service")
    expected_regex = f"^{re.escape(url)}$"
    trusted_url_regex = binding.get("trusted_url_regex")
    if trusted_url_regex != expected_regex:
        raise ValueError("ML Commons trusted URL regex is not exact for the package URL")
    return expected_regex


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
    raise TimeoutError(f"ML Commons task {task_id} timed out")


def _find_existing_model(
    client: OpenSearchClient,
    *,
    name: str,
    version: str,
) -> str | None:
    response = cast(
        dict[str, Any],
        client.request(
            "POST",
            "/_plugins/_ml/models/_search",
            json_body={"size": 100, "query": {"match_all": {}}},
        ),
    )
    hits = cast(list[dict[str, Any]], cast(dict[str, Any], response["hits"])["hits"])
    for hit in hits:
        source = cast(dict[str, Any], hit.get("_source", {}))
        if source.get("name") == name and source.get("model_version") == version:
            return str(hit["_id"])
    return None


def _embedding(response: dict[str, Any]) -> np.ndarray:
    results = cast(list[dict[str, Any]], response.get("inference_results", []))
    for result in results:
        for output in cast(list[dict[str, Any]], result.get("output", [])):
            data = output.get("data")
            if isinstance(data, list):
                return np.asarray(data, dtype=np.float32)
    raise ValueError(f"ML Commons prediction contains no dense embedding: {response}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test Arctic in ML Commons")
    parser.add_argument("--opensearch-version", required=True)
    parser.add_argument("--latency-samples", type=int, default=50)
    parser.add_argument(
        "--package-manifest",
        type=Path,
        default=PACKAGE_MANIFEST,
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.latency_samples <= 0:
        raise ValueError("latency samples must be positive")
    output = (
        ROOT
        / "results/wands/ml-commons"
        / f"arctic-opensearch-{args.opensearch_version}-smoke.json"
    )
    package_path = args.package_manifest
    if not package_path.is_absolute():
        package_path = ROOT / package_path
    package = cast(dict[str, Any], read_json(package_path))
    trusted_url_regex = validate_package_server_binding(
        package, args.opensearch_version
    )
    registration = registration_body_for_version(
        cast(dict[str, Any], package["registration_body"]),
        args.opensearch_version,
    )
    registry = load_model_registry(ROOT / "config/models.toml")
    runtime = load_query_runtime_spec(ROOT / "config/query_runtime.toml")
    model = registry[MODEL_NAME]
    query_texts = ["coffee table"]
    prepared_queries = ROOT / "data/prepared/wands/queries.test.jsonl"
    if prepared_queries.is_file():
        query_texts = [
            str(cast(dict[str, Any], json.loads(line))["query"])
            for line in prepared_queries.read_text().splitlines()
            if line.strip()
        ]
    query = model.render_query(query_texts[0])
    url = os.environ.get("OPENSEARCH_HYBRID_OS_URL", "http://127.0.0.1:9201")
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "started_at": datetime.now(UTC).isoformat(),
        "requested_opensearch_version": args.opensearch_version,
        "host_architecture": platform.machine().lower(),
        "latency_samples": args.latency_samples,
        "host_matches_registered_latency_architecture": (
            platform.machine().lower() in {"x86_64", "amd64"}
        ),
        "eligible_for_decision": False,
        "quality_evidence_eligible_for_decision": False,
        "latency_decision_eligible": False,
        "ineligibility_reason": (
            "diagnostic ML Commons deployment smoke without the registered benchmark "
            "quality or latency evidence contract"
        ),
        "model": MODEL_NAME,
        "source_package": str(package_path.relative_to(ROOT)),
        "registration_body": registration,
        "status": "failed",
    }
    try:
        with OpenSearchClient(url, timeout=300) as client:
            root_response = cast(dict[str, Any], client.request("GET", "/"))
            version = cast(dict[str, Any], root_response["version"])
            live_version = str(version["number"])
            if live_version != args.opensearch_version:
                raise ValueError(
                    f"OpenSearch version {live_version} differs from requested "
                    f"{args.opensearch_version}"
                )
            client.request(
                "PUT",
                "/_cluster/settings",
                json_body={
                    "persistent": {
                        "plugins.ml_commons.allow_registering_model_via_url": "true",
                        "plugins.ml_commons.trusted_url_regex": trusted_url_regex,
                    }
                },
            )
            model_id = _find_existing_model(
                client,
                name=str(registration["name"]),
                version=str(registration["version"]),
            )
            tasks: list[dict[str, Any]] = []
            if model_id is None:
                response = cast(
                    dict[str, Any],
                    client.request(
                        "POST",
                        "/_plugins/_ml/models/_register",
                        json_body=registration,
                    ),
                )
                task = _wait_for_task(client, str(response["task_id"]))
                tasks.append(task)
                model_id = str(task["model_id"])
            live_model = cast(
                dict[str, Any],
                client.request("GET", f"/_plugins/_ml/models/{model_id}"),
            )
            if live_model.get("model_state") != "DEPLOYED":
                response = cast(
                    dict[str, Any],
                    client.request("POST", f"/_plugins/_ml/models/{model_id}/_deploy"),
                )
                tasks.append(_wait_for_task(client, str(response["task_id"])))
                live_model = cast(
                    dict[str, Any],
                    client.request("GET", f"/_plugins/_ml/models/{model_id}"),
                )
            prediction = cast(
                dict[str, Any],
                client.request(
                    "POST",
                    f"/_plugins/_ml/_predict/text_embedding/{model_id}",
                    json_body={
                        "text_docs": [query],
                        "return_number": True,
                        "target_response": ["sentence_embedding"],
                    },
                ),
            )
            durations_ms: list[float] = []
            for index in range(args.latency_samples + 5):
                sample_query = model.render_query(query_texts[index % len(query_texts)])
                started = time.perf_counter()
                client.request(
                    "POST",
                    f"/_plugins/_ml/_predict/text_embedding/{model_id}",
                    json_body={
                        "text_docs": [sample_query],
                        "return_number": True,
                        "target_response": ["sentence_embedding"],
                    },
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if index >= 5:
                    durations_ms.append(elapsed_ms)
        cluster_vector = _embedding(prediction)
        backend = create_int8_query_backend(root=ROOT, model=model, runtime=runtime)
        local_vector = backend.encode([query])[0]
        if cluster_vector.shape != (model.dims,):
            raise ValueError(
                f"ML Commons returned {cluster_vector.shape}, expected {(model.dims,)}"
            )
        cosine = float(np.dot(cluster_vector, local_vector))
        maximum_delta = float(np.max(np.abs(cluster_vector - local_vector)))
        latency = {
            "minimum_ms": min(durations_ms),
            "mean_ms": float(np.mean(durations_ms)),
            "p50_ms": float(np.percentile(durations_ms, 50)),
            "p95_ms": float(np.percentile(durations_ms, 95)),
            "p99_ms": float(np.percentile(durations_ms, 99)),
            "maximum_ms": max(durations_ms),
        }
        evidence.update(
            {
                "completed_at": datetime.now(UTC).isoformat(),
                "status": "passed",
                "model_id": model_id,
                "registered_model": live_model,
                "task_history": tasks,
                "prediction_shape": list(cluster_vector.shape),
                "local_vs_cluster_cosine": cosine,
                "local_vs_cluster_maximum_absolute_delta": maximum_delta,
                "latency": latency,
            }
        )
    except Exception as exc:
        evidence["completed_at"] = datetime.now(UTC).isoformat()
        evidence["error_type"] = type(exc).__name__
        evidence["error"] = str(exc)
        write_json(output, evidence)
        raise
    write_json(output, evidence)
    print(
        f"ML Commons Arctic smoke passed on {args.opensearch_version}: "
        f"shape={evidence['prediction_shape']}, "
        f"cosine={evidence['local_vs_cluster_cosine']:.9f}, "
        f"p95={latency['p95_ms']:.3f} ms"
    )


if __name__ == "__main__":
    main()
