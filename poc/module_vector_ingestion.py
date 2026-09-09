from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import re
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import httpx
import numpy as np

from poc.latency import summarize_latency_ms
from poc.manifest import canonical_sha256
from poc.os_client import OpenSearchClient

WireFormat = Literal["numeric", "base64"]


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    document_count: int
    dimension: int
    batch_sizes: tuple[int, ...]
    trials: int
    warmup_documents: int
    seed: int
    decision_run: bool

    def __post_init__(self) -> None:
        if self.document_count <= 0:
            raise ValueError("document count must be positive")
        if self.dimension != 256:
            raise ValueError("the module benchmark is frozen at 256 dimensions")
        if not self.batch_sizes or any(batch_size <= 0 for batch_size in self.batch_sizes):
            raise ValueError("every batch size must be positive")
        if len(set(self.batch_sizes)) != len(self.batch_sizes):
            raise ValueError("batch sizes must be unique")
        if self.trials <= 0:
            raise ValueError("trial count must be positive")
        if self.warmup_documents < 0:
            raise ValueError("warmup document count cannot be negative")
        if self.decision_run and self.document_count < 50_000:
            raise ValueError("a decision run requires at least 50,000 documents")
        if self.decision_run and self.trials < 5:
            raise ValueError("a decision run requires at least five trials")
        if self.decision_run and len(self.batch_sizes) != 1:
            raise ValueError("a decision run requires one frozen batch size")


def assert_decision_environment(
    *,
    decision_run: bool,
    topology: str,
    machine: str,
    system: str,
    opensearch_image: str,
) -> None:
    if not decision_run:
        return
    if topology != "same-host":
        raise ValueError("a decision run requires same-host topology")
    if system != "Linux" or machine.lower() not in ("x86_64", "amd64"):
        raise ValueError("a decision run requires a Linux x86_64 client host")
    if re.search(r"@sha256:[0-9a-f]{64}$", opensearch_image) is None:
        raise ValueError("a decision run requires an immutable OpenSearch image digest")


def generate_vectors(*, document_count: int, dimension: int, seed: int) -> np.ndarray:
    if document_count <= 0:
        raise ValueError("document count must be positive")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    generator = np.random.default_rng(seed)
    vectors = generator.standard_normal((document_count, dimension), dtype=np.float32)
    norms = np.sqrt(np.sum(vectors * vectors, axis=1, dtype=np.float32))
    if bool(np.any(norms == 0.0)):
        raise ValueError("deterministic vector generation produced a zero vector")
    vectors /= norms[:, np.newaxis]
    return np.asarray(vectors, dtype=np.float32)


def build_bulk_payload(
    *,
    index: str,
    start_document: int,
    vectors: np.ndarray,
    wire_format: WireFormat,
) -> str:
    if not index:
        raise ValueError("index must be non-empty")
    if start_document <= 0:
        raise ValueError("start document must be positive")
    if wire_format not in ("numeric", "base64"):
        raise ValueError("wire format must be numeric or base64")
    if vectors.ndim != 2 or vectors.shape[1] != 256:
        raise ValueError("bulk vectors must be a two-dimensional 256-dimension matrix")
    if not bool(np.all(np.isfinite(vectors))):
        raise ValueError("bulk vectors must be finite")

    lines: list[str] = []
    for offset, vector in enumerate(vectors):
        document_id = start_document + offset
        action = {
            "index": {
                "_index": index,
                "_id": str(document_id),
                "version": document_id,
                "version_type": "external_gte",
            }
        }
        source = _document_source(document_id)
        if wire_format == "numeric":
            source["embedding"] = vector.tolist()
        else:
            source["embedding"] = base64.b64encode(
                vector.astype("<f4", copy=False).tobytes()
            ).decode("ascii")
        lines.append(json.dumps(action, separators=(",", ":"), sort_keys=True))
        lines.append(
            json.dumps(source, allow_nan=False, separators=(",", ":"), sort_keys=True)
        )
    return "\n".join(lines) + "\n"


def summarize_comparison(
    trials: list[dict[str, Any]], *, decision_run: bool
) -> dict[str, Any]:
    if not trials:
        raise ValueError("comparison requires trial records")
    by_arm = {
        arm: [trial for trial in trials if trial.get("arm") == arm]
        for arm in ("numeric", "base64")
    }
    if not by_arm["numeric"] or len(by_arm["numeric"]) != len(by_arm["base64"]):
        raise ValueError("comparison requires paired numeric and base64 trials")
    batch_sizes = {int(trial["batch_size"]) for trial in trials}
    if len(batch_sizes) != 1:
        raise ValueError("comparison records must use one batch size")

    numeric_bytes = statistics.median(
        float(trial["serialized_bytes"]) for trial in by_arm["numeric"]
    )
    base64_bytes = statistics.median(
        float(trial["serialized_bytes"]) for trial in by_arm["base64"]
    )
    numeric_throughput = statistics.median(
        float(trial["documents_per_second"]) for trial in by_arm["numeric"]
    )
    base64_throughput = statistics.median(
        float(trial["documents_per_second"]) for trial in by_arm["base64"]
    )
    numeric_p95 = statistics.median(
        float(cast(dict[str, Any], trial["bulk_latency"])["p95_ms"])
        for trial in by_arm["numeric"]
    )
    base64_p95 = statistics.median(
        float(cast(dict[str, Any], trial["bulk_latency"])["p95_ms"])
        for trial in by_arm["base64"]
    )
    if numeric_bytes <= 0 or numeric_throughput <= 0 or numeric_p95 <= 0:
        raise ValueError("numeric comparison medians must be positive")

    payload_reduction = 1.0 - (base64_bytes / numeric_bytes)
    throughput_ratio = base64_throughput / numeric_throughput
    p95_latency_ratio = base64_p95 / numeric_p95
    expected_documents = {
        int(trial["document_count"]) for trial in trials
    }
    complete = (
        len(expected_documents) == 1
        and next(iter(expected_documents)) > 0
        and all(
            int(trial["failed_items"]) == 0
            and int(trial["vector_count"]) == int(trial["document_count"])
            for trial in trials
        )
    )
    gates = {
        "complete_and_error_free": complete,
        "payload_reduction_at_least_60_percent": payload_reduction >= 0.60,
        "throughput_regression_at_most_5_percent": throughput_ratio >= 0.95,
        "p95_latency_regression_at_most_5_percent": p95_latency_ratio <= 1.05,
    }
    document_count = next(iter(expected_documents)) if len(expected_documents) == 1 else 0
    eligible = (
        decision_run
        and document_count >= 50_000
        and len(by_arm["numeric"]) >= 5
        and all(gates.values())
    )
    return {
        "batch_size": next(iter(batch_sizes)),
        "paired_trials": len(by_arm["numeric"]),
        "numeric_median_serialized_bytes": int(numeric_bytes),
        "base64_median_serialized_bytes": int(base64_bytes),
        "payload_reduction_fraction": payload_reduction,
        "numeric_median_documents_per_second": numeric_throughput,
        "base64_median_documents_per_second": base64_throughput,
        "throughput_ratio": throughput_ratio,
        "numeric_median_bulk_p95_ms": numeric_p95,
        "base64_median_bulk_p95_ms": base64_p95,
        "p95_latency_ratio": p95_latency_ratio,
        "gates": gates,
        "eligible_for_decision": eligible,
    }


def run_benchmark(
    client: OpenSearchClient,
    *,
    config: BenchmarkConfig,
    contract: dict[str, Any],
    source_identity: dict[str, Any],
    opensearch_image: str,
    topology: str,
) -> dict[str, Any]:
    version = client.wait_until_ready(expected_version="3.8.0")
    root_info = cast(dict[str, Any], client.request("GET", "/"))
    version_info = cast(dict[str, Any], root_info["version"])
    vectors = generate_vectors(
        document_count=config.document_count,
        dimension=config.dimension,
        seed=config.seed,
    )
    warmup_count = min(config.warmup_documents, config.document_count)
    trial_records: list[dict[str, Any]] = []
    warmup_records: list[dict[str, Any]] = []
    run_id = uuid.uuid4().hex[:12]
    for batch_size in config.batch_sizes:
        if warmup_count:
            for arm in ("numeric", "base64"):
                warmup_records.append(
                    _run_trial(
                        client,
                        contract=contract,
                        vectors=vectors[:warmup_count],
                        batch_size=batch_size,
                        arm=arm,
                        trial=0,
                        run_id=run_id,
                    )
                )
        for trial in range(1, config.trials + 1):
            arm_order = ("numeric", "base64") if trial % 2 else ("base64", "numeric")
            for arm in arm_order:
                trial_records.append(
                    _run_trial(
                        client,
                        contract=contract,
                        vectors=vectors,
                        batch_size=batch_size,
                        arm=arm,
                        trial=trial,
                        run_id=run_id,
                    )
                )

    comparisons = [
        summarize_comparison(
            [record for record in trial_records if record["batch_size"] == batch_size],
            decision_run=config.decision_run,
        )
        for batch_size in config.batch_sizes
    ]
    selected = max(
        comparisons,
        key=lambda comparison: (
            min(
                float(comparison["numeric_median_documents_per_second"]),
                float(comparison["base64_median_documents_per_second"]),
            ),
            -int(comparison["batch_size"]),
        ),
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": "mageos-opensearch-hybrid-base64-ingestion",
        "declared_variable": "numeric JSON array versus little-endian float32 Base64",
        "config": {
            "document_count": config.document_count,
            "dimension": config.dimension,
            "batch_sizes": list(config.batch_sizes),
            "trials": config.trials,
            "warmup_documents": warmup_count,
            "seed": config.seed,
            "decision_run": config.decision_run,
        },
        "environment": {
            "opensearch_version": version,
            "opensearch_build_hash": str(version_info["build_hash"]),
            "opensearch_distribution": str(version_info["distribution"]),
            "lucene_version": str(version_info["lucene_version"]),
            "opensearch_image": opensearch_image,
            "topology": topology,
            "python_version": sys.version.split()[0],
            "httpx_version": httpx.__version__,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "cpu_model": _cpu_model(),
            "memory_total_bytes": _memory_total_bytes(),
        },
        "identity": {
            "source": source_identity,
            "contract_sha256": canonical_sha256(contract),
            "mapping_sha256": canonical_sha256(contract["mapping"]),
            "result_contract_sha256": canonical_sha256(contract["result_contract"]),
            "vector_float32_sha256": hashlib.sha256(
                vectors.astype("<f4", copy=False).tobytes()
            ).hexdigest(),
        },
        "warmups": warmup_records,
        "trials": trial_records,
        "comparisons": comparisons,
        "selected_batch_size": int(selected["batch_size"]),
        "eligible_for_decision": bool(selected["eligible_for_decision"]),
    }


def _document_source(document_id: int) -> dict[str, Any]:
    source_identity = hashlib.sha256(f"fixture-product-{document_id}".encode()).hexdigest()
    return {
        "brand": "fixture",
        "category": f"category {document_id % 100}",
        "customer_group_prices": [],
        "description": f"deterministic benchmark product {document_id}",
        "embedding_eligible": True,
        "entity_id": document_id,
        "features": f"feature {document_id % 20}",
        "generation_id": 1,
        "is_salable": document_id % 10 != 0,
        "model_revision": "a" * 64,
        "price": float((document_id % 10_000) + 1) / 100.0,
        "product_class": "simple",
        "revision": document_id,
        "sku": f"BENCH-{document_id:08d}",
        "sku_normalized": f"bench-{document_id:08d}",
        "source_hash": source_identity,
        "status": 1,
        "stock_id": 1,
        "store_id": 1,
        "title": f"Benchmark Product {document_id}",
        "title_normalized": f"benchmark product {document_id}",
        "visibility": 4,
    }


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _memory_total_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) * 1024
    return None


def _run_trial(
    client: OpenSearchClient,
    *,
    contract: dict[str, Any],
    vectors: np.ndarray,
    batch_size: int,
    arm: str,
    trial: int,
    run_id: str,
) -> dict[str, Any]:
    if arm not in ("numeric", "base64"):
        raise ValueError("benchmark arm must be numeric or base64")
    arm_code = "n" if arm == "numeric" else "b"
    index = f"mageos-hybrid-wire-{run_id}-{batch_size}-{arm_code}-{trial}"
    definition = {
        "settings": contract["index_settings"],
        "mappings": contract["mapping"],
    }
    client.delete_index(index)
    client.create_index(index, definition)
    node_before = _node_snapshot(client)
    index_before = _index_snapshot(client, index)
    serialized_bytes = 0
    encoding_seconds = 0.0
    bulk_seconds = 0.0
    bulk_latency_ms: list[float] = []
    failed_items = 0
    worker_before = _worker_memory_snapshot()
    worker_cpu_started = time.process_time()
    build_started = time.perf_counter()
    try:
        for start in range(0, len(vectors), batch_size):
            encoding_started = time.perf_counter()
            payload = build_bulk_payload(
                index=index,
                start_document=start + 1,
                vectors=vectors[start : start + batch_size],
                wire_format=cast(WireFormat, arm),
            )
            encoding_seconds += time.perf_counter() - encoding_started
            serialized_bytes += len(payload.encode())
            bulk_started = time.perf_counter()
            response = cast(
                dict[str, Any], client.request("POST", "/_bulk", content=payload)
            )
            elapsed = time.perf_counter() - bulk_started
            bulk_seconds += elapsed
            bulk_latency_ms.append(elapsed * 1000.0)
            items = cast(list[dict[str, Any]], response.get("items", []))
            if response.get("errors"):
                failed_items += sum(
                    1
                    for item in items
                    if int(cast(dict[str, Any], item.get("index", {})).get("status", 500))
                    not in (200, 201, 409)
                )
                raise RuntimeError(
                    f"OpenSearch bulk arm {arm} failed with {failed_items} failed items"
                )
        client.refresh(index)
        readiness_started = time.perf_counter()
        document_count = _count(client, index)
        vector_count = _count(client, index, query={"exists": {"field": "embedding"}})
        readiness_seconds = time.perf_counter() - readiness_started
        full_build_seconds = time.perf_counter() - build_started
        worker_process_cpu_seconds = time.process_time() - worker_cpu_started
        worker_after = _worker_memory_snapshot()
        if document_count != len(vectors) or vector_count != len(vectors):
            raise RuntimeError(
                "OpenSearch benchmark validation found missing documents or vectors"
            )
        mapping = cast(
            dict[str, Any], client.request("GET", f"/{index}/_mapping")
        )[index]["mappings"]
        index_after = _index_snapshot(client, index)
        node_after = _node_snapshot(client)
        return {
            "arm": arm,
            "batch_size": batch_size,
            "trial": trial,
            "document_count": document_count,
            "vector_count": vector_count,
            "failed_items": failed_items,
            "serialized_bytes": serialized_bytes,
            "serialized_bytes_per_document": serialized_bytes / len(vectors),
            "client_encoding_seconds": encoding_seconds,
            "bulk_request_seconds": bulk_seconds,
            "bulk_latency": summarize_latency_ms(bulk_latency_ms),
            "full_build_seconds": full_build_seconds,
            "time_to_validated_readiness_seconds": readiness_seconds,
            "documents_per_second": len(vectors) / full_build_seconds,
            "worker_process_cpu_seconds": worker_process_cpu_seconds,
            "worker_process_cpu_fraction": worker_process_cpu_seconds / full_build_seconds,
            "worker_rss_before_bytes": worker_before["rss_bytes"],
            "worker_rss_bytes": worker_after["rss_bytes"],
            "worker_peak_rss_bytes": worker_after["peak_rss_bytes"],
            "transport_mode": "direct_bulk_no_queue",
            "retry_count": 0,
            "dead_letter_count": 0,
            "mapping_sha256": canonical_sha256(mapping),
            "index_stats_before": index_before,
            "index_stats_after": index_after,
            "node_stats_before": node_before,
            "node_stats_after": node_after,
        }
    finally:
        client.delete_index(index)


def _count(
    client: OpenSearchClient, index: str, *, query: dict[str, Any] | None = None
) -> int:
    body = {"query": query} if query is not None else None
    response = cast(
        dict[str, Any], client.request("POST", f"/{index}/_count", json_body=body)
    )
    return int(response["count"])


def _index_snapshot(client: OpenSearchClient, index: str) -> dict[str, int]:
    response = cast(
        dict[str, Any],
        client.request("GET", f"/{index}/_stats/docs,indexing,refresh,merge,segments"),
    )
    primaries = cast(dict[str, Any], cast(dict[str, Any], response["_all"])["primaries"])
    docs = cast(dict[str, Any], primaries["docs"])
    indexing = cast(dict[str, Any], primaries["indexing"])
    refresh = cast(dict[str, Any], primaries["refresh"])
    merges = cast(dict[str, Any], primaries["merges"])
    segments = cast(dict[str, Any], primaries["segments"])
    return {
        "documents": int(docs["count"]),
        "deleted_documents": int(docs["deleted"]),
        "index_total": int(indexing["index_total"]),
        "index_time_millis": int(indexing["index_time_in_millis"]),
        "refresh_total": int(refresh["total"]),
        "refresh_time_millis": int(refresh["total_time_in_millis"]),
        "merge_total": int(merges["total"]),
        "merge_time_millis": int(merges["total_time_in_millis"]),
        "segments": int(segments["count"]),
    }


def _node_snapshot(client: OpenSearchClient) -> dict[str, int]:
    response = cast(
        dict[str, Any],
        client.request("GET", "/_nodes/stats/jvm,process,thread_pool"),
    )
    nodes = cast(dict[str, dict[str, Any]], response["nodes"])
    heap_used = 0
    heap_max = 0
    cpu_percent = 0
    rejected = 0
    for node in nodes.values():
        jvm = cast(dict[str, Any], node["jvm"])
        memory = cast(dict[str, Any], jvm["mem"])
        process = cast(dict[str, Any], node["process"])
        cpu = cast(dict[str, Any], process["cpu"])
        thread_pool = cast(dict[str, dict[str, Any]], node["thread_pool"])
        heap_used += int(memory["heap_used_in_bytes"])
        heap_max += int(memory["heap_max_in_bytes"])
        cpu_percent = max(cpu_percent, int(cpu["percent"]))
        rejected += sum(
            int(pool.get("rejected", 0))
            for name, pool in thread_pool.items()
            if name in ("bulk", "write")
        )
    return {
        "heap_used_bytes": heap_used,
        "heap_max_bytes": heap_max,
        "maximum_process_cpu_percent": cpu_percent,
        "bulk_or_write_rejected": rejected,
    }


def _worker_memory_snapshot() -> dict[str, int | None]:
    status = Path("/proc/self/status")
    if not status.is_file():
        return {"rss_bytes": None, "peak_rss_bytes": None}
    values: dict[str, int | None] = {"rss_bytes": None, "peak_rss_bytes": None}
    keys = {"VmRSS:": "rss_bytes", "VmHWM:": "peak_rss_bytes"}
    for line in status.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in keys:
            values[keys[parts[0]]] = int(parts[1]) * 1024

    return values
