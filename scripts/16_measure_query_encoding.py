from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import numpy as np

from poc.bm25 import load_prepared_queries
from poc.config import WANDS_INT8_MODEL_NAMES, load_model_registry
from poc.latency import summarize_latency_ms
from poc.manifest import write_json
from poc.provenance import collect_manifest_provenance
from poc.query_runtime import (
    create_fp32_onnx_query_backend,
    create_int8_query_backend,
    finalize_query_encoding_latency_artifact,
    load_query_runtime_spec,
    verify_query_encoding_latency_artifact,
)
from poc.trec import identifier_sort_key

ROOT = Path(__file__).resolve().parents[1]
WARMUP_CALLS = 5


def _measure(backend: Any, texts: list[str], *, dimensions: int) -> dict[str, object]:
    for _ in range(WARMUP_CALLS):
        backend.encode([texts[0]])
    samples: list[float] = []
    for text in texts:
        started = time.perf_counter_ns()
        vectors = np.asarray(backend.encode([text]), dtype=np.float32)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        if vectors.shape != (1, dimensions) or not np.isfinite(vectors).all():
            raise ValueError("query latency backend returned invalid vectors")
        samples.append(elapsed_ms)
    return {
        "runtime": backend.runtime,
        "artifact_sha256": backend.model_artifact_sha256,
        "warmup_calls": WARMUP_CALLS,
        "batch_size": 1,
        "distinct_queries": len(texts),
        "summary": summarize_latency_ms(samples),
        "samples_ms": samples,
    }


def main() -> None:
    benchmark_provenance = collect_manifest_provenance(ROOT)
    registry = load_model_registry(ROOT / "config/models.toml")
    runtime = load_query_runtime_spec(ROOT / "config/query_runtime.toml")
    queries: dict[str, str] = {}
    for split in ("dev", "test"):
        queries.update(
            load_prepared_queries(
                ROOT / f"data/prepared/wands/queries.{split}.jsonl",
                expected_split=split,
            )
        )
    query_ids = sorted(queries, key=identifier_sort_key)

    for model_name in WANDS_INT8_MODEL_NAMES:
        model = registry[model_name]
        texts = [model.render_query(queries[query_id]) for query_id in query_ids]
        fp32 = create_fp32_onnx_query_backend(root=ROOT, model=model, runtime=runtime)
        fp32_measurement = _measure(fp32, texts, dimensions=model.dims)
        del fp32
        gc.collect()
        int8 = create_int8_query_backend(root=ROOT, model=model, runtime=runtime)
        int8_measurement = _measure(int8, texts, dimensions=model.dims)
        del int8
        gc.collect()
        fp32_summary = fp32_measurement["summary"]
        int8_summary = int8_measurement["summary"]
        if not isinstance(fp32_summary, dict) or not isinstance(int8_summary, dict):
            raise TypeError("latency measurement summary has the wrong type")
        fp32_p95 = float(fp32_summary["p95_ms"])
        int8_p95 = float(int8_summary["p95_ms"])
        artifact = finalize_query_encoding_latency_artifact(
            root=ROOT,
            model=model,
            runtime=runtime,
            query_ids=query_ids,
            fp32_measurement=fp32_measurement,
            int8_measurement=int8_measurement,
            benchmark_provenance=benchmark_provenance,
        )
        output = ROOT / f"results/wands/query-runtime/{model_name}.batch1-latency.json"
        write_json(output, artifact)
        verify_query_encoding_latency_artifact(
            root=ROOT,
            model=model,
            runtime=runtime,
        )
        print(
            f"{model_name}: fp32 p95={fp32_p95:.3f} ms, "
            f"int8 p95={int8_p95:.3f} ms, "
            f"ratio={fp32_p95 / int8_p95:.2f}x, "
            "decision_eligible="
            f"{artifact['latency_decision_eligible']}"
        )


if __name__ == "__main__":
    main()
