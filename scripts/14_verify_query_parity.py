from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

from poc.bm25 import load_prepared_queries
from poc.config import WANDS_INT8_MODEL_NAMES, load_model_registry
from poc.datasets import DatasetIntegrityError
from poc.dense import encode_queries
from poc.embedding_cache import load_cached_vectors, load_embedding_inputs
from poc.manifest import canonical_sha256, read_json, write_json
from poc.model_runtime import (
    compare_embedding_vectors,
    create_bulk_backend,
    release_device_memory,
)
from poc.provenance import collect_manifest_provenance
from poc.query_runtime import (
    assert_document_runtime_parity,
    assert_query_runtime_parity,
    create_fp32_onnx_query_backend,
    create_int8_query_backend,
    finalize_query_runtime_artifact,
    load_query_runtime_spec,
    onnx_manifest_path,
    verify_query_parity_artifact,
)
from poc.trec import identifier_sort_key

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = WANDS_INT8_MODEL_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify fp32 and int8 query parity")
    parser.add_argument(
        "--record-only",
        action="store_true",
        help="write failed gate artifacts without returning a nonzero status",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    if len(query_ids) != runtime.parity_sample_size:
        raise ValueError(
            f"registered parity sample requires {runtime.parity_sample_size} queries, "
            f"found {len(query_ids)}"
        )

    products_path = ROOT / "data/prepared/wands/products.jsonl"
    failures: list[str] = []
    for name in MODEL_NAMES:
        model = registry[name]
        fp32_backend = create_bulk_backend(root=ROOT, model=model, device="cpu")
        onnx_fp32_backend = create_fp32_onnx_query_backend(
            root=ROOT,
            model=model,
            runtime=runtime,
        )
        int8_backend = create_int8_query_backend(root=ROOT, model=model, runtime=runtime)

        fp32_started = time.perf_counter()
        fp32_by_query = encode_queries(fp32_backend, model=model, queries=queries)
        fp32_seconds = time.perf_counter() - fp32_started
        onnx_fp32_by_query = encode_queries(
            onnx_fp32_backend,
            model=model,
            queries=queries,
        )
        int8_started = time.perf_counter()
        int8_by_query = encode_queries(
            int8_backend,
            model=model,
            queries=queries,
            batch_size=runtime.batch_size,
        )
        int8_seconds = time.perf_counter() - int8_started
        fp32_queries = np.stack([fp32_by_query[query_id] for query_id in query_ids])
        onnx_fp32_queries = np.stack(
            [onnx_fp32_by_query[query_id] for query_id in query_ids]
        )
        int8_queries = np.stack([int8_by_query[query_id] for query_id in query_ids])
        query_comparison = compare_embedding_vectors(fp32_queries, int8_queries)
        export_comparison = compare_embedding_vectors(fp32_queries, onnx_fp32_queries)
        quantization_comparison = compare_embedding_vectors(
            onnx_fp32_queries,
            int8_queries,
        )

        document_inputs = load_embedding_inputs(products_path, model)
        _, cached_document_vectors = load_cached_vectors(
            root=ROOT,
            products_path=products_path,
            model=model,
        )
        sample_size = runtime.document_drift_sample_size
        if len(document_inputs) < sample_size:
            raise ValueError(f"not enough documents for runtime drift check: {name}")
        current_document_vectors = fp32_backend.encode(
            [item.text for item in document_inputs[:sample_size]]
        )
        document_comparison = compare_embedding_vectors(
            cached_document_vectors[:sample_size],
            current_document_vectors,
        )
        status = "passed"
        error: str | None = None
        try:
            assert_query_runtime_parity(query_comparison, runtime)
            assert_document_runtime_parity(document_comparison, runtime)
        except DatasetIntegrityError as exception:
            status = "failed"
            error = str(exception)

        artifact_manifest = cast(
            dict[str, Any],
            read_json(onnx_manifest_path(ROOT, model, runtime)),
        )
        measurements = {
            "query_ids_sha256": canonical_sha256(query_ids),
            "query_sample_size": len(query_ids),
            "query_comparison": query_comparison,
            "torch_to_onnx_fp32_comparison": export_comparison,
            "onnx_fp32_to_int8_comparison": quantization_comparison,
            "worst_query": _worst_query(
                query_ids,
                fp32_queries,
                int8_queries,
            ),
            "document_drift_sample_size": sample_size,
            "document_runtime_comparison": document_comparison,
            "fp32_query_runtime": fp32_backend.runtime,
            "fp32_query_seconds": fp32_seconds,
            "int8_query_runtime": int8_backend.runtime,
            "int8_query_seconds": int8_seconds,
            "source_model_sha256": int8_backend.source_model_artifact_sha256,
            "int8_artifact_sha256": int8_backend.model_artifact_sha256,
            "int8_export_packages": artifact_manifest["export_packages"],
            "status": status,
            "error": error,
        }
        output = finalize_query_runtime_artifact(
            root=ROOT,
            model=model,
            runtime=runtime,
            measurements=measurements,
            benchmark_provenance=benchmark_provenance,
            artifact_type="parity",
        )
        write_json(
            ROOT / f"results/wands/query-runtime/{name}.parity.json",
            output,
        )
        verify_query_parity_artifact(root=ROOT, model=model, runtime=runtime)
        print(
            f"{name} parity {status}: min cosine="
            f"{query_comparison['minimum_cosine_similarity']:.9f}, "
            f"mean cosine={query_comparison['mean_cosine_similarity']:.9f}, "
            f"int8={int8_seconds:.3f}s, fp32={fp32_seconds:.3f}s"
        )
        del int8_backend
        del onnx_fp32_backend
        del fp32_backend
        release_device_memory()
        if error is not None:
            failures.append(f"{name}: {error}")
    if failures and not args.record_only:
        raise DatasetIntegrityError("; ".join(failures))


def _worst_query(
    query_ids: list[str],
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, object]:
    similarities = np.sum(reference * candidate, axis=1) / (
        np.linalg.norm(reference, axis=1) * np.linalg.norm(candidate, axis=1)
    )
    index = int(np.argmin(similarities))
    return {
        "query_id": query_ids[index],
        "cosine_similarity": float(similarities[index]),
        "maximum_absolute_delta": float(
            np.max(np.abs(reference[index] - candidate[index]))
        ),
    }


if __name__ == "__main__":
    main()
