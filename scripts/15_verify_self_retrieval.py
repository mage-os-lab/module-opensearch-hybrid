from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from poc.config import WANDS_INT8_MODEL_NAMES, load_model_registry
from poc.datasets import DatasetIntegrityError, file_facts
from poc.embedding_cache import load_cached_vectors
from poc.manifest import canonical_sha256, read_json, write_json
from poc.model_runtime import create_bulk_backend, release_device_memory
from poc.provenance import collect_manifest_provenance
from poc.query_correctness import select_self_retrieval_ids, top1_self_retrieval
from poc.query_runtime import (
    create_int8_query_backend,
    finalize_query_runtime_artifact,
    load_query_runtime_spec,
    load_registered_product_titles,
    onnx_manifest_path,
    verify_self_retrieval_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = WANDS_INT8_MODEL_NAMES


def main() -> None:
    benchmark_provenance = collect_manifest_provenance(ROOT)
    registry = load_model_registry(ROOT / "config/models.toml")
    runtime = load_query_runtime_spec(ROOT / "config/query_runtime.toml")
    products_path = ROOT / "data/prepared/wands/products.jsonl"
    titles = load_registered_product_titles(products_path)
    sample_ids = select_self_retrieval_ids(
        list(titles),
        sample_size=runtime.self_retrieval_sample_size,
    )
    failures: list[str] = []
    for name in MODEL_NAMES:
        model = registry[name]
        document_ids, document_vectors = load_cached_vectors(
            root=ROOT,
            products_path=products_path,
            model=model,
        )
        texts = [model.render_query(titles[document_id]) for document_id in sample_ids]
        fp32_backend = create_bulk_backend(root=ROOT, model=model, device="cpu")
        int8_backend = create_int8_query_backend(root=ROOT, model=model, runtime=runtime)
        fp32_vectors = _encode_batches(fp32_backend, texts)
        int8_vectors = _encode_batches(
            int8_backend,
            texts,
            batch_size=runtime.batch_size,
        )
        fp32_result = top1_self_retrieval(
            query_document_ids=sample_ids,
            query_vectors=fp32_vectors,
            document_ids=document_ids,
            document_vectors=document_vectors,
        )
        int8_result = top1_self_retrieval(
            query_document_ids=sample_ids,
            query_vectors=int8_vectors,
            document_ids=document_ids,
            document_vectors=document_vectors,
        )
        int8_rate = cast(float, int8_result["top1_rate"])
        passed = int8_rate >= runtime.minimum_self_retrieval_top1
        if not passed:
            failures.append(
                f"{name} self-retrieval {int8_rate:.6f} < "
                f"{runtime.minimum_self_retrieval_top1:.6f}"
            )
        artifact_manifest_path = onnx_manifest_path(ROOT, model, runtime)
        artifact_manifest = cast(dict[str, Any], read_json(artifact_manifest_path))
        output = finalize_query_runtime_artifact(
            root=ROOT,
            model=model,
            runtime=runtime,
            benchmark_provenance=benchmark_provenance,
            artifact_type="self_retrieval",
            measurements={
                "query_recipe": "registered_query_template_applied_to_product_title",
                "document_recipe": "registered_full_document_template",
                "sample_method": "sha256_product_id",
                "sample_ids_sha256": canonical_sha256(sample_ids),
                "sample_size": len(sample_ids),
                "minimum_top1_rate": runtime.minimum_self_retrieval_top1,
                "fp32": fp32_result,
                "int8": int8_result,
                "int8_runtime": int8_backend.runtime,
                "int8_artifact_sha256": artifact_manifest["artifact_sha256"],
                "int8_manifest_sha256": file_facts(artifact_manifest_path).sha256,
                "status": "passed" if passed else "failed",
            },
        )
        write_json(
            ROOT / f"results/wands/query-runtime/{name}.self-retrieval.json",
            output,
        )
        verify_self_retrieval_artifact(root=ROOT, model=model, runtime=runtime)
        print(
            f"{name} self-retrieval: "
            f"fp32={cast(float, fp32_result['top1_rate']):.6f}, "
            f"int8={int8_rate:.6f}, status={'passed' if passed else 'failed'}"
        )
        del int8_backend
        del fp32_backend
        release_device_memory()
    if failures:
        raise DatasetIntegrityError("; ".join(failures))


def _encode_batches(backend: Any, texts: list[str], batch_size: int = 32) -> np.ndarray:
    return np.concatenate(
        [
            backend.encode(texts[start : start + batch_size])
            for start in range(0, len(texts), batch_size)
        ]
    )


if __name__ == "__main__":
    main()
