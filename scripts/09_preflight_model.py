from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from poc.config import ModelSpec, load_model_registry
from poc.datasets import DatasetIntegrityError
from poc.embedding_cache import (
    PREFLIGHT_MAXIMUM_ABSOLUTE_DELTA,
    PREFLIGHT_MINIMUM_COSINE,
    EmbeddingBackend,
    finalize_embedding_preflight_artifact,
    load_embedding_inputs,
    verify_embedding_preflight_artifact,
)
from poc.manifest import canonical_sha256, write_json
from poc.model_runtime import (
    compare_embedding_vectors,
    create_bulk_backend,
    release_device_memory,
    select_bulk_device,
)
from poc.provenance import collect_manifest_provenance

ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    "gte_modernbert_base",
    "granite_embedding_english_r2",
    "arctic_embed_m_v2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CPU and MPS model outputs")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--candidate-device", choices=("mps", "cpu"), default="mps")
    return parser.parse_args()


def encode_samples(
    backend: EmbeddingBackend,
    texts: list[str],
    *,
    batch_size: int,
) -> NDArray[np.float32]:
    batches = [
        backend.encode(texts[start : start + batch_size])
        for start in range(0, len(texts), batch_size)
    ]
    return np.concatenate(batches)


def run_backend(
    model: ModelSpec,
    texts: list[str],
    *,
    device: str,
    batch_size: int,
) -> tuple[NDArray[np.float32], dict[str, object]]:
    started = time.perf_counter()
    backend = create_bulk_backend(root=ROOT, model=model, device=device)
    vectors = encode_samples(backend, texts, batch_size=batch_size)
    elapsed = time.perf_counter() - started
    facts: dict[str, object] = {
        "device": backend.device,
        "runtime": backend.runtime,
        "model_artifact_sha256": backend.model_artifact_sha256,
        "seconds_including_load": elapsed,
    }
    del backend
    release_device_memory()
    return vectors, facts


def main() -> None:
    args = parse_args()
    benchmark_provenance = collect_manifest_provenance(ROOT)
    if args.sample_size <= 0 or args.batch_size <= 0:
        raise ValueError("sample and batch sizes must be positive")
    if args.candidate_device == "mps" and select_bulk_device("auto") != "mps":
        raise DatasetIntegrityError("CPU/MPS parity preflight requires MPS")
    model = load_model_registry(ROOT / "config/models.toml")[args.model]
    products_path = ROOT / "data/prepared/wands/products.jsonl"
    inputs = load_embedding_inputs(products_path, model)
    samples = inputs[: args.sample_size]
    texts = [item.text for item in samples]

    print(f"preflighting {model.name}: {len(texts)} documents on CPU", flush=True)
    cpu_vectors, cpu_facts = run_backend(model, texts, device="cpu", batch_size=args.batch_size)
    print(
        f"preflighting {model.name}: {len(texts)} documents on {args.candidate_device}",
        flush=True,
    )
    candidate_vectors, candidate_facts = run_backend(
        model,
        texts,
        device=args.candidate_device,
        batch_size=args.batch_size,
    )
    output_path = (
        ROOT / f"results/wands/embeddings/{model.name}.preflight.{args.candidate_device}.json"
    )
    try:
        comparison = compare_embedding_vectors(cpu_vectors, candidate_vectors)
    except DatasetIntegrityError as comparison_error:
        output = finalize_embedding_preflight_artifact(
            root=ROOT,
            products_path=products_path,
            model=model,
            candidate_device=args.candidate_device,
            benchmark_provenance=benchmark_provenance,
            measurements={
                "status": "failed",
                "sample_size": len(samples),
                "sample_cache_keys_sha256": canonical_sha256(
                    [item.cache_key for item in samples]
                ),
                "cpu": cpu_facts,
                "candidate": candidate_facts,
                "error": str(comparison_error),
            },
        )
        write_json(
            output_path,
            output,
        )
        verify_embedding_preflight_artifact(
            root=ROOT,
            products_path=products_path,
            model=model,
            candidate_device=args.candidate_device,
        )
        raise
    failure_reason: str | None = None
    if cpu_facts["model_artifact_sha256"] != candidate_facts["model_artifact_sha256"]:
        failure_reason = "CPU and MPS loaded different model artifacts"
    elif comparison["minimum_cosine_similarity"] < PREFLIGHT_MINIMUM_COSINE:
        failure_reason = f"CPU/MPS cosine parity failed for {model.name}: {comparison}"
    elif comparison["maximum_absolute_delta"] > PREFLIGHT_MAXIMUM_ABSOLUTE_DELTA:
        failure_reason = f"CPU/MPS absolute parity failed for {model.name}: {comparison}"

    output = finalize_embedding_preflight_artifact(
        root=ROOT,
        products_path=products_path,
        model=model,
        candidate_device=args.candidate_device,
        benchmark_provenance=benchmark_provenance,
        measurements={
            "status": "failed" if failure_reason is not None else "passed",
            "sample_size": len(samples),
            "sample_cache_keys_sha256": canonical_sha256(
                [item.cache_key for item in samples]
            ),
            "comparison": comparison,
            "cpu": cpu_facts,
            "candidate": candidate_facts,
            "error": failure_reason,
        },
    )
    write_json(output_path, output)
    verify_embedding_preflight_artifact(
        root=ROOT,
        products_path=products_path,
        model=model,
        candidate_device=args.candidate_device,
    )
    if failure_reason is not None:
        raise DatasetIntegrityError(failure_reason)
    print(
        f"{model.name} CPU/{args.candidate_device} parity passed: min cosine "
        f"{comparison['minimum_cosine_similarity']:.9f}, max delta "
        f"{comparison['maximum_absolute_delta']:.9f}"
    )


if __name__ == "__main__":
    main()
