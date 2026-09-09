from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from poc.datasets import DatasetIntegrityError, file_facts
from poc.manifest import read_json, write_json
from poc.provenance import collect_manifest_provenance, verify_decision_provenance
from poc.trec_product_search import iter_trec_products
from poc.trec_sparse import (
    TrecSparseSpec,
    iter_sparse_checkpoint_lines,
    load_trec_sparse_config,
    trec_sparse_document_text,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/trec_neural_sparse.toml"
MODEL_DIRECTORY = ROOT / "data/cache/models/neural-sparse/doc-v2-distill"
MODEL_PATH = (
    MODEL_DIRECTORY / "opensearch-neural-sparse-encoding-doc-v2-distill.pt"
)
TOKENIZER_PATH = MODEL_DIRECTORY / "tokenizer.json"
PACKAGE_PATH = ROOT / "data/cache/models/neural-sparse/doc-v2-distill.zip"
CORPUS_PATH = ROOT / "data/raw/trec-product-search-2024/collection.trec.gz"
OUTPUT_DIRECTORY = ROOT / "data/cache/neural-sparse"
RESULT_DIRECTORY = ROOT / "results/trec-product-search/neural-sparse"
PROFILE_PATH = ROOT / "config/benchmark-2.19.toml"
ENVIRONMENT_PATH = (
    ROOT / "results/environment/benchmark-profile-opensearch-2.19.json"
)
CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_ARTIFACT_TYPE = "trec_sparse_precompute_checkpoint_binding"
CHECKPOINT_RECORD_INTERVAL = 10_000


def main() -> None:
    generation_provenance = collect_manifest_provenance(
        ROOT,
        profile_path=PROFILE_PATH,
        environment_path=ENVIRONMENT_PATH,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-documents", type=int)
    args = parser.parse_args()
    if args.preflight_documents is not None and args.preflight_documents <= 0:
        parser.error("--preflight-documents must be positive")
    spec = load_trec_sparse_config(CONFIG_PATH)
    _verify_official_package(spec)
    suffix = "preflight" if args.preflight_documents is not None else "full"
    output = OUTPUT_DIRECTORY / f"trec-product-search-doc-v2-distill-{suffix}.jsonl"
    manifest_path = RESULT_DIRECTORY / f"precompute-{suffix}-manifest.json"
    result = precompute(
        spec,
        output=output,
        manifest_path=manifest_path,
        limit=cast(int | None, args.preflight_documents),
        generation_provenance=generation_provenance,
    )
    print(
        f"precomputed {result['records']} TREC sparse documents at "
        f"{cast(float, result['documents_per_second']):.1f} docs/s"
    )


def precompute(
    spec: TrecSparseSpec,
    *,
    output: Path,
    manifest_path: Path,
    limit: int | None,
    generation_provenance: dict[str, Any],
) -> dict[str, object]:
    checkpoint_binding = _checkpoint_binding(
        spec,
        output=output,
        limit=limit,
        generation_provenance=generation_provenance,
    )
    temporary = output.with_name(f".{output.name}.tmp")
    checkpoint_path = _checkpoint_sidecar_path(output)
    resume_attempted = temporary.exists()
    resume_verified = False
    recorded_checkpoint_state: dict[str, object] | None = None
    if resume_attempted:
        if not checkpoint_path.is_file():
            raise DatasetIntegrityError(
                "TREC sparse resume checkpoint sidecar is missing"
            )
        try:
            recorded_checkpoint = read_json(checkpoint_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DatasetIntegrityError(
                "TREC sparse resume checkpoint sidecar is not readable JSON"
            ) from error
        recorded_checkpoint_state = _checkpoint_state(
            recorded_checkpoint,
            expected_binding=checkpoint_binding,
        )
        recorded_bytes = cast(int, recorded_checkpoint_state["bytes"])
        actual_bytes = temporary.stat().st_size
        if actual_bytes < recorded_bytes:
            raise DatasetIntegrityError(
                "TREC sparse resume checkpoint is shorter than its attested prefix"
            )
        if actual_bytes > recorded_bytes:
            with temporary.open("r+b") as handle:
                handle.truncate(recorded_bytes)
    else:
        write_json(
            checkpoint_path,
            _checkpoint_payload(
                checkpoint_binding,
                records=0,
                bytes_written=0,
                stream_sha256=hashlib.sha256().hexdigest(),
            ),
        )

    torch = importlib.import_module("torch")
    tokenizers = importlib.import_module("tokenizers")
    tokenizer: Any = tokenizers.Tokenizer.from_file(str(TOKENIZER_PATH))
    tokenizer.enable_truncation(max_length=spec.maximum_token_length)
    tokenizer.enable_padding()
    model = torch.jit.load(str(MODEL_PATH), map_location="cpu").eval().float()
    output_tokens = [
        tokenizer.id_to_token(index) for index in range(tokenizer.get_vocab_size())
    ]
    if any(token is None for token in output_tokens):
        raise DatasetIntegrityError("TREC sparse tokenizer has unmapped token IDs")

    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    pending: list[tuple[str, str]] = []
    products = iter_trec_products(CORPUS_PATH)
    records = 0
    if temporary.exists():
        for checkpoint_product_id, line in iter_sparse_checkpoint_lines(temporary):
            try:
                corpus_product_id, _ = next(products)
            except StopIteration as error:
                raise DatasetIntegrityError(
                    "sparse checkpoint contains more records than the source corpus"
                ) from error
            if checkpoint_product_id != corpus_product_id:
                raise DatasetIntegrityError(
                    "sparse checkpoint product order differs from the source corpus"
                )
            digest.update(line)
            records += 1
        if limit is not None and records > limit:
            raise DatasetIntegrityError(
                "sparse checkpoint exceeds the requested record limit"
            )
        assert recorded_checkpoint_state is not None
        if (
            recorded_checkpoint_state["records"] != records
            or recorded_checkpoint_state["bytes"] != temporary.stat().st_size
            or recorded_checkpoint_state["stream_sha256"] != digest.hexdigest()
        ):
            raise DatasetIntegrityError(
                "TREC sparse resume checkpoint prefix attestation differs"
            )
        resume_verified = True
    resumed_records = records
    last_checkpoint_records = records
    started = time.perf_counter()
    with temporary.open("a", encoding="utf-8") as handle:
        for product_id, document in products:
            if limit is not None and records >= limit:
                break
            pending.append((product_id, trec_sparse_document_text(document)))
            target_batch_size = (
                spec.inference_batch_size
                if limit is None
                else min(spec.inference_batch_size, limit - records)
            )
            if len(pending) == target_batch_size:
                records += _encode_batch(
                    pending,
                    model=model,
                    tokenizer=tokenizer,
                    torch=torch,
                    output_tokens=output_tokens,
                    maximum_value_ratio=spec.maximum_value_ratio,
                    handle=handle,
                    digest=digest,
                )
                pending.clear()
                if records - last_checkpoint_records >= CHECKPOINT_RECORD_INTERVAL:
                    _write_checkpoint(
                        handle,
                        temporary=temporary,
                        checkpoint_path=checkpoint_path,
                        checkpoint_binding=checkpoint_binding,
                        records=records,
                        digest=digest,
                    )
                    last_checkpoint_records = records
            if limit is not None and records >= limit:
                break
        if pending and (limit is None or records < limit):
            remaining = pending if limit is None else pending[: limit - records]
            records += _encode_batch(
                remaining,
                model=model,
                tokenizer=tokenizer,
                torch=torch,
                output_tokens=output_tokens,
                maximum_value_ratio=spec.maximum_value_ratio,
                handle=handle,
                digest=digest,
            )
        _write_checkpoint(
            handle,
            temporary=temporary,
            checkpoint_path=checkpoint_path,
            checkpoint_binding=checkpoint_binding,
            records=records,
            digest=digest,
        )
    temporary.replace(output)
    elapsed = time.perf_counter() - started
    encoded_records = records - resumed_records
    documents_per_second = encoded_records / elapsed if encoded_records else 0.0
    generation_quality_eligible = (
        limit is None
        and (not resume_attempted or resume_verified)
        and verify_decision_provenance(
            generation_provenance,
            root=ROOT,
            profile_path=PROFILE_PATH,
            environment_path=ENVIRONMENT_PATH,
            require_current_code_revision=True,
        )
    )
    result: dict[str, object] = {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "generation_provenance": generation_provenance,
        "benchmark_provenance": generation_provenance,
        "quality_evidence_eligible_for_decision": generation_quality_eligible,
        "dataset": "TREC Product Search 2024",
        "scope": "preflight" if limit is not None else "full_corpus",
        "record_limit": limit,
        "document_model": {
            "name": spec.document_model_name,
            "version": spec.document_model_version,
            "format": spec.document_model_format,
            "package_url": spec.document_model_url,
            "package": _artifact(PACKAGE_PATH),
            "torchscript": _artifact(MODEL_PATH),
            "tokenizer": _artifact(TOKENIZER_PATH),
        },
        "source_corpus": _artifact(CORPUS_PATH),
        "runtime": "torchscript_cpu",
        "batch_size": spec.inference_batch_size,
        "maximum_token_length": spec.maximum_token_length,
        "pruning": {
            "type": "max_ratio",
            "maximum_value_ratio": spec.maximum_value_ratio,
            "implementation": "local_precomputed_equivalent",
        },
        "document_recipe": "title.description",
        "records": records,
        "resumed_records": resumed_records,
        "checkpoint_resume_verified": resume_verified,
        "checkpoint_binding": _artifact(checkpoint_path),
        "encoded_records_this_session": encoded_records,
        "elapsed_seconds": elapsed,
        "documents_per_second": documents_per_second,
        "throughput_scope": "current_process_session",
        "minimum_preflight_documents_per_second": (
            spec.minimum_preflight_documents_per_second
        ),
        "meets_preflight_throughput": (
            resumed_records == 0
            and documents_per_second >= spec.minimum_preflight_documents_per_second
        ),
        "embeddings": {
            **_artifact(output),
            "stream_sha256": digest.hexdigest(),
        },
    }
    write_json(manifest_path, result)
    return result


def _checkpoint_sidecar_path(output: Path) -> Path:
    return output.with_name(f".{output.name}.checkpoint.json")


def _checkpoint_binding(
    spec: TrecSparseSpec,
    *,
    output: Path,
    limit: int | None,
    generation_provenance: dict[str, Any],
) -> dict[str, object]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "artifact_type": CHECKPOINT_ARTIFACT_TYPE,
        "dataset": "TREC Product Search 2024",
        "scope": "preflight" if limit is not None else "full_corpus",
        "record_limit": limit,
        "output_filename": output.name,
        "generation_provenance": generation_provenance,
        "source_corpus": _artifact(CORPUS_PATH),
        "document_model": {
            "name": spec.document_model_name,
            "version": spec.document_model_version,
            "format": spec.document_model_format,
            "package_url": spec.document_model_url,
            "package": _artifact(PACKAGE_PATH),
            "torchscript": _artifact(MODEL_PATH),
            "tokenizer": _artifact(TOKENIZER_PATH),
        },
        "runtime": "torchscript_cpu",
        "batch_size": spec.inference_batch_size,
        "maximum_token_length": spec.maximum_token_length,
        "minimum_preflight_documents_per_second": (
            spec.minimum_preflight_documents_per_second
        ),
        "pruning": {
            "type": "max_ratio",
            "maximum_value_ratio": spec.maximum_value_ratio,
            "implementation": "local_precomputed_equivalent",
        },
        "document_recipe": "title.description",
    }


def _checkpoint_payload(
    binding: dict[str, object],
    *,
    records: int,
    bytes_written: int,
    stream_sha256: str,
) -> dict[str, object]:
    return {
        **binding,
        "checkpoint_prefix": {
            "records": records,
            "bytes": bytes_written,
            "stream_sha256": stream_sha256,
        },
    }


def _checkpoint_state(
    value: object,
    *,
    expected_binding: dict[str, object],
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise DatasetIntegrityError("TREC sparse resume checkpoint must be an object")
    binding = dict(value)
    state = binding.pop("checkpoint_prefix", None)
    if binding != expected_binding:
        raise DatasetIntegrityError("TREC sparse resume checkpoint binding differs")
    if not isinstance(state, dict) or set(state) != {
        "records",
        "bytes",
        "stream_sha256",
    }:
        raise DatasetIntegrityError("TREC sparse resume checkpoint prefix is missing")
    records = state.get("records")
    bytes_written = state.get("bytes")
    stream_sha256 = state.get("stream_sha256")
    if (
        not isinstance(records, int)
        or isinstance(records, bool)
        or records < 0
        or not isinstance(bytes_written, int)
        or isinstance(bytes_written, bool)
        or bytes_written < 0
        or not isinstance(stream_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", stream_sha256) is None
    ):
        raise DatasetIntegrityError("TREC sparse resume checkpoint prefix is invalid")
    return cast(dict[str, object], state)


def _write_checkpoint(
    handle: Any,
    *,
    temporary: Path,
    checkpoint_path: Path,
    checkpoint_binding: dict[str, object],
    records: int,
    digest: Any,
) -> None:
    handle.flush()
    os.fsync(handle.fileno())
    write_json(
        checkpoint_path,
        _checkpoint_payload(
            checkpoint_binding,
            records=records,
            bytes_written=temporary.stat().st_size,
            stream_sha256=digest.hexdigest(),
        ),
    )


def _encode_batch(
    batch: list[tuple[str, str]],
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    output_tokens: list[str | None],
    maximum_value_ratio: float,
    handle: Any,
    digest: Any,
) -> int:
    encoded = tokenizer.encode_batch([text for _, text in batch])
    inputs = {
        "input_ids": torch.tensor([item.ids for item in encoded], dtype=torch.long),
        "attention_mask": torch.tensor(
            [item.attention_mask for item in encoded], dtype=torch.long
        ),
    }
    with torch.inference_mode():
        values = model(inputs)["output"].detach().cpu().numpy()
    if values.shape != (len(batch), len(output_tokens)) or not np.isfinite(values).all():
        raise DatasetIntegrityError("official TREC sparse model returned invalid output")
    for (product_id, _), row in zip(batch, values, strict=True):
        threshold = float(row.max()) * maximum_value_ratio
        indexes = np.flatnonzero((row > 0.0) & (row >= threshold))
        embedding = {
            str(output_tokens[index]): float(row[index])
            for index in indexes
            if output_tokens[index] is not None
        }
        if not embedding:
            raise DatasetIntegrityError(
                f"official TREC sparse model returned no tokens for {product_id}"
            )
        line = json.dumps(
            {"product_id": product_id, "embedding": embedding},
            separators=(",", ":"),
            sort_keys=True,
        ) + "\n"
        handle.write(line)
        digest.update(line.encode())
    return len(batch)


def _verify_official_package(spec: TrecSparseSpec) -> None:
    facts = file_facts(PACKAGE_PATH)
    if facts.sha256 != spec.document_model_sha256:
        raise DatasetIntegrityError("official TREC sparse model package hash differs")
    if facts.bytes != spec.document_model_bytes:
        raise DatasetIntegrityError("official TREC sparse model package size differs")


def _artifact(path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }


if __name__ == "__main__":
    main()
