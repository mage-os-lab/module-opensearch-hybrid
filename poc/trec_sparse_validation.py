from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from poc.datasets import (
    DatasetIntegrityError,
    TrecProductSearchDatasetSpec,
    file_facts,
)
from poc.manifest import canonical_sha256, read_json, write_json
from poc.provenance import collect_manifest_provenance, verify_decision_provenance
from poc.trec_product_search import iter_trec_products
from poc.trec_sparse import TrecSparseSpec, trec_sparse_document_text

VALIDATION_SCHEMA_VERSION = 2
VALIDATION_ARTIFACT_TYPE = "trec_sparse_full_stream_validation"
CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_ARTIFACT_TYPE = "trec_sparse_precompute_checkpoint_binding"
DEFAULT_SAMPLE_SIZE = 1000
SAMPLE_ALGORITHM = "evenly_spaced_zero_based_v1"
SAMPLE_COMPARISON = "exact_canonical_json_values"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_IMPLEMENTATION_PATHS = (
    "poc/datasets.py",
    "poc/manifest.py",
    "poc/provenance.py",
    "poc/trec_product_search.py",
    "poc/trec_sparse.py",
    "poc/trec_sparse_validation.py",
    "scripts/32_precompute_trec_sparse.py",
    "scripts/32_validate_trec_sparse.py",
    "scripts/32_verify_trec_sparse_validation.py",
)
_RECORD_SCHEMA: dict[str, object] = {
    "encoding": "utf-8",
    "framing": "json_lines_with_terminal_lf_per_record",
    "top_level_keys": ["embedding", "product_id"],
    "product_id": "nonempty_string_equal_to_corpus_position",
    "embedding": "nonempty_object",
    "embedding_token": "nonempty_string",
    "embedding_weight": "finite_positive_json_number_excluding_boolean",
}


@dataclass(frozen=True, slots=True)
class SampledSparseRecord:
    position: int
    product_id: str
    document_text: str
    canonical_line: bytes


@dataclass(frozen=True, slots=True)
class SparseStreamScan:
    records: int
    corpus_records: int
    embedding_values: int
    embeddings_bytes: int
    embeddings_sha256: str
    samples: tuple[SampledSparseRecord, ...]


@dataclass(frozen=True, slots=True)
class CompletionBindings:
    manifest: dict[str, Any]
    completion_manifest: dict[str, object]
    source_corpus: dict[str, object]
    embeddings: dict[str, object]
    document_model: dict[str, object]
    generation_provenance_status: str
    completion_provenance_interpretation: str
    generation_provenance_eligible_for_decision: bool
    completion_quality_eligible_for_decision: bool
    quality_ineligibility_reason: str | None


def deterministic_sample_positions(
    total_records: int,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> tuple[int, ...]:
    if total_records <= 0 or sample_size <= 0:
        raise ValueError("TREC sparse record and sample counts must be positive")
    if sample_size > total_records:
        raise ValueError("TREC sparse sample size cannot exceed the record count")
    if sample_size == 1:
        return (0,)
    return tuple(
        index * (total_records - 1) // (sample_size - 1)
        for index in range(sample_size)
    )


def scan_trec_sparse_stream(
    corpus_path: Path,
    embeddings_path: Path,
    *,
    expected_records: int,
    sample_positions: Sequence[int],
) -> SparseStreamScan:
    expected_positions = tuple(sample_positions)
    if expected_positions != tuple(sorted(set(expected_positions))):
        raise ValueError("TREC sparse sample positions must be sorted and unique")
    if expected_positions and (
        expected_positions[0] < 0 or expected_positions[-1] >= expected_records
    ):
        raise ValueError("TREC sparse sample position is outside the expected stream")
    selected = set(expected_positions)
    samples: list[SampledSparseRecord] = []
    digest = hashlib.sha256()
    byte_count = 0
    embedding_values = 0
    corpus_records = 0
    with embeddings_path.open("rb") as embeddings:
        for position, (product_id, document) in enumerate(
            iter_trec_products(corpus_path)
        ):
            corpus_records += 1
            line = embeddings.readline()
            if not line:
                raise DatasetIntegrityError(
                    "TREC sparse embeddings end before the source corpus"
                )
            digest.update(line)
            byte_count += len(line)
            record = _parse_sparse_record(line, position=position)
            recorded_product_id = cast(str, record["product_id"])
            if recorded_product_id != product_id:
                raise DatasetIntegrityError(
                    "TREC sparse product order differs at zero-based position "
                    f"{position}: expected {product_id!r}, found "
                    f"{recorded_product_id!r}"
                )
            embedding = cast(dict[str, float], record["embedding"])
            embedding_values += len(embedding)
            if position in selected:
                samples.append(
                    SampledSparseRecord(
                        position=position,
                        product_id=product_id,
                        document_text=trec_sparse_document_text(document),
                        canonical_line=_canonical_sparse_line(product_id, embedding),
                    )
                )
        extra = embeddings.readline()
        if extra:
            raise DatasetIntegrityError(
                "TREC sparse embeddings contain records beyond the source corpus"
            )
    if corpus_records != expected_records:
        raise DatasetIntegrityError(
            "TREC sparse source corpus record count differs: "
            f"expected {expected_records}, found {corpus_records}"
        )
    if len(samples) != len(expected_positions):
        raise DatasetIntegrityError("TREC sparse validation sample is incomplete")
    return SparseStreamScan(
        records=corpus_records,
        corpus_records=corpus_records,
        embedding_values=embedding_values,
        embeddings_bytes=byte_count,
        embeddings_sha256=digest.hexdigest(),
        samples=tuple(samples),
    )


def validate_recovered_trec_sparse(
    *,
    root: Path,
    dataset_spec: TrecProductSearchDatasetSpec,
    sparse_spec: TrecSparseSpec,
    corpus_path: Path,
    embeddings_path: Path,
    completion_manifest_path: Path,
    package_path: Path,
    model_path: Path,
    tokenizer_path: Path,
    validation_manifest_path: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> dict[str, object]:
    profile_path = root / "config/benchmark-2.19.toml"
    environment_path = (
        root / "results/environment/benchmark-profile-opensearch-2.19.json"
    )
    starting_provenance = collect_manifest_provenance(
        root,
        profile_path=profile_path,
        environment_path=environment_path,
    )
    _require_validator_provenance(
        starting_provenance,
        root=root,
        profile_path=profile_path,
        environment_path=environment_path,
        require_current_code_revision=True,
    )
    starting_fingerprints = _implementation_fingerprints(root)
    completion = _verify_completion_and_artifacts(
        root=root,
        dataset_spec=dataset_spec,
        sparse_spec=sparse_spec,
        corpus_path=corpus_path,
        embeddings_path=embeddings_path,
        completion_manifest_path=completion_manifest_path,
        package_path=package_path,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )
    positions = deterministic_sample_positions(
        dataset_spec.expected_products, sample_size
    )
    scan = scan_trec_sparse_stream(
        corpus_path,
        embeddings_path,
        expected_records=dataset_spec.expected_products,
        sample_positions=positions,
    )
    expected_embeddings = completion.embeddings
    if (
        scan.embeddings_sha256 != expected_embeddings["sha256"]
        or scan.embeddings_bytes != expected_embeddings["bytes"]
    ):
        raise DatasetIntegrityError(
            "TREC sparse full scan facts differ from the embeddings artifact"
        )
    reencoded_lines, runtime = reencode_trec_sparse_sample(
        scan.samples,
        sparse_spec=sparse_spec,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )
    reencoded = tuple(reencoded_lines)
    if len(reencoded) != len(scan.samples):
        raise DatasetIntegrityError(
            "TREC sparse sample re-encoding returned the wrong record count"
        )
    for sample, actual in zip(scan.samples, reencoded, strict=True):
        if actual != sample.canonical_line:
            raise DatasetIntegrityError(
                "TREC sparse sample re-encoding differs at zero-based position "
                f"{sample.position} for product {sample.product_id}"
            )
    finishing_provenance = collect_manifest_provenance(
        root,
        profile_path=profile_path,
        environment_path=environment_path,
    )
    _require_validator_provenance(
        finishing_provenance,
        root=root,
        profile_path=profile_path,
        environment_path=environment_path,
        require_current_code_revision=True,
    )
    if (
        finishing_provenance.get("code_revision")
        != starting_provenance.get("code_revision")
    ):
        raise DatasetIntegrityError(
            "TREC sparse validator source revision changed during validation"
        )
    finishing_fingerprints = _implementation_fingerprints(root)
    if finishing_fingerprints != starting_fingerprints:
        raise DatasetIntegrityError(
            "TREC sparse validator implementation changed during validation"
        )
    checks = {
        "completion_manifest_binding": True,
        "official_model_artifact_binding": True,
        "full_structural_scan": True,
        "deterministic_sample_reencoding": True,
        "clean_validator_provenance": True,
        "implementation_fingerprints": True,
    }
    all_checks_passed = all(checks.values())
    quality_eligible = (
        all_checks_passed
        and completion.generation_provenance_eligible_for_decision
        and completion.completion_quality_eligible_for_decision
    )
    result: dict[str, object] = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "artifact_type": VALIDATION_ARTIFACT_TYPE,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": "TREC Product Search 2024",
        "source_revision": dataset_spec.revision,
        "generation_provenance_status": completion.generation_provenance_status,
        "completion_manifest_provenance_interpretation": (
            completion.completion_provenance_interpretation
        ),
        "completion_manifest": completion.completion_manifest,
        "source_corpus": {
            **completion.source_corpus,
            "records": dataset_spec.expected_products,
        },
        "embeddings": {
            **expected_embeddings,
            "records": scan.records,
        },
        "document_model": completion.document_model,
        "completion_schema_version": completion.manifest["schema_version"],
        "record_schema": _RECORD_SCHEMA,
        "implementation_fingerprints": starting_fingerprints,
        "structural_scan": {
            "passed": True,
            "records": scan.records,
            "corpus_records": scan.corpus_records,
            "embedding_values": scan.embedding_values,
            "product_order_matches": True,
            "all_embeddings_nonempty": True,
            "all_weights_finite_positive": True,
            "stream_bytes": scan.embeddings_bytes,
            "stream_sha256": scan.embeddings_sha256,
        },
        "sample_reencoding": {
            "passed": True,
            "algorithm": SAMPLE_ALGORITHM,
            "comparison": SAMPLE_COMPARISON,
            "requested_records": sample_size,
            "compared_records": len(scan.samples),
            "positions": list(positions),
            "positions_sha256": canonical_sha256(list(positions)),
            "product_ids_sha256": canonical_sha256(
                [sample.product_id for sample in scan.samples]
            ),
            "runtime": dict(runtime),
        },
        "checks": checks,
        "all_checks_passed": all_checks_passed,
        "generation_provenance_eligible_for_decision": (
            completion.generation_provenance_eligible_for_decision
        ),
        "completion_quality_evidence_eligible_for_decision": (
            completion.completion_quality_eligible_for_decision
        ),
        "quality_evidence_eligible_for_decision": quality_eligible,
        "quality_ineligibility_reason": (
            None if quality_eligible else completion.quality_ineligibility_reason
        ),
        "latency_evidence_eligible_for_decision": False,
        "benchmark_provenance": finishing_provenance,
    }
    write_json(validation_manifest_path, result)
    return result


def verify_trec_sparse_validation_manifest(
    *,
    root: Path,
    dataset_spec: TrecProductSearchDatasetSpec,
    sparse_spec: TrecSparseSpec,
    corpus_path: Path,
    embeddings_path: Path,
    completion_manifest_path: Path,
    package_path: Path,
    model_path: Path,
    tokenizer_path: Path,
    validation_manifest_path: Path,
    expected_sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> dict[str, Any]:
    if not validation_manifest_path.is_file():
        raise DatasetIntegrityError(
            "TREC sparse validation manifest is required before indexing"
        )
    completion = _verify_completion_and_artifacts(
        root=root,
        dataset_spec=dataset_spec,
        sparse_spec=sparse_spec,
        corpus_path=corpus_path,
        embeddings_path=embeddings_path,
        completion_manifest_path=completion_manifest_path,
        package_path=package_path,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )
    manifest = _read_manifest_object(
        validation_manifest_path, label="validation"
    )
    if (
        manifest.get("schema_version") != VALIDATION_SCHEMA_VERSION
        or manifest.get("artifact_type") != VALIDATION_ARTIFACT_TYPE
    ):
        raise DatasetIntegrityError("unsupported TREC sparse validation schema")
    if (
        manifest.get("dataset") != "TREC Product Search 2024"
        or manifest.get("source_revision") != dataset_spec.revision
    ):
        raise DatasetIntegrityError("TREC sparse validation dataset binding differs")
    if (
        manifest.get("generation_provenance_status")
        != completion.generation_provenance_status
        or manifest.get("completion_manifest_provenance_interpretation")
        != completion.completion_provenance_interpretation
    ):
        raise DatasetIntegrityError(
            "TREC sparse completion-time provenance interpretation differs"
        )
    expected_artifacts: tuple[tuple[str, dict[str, object]], ...] = (
        ("completion_manifest", completion.completion_manifest),
        (
            "source_corpus",
            {
                **completion.source_corpus,
                "records": dataset_spec.expected_products,
            },
        ),
        (
            "embeddings",
            {
                **completion.embeddings,
                "records": dataset_spec.expected_products,
            },
        ),
    )
    for name, expected in expected_artifacts:
        if manifest.get(name) != expected:
            raise DatasetIntegrityError(
                f"TREC sparse validation {name.replace('_', ' ')} artifact differs"
            )
    if manifest.get("document_model") != completion.document_model:
        raise DatasetIntegrityError("TREC sparse validation document model differs")
    if manifest.get("completion_schema_version") != completion.manifest.get(
        "schema_version"
    ):
        raise DatasetIntegrityError("TREC sparse completion schema binding differs")
    if manifest.get("record_schema") != _RECORD_SCHEMA:
        raise DatasetIntegrityError("TREC sparse validation record schema differs")
    if manifest.get("implementation_fingerprints") != _implementation_fingerprints(
        root
    ):
        raise DatasetIntegrityError(
            "TREC sparse validator implementation fingerprints differ"
        )
    structural = manifest.get("structural_scan")
    if not isinstance(structural, Mapping):
        raise DatasetIntegrityError("TREC sparse structural scan is missing")
    expected_structural_values: dict[str, object] = {
        "passed": True,
        "records": dataset_spec.expected_products,
        "corpus_records": dataset_spec.expected_products,
        "product_order_matches": True,
        "all_embeddings_nonempty": True,
        "all_weights_finite_positive": True,
        "stream_bytes": completion.embeddings["bytes"],
        "stream_sha256": completion.embeddings["sha256"],
    }
    if any(structural.get(key) != value for key, value in expected_structural_values.items()):
        raise DatasetIntegrityError("TREC sparse structural validation differs")
    embedding_values = structural.get("embedding_values")
    if (
        not isinstance(embedding_values, int)
        or isinstance(embedding_values, bool)
        or embedding_values < dataset_spec.expected_products
    ):
        raise DatasetIntegrityError("TREC sparse structural weight count is invalid")
    positions = deterministic_sample_positions(
        dataset_spec.expected_products, expected_sample_size
    )
    sample = manifest.get("sample_reencoding")
    if not isinstance(sample, Mapping):
        raise DatasetIntegrityError("TREC sparse sample validation is missing")
    expected_sample_values: dict[str, object] = {
        "passed": True,
        "algorithm": SAMPLE_ALGORITHM,
        "comparison": SAMPLE_COMPARISON,
        "requested_records": expected_sample_size,
        "compared_records": expected_sample_size,
        "positions": list(positions),
        "positions_sha256": canonical_sha256(list(positions)),
    }
    if any(sample.get(key) != value for key, value in expected_sample_values.items()):
        raise DatasetIntegrityError("TREC sparse sample validation differs")
    product_ids_sha256 = sample.get("product_ids_sha256")
    runtime = sample.get("runtime")
    if (
        not isinstance(product_ids_sha256, str)
        or _SHA256.fullmatch(product_ids_sha256) is None
        or not isinstance(runtime, Mapping)
        or not runtime
    ):
        raise DatasetIntegrityError("TREC sparse sample evidence is incomplete")
    expected_runtime: dict[str, object] = {
        "engine": "torchscript_cpu",
        "inference_batch_size": sparse_spec.inference_batch_size,
        "maximum_token_length": sparse_spec.maximum_token_length,
        "maximum_value_ratio": sparse_spec.maximum_value_ratio,
    }
    if any(runtime.get(key) != value for key, value in expected_runtime.items()):
        raise DatasetIntegrityError("TREC sparse sample runtime differs")
    for version_key in ("torch_version", "tokenizers_version", "numpy_version"):
        version = runtime.get(version_key)
        if not isinstance(version, str) or not version:
            raise DatasetIntegrityError("TREC sparse sample runtime is incomplete")

    replayed_scan = scan_trec_sparse_stream(
        corpus_path,
        embeddings_path,
        expected_records=dataset_spec.expected_products,
        sample_positions=positions,
    )
    replayed_structural_values: dict[str, object] = {
        "records": replayed_scan.records,
        "corpus_records": replayed_scan.corpus_records,
        "embedding_values": replayed_scan.embedding_values,
        "stream_bytes": replayed_scan.embeddings_bytes,
        "stream_sha256": replayed_scan.embeddings_sha256,
    }
    if any(
        structural.get(key) != value
        for key, value in replayed_structural_values.items()
    ):
        raise DatasetIntegrityError(
            "TREC sparse structural validation does not replay"
        )
    replayed_lines, replayed_runtime = reencode_trec_sparse_sample(
        replayed_scan.samples,
        sparse_spec=sparse_spec,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )
    if len(replayed_lines) != len(replayed_scan.samples):
        raise DatasetIntegrityError(
            "TREC sparse sample replay returned the wrong record count"
        )
    for sampled_record, replayed_line in zip(
        replayed_scan.samples, replayed_lines, strict=True
    ):
        if replayed_line != sampled_record.canonical_line:
            raise DatasetIntegrityError(
                "TREC sparse sample re-encoding does not replay at zero-based "
                f"position {sampled_record.position} for product "
                f"{sampled_record.product_id}"
            )
    if (
        dict(replayed_runtime) != dict(runtime)
        or sample.get("product_ids_sha256")
        != canonical_sha256(
            [sampled.product_id for sampled in replayed_scan.samples]
        )
    ):
        raise DatasetIntegrityError("TREC sparse sample evidence does not replay")
    expected_checks = {
        "completion_manifest_binding": True,
        "official_model_artifact_binding": True,
        "full_structural_scan": True,
        "deterministic_sample_reencoding": True,
        "clean_validator_provenance": True,
        "implementation_fingerprints": True,
    }
    if manifest.get("checks") != expected_checks:
        raise DatasetIntegrityError("TREC sparse validation checks differ")
    expected_quality = (
        completion.generation_provenance_eligible_for_decision
        and completion.completion_quality_eligible_for_decision
    )
    expected_reason = None if expected_quality else completion.quality_ineligibility_reason
    if (
        manifest.get("all_checks_passed") is not True
        or manifest.get("generation_provenance_eligible_for_decision")
        is not completion.generation_provenance_eligible_for_decision
        or manifest.get("completion_quality_evidence_eligible_for_decision")
        is not completion.completion_quality_eligible_for_decision
        or manifest.get("quality_evidence_eligible_for_decision") is not expected_quality
        or manifest.get("quality_ineligibility_reason") != expected_reason
        or manifest.get("latency_evidence_eligible_for_decision") is not False
    ):
        raise DatasetIntegrityError("TREC sparse validation eligibility differs")
    provenance = manifest.get("benchmark_provenance")
    if not isinstance(provenance, Mapping):
        raise DatasetIntegrityError("TREC sparse validator provenance is missing")
    _require_validator_provenance(
        provenance,
        root=root,
        profile_path=root / "config/benchmark-2.19.toml",
        environment_path=(
            root / "results/environment/benchmark-profile-opensearch-2.19.json"
        ),
        require_current_code_revision=True,
    )
    return manifest


def reencode_trec_sparse_sample(
    samples: Sequence[SampledSparseRecord],
    *,
    sparse_spec: TrecSparseSpec,
    model_path: Path,
    tokenizer_path: Path,
) -> tuple[tuple[bytes, ...], dict[str, object]]:
    torch = importlib.import_module("torch")
    tokenizers = importlib.import_module("tokenizers")
    tokenizer: Any = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=sparse_spec.maximum_token_length)
    tokenizer.enable_padding()
    model = torch.jit.load(str(model_path), map_location="cpu").eval().float()
    output_tokens = [
        tokenizer.id_to_token(index) for index in range(tokenizer.get_vocab_size())
    ]
    if any(token is None for token in output_tokens):
        raise DatasetIntegrityError("TREC sparse tokenizer has unmapped token IDs")
    lines: list[bytes] = []
    for start in range(0, len(samples), sparse_spec.inference_batch_size):
        batch = samples[start : start + sparse_spec.inference_batch_size]
        encoded = tokenizer.encode_batch([sample.document_text for sample in batch])
        inputs = {
            "input_ids": torch.tensor(
                [item.ids for item in encoded], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [item.attention_mask for item in encoded], dtype=torch.long
            ),
        }
        with torch.inference_mode():
            values = model(inputs)["output"].detach().cpu().numpy()
        if values.shape != (len(batch), len(output_tokens)) or not np.isfinite(
            values
        ).all():
            raise DatasetIntegrityError(
                "official TREC sparse model returned invalid validation output"
            )
        for sample, row in zip(batch, values, strict=True):
            threshold = float(row.max()) * sparse_spec.maximum_value_ratio
            indexes = np.flatnonzero((row > 0.0) & (row >= threshold))
            embedding = {
                str(output_tokens[index]): float(row[index])
                for index in indexes
                if output_tokens[index] is not None
            }
            if not embedding:
                raise DatasetIntegrityError(
                    "official TREC sparse model returned no validation tokens for "
                    f"{sample.product_id}"
                )
            lines.append(_canonical_sparse_line(sample.product_id, embedding))
    return tuple(lines), {
        "engine": "torchscript_cpu",
        "torch_version": str(torch.__version__),
        "tokenizers_version": str(tokenizers.__version__),
        "numpy_version": np.__version__,
        "inference_batch_size": sparse_spec.inference_batch_size,
        "maximum_token_length": sparse_spec.maximum_token_length,
        "maximum_value_ratio": sparse_spec.maximum_value_ratio,
    }


def _verify_completion_and_artifacts(
    *,
    root: Path,
    dataset_spec: TrecProductSearchDatasetSpec,
    sparse_spec: TrecSparseSpec,
    corpus_path: Path,
    embeddings_path: Path,
    completion_manifest_path: Path,
    package_path: Path,
    model_path: Path,
    tokenizer_path: Path,
) -> CompletionBindings:
    if not completion_manifest_path.is_file():
        raise DatasetIntegrityError("TREC sparse completion manifest is missing")
    manifest = _read_manifest_object(
        completion_manifest_path, label="completion"
    )
    schema_version = manifest.get("schema_version")
    if schema_version not in {1, 2}:
        raise DatasetIntegrityError("unsupported TREC sparse completion schema")
    expected_header: dict[str, object] = {
        "dataset": "TREC Product Search 2024",
        "scope": "full_corpus",
        "records": dataset_spec.expected_products,
        "record_limit": None,
        "runtime": "torchscript_cpu",
        "batch_size": sparse_spec.inference_batch_size,
        "maximum_token_length": sparse_spec.maximum_token_length,
        "minimum_preflight_documents_per_second": (
            sparse_spec.minimum_preflight_documents_per_second
        ),
        "document_recipe": "title.description",
    }
    if any(manifest.get(key) != value for key, value in expected_header.items()):
        raise DatasetIntegrityError("TREC sparse completion manifest schema differs")
    expected_pruning = {
        "type": "max_ratio",
        "maximum_value_ratio": sparse_spec.maximum_value_ratio,
        "implementation": "local_precomputed_equivalent",
    }
    if manifest.get("pruning") != expected_pruning:
        raise DatasetIntegrityError("TREC sparse completion pruning recipe differs")
    corpus_registry = dataset_spec.files.get("corpus")
    corpus_artifact = _artifact(root, corpus_path)
    if (
        corpus_registry is None
        or corpus_artifact["sha256"] != corpus_registry.sha256
        or corpus_artifact["bytes"] != corpus_registry.bytes
    ):
        raise DatasetIntegrityError("TREC sparse corpus differs from the registry")
    if manifest.get("source_corpus") != corpus_artifact:
        raise DatasetIntegrityError("TREC sparse completion corpus binding differs")
    embeddings = _artifact(root, embeddings_path)
    expected_embeddings = {**embeddings, "stream_sha256": embeddings["sha256"]}
    if manifest.get("embeddings") != expected_embeddings:
        raise DatasetIntegrityError("TREC sparse completion embeddings binding differs")
    validated_model = _validated_model_record(
        root=root,
        sparse_spec=sparse_spec,
        package_path=package_path,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )
    model_record = cast(dict[str, Any], manifest.get("document_model", {}))
    expected_model = {
        "name": sparse_spec.document_model_name,
        "version": sparse_spec.document_model_version,
        "format": sparse_spec.document_model_format,
        "package_url": sparse_spec.document_model_url,
        "package": validated_model["package"],
        "torchscript": validated_model["torchscript"],
        "tokenizer": validated_model["tokenizer"],
    }
    if model_record != expected_model:
        raise DatasetIntegrityError("TREC sparse completion model binding differs")
    generation_provenance_status = "unknown_pre_commit_process"
    provenance_interpretation = "completion_time_only_not_generator_attribution"
    generation_provenance_eligible = False
    completion_quality_eligible = False
    quality_ineligibility_reason: str | None = (
        "legacy schema v1 generation provenance is unknown"
    )
    if schema_version == 2:
        generation_provenance = manifest.get("generation_provenance")
        if not isinstance(generation_provenance, Mapping):
            raise DatasetIntegrityError(
                "TREC sparse schema v2 generation provenance is missing"
            )
        if manifest.get("benchmark_provenance") != generation_provenance:
            raise DatasetIntegrityError(
                "TREC sparse schema v2 provenance aliases differ"
            )
        generation_provenance_eligible = verify_decision_provenance(
            generation_provenance,
            root=root,
            profile_path=root / "config/benchmark-2.19.toml",
            environment_path=(
                root
                / "results/environment/benchmark-profile-opensearch-2.19.json"
            ),
        )
        completion_quality = manifest.get(
            "quality_evidence_eligible_for_decision"
        )
        if not isinstance(completion_quality, bool):
            raise DatasetIntegrityError(
                "TREC sparse schema v2 completion quality flag is missing"
            )
        if completion_quality and not generation_provenance_eligible:
            raise DatasetIntegrityError(
                "TREC sparse schema v2 completion overstates provenance eligibility"
            )
        completion_quality_eligible = completion_quality
        generation_provenance_status = (
            "verified_clean_schema_v2_start"
            if generation_provenance_eligible
            else "ineligible_schema_v2_start"
        )
        provenance_interpretation = "immutable_generation_start_snapshot"
        if not generation_provenance_eligible:
            quality_ineligibility_reason = (
                "schema v2 generation provenance is not decision eligible"
            )
        elif not completion_quality_eligible:
            quality_ineligibility_reason = (
                "schema v2 completion did not declare quality eligibility"
            )
        else:
            quality_ineligibility_reason = None
        _verify_schema_v2_checkpoint(
            root=root,
            sparse_spec=sparse_spec,
            embeddings_path=embeddings_path,
            manifest=manifest,
            generation_provenance=generation_provenance,
            corpus_artifact=corpus_artifact,
            document_model=expected_model,
        )
    return CompletionBindings(
        manifest=manifest,
        completion_manifest=_artifact(root, completion_manifest_path),
        source_corpus=corpus_artifact,
        embeddings=embeddings,
        document_model=validated_model,
        generation_provenance_status=generation_provenance_status,
        completion_provenance_interpretation=provenance_interpretation,
        generation_provenance_eligible_for_decision=(
            generation_provenance_eligible
        ),
        completion_quality_eligible_for_decision=completion_quality_eligible,
        quality_ineligibility_reason=quality_ineligibility_reason,
    )


def _validated_model_record(
    *,
    root: Path,
    sparse_spec: TrecSparseSpec,
    package_path: Path,
    model_path: Path,
    tokenizer_path: Path,
) -> dict[str, object]:
    package_artifact = _artifact(root, package_path)
    if (
        package_artifact["sha256"] != sparse_spec.document_model_sha256
        or package_artifact["bytes"] != sparse_spec.document_model_bytes
    ):
        raise DatasetIntegrityError("official TREC sparse model package differs")
    members = _official_package_member_facts(
        package_path,
        member_names=(model_path.name, tokenizer_path.name),
    )
    model_artifact = _artifact(root, model_path)
    tokenizer_artifact = _artifact(root, tokenizer_path)
    if (
        members[model_path.name]["sha256"] != model_artifact["sha256"]
        or members[model_path.name]["bytes"] != model_artifact["bytes"]
    ):
        raise DatasetIntegrityError(
            "TREC sparse TorchScript differs from the pinned official package"
        )
    if (
        members[tokenizer_path.name]["sha256"] != tokenizer_artifact["sha256"]
        or members[tokenizer_path.name]["bytes"] != tokenizer_artifact["bytes"]
    ):
        raise DatasetIntegrityError(
            "TREC sparse tokenizer differs from the pinned official package"
        )
    return {
        "name": sparse_spec.document_model_name,
        "version": sparse_spec.document_model_version,
        "format": sparse_spec.document_model_format,
        "package_url": sparse_spec.document_model_url,
        "package": package_artifact,
        "torchscript": model_artifact,
        "tokenizer": tokenizer_artifact,
        "extracted_members_match_package": True,
    }


def _verify_schema_v2_checkpoint(
    *,
    root: Path,
    sparse_spec: TrecSparseSpec,
    embeddings_path: Path,
    manifest: Mapping[str, Any],
    generation_provenance: Mapping[str, Any],
    corpus_artifact: dict[str, object],
    document_model: dict[str, object],
) -> None:
    resumed_records = manifest.get("resumed_records")
    encoded_records = manifest.get("encoded_records_this_session")
    resume_verified = manifest.get("checkpoint_resume_verified")
    records = manifest.get("records")
    if (
        not isinstance(resumed_records, int)
        or isinstance(resumed_records, bool)
        or resumed_records < 0
        or not isinstance(encoded_records, int)
        or isinstance(encoded_records, bool)
        or encoded_records < 0
        or resumed_records + encoded_records != records
        or not isinstance(resume_verified, bool)
        or (resumed_records > 0 and not resume_verified)
        or manifest.get("throughput_scope") != "current_process_session"
    ):
        raise DatasetIntegrityError("TREC sparse schema v2 resume evidence differs")
    checkpoint_path = embeddings_path.with_name(
        f".{embeddings_path.name}.checkpoint.json"
    )
    checkpoint_artifact = _artifact(root, checkpoint_path)
    if manifest.get("checkpoint_binding") != checkpoint_artifact:
        raise DatasetIntegrityError(
            "TREC sparse schema v2 checkpoint artifact binding differs"
        )
    checkpoint = _read_manifest_object(checkpoint_path, label="checkpoint")
    expected = _expected_checkpoint_binding(
        sparse_spec=sparse_spec,
        embeddings_path=embeddings_path,
        generation_provenance=generation_provenance,
        corpus_artifact=corpus_artifact,
        document_model=document_model,
    )
    embeddings = _artifact(root, embeddings_path)
    expected["checkpoint_prefix"] = {
        "records": records,
        "bytes": embeddings["bytes"],
        "stream_sha256": embeddings["sha256"],
    }
    if checkpoint != expected:
        raise DatasetIntegrityError(
            "TREC sparse schema v2 checkpoint content binding differs"
        )


def _expected_checkpoint_binding(
    *,
    sparse_spec: TrecSparseSpec,
    embeddings_path: Path,
    generation_provenance: Mapping[str, Any],
    corpus_artifact: dict[str, object],
    document_model: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "artifact_type": CHECKPOINT_ARTIFACT_TYPE,
        "dataset": "TREC Product Search 2024",
        "scope": "full_corpus",
        "record_limit": None,
        "output_filename": embeddings_path.name,
        "generation_provenance": dict(generation_provenance),
        "source_corpus": corpus_artifact,
        "document_model": document_model,
        "runtime": "torchscript_cpu",
        "batch_size": sparse_spec.inference_batch_size,
        "maximum_token_length": sparse_spec.maximum_token_length,
        "minimum_preflight_documents_per_second": (
            sparse_spec.minimum_preflight_documents_per_second
        ),
        "pruning": {
            "type": "max_ratio",
            "maximum_value_ratio": sparse_spec.maximum_value_ratio,
            "implementation": "local_precomputed_equivalent",
        },
        "document_recipe": "title.description",
    }


def _official_package_member_facts(
    package_path: Path,
    *,
    member_names: Sequence[str],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    try:
        with zipfile.ZipFile(package_path) as archive:
            names = archive.namelist()
            for member_name in member_names:
                matches = [name for name in names if Path(name).name == member_name]
                if len(matches) != 1:
                    raise DatasetIntegrityError(
                        "official TREC sparse package must contain exactly one "
                        f"{member_name}"
                    )
                info = archive.getinfo(matches[0])
                digest = hashlib.sha256()
                byte_count = 0
                with archive.open(info) as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                        byte_count += len(chunk)
                result[member_name] = {
                    "sha256": digest.hexdigest(),
                    "bytes": byte_count,
                }
    except (OSError, zipfile.BadZipFile) as error:
        raise DatasetIntegrityError(
            "official TREC sparse model package is not a readable ZIP"
        ) from error
    return result


def _parse_sparse_record(line: bytes, *, position: int) -> dict[str, object]:
    if not line.endswith(b"\n") or line.endswith(b"\r\n"):
        raise DatasetIntegrityError(
            "TREC sparse record must end in one LF at zero-based position "
            f"{position}"
        )
    try:
        raw = json.loads(line, object_pairs_hook=_object_without_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateKeyError) as error:
        raise DatasetIntegrityError(
            f"invalid TREC sparse JSON at zero-based position {position}"
        ) from error
    if not isinstance(raw, dict) or set(raw) != {"embedding", "product_id"}:
        raise DatasetIntegrityError(
            f"invalid TREC sparse record schema at zero-based position {position}"
        )
    product_id = raw.get("product_id")
    embedding = raw.get("embedding")
    if not isinstance(product_id, str) or not product_id:
        raise DatasetIntegrityError(
            f"invalid TREC sparse product ID at zero-based position {position}"
        )
    if not isinstance(embedding, dict) or not embedding:
        raise DatasetIntegrityError(
            f"empty TREC sparse embedding at zero-based position {position}"
        )
    validated: dict[str, float] = {}
    for token, value in embedding.items():
        if not isinstance(token, str) or not token:
            raise DatasetIntegrityError(
                "TREC sparse embedding token must be nonempty at zero-based "
                f"position {position}"
            )
        try:
            numeric = float(value)
        except (OverflowError, TypeError, ValueError):
            numeric = math.nan
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(numeric)
            or numeric <= 0.0
        ):
            raise DatasetIntegrityError(
                "TREC sparse embedding weights must be finite positive numbers at "
                f"zero-based position {position}"
            )
        validated[token] = numeric
    return {"product_id": product_id, "embedding": validated}


def _canonical_sparse_line(product_id: str, embedding: Mapping[str, float]) -> bytes:
    return (
        json.dumps(
            {"product_id": product_id, "embedding": dict(embedding)},
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


class _DuplicateKeyError(ValueError):
    pass


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(key)
        result[key] = value
    return result


def _require_clean_validator_provenance(provenance: Mapping[str, Any]) -> None:
    revision = provenance.get("code_revision")
    if not isinstance(revision, Mapping):
        raise DatasetIntegrityError("TREC sparse validator code provenance is missing")
    commit = revision.get("git_commit")
    source_tree = revision.get("source_tree_sha256")
    if (
        not isinstance(commit, str)
        or _GIT_COMMIT.fullmatch(commit) is None
        or not isinstance(source_tree, str)
        or _SHA256.fullmatch(source_tree) is None
        or revision.get("source_dirty") is not False
    ):
        raise DatasetIntegrityError(
            "TREC sparse validation requires a clean committed validator source tree"
        )


def _require_validator_provenance(
    provenance: Mapping[str, Any],
    *,
    root: Path,
    profile_path: Path,
    environment_path: Path,
    require_current_code_revision: bool,
) -> None:
    _require_clean_validator_provenance(provenance)
    if not verify_decision_provenance(
        provenance,
        root=root,
        profile_path=profile_path,
        environment_path=environment_path,
        require_current_code_revision=require_current_code_revision,
    ):
        raise DatasetIntegrityError(
            "TREC sparse validator provenance is not decision eligible"
        )


def _implementation_fingerprints(root: Path) -> dict[str, dict[str, object]]:
    fingerprints: dict[str, dict[str, object]] = {}
    for relative in _IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.is_file():
            raise DatasetIntegrityError(
                f"TREC sparse validator implementation file is missing: {relative}"
            )
        fingerprints[relative] = _artifact(root, path)
    return fingerprints


def _read_manifest_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = read_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetIntegrityError(
            f"TREC sparse {label} manifest is not readable JSON"
        ) from error
    if not isinstance(value, dict):
        raise DatasetIntegrityError(
            f"TREC sparse {label} manifest must be a JSON object"
        )
    return cast(dict[str, Any], value)


def _artifact(root: Path, path: Path) -> dict[str, object]:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise DatasetIntegrityError(
            f"TREC sparse artifact is outside the repository root: {path}"
        ) from error
    facts = file_facts(path)
    return {
        "path": relative.as_posix(),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }
