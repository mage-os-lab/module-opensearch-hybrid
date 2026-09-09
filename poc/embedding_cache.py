from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sqlite3
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from poc.config import ConfigError, ModelSpec, load_model_registry
from poc.datasets import DatasetIntegrityError, file_facts
from poc.manifest import canonical_sha256, read_json, write_json
from poc.model_runtime import (
    create_bulk_backend,
    model_snapshot_directory,
    release_device_memory,
    verify_registered_model_snapshot,
)
from poc.provenance import (
    collect_code_revision,
    collect_manifest_provenance,
    verify_decision_provenance,
)

ATTRIBUTES_RECIPE = (
    "product_class labeled as product class; category labeled as category; "
    "features labeled as features"
)
NORM_TOLERANCE = 1e-4
PREFLIGHT_MINIMUM_COSINE = 0.9999
PREFLIGHT_MAXIMUM_ABSOLUTE_DELTA = 0.002
CACHE_GENERATOR_CONTRACT = "opensearch-hybrid.embedding-cache-v2"
CACHE_TABLE = "embedding_cache_v2"
EMBEDDING_REPLAY_SAMPLE_SIZE = 256
EMBEDDING_REPLAY_BATCH_SIZE = 32
EMBEDDING_REPLAY_RELATIVE_TOLERANCE = 1e-6
EMBEDDING_REPLAY_ABSOLUTE_TOLERANCE = 1e-6


class EmbeddingBackend(Protocol):
    device: str
    runtime: str
    model_artifact_sha256: str

    def encode(self, texts: list[str]) -> NDArray[np.float32]: ...


@dataclass(frozen=True, slots=True)
class EmbeddingInput:
    document_id: str
    text: str
    cache_key: str


@dataclass(frozen=True, slots=True)
class CacheEntry:
    cache_key: str
    model_fingerprint: str
    generator_fingerprint: str
    generator_contract: str
    backend_implementation: str
    cache_implementation_sha256: str
    backend_implementation_sha256: str
    model_artifact_sha256: str
    encoder_runtime: str
    device: str
    shard_path: str
    row_index: int
    dims: int
    vector_sha256: str


@dataclass(frozen=True, slots=True)
class EmbeddingCacheSummary:
    model_name: str
    model_fingerprint: str
    document_count: int
    unique_cache_keys: int
    cache_hits: int
    encoded_vectors: int
    shard_count: int
    manifest_path: Path


def load_embedding_inputs(products_path: Path, model: ModelSpec) -> list[EmbeddingInput]:
    inputs: list[EmbeddingInput] = []
    document_ids: set[str] = set()
    model_values = cache_model_values(model)
    for line_number, line in enumerate(products_path.read_text().splitlines(), start=1):
        values = cast(dict[str, Any], json.loads(line))
        document_id = _required_string(values, "product_id", products_path, line_number)
        title = _required_string(values, "title", products_path, line_number)
        description = _optional_string(values, "description", products_path, line_number)
        product_class = _optional_string(values, "product_class", products_path, line_number)
        category = _optional_string(values, "category", products_path, line_number)
        features = _optional_string(values, "features", products_path, line_number)
        if document_id in document_ids:
            raise DatasetIntegrityError(f"duplicate product ID {document_id!r} in {products_path}")
        document_ids.add(document_id)
        attributes = ". ".join(
            value
            for value in (
                f"product class: {product_class}" if product_class else "",
                f"category: {category}" if category else "",
                f"features: {features}" if features else "",
            )
            if value
        )
        text = model.render_document(
            title=title,
            description=description,
            attributes=attributes,
        )
        cache_key = canonical_sha256({"model": model_values, "document_text": text})
        inputs.append(EmbeddingInput(document_id=document_id, text=text, cache_key=cache_key))
    if not inputs:
        raise DatasetIntegrityError(f"no embedding inputs in {products_path}")
    return inputs


def build_embedding_cache(
    *,
    root: Path,
    products_path: Path,
    model: ModelSpec,
    backend: EmbeddingBackend,
    batch_size: int = 32,
    shard_size: int = 2048,
    generation_provenance: Mapping[str, Any] | None = None,
) -> EmbeddingCacheSummary:
    benchmark_provenance = _capture_provenance(root, generation_provenance)
    if batch_size <= 0 or shard_size <= 0:
        raise ValueError("embedding batch and shard sizes must be positive")
    inputs = load_embedding_inputs(products_path, model)
    unique_inputs = _unique_inputs(inputs)
    model_fingerprint = canonical_sha256(cache_model_values(model))
    generator = cache_generator_identity(
        model_fingerprint=model_fingerprint,
        backend=backend,
    )
    _verify_backend_generator_identity(root=root, model=model, generator=generator)
    generator_fingerprint = canonical_sha256(generator)
    manifest_path = root / f"results/wands/embeddings/{model.name}.manifest.json"
    _preserve_unbound_manifest(manifest_path)
    database_path = root / "data/cache/embeddings/wands.sqlite3"
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(database_path) as connection:
        entries = _read_entries(
            connection,
            [item.cache_key for item in unique_inputs],
            generator_fingerprint=generator_fingerprint,
        )
        missing = [item for item in unique_inputs if item.cache_key not in entries]
        encoded_vectors = _encode_missing(
            connection,
            root=root,
            model=model,
            model_fingerprint=model_fingerprint,
            generator=generator,
            generator_fingerprint=generator_fingerprint,
            backend=backend,
            missing=missing,
            batch_size=batch_size,
            shard_size=shard_size,
        )
        entries = _read_entries(
            connection,
            [item.cache_key for item in unique_inputs],
            generator_fingerprint=generator_fingerprint,
        )
        if len(entries) != len(unique_inputs):
            raise DatasetIntegrityError("embedding cache did not persist every unique input")

    _write_manifest(
        root=root,
        products_path=products_path,
        model=model,
        backend=backend,
        model_fingerprint=model_fingerprint,
        generator=generator,
        generator_fingerprint=generator_fingerprint,
        inputs=inputs,
        entries=entries,
        manifest_path=manifest_path,
        cache_hits=len(unique_inputs) - len(missing),
        encoded_vectors=encoded_vectors,
        benchmark_provenance=benchmark_provenance,
    )
    verify_embedding_cache(root=root, products_path=products_path, model=model)
    return EmbeddingCacheSummary(
        model_name=model.name,
        model_fingerprint=model_fingerprint,
        document_count=len(inputs),
        unique_cache_keys=len(unique_inputs),
        cache_hits=len(unique_inputs) - len(missing),
        encoded_vectors=encoded_vectors,
        shard_count=len({entry.shard_path for entry in entries.values()}),
        manifest_path=manifest_path,
    )


def verify_embedding_cache(*, root: Path, products_path: Path, model: ModelSpec) -> None:
    _verify_registered_embedding_inputs(
        root=root,
        products_path=products_path,
        model=model,
    )
    inputs = load_embedding_inputs(products_path, model)
    unique_inputs = _unique_inputs(inputs)
    manifest_path = root / f"results/wands/embeddings/{model.name}.manifest.json"
    manifest = cast(dict[str, Any], read_json(manifest_path))
    if manifest.get("schema_version") != 2 or manifest.get("dataset") != "WANDS":
        raise DatasetIntegrityError(f"embedding manifest schema differs for {model.name}")
    model_fingerprint = canonical_sha256(cache_model_values(model))
    if manifest.get("model") != asdict(model):
        raise DatasetIntegrityError(f"embedding manifest model differs for {model.name}")
    if manifest.get("model_fingerprint") != model_fingerprint:
        raise DatasetIntegrityError(f"embedding model fingerprint differs for {model.name}")
    generator, generator_fingerprint = _manifest_generator_identity(
        root=root,
        model=model,
        manifest=manifest,
        model_fingerprint=model_fingerprint,
    )
    cache_hits = manifest.get("cache_hits")
    encoded_vectors = manifest.get("encoded_vectors")
    expected_metadata = {
        "products_path": str(products_path.relative_to(root)),
        "document_count": len(inputs),
        "unique_cache_keys": len(unique_inputs),
        "document_attributes_recipe": ATTRIBUTES_RECIPE,
        "rendered_template_samples": [
            {"document_id": item.document_id, "text": item.text}
            for item in inputs[:3]
        ],
        "dtype": "float32",
        "dimensions": model.dims,
        "normalized": model.normalize,
        "cache_table": CACHE_TABLE,
    }
    if any(manifest.get(key) != value for key, value in expected_metadata.items()):
        raise DatasetIntegrityError(f"embedding manifest metadata differs for {model.name}")
    if (
        not isinstance(cache_hits, int)
        or isinstance(cache_hits, bool)
        or cache_hits < 0
        or not isinstance(encoded_vectors, int)
        or isinstance(encoded_vectors, bool)
        or encoded_vectors < 0
        or cache_hits + encoded_vectors != len(unique_inputs)
        or not isinstance(manifest.get("encoder_runtime"), str)
        or not manifest.get("encoder_runtime")
        or not isinstance(manifest.get("device"), str)
        or not manifest.get("device")
        or not _is_sha256(manifest.get("model_artifact_sha256"))
    ):
        raise DatasetIntegrityError(f"embedding manifest metadata differs for {model.name}")
    expected_input_hash = canonical_sha256(
        [(item.document_id, item.cache_key) for item in inputs]
    )
    if manifest.get("input_cache_keys_sha256") != expected_input_hash:
        raise DatasetIntegrityError(f"embedding inputs differ for {model.name}")
    product_facts = file_facts(products_path)
    if manifest.get("products_sha256") != product_facts.sha256:
        raise DatasetIntegrityError(f"embedding product source differs for {model.name}")
    if manifest.get("products") != _artifact(root, products_path):
        raise DatasetIntegrityError(f"embedding product binding differs for {model.name}")
    if manifest.get("model_registry") != _optional_artifact(
        root, root / "config/models.toml"
    ):
        raise DatasetIntegrityError(f"embedding model registry differs for {model.name}")
    if manifest.get("model_artifact_registry") != _optional_artifact(
        root, root / "config/model_artifacts.toml"
    ):
        raise DatasetIntegrityError(
            f"embedding model artifact registry differs for {model.name}"
        )
    if manifest.get("prepared_manifest") != _optional_artifact(
        root, products_path.parent / "manifest.json"
    ):
        raise DatasetIntegrityError(f"embedding preparation manifest differs for {model.name}")
    snapshot_path = model_snapshot_directory(root, model)
    if snapshot_path.is_dir():
        current_model_sha256 = verify_registered_model_snapshot(
            root=root,
            model=model,
            snapshot=snapshot_path,
        )
        if manifest.get("model_artifact_sha256") != current_model_sha256:
            raise DatasetIntegrityError(f"embedding model bytes differ for {model.name}")

    database_path = root / "data/cache/embeddings/wands.sqlite3"
    with _connect(database_path) as connection:
        entries = _read_entries(
            connection,
            [item.cache_key for item in unique_inputs],
            generator_fingerprint=generator_fingerprint,
        )
    if len(entries) != len(unique_inputs):
        raise DatasetIntegrityError(
            f"schema-v2 embedding cache is incomplete for {model.name}; "
            "legacy rows are not eligible"
        )
    _verify_entries(
        root,
        entries,
        model,
        model_fingerprint=model_fingerprint,
        generator=generator,
        generator_fingerprint=generator_fingerprint,
    )

    recorded_shards = cast(list[dict[str, Any]], manifest.get("shards"))
    actual_shards = _shard_facts(root, entries)
    if recorded_shards != actual_shards:
        raise DatasetIntegrityError(f"embedding shard manifest differs for {model.name}")
    expected_entries_hash = canonical_sha256(
        [asdict(entries[key]) for key in sorted(entries)]
    )
    if manifest.get("cache_entries_sha256") != expected_entries_hash:
        raise DatasetIntegrityError(f"embedding cache entries differ for {model.name}")
    replay_verified = _verify_embedding_cache_replay(
        root=root,
        products_path=products_path,
        model=model,
        inputs=unique_inputs,
        entries=entries,
        generator=generator,
        manifest=manifest,
    )
    provenance_valid = _recorded_provenance_valid(
        root,
        manifest.get("benchmark_provenance"),
        manifest.get("completion_code_revision"),
    )
    device_evidence_eligible, preflight_binding = _embedding_device_evidence(
        root=root,
        products_path=products_path,
        model=model,
        device=str(manifest.get("device", "")),
        model_artifact_sha256=str(manifest.get("model_artifact_sha256", "")),
    )
    if (
        manifest.get("device_evidence_eligible") is not device_evidence_eligible
        or manifest.get("device_preflight") != preflight_binding
    ):
        raise DatasetIntegrityError(f"embedding device evidence differs for {model.name}")
    expected_eligible = (
        provenance_valid and device_evidence_eligible and replay_verified
    )
    if manifest.get("quality_evidence_eligible_for_decision") is not expected_eligible:
        raise DatasetIntegrityError(f"embedding cache eligibility differs for {model.name}")


def load_cached_vectors(
    *,
    root: Path,
    products_path: Path,
    model: ModelSpec,
) -> tuple[list[str], NDArray[np.float32]]:
    verify_embedding_cache(root=root, products_path=products_path, model=model)
    inputs = load_embedding_inputs(products_path, model)
    manifest = cast(
        dict[str, Any],
        read_json(root / f"results/wands/embeddings/{model.name}.manifest.json"),
    )
    generator_fingerprint = str(manifest["cache_generator_fingerprint"])
    database_path = root / "data/cache/embeddings/wands.sqlite3"
    with _connect(database_path) as connection:
        entries = _read_entries(
            connection,
            [item.cache_key for item in inputs],
            generator_fingerprint=generator_fingerprint,
        )
    arrays = _load_shards(root, entries)
    vectors = np.stack(
        [
            arrays[entries[item.cache_key].shard_path][entries[item.cache_key].row_index]
            for item in inputs
        ]
    ).astype(np.float32, copy=False)
    return [item.document_id for item in inputs], vectors


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    # Keep the original table untouched as recoverable legacy evidence. Schema-v2
    # manifests can only reference rows from the identity-bound table below.
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS embedding_cache (
            cache_key TEXT PRIMARY KEY,
            model_fingerprint TEXT NOT NULL,
            shard_path TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            dims INTEGER NOT NULL,
            vector_sha256 TEXT NOT NULL
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {CACHE_TABLE} (
            cache_key TEXT NOT NULL,
            model_fingerprint TEXT NOT NULL,
            generator_fingerprint TEXT NOT NULL,
            generator_contract TEXT NOT NULL,
            backend_implementation TEXT NOT NULL,
            cache_implementation_sha256 TEXT NOT NULL,
            backend_implementation_sha256 TEXT NOT NULL,
            model_artifact_sha256 TEXT NOT NULL,
            encoder_runtime TEXT NOT NULL,
            device TEXT NOT NULL,
            shard_path TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            dims INTEGER NOT NULL,
            vector_sha256 TEXT NOT NULL,
            PRIMARY KEY (cache_key, generator_fingerprint)
        )
        """
    )
    return connection


def _preserve_unbound_manifest(manifest_path: Path) -> None:
    if not manifest_path.is_file():
        return
    manifest = read_json(manifest_path)
    if (
        isinstance(manifest, dict)
        and manifest.get("cache_table") == CACHE_TABLE
        and isinstance(manifest.get("cache_generator"), dict)
        and _is_sha256(manifest.get("cache_generator_fingerprint"))
    ):
        return
    facts = file_facts(manifest_path)
    preserved_path = manifest_path.with_name(
        f"{manifest_path.stem}.legacy-unbound-{facts.sha256[:12]}.json"
    )
    if preserved_path.is_file():
        if preserved_path.read_bytes() != manifest_path.read_bytes():
            raise DatasetIntegrityError(
                f"legacy embedding manifest preservation conflict: {preserved_path}"
            )
        return
    preserved_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, preserved_path)


def _read_entries(
    connection: sqlite3.Connection,
    cache_keys: list[str],
    *,
    generator_fingerprint: str,
) -> dict[str, CacheEntry]:
    entries: dict[str, CacheEntry] = {}
    for start in range(0, len(cache_keys), 500):
        chunk = cache_keys[start : start + 500]
        placeholders = ",".join("?" for _ in chunk)
        rows = connection.execute(
            f"SELECT cache_key, model_fingerprint, generator_fingerprint, "
            f"generator_contract, backend_implementation, "
            f"cache_implementation_sha256, backend_implementation_sha256, "
            f"model_artifact_sha256, encoder_runtime, device, shard_path, row_index, "
            f"dims, vector_sha256 "
            f"FROM {CACHE_TABLE} WHERE generator_fingerprint = ? "
            f"AND cache_key IN ({placeholders})",
            [generator_fingerprint, *chunk],
        )
        for (
            cache_key,
            model_fingerprint,
            recorded_generator_fingerprint,
            generator_contract,
            backend_implementation,
            cache_implementation_sha256,
            backend_implementation_sha256,
            model_artifact_sha256,
            encoder_runtime,
            device,
            shard_path,
            row_index,
            dims,
            vector_sha256,
        ) in rows:
            entries[str(cache_key)] = CacheEntry(
                cache_key=str(cache_key),
                model_fingerprint=str(model_fingerprint),
                generator_fingerprint=str(recorded_generator_fingerprint),
                generator_contract=str(generator_contract),
                backend_implementation=str(backend_implementation),
                cache_implementation_sha256=str(cache_implementation_sha256),
                backend_implementation_sha256=str(backend_implementation_sha256),
                model_artifact_sha256=str(model_artifact_sha256),
                encoder_runtime=str(encoder_runtime),
                device=str(device),
                shard_path=str(shard_path),
                row_index=int(row_index),
                dims=int(dims),
                vector_sha256=str(vector_sha256),
            )
    return entries


def _encode_missing(
    connection: sqlite3.Connection,
    *,
    root: Path,
    model: ModelSpec,
    model_fingerprint: str,
    generator: dict[str, object],
    generator_fingerprint: str,
    backend: EmbeddingBackend,
    missing: list[EmbeddingInput],
    batch_size: int,
    shard_size: int,
) -> int:
    if not missing:
        return 0
    shard_directory = (
        root
        / "data/cache/embeddings/shards/v2"
        / model_fingerprint
        / generator_fingerprint
    )
    shard_directory.mkdir(parents=True, exist_ok=True)
    encoded = 0
    for start in range(0, len(missing), shard_size):
        shard_inputs = missing[start : start + shard_size]
        batches = []
        for batch_start in range(0, len(shard_inputs), batch_size):
            texts = [item.text for item in shard_inputs[batch_start : batch_start + batch_size]]
            batch = np.asarray(backend.encode(texts), dtype=np.float32)
            _validate_vectors(batch, expected_rows=len(texts), model=model)
            batches.append(batch)
        vectors = np.concatenate(batches)
        shard_identifier = canonical_sha256([item.cache_key for item in shard_inputs])[:16]
        shard_path = shard_directory / f"shard-{shard_identifier}.npy"
        if shard_path.exists():
            raise DatasetIntegrityError(f"orphan or conflicting embedding shard: {shard_path}")
        temporary = shard_path.with_name(f".{shard_path.name}.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, vectors, allow_pickle=False)
        temporary.replace(shard_path)
        relative_path = str(shard_path.relative_to(root))
        rows = [
            (
                item.cache_key,
                model_fingerprint,
                generator_fingerprint,
                generator["contract"],
                generator["backend_implementation"],
                generator["cache_implementation_sha256"],
                generator["backend_implementation_sha256"],
                generator["model_artifact_sha256"],
                generator["encoder_runtime"],
                generator["device"],
                relative_path,
                row_index,
                model.dims,
                hashlib.sha256(vectors[row_index].tobytes()).hexdigest(),
            )
            for row_index, item in enumerate(shard_inputs)
        ]
        connection.executemany(
            f"INSERT INTO {CACHE_TABLE} "
            "(cache_key, model_fingerprint, generator_fingerprint, "
            "generator_contract, backend_implementation, "
            "cache_implementation_sha256, backend_implementation_sha256, "
            "model_artifact_sha256, encoder_runtime, device, shard_path, row_index, "
            "dims, vector_sha256) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        connection.commit()
        encoded += len(shard_inputs)
    return encoded


def _validate_vectors(
    vectors: NDArray[np.float32],
    *,
    expected_rows: int,
    model: ModelSpec,
) -> None:
    if vectors.shape != (expected_rows, model.dims):
        raise DatasetIntegrityError(
            f"{model.name} returned vector shape {vectors.shape}, expected "
            f"({expected_rows}, {model.dims})"
        )
    if not np.isfinite(vectors).all():
        raise DatasetIntegrityError(f"{model.name} returned non-finite vectors")
    if model.normalize:
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, rtol=NORM_TOLERANCE, atol=NORM_TOLERANCE):
            raise DatasetIntegrityError(f"{model.name} returned non-normalized vectors")


def _verify_entries(
    root: Path,
    entries: dict[str, CacheEntry],
    model: ModelSpec,
    *,
    model_fingerprint: str,
    generator: dict[str, object],
    generator_fingerprint: str,
) -> None:
    arrays = _load_shards(root, entries)
    for cache_key, entry in entries.items():
        if (
            entry.model_fingerprint != model_fingerprint
            or entry.generator_fingerprint != generator_fingerprint
            or entry.generator_contract != generator["contract"]
            or entry.backend_implementation != generator["backend_implementation"]
            or entry.cache_implementation_sha256
            != generator["cache_implementation_sha256"]
            or entry.backend_implementation_sha256
            != generator["backend_implementation_sha256"]
            or entry.model_artifact_sha256 != generator["model_artifact_sha256"]
            or entry.encoder_runtime != generator["encoder_runtime"]
            or entry.device != generator["device"]
        ):
            raise DatasetIntegrityError(
                f"embedding cache generator identity differs for {cache_key}"
            )
        if entry.dims != model.dims:
            raise DatasetIntegrityError(f"embedding dimensions differ for {cache_key}")
        array = arrays[entry.shard_path]
        if entry.row_index < 0 or entry.row_index >= len(array):
            raise DatasetIntegrityError(f"embedding shard row is invalid for {cache_key}")
        vector = np.asarray(array[entry.row_index], dtype=np.float32)
        if hashlib.sha256(vector.tobytes()).hexdigest() != entry.vector_sha256:
            raise DatasetIntegrityError(f"embedding vector checksum differs for {cache_key}")
        _validate_vectors(vector.reshape(1, -1), expected_rows=1, model=model)


def _load_shards(
    root: Path,
    entries: dict[str, CacheEntry],
) -> dict[str, NDArray[np.float32]]:
    arrays: dict[str, NDArray[np.float32]] = {}
    for shard_path in sorted({entry.shard_path for entry in entries.values()}):
        path = root / shard_path
        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise DatasetIntegrityError(f"invalid embedding shard: {path}") from error
        if array.dtype != np.float32 or array.ndim != 2:
            raise DatasetIntegrityError(f"invalid embedding shard layout: {path}")
        arrays[shard_path] = array
    return arrays


def _write_manifest(
    *,
    root: Path,
    products_path: Path,
    model: ModelSpec,
    backend: EmbeddingBackend,
    model_fingerprint: str,
    generator: dict[str, object],
    generator_fingerprint: str,
    inputs: list[EmbeddingInput],
    entries: dict[str, CacheEntry],
    manifest_path: Path,
    cache_hits: int,
    encoded_vectors: int,
    benchmark_provenance: dict[str, Any],
) -> None:
    product_facts = file_facts(products_path)
    completion_revision = asdict(collect_code_revision(root))
    provenance_valid = _recorded_provenance_valid(
        root,
        benchmark_provenance,
        completion_revision,
        require_current=True,
    )
    device_evidence_eligible, preflight_binding = _embedding_device_evidence(
        root=root,
        products_path=products_path,
        model=model,
        device=backend.device,
        model_artifact_sha256=backend.model_artifact_sha256,
    )
    write_json(
        manifest_path,
        {
            "schema_version": 2,
            "dataset": "WANDS",
            "model": asdict(model),
            "model_fingerprint": model_fingerprint,
            "cache_table": CACHE_TABLE,
            "cache_generator": generator,
            "cache_generator_fingerprint": generator_fingerprint,
            "model_artifact_sha256": backend.model_artifact_sha256,
            "encoder_runtime": backend.runtime,
            "device": backend.device,
            "device_evidence_eligible": device_evidence_eligible,
            "device_preflight": preflight_binding,
            "products_path": str(products_path.relative_to(root)),
            "products_sha256": product_facts.sha256,
            "products": _artifact(root, products_path),
            "model_registry": _optional_artifact(root, root / "config/models.toml"),
            "model_artifact_registry": _optional_artifact(
                root, root / "config/model_artifacts.toml"
            ),
            "prepared_manifest": _optional_artifact(
                root, products_path.parent / "manifest.json"
            ),
            "document_count": len(inputs),
            "unique_cache_keys": len({item.cache_key for item in inputs}),
            "input_cache_keys_sha256": canonical_sha256(
                [(item.document_id, item.cache_key) for item in inputs]
            ),
            "document_attributes_recipe": ATTRIBUTES_RECIPE,
            "rendered_template_samples": [
                {"document_id": item.document_id, "text": item.text} for item in inputs[:3]
            ],
            "dtype": "float32",
            "dimensions": model.dims,
            "normalized": model.normalize,
            "cache_hits": cache_hits,
            "encoded_vectors": encoded_vectors,
            "shards": _shard_facts(root, entries),
            "cache_entries_sha256": canonical_sha256(
                [asdict(entries[key]) for key in sorted(entries)]
            ),
            "deterministic_replay": _embedding_replay_contract(
                _unique_inputs(inputs)
            ),
            "benchmark_provenance": benchmark_provenance,
            "completion_code_revision": completion_revision,
            "quality_evidence_eligible_for_decision": (
                provenance_valid
                and device_evidence_eligible
                and _registered_embedding_replay_available(
                    root=root,
                    products_path=products_path,
                    model=model,
                )
            ),
        },
    )


def finalize_embedding_preflight_artifact(
    *,
    root: Path,
    products_path: Path,
    model: ModelSpec,
    candidate_device: str,
    measurements: Mapping[str, Any],
    benchmark_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    completion_revision = asdict(collect_code_revision(root))
    output = cast(dict[str, Any], json.loads(json.dumps(dict(measurements))))
    output.update(
        {
            "schema_version": 2,
            "model": asdict(model),
            "candidate_device": candidate_device,
            "thresholds": {
                "minimum_cosine_similarity": PREFLIGHT_MINIMUM_COSINE,
                "maximum_absolute_delta": PREFLIGHT_MAXIMUM_ABSOLUTE_DELTA,
            },
            "model_registry": _artifact(root, root / "config/models.toml"),
            "model_artifact_registry": _artifact(
                root, root / "config/model_artifacts.toml"
            ),
            "products": _artifact(root, products_path),
            "benchmark_provenance": cast(
                dict[str, Any], json.loads(json.dumps(dict(benchmark_provenance)))
            ),
            "completion_code_revision": completion_revision,
        }
    )
    output["quality_evidence_eligible_for_decision"] = _recorded_provenance_valid(
        root,
        output["benchmark_provenance"],
        completion_revision,
    )
    return output


def verify_embedding_preflight_artifact(
    *,
    root: Path,
    products_path: Path,
    model: ModelSpec,
    candidate_device: str,
) -> dict[str, Any]:
    path = (
        root
        / f"results/wands/embeddings/{model.name}.preflight.{candidate_device}.json"
    )
    artifact = cast(dict[str, Any], read_json(path))
    if (
        artifact.get("schema_version") != 2
        or artifact.get("model") != asdict(model)
        or artifact.get("candidate_device") != candidate_device
        or artifact.get("model_registry")
        != _artifact(root, root / "config/models.toml")
        or artifact.get("model_artifact_registry")
        != _artifact(root, root / "config/model_artifacts.toml")
        or artifact.get("products") != _artifact(root, products_path)
        or artifact.get("thresholds")
        != {
            "minimum_cosine_similarity": PREFLIGHT_MINIMUM_COSINE,
            "maximum_absolute_delta": PREFLIGHT_MAXIMUM_ABSOLUTE_DELTA,
        }
    ):
        raise DatasetIntegrityError(f"embedding preflight metadata differs for {model.name}")
    sample_size = artifact.get("sample_size")
    if not isinstance(sample_size, int) or isinstance(sample_size, bool) or sample_size <= 0:
        raise DatasetIntegrityError(f"embedding preflight sample differs for {model.name}")
    inputs = load_embedding_inputs(products_path, model)
    samples = inputs[:sample_size]
    if (
        len(samples) != sample_size
        or artifact.get("sample_cache_keys_sha256")
        != canonical_sha256([item.cache_key for item in samples])
    ):
        raise DatasetIntegrityError(f"embedding preflight sample differs for {model.name}")
    comparison = artifact.get("comparison")
    comparison_passed = False
    if isinstance(comparison, dict):
        minimum = comparison.get("minimum_cosine_similarity")
        maximum_delta = comparison.get("maximum_absolute_delta")
        if (
            not isinstance(minimum, int | float)
            or isinstance(minimum, bool)
            or not isinstance(maximum_delta, int | float)
            or isinstance(maximum_delta, bool)
        ):
            raise DatasetIntegrityError(
                f"embedding preflight comparison differs for {model.name}"
            )
        if (
            float(minimum) >= PREFLIGHT_MINIMUM_COSINE
            and float(maximum_delta) <= PREFLIGHT_MAXIMUM_ABSOLUTE_DELTA
        ):
            comparison_passed = True
    reference = artifact.get("cpu")
    candidate = artifact.get("candidate")
    if not isinstance(reference, dict) or not isinstance(candidate, dict):
        raise DatasetIntegrityError(f"embedding preflight runtimes differ for {model.name}")
    model_hash = reference.get("model_artifact_sha256")
    if (
        not _is_sha256(model_hash)
        or not _is_sha256(candidate.get("model_artifact_sha256"))
        or not isinstance(reference.get("runtime"), str)
        or not reference.get("runtime")
        or not isinstance(candidate.get("runtime"), str)
        or not candidate.get("runtime")
        or not isinstance(reference.get("device"), str)
        or not str(reference.get("device")).startswith("cpu")
        or not isinstance(candidate.get("device"), str)
        or not str(candidate.get("device")).startswith(candidate_device)
    ):
        raise DatasetIntegrityError(f"embedding preflight runtimes differ for {model.name}")
    expected_status = (
        "passed"
        if comparison_passed and candidate.get("model_artifact_sha256") == model_hash
        else "failed"
    )
    if artifact.get("status") != expected_status:
        raise DatasetIntegrityError(f"embedding preflight status differs for {model.name}")
    eligible = _recorded_provenance_valid(
        root,
        artifact.get("benchmark_provenance"),
        artifact.get("completion_code_revision"),
    )
    if artifact.get("quality_evidence_eligible_for_decision") is not eligible:
        raise DatasetIntegrityError(f"embedding preflight eligibility differs for {model.name}")
    return artifact


def _embedding_device_evidence(
    *,
    root: Path,
    products_path: Path,
    model: ModelSpec,
    device: str,
    model_artifact_sha256: str,
) -> tuple[bool, dict[str, object] | None]:
    if device == "cpu" or device.startswith("cpu:"):
        return True, None
    if device == "mps" or device.startswith("mps:"):
        artifact = verify_embedding_preflight_artifact(
            root=root,
            products_path=products_path,
            model=model,
            candidate_device="mps",
        )
        path = root / f"results/wands/embeddings/{model.name}.preflight.mps.json"
        candidate = artifact.get("candidate")
        eligible = (
            artifact.get("status") == "passed"
            and artifact.get("quality_evidence_eligible_for_decision") is True
            and isinstance(candidate, dict)
            and candidate.get("model_artifact_sha256") == model_artifact_sha256
        )
        return eligible, _artifact(root, path)
    return False, None


def _capture_provenance(
    root: Path,
    supplied: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if supplied is not None:
        return cast(dict[str, Any], json.loads(json.dumps(dict(supplied))))
    try:
        return collect_manifest_provenance(root)
    except (ConfigError, OSError, ValueError):
        return {"status": "unavailable"}


def _recorded_provenance_valid(
    root: Path,
    provenance: object,
    completion_revision: object,
    *,
    require_current: bool = False,
) -> bool:
    if not isinstance(provenance, Mapping) or not isinstance(
        completion_revision, Mapping
    ):
        return False
    if provenance.get("status") == "unavailable":
        return False
    start_revision = provenance.get("code_revision")
    if not isinstance(start_revision, Mapping) or dict(start_revision) != dict(
        completion_revision
    ):
        return False
    return verify_decision_provenance(
        provenance,
        root=root,
        profile_path=root / "config/benchmark.toml",
        environment_path=root / "results/environment/benchmark-profile.json",
        require_current_code_revision=require_current,
    )


def _verify_registered_embedding_inputs(
    *,
    root: Path,
    products_path: Path,
    model: ModelSpec,
) -> None:
    registry_path = root / "config/models.toml"
    if registry_path.is_file():
        registry = load_model_registry(registry_path)
        if registry.get(model.name) != model:
            raise DatasetIntegrityError(
                f"embedding model differs from the registered model: {model.name}"
            )
        expected_products = root / "data/prepared/wands/products.jsonl"
        if expected_products.is_file() and products_path.resolve() != expected_products.resolve():
            raise DatasetIntegrityError("embedding products path is not canonical WANDS data")


def _embedding_replay_contract(
    inputs: list[EmbeddingInput],
) -> dict[str, object]:
    indexes = _spanning_indexes(len(inputs), EMBEDDING_REPLAY_SAMPLE_SIZE)
    samples = [inputs[index] for index in indexes]
    return {
        "method": "fixed_evenly_spaced_unique_inputs_including_endpoints",
        "sample_size": len(samples),
        "sample_cache_keys_sha256": canonical_sha256(
            [item.cache_key for item in samples]
        ),
        "batch_size": EMBEDDING_REPLAY_BATCH_SIZE,
        "relative_tolerance": EMBEDDING_REPLAY_RELATIVE_TOLERANCE,
        "absolute_tolerance": EMBEDDING_REPLAY_ABSOLUTE_TOLERANCE,
    }


def _verify_embedding_cache_replay(
    *,
    root: Path,
    products_path: Path,
    model: ModelSpec,
    inputs: list[EmbeddingInput],
    entries: dict[str, CacheEntry],
    generator: Mapping[str, object],
    manifest: Mapping[str, Any],
) -> bool:
    expected_contract = _embedding_replay_contract(inputs)
    if manifest.get("deterministic_replay") != expected_contract:
        raise DatasetIntegrityError(
            f"embedding deterministic replay contract differs for {model.name}"
        )
    if not _registered_embedding_replay_available(
        root=root,
        products_path=products_path,
        model=model,
    ):
        return False
    snapshot_sha256 = verify_registered_model_snapshot(root=root, model=model)
    if generator.get("model_artifact_sha256") != snapshot_sha256:
        raise DatasetIntegrityError(
            f"embedding replay model bytes differ for {model.name}"
        )
    indexes = _spanning_indexes(len(inputs), EMBEDDING_REPLAY_SAMPLE_SIZE)
    samples = [inputs[index] for index in indexes]
    arrays = _load_shards(root, {item.cache_key: entries[item.cache_key] for item in samples})
    cached = np.stack(
        [
            arrays[entries[item.cache_key].shard_path][
                entries[item.cache_key].row_index
            ]
            for item in samples
        ]
    ).astype(np.float32, copy=False)
    backend: EmbeddingBackend | None = None
    batches: list[NDArray[np.float32]] = []
    try:
        backend = cast(
            EmbeddingBackend,
            create_bulk_backend(
                root=root,
                model=model,
                device=cast(str, generator["device"]),
            ),
        )
        if cache_generator_identity(
            model_fingerprint=canonical_sha256(cache_model_values(model)),
            backend=backend,
        ) != dict(generator):
            raise DatasetIntegrityError(
                f"embedding replay backend identity differs for {model.name}"
            )
        for offset in range(0, len(samples), EMBEDDING_REPLAY_BATCH_SIZE):
            texts = [
                item.text
                for item in samples[offset : offset + EMBEDDING_REPLAY_BATCH_SIZE]
            ]
            batch = np.asarray(backend.encode(texts), dtype=np.float32)
            _validate_vectors(batch, expected_rows=len(texts), model=model)
            batches.append(batch)
    finally:
        del backend
        release_device_memory()
    replayed = np.concatenate(batches)
    if not np.allclose(
        cached,
        replayed,
        rtol=EMBEDDING_REPLAY_RELATIVE_TOLERANCE,
        atol=EMBEDDING_REPLAY_ABSOLUTE_TOLERANCE,
    ):
        maximum_delta = float(np.max(np.abs(cached - replayed)))
        raise DatasetIntegrityError(
            f"embedding deterministic sample re-encoding differs for {model.name}; "
            f"maximum absolute delta {maximum_delta:.9g}"
        )
    return True


def _registered_embedding_replay_available(
    *,
    root: Path,
    products_path: Path,
    model: ModelSpec,
) -> bool:
    registry_path = root / "config/models.toml"
    artifact_registry_path = root / "config/model_artifacts.toml"
    canonical_products = root / "data/prepared/wands/products.jsonl"
    snapshot = model_snapshot_directory(root, model)
    if (
        not registry_path.is_file()
        or not artifact_registry_path.is_file()
        or not canonical_products.is_file()
        or products_path.resolve() != canonical_products.resolve()
        or not snapshot.is_dir()
    ):
        return False
    try:
        if load_model_registry(registry_path).get(model.name) != model:
            return False
        verify_registered_model_snapshot(
            root=root,
            model=model,
            snapshot=snapshot,
        )
        return True
    except (ConfigError, DatasetIntegrityError, OSError, ValueError):
        return False


def _spanning_indexes(length: int, sample_size: int) -> list[int]:
    if length <= 0 or sample_size <= 0:
        raise ValueError("spanning sample dimensions must be positive")
    if length <= sample_size:
        return list(range(length))
    return [
        position * (length - 1) // (sample_size - 1)
        for position in range(sample_size)
    ]


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }


def _optional_artifact(root: Path, path: Path) -> dict[str, object] | None:
    return _artifact(root, path) if path.is_file() else None


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _shard_facts(root: Path, entries: dict[str, CacheEntry]) -> list[dict[str, object]]:
    facts = []
    for shard_path in sorted({entry.shard_path for entry in entries.values()}):
        file = file_facts(root / shard_path)
        rows = sum(entry.shard_path == shard_path for entry in entries.values())
        facts.append(
            {"path": shard_path, "sha256": file.sha256, "bytes": file.bytes, "rows": rows}
        )
    return facts


def _unique_inputs(inputs: list[EmbeddingInput]) -> list[EmbeddingInput]:
    unique: dict[str, EmbeddingInput] = {}
    for item in inputs:
        unique.setdefault(item.cache_key, item)
    return list(unique.values())


def cache_model_values(model: ModelSpec) -> dict[str, object]:
    values = cast(dict[str, object], asdict(model))
    if not model.trust_remote_code:
        values.pop("use_memory_efficient_attention")
    return values


def cache_generator_identity(
    *,
    model_fingerprint: str,
    backend: EmbeddingBackend,
) -> dict[str, object]:
    backend_implementation = f"{type(backend).__module__}.{type(backend).__qualname__}"
    return {
        "schema_version": 1,
        "contract": CACHE_GENERATOR_CONTRACT,
        "model_fingerprint": model_fingerprint,
        "backend_implementation": backend_implementation,
        "cache_implementation_sha256": file_facts(Path(__file__)).sha256,
        "backend_implementation_sha256": _backend_module_sha256(
            backend_implementation
        ),
        "model_artifact_sha256": backend.model_artifact_sha256,
        "encoder_runtime": backend.runtime,
        "device": backend.device,
    }


def _manifest_generator_identity(
    *,
    root: Path,
    model: ModelSpec,
    manifest: dict[str, Any],
    model_fingerprint: str,
) -> tuple[dict[str, object], str]:
    value = manifest.get("cache_generator")
    if not isinstance(value, dict):
        raise DatasetIntegrityError(
            f"schema-v2 embedding generator identity is missing for {model.name}"
        )
    generator = cast(dict[str, object], value)
    _verify_backend_generator_identity(root=root, model=model, generator=generator)
    expected = {
        "schema_version": 1,
        "contract": CACHE_GENERATOR_CONTRACT,
        "model_fingerprint": model_fingerprint,
        "backend_implementation": generator.get("backend_implementation"),
        "cache_implementation_sha256": generator.get(
            "cache_implementation_sha256"
        ),
        "backend_implementation_sha256": generator.get(
            "backend_implementation_sha256"
        ),
        "model_artifact_sha256": manifest.get("model_artifact_sha256"),
        "encoder_runtime": manifest.get("encoder_runtime"),
        "device": manifest.get("device"),
    }
    generator_fingerprint = canonical_sha256(generator)
    if (
        generator != expected
        or manifest.get("cache_generator_fingerprint") != generator_fingerprint
    ):
        raise DatasetIntegrityError(
            f"embedding cache generator identity differs for {model.name}"
        )
    return generator, generator_fingerprint


def _verify_backend_generator_identity(
    *,
    root: Path,
    model: ModelSpec,
    generator: Mapping[str, object],
) -> None:
    required_keys = {
        "schema_version",
        "contract",
        "model_fingerprint",
        "backend_implementation",
        "cache_implementation_sha256",
        "backend_implementation_sha256",
        "model_artifact_sha256",
        "encoder_runtime",
        "device",
    }
    if (
        set(generator) != required_keys
        or generator.get("schema_version") != 1
        or generator.get("contract") != CACHE_GENERATOR_CONTRACT
        or generator.get("model_fingerprint")
        != canonical_sha256(cache_model_values(model))
        or not isinstance(generator.get("backend_implementation"), str)
        or not generator.get("backend_implementation")
        or generator.get("cache_implementation_sha256")
        != file_facts(Path(__file__)).sha256
        or not _is_sha256(generator.get("backend_implementation_sha256"))
        or not _is_sha256(generator.get("model_artifact_sha256"))
        or not isinstance(generator.get("encoder_runtime"), str)
        or not generator.get("encoder_runtime")
        or not isinstance(generator.get("device"), str)
        or not generator.get("device")
    ):
        raise DatasetIntegrityError(
            f"embedding cache generator identity is invalid for {model.name}"
        )
    backend_implementation = cast(str, generator["backend_implementation"])
    if (root / "config/models.toml").is_file():
        expected_backend = (
            "poc.model_runtime.TransformersClsBackend"
            if model.name == "arctic_embed_m_v2"
            else "poc.model_runtime.SentenceTransformerBackend"
        )
        if generator.get("backend_implementation") != expected_backend:
            raise DatasetIntegrityError(
                f"embedding backend implementation differs for {model.name}"
            )
    if generator.get("backend_implementation_sha256") != _backend_module_sha256(
        backend_implementation
    ):
        raise DatasetIntegrityError(
            f"embedding backend implementation bytes differ for {model.name}"
        )


def _backend_module_sha256(backend_implementation: str) -> str:
    module_name, separator, _ = backend_implementation.rpartition(".")
    if not separator or not module_name:
        raise DatasetIntegrityError("embedding backend implementation is invalid")
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise DatasetIntegrityError(
            f"embedding backend module has no source file: {module_name}"
        )
    return file_facts(Path(module_file)).sha256


def _required_string(values: dict[str, Any], key: str, path: Path, line_number: int) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise DatasetIntegrityError(f"invalid {key} in {path}:{line_number}")
    return value


def _optional_string(values: dict[str, Any], key: str, path: Path, line_number: int) -> str:
    value = values.get(key, "")
    if not isinstance(value, str):
        raise DatasetIntegrityError(f"invalid {key} in {path}:{line_number}")
    return value
