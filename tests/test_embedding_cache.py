from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

import poc.embedding_cache as embedding_cache_module
from poc.config import ModelSpec
from poc.datasets import DatasetIntegrityError
from poc.embedding_cache import (
    build_embedding_cache,
    cache_model_values,
    load_embedding_inputs,
    verify_embedding_cache,
)
from poc.manifest import canonical_sha256, read_json


class FakeBackend:
    device = "test"

    def __init__(
        self,
        dims: int,
        *,
        reject_calls: bool = False,
        runtime: str = "fake-v1",
        model_artifact_sha256: str = "a" * 64,
    ) -> None:
        self.dims = dims
        self.reject_calls = reject_calls
        self.runtime = runtime
        self.model_artifact_sha256 = model_artifact_sha256
        self.encoded_texts: list[str] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.reject_calls:
            raise AssertionError("a complete cache must not call the encoder")
        self.encoded_texts.extend(texts)
        vectors = []
        for text in texts:
            raw = np.frombuffer(hashlib.sha256(text.encode()).digest(), dtype=np.uint8)
            vector = raw[: self.dims].astype(np.float32) + 1.0
            vectors.append(vector / np.linalg.norm(vector))
        return np.stack(vectors)


def model_spec(*, document_prefix: str = "") -> ModelSpec:
    return ModelSpec.from_mapping(
        "test_model",
        {
            "hf_id": "example/test-model",
            "revision": "1" * 40,
            "dims": 8,
            "normalize": True,
            "max_seq_length": 512,
            "bulk_dtype": "float32",
            "use_memory_efficient_attention": False,
            "trust_remote_code": False,
            "query_prefix": "",
            "document_prefix": document_prefix,
            "query_template": "{query}",
            "document_template": "{title}. {description}. {attributes}",
            "license": "Apache-2.0",
            "decision_eligible": True,
            "contamination": "none_known",
        },
    )


def write_products(path: Path) -> None:
    records = [
        {
            "product_id": str(index),
            "title": f"Product {index}",
            "description": f"Description {index}",
            "product_class": "Class",
            "category": "Root / Class",
            "features": f"color:{color}",
        }
        for index, color in enumerate(("red", "green", "blue"), start=1)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_embedding_cache_is_keyed_by_model_config_and_document_text(tmp_path: Path) -> None:
    products_path = tmp_path / "products.jsonl"
    write_products(products_path)
    model = model_spec()
    first_backend = FakeBackend(model.dims)

    first = build_embedding_cache(
        root=tmp_path,
        products_path=products_path,
        model=model,
        backend=first_backend,
        batch_size=2,
        shard_size=2,
    )
    assert first.document_count == 3
    assert first.encoded_vectors == 3
    assert first.cache_hits == 0
    assert len(first_backend.encoded_texts) == 3

    second = build_embedding_cache(
        root=tmp_path,
        products_path=products_path,
        model=model,
        backend=FakeBackend(model.dims, reject_calls=True),
        batch_size=2,
        shard_size=2,
    )
    assert second.encoded_vectors == 0
    assert second.cache_hits == 3
    verify_embedding_cache(root=tmp_path, products_path=products_path, model=model)

    changed = build_embedding_cache(
        root=tmp_path,
        products_path=products_path,
        model=model_spec(document_prefix="passage: "),
        backend=FakeBackend(model.dims),
        batch_size=2,
        shard_size=2,
    )
    assert changed.encoded_vectors == 3


def test_embedding_cache_verifier_rejects_corrupt_shard(tmp_path: Path) -> None:
    products_path = tmp_path / "products.jsonl"
    write_products(products_path)
    model = model_spec()
    summary = build_embedding_cache(
        root=tmp_path,
        products_path=products_path,
        model=model,
        backend=FakeBackend(model.dims),
        batch_size=2,
        shard_size=2,
    )
    manifest = read_json(summary.manifest_path)
    assert isinstance(manifest, dict)
    shards = manifest["shards"]
    assert isinstance(shards, list)
    first_shard = shards[0]
    assert isinstance(first_shard, dict)
    shard_path = tmp_path / str(first_shard["path"])
    shard_path.write_bytes(b"corrupt")

    with pytest.raises(DatasetIntegrityError, match="shard"):
        verify_embedding_cache(root=tmp_path, products_path=products_path, model=model)


def test_embedding_manifest_is_schema_v2_and_eligibility_is_derived(
    tmp_path: Path,
) -> None:
    products_path = tmp_path / "products.jsonl"
    write_products(products_path)
    model = model_spec()
    summary = build_embedding_cache(
        root=tmp_path,
        products_path=products_path,
        model=model,
        backend=FakeBackend(model.dims),
        batch_size=2,
        shard_size=2,
    )
    manifest = read_json(summary.manifest_path)
    assert isinstance(manifest, dict)

    assert manifest["schema_version"] == 2
    assert manifest["quality_evidence_eligible_for_decision"] is False
    assert manifest["benchmark_provenance"]["status"] == "unavailable"

    manifest["quality_evidence_eligible_for_decision"] = True
    summary.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(DatasetIntegrityError, match="eligibility"):
        verify_embedding_cache(root=tmp_path, products_path=products_path, model=model)


def test_embedding_manifest_verifier_rejects_recorded_metadata_tampering(
    tmp_path: Path,
) -> None:
    products_path = tmp_path / "products.jsonl"
    write_products(products_path)
    model = model_spec()
    summary = build_embedding_cache(
        root=tmp_path,
        products_path=products_path,
        model=model,
        backend=FakeBackend(model.dims),
        batch_size=2,
        shard_size=2,
    )
    manifest = read_json(summary.manifest_path)
    assert isinstance(manifest, dict)
    manifest["dimensions"] = model.dims + 1
    summary.manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(DatasetIntegrityError, match="metadata"):
        verify_embedding_cache(root=tmp_path, products_path=products_path, model=model)


def test_schema_v2_build_does_not_promote_unbound_legacy_cache_rows(
    tmp_path: Path,
) -> None:
    products_path = tmp_path / "products.jsonl"
    write_products(products_path)
    model = model_spec()
    inputs = load_embedding_inputs(products_path, model)
    legacy_backend = FakeBackend(model.dims)
    vectors = legacy_backend.encode([item.text for item in inputs])
    shard_path = tmp_path / "data/cache/embeddings/shards/legacy/shard.npy"
    shard_path.parent.mkdir(parents=True)
    np.save(shard_path, vectors, allow_pickle=False)
    database_path = tmp_path / "data/cache/embeddings/wands.sqlite3"
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE embedding_cache (
                cache_key TEXT PRIMARY KEY,
                model_fingerprint TEXT NOT NULL,
                shard_path TEXT NOT NULL,
                row_index INTEGER NOT NULL,
                dims INTEGER NOT NULL,
                vector_sha256 TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO embedding_cache VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    item.cache_key,
                    canonical_sha256(cache_model_values(model)),
                    str(shard_path.relative_to(tmp_path)),
                    row_index,
                    model.dims,
                    hashlib.sha256(vectors[row_index].tobytes()).hexdigest(),
                )
                for row_index, item in enumerate(inputs)
            ],
        )
    legacy_manifest = tmp_path / "results/wands/embeddings/test_model.manifest.json"
    legacy_manifest.parent.mkdir(parents=True)
    legacy_manifest.write_text('{"schema_version": 2, "legacy": true}\n')
    legacy_manifest_bytes = legacy_manifest.read_bytes()

    current_backend = FakeBackend(model.dims)
    summary = build_embedding_cache(
        root=tmp_path,
        products_path=products_path,
        model=model,
        backend=current_backend,
        batch_size=2,
        shard_size=2,
    )

    assert summary.cache_hits == 0
    assert summary.encoded_vectors == len(inputs)
    assert len(current_backend.encoded_texts) == len(inputs)
    preserved = list(legacy_manifest.parent.glob("*.legacy-unbound-*.json"))
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == legacy_manifest_bytes
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT count(*) FROM embedding_cache").fetchone() == (
            len(inputs),
        )
        assert connection.execute(
            "SELECT count(*) FROM embedding_cache_v2"
        ).fetchone() == (len(inputs),)


@pytest.mark.parametrize(
    ("runtime", "artifact_sha256"),
    (("fake-v2", "a" * 64), ("fake-v1", "b" * 64)),
)
def test_cache_hits_require_exact_generator_runtime_and_model_bytes(
    tmp_path: Path,
    runtime: str,
    artifact_sha256: str,
) -> None:
    products_path = tmp_path / "products.jsonl"
    write_products(products_path)
    model = model_spec()
    build_embedding_cache(
        root=tmp_path,
        products_path=products_path,
        model=model,
        backend=FakeBackend(model.dims),
        batch_size=2,
        shard_size=2,
    )

    changed_backend = FakeBackend(
        model.dims,
        runtime=runtime,
        model_artifact_sha256=artifact_sha256,
    )
    changed = build_embedding_cache(
        root=tmp_path,
        products_path=products_path,
        model=model,
        backend=changed_backend,
        batch_size=2,
        shard_size=2,
    )

    assert changed.cache_hits == 0
    assert changed.encoded_vectors == 3
    assert len(changed_backend.encoded_texts) == 3


def test_embedding_verifier_reencodes_spanning_sample_after_coherent_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    products_path = tmp_path / "products.jsonl"
    write_products(products_path)
    model = model_spec()
    summary = build_embedding_cache(
        root=tmp_path,
        products_path=products_path,
        model=model,
        backend=FakeBackend(model.dims),
        batch_size=2,
        shard_size=2,
    )
    monkeypatch.setattr(
        embedding_cache_module,
        "_registered_embedding_replay_available",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        embedding_cache_module,
        "verify_registered_model_snapshot",
        lambda **_kwargs: "a" * 64,
    )
    monkeypatch.setattr(
        embedding_cache_module,
        "create_bulk_backend",
        lambda **_kwargs: FakeBackend(model.dims),
    )

    verify_embedding_cache(root=tmp_path, products_path=products_path, model=model)

    manifest = read_json(summary.manifest_path)
    assert isinstance(manifest, dict)
    first_shard = manifest["shards"][0]
    assert isinstance(first_shard, dict)
    shard_path = tmp_path / str(first_shard["path"])
    vectors = np.load(shard_path, allow_pickle=False)
    vectors[0] = np.roll(vectors[0], 1)
    np.save(shard_path, vectors, allow_pickle=False)

    generator_fingerprint = str(manifest["cache_generator_fingerprint"])
    inputs = load_embedding_inputs(products_path, model)
    first_key = inputs[0].cache_key
    vector_sha256 = hashlib.sha256(vectors[0].tobytes()).hexdigest()
    database_path = tmp_path / "data/cache/embeddings/wands.sqlite3"
    with embedding_cache_module._connect(database_path) as connection:
        connection.execute(
            "UPDATE embedding_cache_v2 SET vector_sha256 = ? "
            "WHERE cache_key = ? AND generator_fingerprint = ?",
            (vector_sha256, first_key, generator_fingerprint),
        )
        connection.commit()
        entries = embedding_cache_module._read_entries(
            connection,
            [item.cache_key for item in inputs],
            generator_fingerprint=generator_fingerprint,
        )
    manifest["shards"] = embedding_cache_module._shard_facts(tmp_path, entries)
    manifest["cache_entries_sha256"] = canonical_sha256(
        [asdict(entries[key]) for key in sorted(entries)]
    )
    summary.manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(DatasetIntegrityError, match="re-encoding"):
        verify_embedding_cache(root=tmp_path, products_path=products_path, model=model)
