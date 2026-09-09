from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from poc.config import ModelSpec
from poc.datasets import DatasetIntegrityError
from poc.model_runtime import (
    SentenceTransformerBackend,
    compare_embedding_vectors,
    finalize_cls_embeddings,
    hash_model_snapshot,
    is_mps_device,
    patch_biasless_layer_norms,
    prepare_mps_environment,
    reset_position_id_buffers,
    resolve_cached_model_source,
    verify_registered_model_snapshot,
)


def test_cached_sentence_transformer_loads_resolved_snapshot_path_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ModelSpec.from_mapping(
        "test_model",
        {
            "hf_id": "example/test-model",
            "revision": "a" * 40,
            "dims": 2,
            "normalize": True,
            "max_seq_length": 512,
            "bulk_dtype": "float32",
            "use_memory_efficient_attention": False,
            "trust_remote_code": False,
            "query_prefix": "",
            "document_prefix": "",
            "query_template": "{query}",
            "document_template": "{title}",
            "license": "Apache-2.0",
            "decision_eligible": True,
            "contamination": "none_known",
        },
    )
    snapshot = (
        tmp_path
        / "data/cache/models/sentence-transformers/models--example--test-model/snapshots"
        / model.revision
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n")
    expected_snapshot_sha256 = hash_model_snapshot(snapshot)
    (tmp_path / "config").mkdir()
    (tmp_path / "config/model_artifacts.toml").write_text(
        "\n".join(
            (
                "schema_version = 1",
                "",
                "[models.test_model]",
                'hf_id = "example/test-model"',
                f'revision = "{model.revision}"',
                f'snapshot_sha256 = "{expected_snapshot_sha256}"',
                'files = ["config.json"]',
                "",
            )
        )
    )
    loaded_paths: list[str] = []

    class FakeEncoder:
        device = "cpu"

        def __init__(self, model_name_or_path: str, **kwargs: Any) -> None:
            loaded_paths.append(model_name_or_path)

        def get_embedding_dimension(self) -> int:
            return 2

    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(SentenceTransformer=FakeEncoder)
            if name == "sentence_transformers"
            else real_import(name)
        ),
    )

    SentenceTransformerBackend(root=tmp_path, model=model, device="cpu")

    assert loaded_paths == [str(snapshot)]


def test_cached_transformers_model_resolves_snapshot_path_without_hub_lookup(
    tmp_path: Path,
) -> None:
    model = ModelSpec.from_mapping(
        "test_model",
        {
            "hf_id": "example/test-model",
            "revision": "a" * 40,
            "dims": 2,
            "normalize": True,
            "max_seq_length": 512,
            "bulk_dtype": "float32",
            "use_memory_efficient_attention": False,
            "trust_remote_code": True,
            "query_prefix": "query: ",
            "document_prefix": "",
            "query_template": "{query}",
            "document_template": "{title}",
            "license": "Apache-2.0",
            "decision_eligible": True,
            "contamination": "none_known",
        },
    )
    cache = tmp_path / "sentence-transformers"
    snapshot = (
        cache
        / "models--example--test-model/snapshots"
        / model.revision
    )
    snapshot.mkdir(parents=True)
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        (snapshot / name).write_text("cached\n")

    source, local_only = resolve_cached_model_source(
        cache,
        model,
        required_files=("config.json", "model.safetensors", "tokenizer.json"),
    )

    assert source == str(snapshot)
    assert local_only


def test_model_snapshot_hash_covers_paths_and_contents(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"model":"test"}\n')
    nested = tmp_path / "1_Pooling"
    nested.mkdir()
    (nested / "config.json").write_text('{"pooling":"cls"}\n')

    first = hash_model_snapshot(tmp_path)
    second = hash_model_snapshot(tmp_path)
    assert first == second
    assert len(first) == 64

    (nested / "config.json").write_text('{"pooling":"mean"}\n')
    assert hash_model_snapshot(tmp_path) != first


def test_registered_model_snapshot_rejects_relabelled_bytes(tmp_path: Path) -> None:
    model = ModelSpec.from_mapping(
        "test_model",
        {
            "hf_id": "example/test-model",
            "revision": "a" * 40,
            "dims": 2,
            "normalize": True,
            "max_seq_length": 512,
            "bulk_dtype": "float32",
            "use_memory_efficient_attention": False,
            "trust_remote_code": False,
            "query_prefix": "",
            "document_prefix": "",
            "query_template": "{query}",
            "document_template": "{title}",
            "license": "Apache-2.0",
            "decision_eligible": True,
            "contamination": "none_known",
        },
    )
    snapshot = (
        tmp_path
        / "data/cache/models/sentence-transformers/models--example--test-model/snapshots"
        / model.revision
    )
    snapshot.mkdir(parents=True)
    artifact = snapshot / "model.safetensors"
    artifact.write_bytes(b"official")
    expected = hash_model_snapshot(snapshot)
    (tmp_path / "config").mkdir()
    (tmp_path / "config/model_artifacts.toml").write_text(
        "\n".join(
            (
                "schema_version = 1",
                "",
                "[models.test_model]",
                'hf_id = "example/test-model"',
                f'revision = "{model.revision}"',
                f'snapshot_sha256 = "{expected}"',
                'files = ["model.safetensors"]',
                "",
            )
        )
    )

    assert verify_registered_model_snapshot(root=tmp_path, model=model) == expected

    artifact.write_bytes(b"substituted")
    with pytest.raises(DatasetIntegrityError, match="snapshot bytes differ"):
        verify_registered_model_snapshot(root=tmp_path, model=model)


def test_mps_compatibility_shim_adds_only_zero_layer_norm_bias() -> None:
    torch = pytest.importorskip("torch")
    layer_norm = torch.nn.LayerNorm(4, bias=False)
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    before = layer_norm(values).detach().numpy()

    patched = patch_biasless_layer_norms(layer_norm)

    assert patched == 1
    assert layer_norm.bias is not None
    assert np.array_equal(layer_norm.bias.detach().numpy(), np.zeros(4, dtype=np.float32))
    assert np.array_equal(layer_norm(values).detach().numpy(), before)


def test_embedding_runtime_comparison_reports_worst_vector() -> None:
    reference = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    candidate = np.asarray([[1.0, 0.0], [0.01, 0.99995]], dtype=np.float32)
    candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)

    comparison = compare_embedding_vectors(reference, candidate)

    assert comparison["maximum_absolute_delta"] > 0
    assert comparison["minimum_cosine_similarity"] < 1
    assert comparison["minimum_cosine_similarity"] > 0.9999


def test_mps_device_detection_accepts_indexed_device_name() -> None:
    assert is_mps_device("mps")
    assert is_mps_device("mps:0")
    assert not is_mps_device("cpu")


def test_mps_environment_removes_inherited_metal_debug_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("METAL_DEVICE_WRAPPER_TYPE", "1")

    assert prepare_mps_environment()
    assert "METAL_DEVICE_WRAPPER_TYPE" not in __import__("os").environ


def test_cls_mrl_embeddings_are_truncated_before_normalization() -> None:
    torch = pytest.importorskip("torch")
    hidden_state = torch.tensor([[[3.0, 4.0, 12.0]]])

    vector = finalize_cls_embeddings(hidden_state, dims=2, normalize=True)

    assert np.allclose(vector, np.asarray([[0.6, 0.8]], dtype=np.float32))


def test_remote_model_position_ids_are_restored_to_declared_initializer() -> None:
    torch = pytest.importorskip("torch")
    module = torch.nn.Module()
    module.register_buffer(
        "position_ids",
        torch.full((4,), 999, dtype=torch.long),
        persistent=False,
    )

    assert reset_position_id_buffers(module) == 1
    assert np.array_equal(module.position_ids.numpy(), np.arange(4))
