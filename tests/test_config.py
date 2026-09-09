from __future__ import annotations

from pathlib import Path

import pytest

from poc.config import (
    WANDS_FINALIST_MODEL_NAMES,
    WANDS_INT8_MODEL_NAMES,
    ConfigError,
    ModelSpec,
    load_model_registry,
)

ROOT = Path(__file__).resolve().parents[1]


def valid_model() -> dict[str, object]:
    return {
        "hf_id": "example/model",
        "revision": "a" * 40,
        "dims": 32,
        "normalize": True,
        "max_seq_length": 512,
        "bulk_dtype": "float32",
        "use_memory_efficient_attention": False,
        "trust_remote_code": False,
        "query_prefix": "",
        "document_prefix": "",
        "query_template": "{query}",
        "document_template": "{title}. {description}. {attributes}",
        "license": "Apache-2.0",
        "decision_eligible": True,
        "contamination": "none_known",
    }


def test_registry_has_pinned_revisions_and_explicit_templates() -> None:
    registry = load_model_registry(ROOT / "config/models.toml")
    assert set(registry) == {
        "gte_modernbert_base",
        "granite_embedding_english_r2",
        "arctic_embed_m_v2",
        "embeddinggemma_300m",
        "esci_minilm",
    }
    assert all(len(model.revision) == 40 for model in registry.values())
    assert all(model.max_seq_length == 512 for model in registry.values())
    assert all(model.bulk_dtype == "float32" for model in registry.values())
    assert not registry["arctic_embed_m_v2"].use_memory_efficient_attention
    assert registry["arctic_embed_m_v2"].trust_remote_code
    assert not registry["gte_modernbert_base"].trust_remote_code
    assert registry["arctic_embed_m_v2"].render_query("trail shoe").startswith("query: ")
    assert not registry["embeddinggemma_300m"].decision_eligible
    assert not registry["esci_minilm"].decision_eligible
    assert WANDS_FINALIST_MODEL_NAMES == (
        "gte_modernbert_base",
        "granite_embedding_english_r2",
        "arctic_embed_m_v2",
    )
    assert WANDS_INT8_MODEL_NAMES == (
        "gte_modernbert_base",
        "granite_embedding_english_r2",
        "arctic_embed_m_v2",
    )


@pytest.mark.parametrize(
    "missing",
    [
        "query_prefix",
        "document_prefix",
        "max_seq_length",
        "bulk_dtype",
        "use_memory_efficient_attention",
        "trust_remote_code",
    ],
)
def test_missing_prefix_is_rejected_even_when_empty_is_valid(missing: str) -> None:
    values = valid_model()
    del values[missing]
    with pytest.raises(ConfigError, match="mandatory keys"):
        ModelSpec.from_mapping("broken", values)


def test_moving_revision_is_rejected() -> None:
    values = valid_model()
    values["revision"] = "main"
    with pytest.raises(ConfigError, match="immutable"):
        ModelSpec.from_mapping("broken", values)
