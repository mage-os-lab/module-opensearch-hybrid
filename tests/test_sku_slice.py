from __future__ import annotations

from pathlib import Path

from poc.neural_sparse import load_neural_sparse_spec
from poc.sku_slice import (
    assess_known_item_regression,
    build_known_item_lexical_query,
    build_sku_index_definition,
    known_item_route,
    load_sku_slice_spec,
    synthetic_sku,
)

ROOT = Path(__file__).resolve().parents[1]


def test_sku_slice_is_explicitly_synthetic_and_registered() -> None:
    spec = load_sku_slice_spec(ROOT / "config/sku_slice.toml")

    assert "synthetic" in spec.dataset.lower()
    assert spec.sku_queries == 100
    assert spec.exact_title_queries == 100
    assert spec.lexical_weight == 0.3
    assert spec.maximum_mrr_regression == 0.01


def test_sku_index_has_case_normalized_exact_field_and_sparse_features() -> None:
    sparse = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")
    slice_spec = load_sku_slice_spec(ROOT / "config/sku_slice.toml")

    definition = build_sku_index_definition(sparse, slice_spec)

    sku = definition["mappings"]["properties"]["sku"]
    assert sku == {"type": "keyword", "normalizer": "sku_normalizer"}
    assert definition["mappings"]["properties"]["sparse_embedding"] == {
        "type": "rank_features"
    }
    assert definition["mappings"]["properties"]["category"]["fields"]["facet"] == {
        "type": "keyword",
        "ignore_above": 256,
    }
    assert "default_pipeline" not in definition["settings"]["index"]


def test_known_item_query_preserves_exact_sku_and_title_paths() -> None:
    query = build_known_item_lexical_query("RW-123")

    clauses = query["bool"]["should"]
    assert clauses[0] == {"term": {"sku": {"value": "RW-123", "boost": 100.0}}}
    assert clauses[1]["match_phrase"]["title"]["query"] == "RW-123"
    assert synthetic_sku("123", prefix="RW") == "RW-123"


def test_known_item_regression_gate_handles_ties_and_material_losses() -> None:
    tied = assess_known_item_regression(
        baseline={"one": 1.0, "two": 0.5},
        candidate={"one": 1.0, "two": 0.5},
        maximum_mrr_regression=0.01,
        alpha=0.05,
    )
    assert tied["passes_no_significant_regression_gate"] is True
    assert tied["paired_sign_test_p_value"] == 1.0

    regressed = assess_known_item_regression(
        baseline={str(index): 1.0 for index in range(20)},
        candidate={str(index): 0.0 for index in range(20)},
        maximum_mrr_regression=0.01,
        alpha=0.05,
    )
    assert regressed["passes_no_significant_regression_gate"] is False
    assert regressed["losses"] == 20


def test_known_item_router_bypasses_semantic_fusion_for_high_confidence_queries() -> None:
    assert (
        known_item_route("rw-123", sku_prefix="RW", lexical_top_title=None)
        == "lexical_only_sku"
    )
    assert (
        known_item_route(
            "Exact Product Title",
            sku_prefix="RW",
            lexical_top_title=" exact   product title ",
        )
        == "lexical_only_exact_title"
    )
    assert (
        known_item_route(
            "comfortable running shoe",
            sku_prefix="RW",
            lexical_top_title="Different title",
        )
        == "hybrid"
    )
