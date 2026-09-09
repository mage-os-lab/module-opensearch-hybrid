from __future__ import annotations

from pathlib import Path
from shutil import copytree
from typing import Any, cast

import pytest

from poc.datasets import DatasetIntegrityError
from poc.experiments import BM25Profile
from poc.facet_semantics import (
    PIPELINE_ID,
    _verify_live_facet_pipeline,
    build_facet_requests,
    facet_evidence_status,
    facet_method_config,
    verify_registered_facet_configuration,
)
from poc.manifest import canonical_sha256, write_json
from poc.neural_sparse import load_neural_sparse_spec
from poc.os_client import OpenSearchClient
from poc.sku_slice import load_sku_slice_spec

ROOT = Path(__file__).resolve().parents[1]


def test_facet_semantics_separates_hybrid_ranking_from_lexical_counts(
    bm25_selection: dict[str, Any],
) -> None:
    selection = bm25_selection
    profile = BM25Profile.from_mapping(
        cast(dict[str, Any], selection["selected_profile"])
    )
    sparse = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")

    hybrid, lexical = build_facet_requests(
        "running shoe",
        profile=profile,
        sparse=sparse,
    )

    assert "hybrid" in hybrid["query"]
    assert "hybrid" not in lexical["query"]
    assert hybrid["size"] == 10
    assert lexical["size"] == 0
    assert hybrid["track_total_hits"] is True
    assert lexical["track_total_hits"] is True
    assert hybrid["aggs"] == lexical["aggs"]

    filtered_hybrid, filtered_lexical = build_facet_requests(
        "running shoe",
        profile=profile,
        sparse=sparse,
        active_filter=("category", "Footwear"),
    )
    expected = [{"term": {"category.facet": "Footwear"}}]
    clauses = filtered_hybrid["query"]["hybrid"]["queries"]
    assert all(clause["bool"]["filter"] == expected for clause in clauses)
    assert filtered_lexical["query"]["bool"]["filter"] == expected


def test_facet_method_is_exact_and_never_promoted_to_decision_evidence(
    bm25_selection: dict[str, Any],
) -> None:
    selection = bm25_selection
    profile = BM25Profile.from_mapping(
        cast(dict[str, Any], selection["selected_profile"])
    )
    sparse = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")

    method = facet_method_config(profile=profile, sparse=sparse)
    assert method["bm25_profile"] == selection["selected_profile"]
    assert method["neural_sparse_spec"]["index_name"] == sparse.index_name
    assert method["facet_fields"] == ["brand", "category", "product_class"]
    assert method["count_equivalence_claimed"] is False
    assert method["request_cache"] is False

    assert facet_evidence_status(
        provenance_valid=True,
        upstream_eligible=True,
        upstream_unchanged=True,
    ) == {
        "diagnostic_reproducible": True,
        "quality_evidence_eligible_for_decision": False,
        "latency_evidence_eligible_for_decision": False,
    }


def test_facet_configuration_binds_registered_index_and_profile(
    tmp_path: Path,
    bm25_selection: dict[str, Any],
) -> None:
    copytree(ROOT / "config", tmp_path / "config")
    selection = bm25_selection
    selected_profile = cast(dict[str, Any], selection["selected_profile"])
    selection_path = tmp_path / "results/wands/bm25-selection.json"
    write_json(
        selection_path,
        {
            "schema_version": 2,
            "dataset": "WANDS",
            "selected_profile": selected_profile,
            "selected_profile_sha256": canonical_sha256(selected_profile),
        },
    )
    profile = BM25Profile.from_mapping(selected_profile)
    sparse = load_neural_sparse_spec(tmp_path / "config/neural_sparse.toml")
    sku = load_sku_slice_spec(tmp_path / "config/sku_slice.toml")

    registered = verify_registered_facet_configuration(
        root=tmp_path,
        index=sku.index_name,
        profile=profile,
        sparse=sparse,
    )

    assert registered["dataset"]["revision"]
    assert registered["sku_slice"]["index_name"] == sku.index_name
    assert registered["selected_bm25_profile"] == selected_profile
    assert set(registered["config_artifacts"]) == {
        "datasets",
        "experiments",
        "indexes",
        "neural_sparse",
        "sku_slice",
    }
    with pytest.raises(DatasetIntegrityError, match="registered configuration"):
        verify_registered_facet_configuration(
            root=tmp_path,
            index="not-the-registered-index",
            profile=profile,
            sparse=sparse,
        )


def test_facet_pipeline_identity_is_checked_against_live_definition() -> None:
    expected: dict[str, Any] = {"phase_results_processors": []}

    class FakeClient:
        pipeline = expected

        def request(self, method: str, path: str) -> dict[str, Any]:
            assert method == "GET"
            assert path == f"/_search/pipeline/{PIPELINE_ID}"
            return {PIPELINE_ID: self.pipeline}

    client = FakeClient()
    _verify_live_facet_pipeline(cast(OpenSearchClient, client), expected)
    client.pipeline = {"phase_results_processors": [{"unexpected": {}}]}
    with pytest.raises(DatasetIntegrityError, match="live facet"):
        _verify_live_facet_pipeline(cast(OpenSearchClient, client), expected)
