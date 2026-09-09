from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from poc.ann_recall import (
    assess_ann_recall,
    derive_ann_quality_eligibility,
    select_ef_search,
    verify_registered_ann_configuration,
)
from poc.config import load_model_registry
from poc.datasets import DatasetIntegrityError
from poc.experiments import load_bm25_experiments
from poc.indexing import load_wands_index_config
from poc.query_runtime import load_query_runtime_spec
from poc.three_way import load_three_way_spec

ROOT = Path(__file__).resolve().parents[1]


def test_ann_recall_reports_per_query_and_registered_floor() -> None:
    result = assess_ann_recall(
        exact={"q1": ["a", "b", "c"], "q2": ["d", "e", "f"]},
        approximate={"q1": ["a", "b", "x"], "q2": ["d", "e", "f"]},
        depth=3,
        minimum_recall=0.8,
    )

    assert result["per_query"] == {"q1": 2 / 3, "q2": 1.0}
    assert result["mean_recall@3"] == (2 / 3 + 1.0) / 2
    assert result["minimum_query_recall@3"] == 2 / 3
    assert result["meets_mean_recall_floor"] is True


def test_ann_recall_rejects_incomplete_query_sets() -> None:
    try:
        assess_ann_recall(
            exact={"q1": ["a"]},
            approximate={},
            depth=1,
            minimum_recall=0.98,
        )
    except ValueError as error:
        assert "queries" in str(error)
    else:
        raise AssertionError("incomplete ANN query set should fail")


def test_ann_ef_search_selection_uses_smallest_development_candidate_that_passes() -> None:
    selected, passed = select_ef_search(
        (100, 200, 400, 800),
        {100: 0.91, 200: 0.96, 400: 0.985, 800: 0.99},
        minimum_recall=0.98,
    )

    assert selected == 400
    assert passed is True

    selected, passed = select_ef_search(
        (100, 200), {100: 0.91, 200: 0.96}, minimum_recall=0.98
    )
    assert selected == 200
    assert passed is False


def test_ann_configuration_is_bound_to_registered_inputs() -> None:
    index = load_wands_index_config(ROOT / "config/indexes.toml")
    three_way = load_three_way_spec(ROOT / "config/three_way.toml")
    model = load_model_registry(ROOT / "config/models.toml")[three_way.dense_model]
    runtime = load_query_runtime_spec(ROOT / "config/query_runtime.toml")
    experiments = load_bm25_experiments(ROOT / "config/experiments.toml")

    registered = verify_registered_ann_configuration(
        root=ROOT,
        index=index.name,
        model=model,
        runtime=runtime,
        minimum_recall=experiments.minimum_ann_recall_at_100,
        ef_search_grid=experiments.ann_ef_search_grid,
        depth=100,
    )

    assert registered["dataset"]["revision"]
    assert registered["index"]["name"] == index.name
    assert model.name in registered["int8_model_order"]
    assert registered["model"] == {
        "name": model.name,
        "hf_id": model.hf_id,
        "revision": model.revision,
        "dims": model.dims,
        "normalize": model.normalize,
        "max_seq_length": model.max_seq_length,
        "bulk_dtype": model.bulk_dtype,
        "use_memory_efficient_attention": model.use_memory_efficient_attention,
        "trust_remote_code": model.trust_remote_code,
        "query_prefix": model.query_prefix,
        "document_prefix": model.document_prefix,
        "query_template": model.query_template,
        "document_template": model.document_template,
        "license": model.license,
        "decision_eligible": model.decision_eligible,
        "contamination": model.contamination,
    }
    assert registered["thresholds"] == {
        "minimum_recall@100": experiments.minimum_ann_recall_at_100,
        "ef_search_grid": list(experiments.ann_ef_search_grid),
        "depth": 100,
    }
    assert set(registered["config_artifacts"]) == {
        "datasets",
        "experiments",
        "indexes",
        "models",
        "query_runtime",
        "three_way",
    }

    with pytest.raises(DatasetIntegrityError, match="registered configuration"):
        verify_registered_ann_configuration(
            root=ROOT,
            index=index.name,
            model=model,
            runtime=replace(runtime, intra_op_num_threads=runtime.intra_op_num_threads + 1),
            minimum_recall=experiments.minimum_ann_recall_at_100,
            ef_search_grid=experiments.ann_ef_search_grid,
            depth=100,
        )


def test_ann_eligibility_requires_selected_model_and_all_evidence() -> None:
    assert derive_ann_quality_eligibility(
        development_passed=True,
        held_out_passed=True,
        provenance_valid=True,
        upstream_unchanged=True,
        upstream_eligible=True,
        selected_model_eligible=True,
    ) is True

    assert derive_ann_quality_eligibility(
        development_passed=True,
        held_out_passed=True,
        provenance_valid=True,
        upstream_unchanged=True,
        upstream_eligible=True,
        selected_model_eligible=False,
    ) is False
