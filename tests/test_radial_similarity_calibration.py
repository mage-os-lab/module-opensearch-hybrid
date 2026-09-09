from __future__ import annotations

from typing import Any

import pytest

from poc.radial_similarity_calibration import evaluate_capture
from poc.radial_similarity_capture import radial_capture_qualification
from poc.radial_similarity_judgments import (
    bind_completed_judgments,
    prepare_judgment_template,
)


def test_calibration_evaluator_applies_registered_quality_and_latency_gates() -> None:
    capture = _capture(
        cases=[
            {
                "case_id": "head-shoe",
                "slices": ["head", "simple"],
                "exact_ids": [2, 3],
                "exact_total": 2,
                "radial_ids": [2, 3],
                "radial_latency_ms": [20.0, 22.0, 24.0, 21.0],
                "unsafe_ids": [],
            },
            {
                "case_id": "tail-empty",
                "slices": ["tail", "sparse"],
                "exact_ids": [],
                "exact_total": 0,
                "radial_ids": [],
                "radial_latency_ms": [18.0, 19.0, 20.0, 18.5],
                "unsafe_ids": [],
            },
        ]
    )

    report = evaluate_capture(capture)
    candidate = report["candidates"][0]

    assert candidate["mean_ann_recall_at_limit"] == pytest.approx(1.0)
    assert candidate["ann_false_positive_rate"] == pytest.approx(0.0)
    assert candidate["merchant_quality"] == {
        "mean_precision_at_limit": 1.0,
        "mean_recall_against_judged_pool_at_limit": 1.0,
        "mean_normalized_gain_at_limit": 1.0,
        "false_positive_count": 0,
        "false_positive_rate": 0.0,
    }
    assert candidate["empty_result_rate"] == pytest.approx(0.5)
    assert candidate["latency_p95_ms"] <= 24.0
    assert candidate["slice_ann_recall"] == {
        "head": 1.0,
        "simple": 1.0,
        "sparse": 1.0,
        "tail": 1.0,
    }
    assert all(candidate["gates"].values())
    assert candidate["eligible"] is True
    assert report["quality_evidence_complete"] is True
    assert report["accepted_profile"] is None
    assert report["requires_human_selection"] is True


def test_calibration_evaluator_rejects_false_positives_and_low_slice_recall() -> None:
    capture = _capture(
        cases=[
            {
                "case_id": "unsafe-tail",
                "slices": ["tail"],
                "exact_ids": [2, 3],
                "exact_total": 2,
                "radial_ids": [2, 99],
                "radial_latency_ms": [120.0] * 4,
                "unsafe_ids": [99],
            }
        ]
    )

    candidate = evaluate_capture(capture)["candidates"][0]

    assert candidate["mean_ann_recall_at_limit"] == pytest.approx(0.5)
    assert candidate["ann_false_positive_rate"] == pytest.approx(0.5)
    assert candidate["merchant_quality"]["false_positive_rate"] == pytest.approx(0.5)
    assert candidate["unsafe_hit_count"] == 1
    assert candidate["gates"] == {
        "mean_recall_at_least_0_97": False,
        "every_slice_recall_at_least_0_90": False,
        "latency_p95_at_most_100_ms": False,
        "bounded_result_count": True,
        "no_unsafe_hits": False,
        "complete_merchant_judgments": True,
        "registered_capture_scope": True,
        "representative_merchant_catalog": True,
    }
    assert candidate["eligible"] is False


def test_calibration_evaluator_refuses_truncated_exact_membership() -> None:
    capture = _capture(
        cases=[
            {
                "case_id": "truncated",
                "slices": ["head"],
                "exact_ids": [2, 3],
                "exact_total": 3,
                "radial_ids": [2, 3],
                "radial_latency_ms": [20.0] * 4,
                "unsafe_ids": [],
            }
        ]
    )

    with pytest.raises(ValueError, match="exact membership capture is truncated"):
        evaluate_capture(capture)


def test_holdout_capture_can_verify_only_one_frozen_threshold() -> None:
    capture = _capture(cases=[], bind=False)
    capture["split"] = "holdout"
    capture["thresholds"] = [
        {"min_score": 0.9, "cases": [{}]},
        {"min_score": 0.92, "cases": [{}]},
    ]

    with pytest.raises(ValueError, match="one frozen threshold"):
        evaluate_capture(capture)


def test_unjudged_technical_capture_cannot_nominate_a_threshold() -> None:
    capture = _capture(
        cases=[
            {
                "case_id": "unjudged",
                "slices": ["head"],
                "exact_ids": [2],
                "exact_total": 1,
                "radial_ids": [2],
                "radial_latency_ms": [20.0] * 4,
                "unsafe_ids": [],
            }
        ],
        bind=False,
    )

    report = evaluate_capture(capture)

    assert report["quality_evidence_complete"] is False
    assert report["technical_eligible_candidates"] == [0.9]
    assert report["eligible_candidates"] == []
    assert report["candidates"][0]["gates"]["complete_merchant_judgments"] is False


def test_smoke_capture_cannot_nominate_a_threshold_even_when_quality_passes() -> None:
    capture = _capture(
        cases=[
            {
                "case_id": "fixture-smoke",
                "slices": ["fixture-smoke", "simple"],
                "exact_ids": [2],
                "exact_total": 1,
                "radial_ids": [2],
                "radial_latency_ms": [20.0] * 4,
                "unsafe_ids": [],
            }
        ],
        qualification_scope="smoke",
    )

    report = evaluate_capture(capture)

    assert report["qualification"]["scope"] == "smoke"
    assert report["qualification"]["decision_eligible"] is False
    assert report["technical_eligible_candidates"] == []
    assert report["eligible_candidates"] == []
    assert report["candidates"][0]["gates"]["registered_capture_scope"] is False
    assert report["candidates"][0]["eligible"] is False


def test_synthetic_execution_capture_cannot_nominate_a_merchant_threshold() -> None:
    capture = _capture(
        cases=[
            {
                "case_id": "synthetic-registered-case",
                "slices": ["synthetic-registered-fixture"],
                "exact_ids": [2],
                "exact_total": 1,
                "radial_ids": [2],
                "radial_latency_ms": [20.0] * 4,
                "unsafe_ids": [],
            }
        ]
    )
    capture["capture_provenance"]["catalog_evidence_scope"] = "synthetic_execution"
    capture["qualification"] = radial_capture_qualification(
        "registered",
        capture["capture_provenance"],
    )

    report = evaluate_capture(capture)

    assert report["technical_eligible_candidates"] == [0.9]
    assert report["eligible_candidates"] == []
    assert report["qualification"]["merchant_decision_eligible"] is False
    assert report["candidates"][0]["gates"]["representative_merchant_catalog"] is False


def test_calibration_evaluator_refuses_tampered_qualification_evidence() -> None:
    capture = _capture(
        cases=[
            {
                "case_id": "registered-case",
                "slices": ["head"],
                "exact_ids": [2],
                "exact_total": 1,
                "radial_ids": [2],
                "radial_latency_ms": [20.0] * 4,
                "unsafe_ids": [],
            }
        ]
    )
    capture["qualification"]["decision_eligible"] = False

    with pytest.raises(ValueError, match="qualification evidence is inconsistent"):
        evaluate_capture(capture)


def _capture(
    *,
    cases: list[dict[str, Any]],
    bind: bool = True,
    qualification_scope: str = "registered",
) -> dict[str, Any]:
    for position, case in enumerate(cases, start=1):
        case.setdefault("seed_product_id", 100 + position)
    registered = qualification_scope == "registered"
    provenance = {
        "fixture": True,
        "catalog_evidence_scope": "representative_merchant",
        "opensearch_version": "3.8.0",
        "index_document_count": 50_000 if registered else 148,
        "runtime_system": "Linux",
        "runtime_architecture": "x86_64" if registered else "arm64",
        "loopback_endpoint": registered,
    }
    qualification = radial_capture_qualification(qualification_scope, provenance)
    capture: dict[str, Any] = {
        "schema_version": 1,
        "split": "development",
        "storefront_limit": 12,
        "maximum_result_count": 20,
        "concurrency": 4,
        "identity": {
            "model_id": "fixture/model",
            "model_revision": "revision-1",
            "dimension": 256,
            "similarity": "cosine",
            "source_recipe_version": "recipe-1",
            "store_id": 2,
            "use_case": "substitute",
            "judgment_set_sha256": "a" * 64,
        },
        "thresholds": [{"min_score": 0.9, "cases": cases}],
        "qualification": qualification,
        "capture_provenance": provenance,
    }
    if not bind or not cases:
        return capture
    template = prepare_judgment_template(capture)
    capture_cases = {str(case["case_id"]): case for case in cases}
    for template_case in template["cases"]:
        capture_case = capture_cases[str(template_case["case_id"])]
        exact_ids = set(capture_case["exact_ids"])
        unsafe_ids = set(capture_case["unsafe_ids"])
        for candidate in template_case["candidates"]:
            product_id = candidate["product_id"]
            candidate["label"] = (
                "unsafe"
                if product_id in unsafe_ids
                else "substitute"
                if product_id in exact_ids
                else "irrelevant"
            )
    return bind_completed_judgments(capture, template)
