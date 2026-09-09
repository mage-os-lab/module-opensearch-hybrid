from __future__ import annotations

import copy
from typing import Any

import pytest

from poc.radial_similarity_judgments import (
    bind_completed_judgments,
    judgment_labels,
    prepare_judgment_template,
)


def test_template_blinds_scores_and_threshold_membership() -> None:
    capture = _capture()

    template = prepare_judgment_template(capture)

    assert template["capture_sha256"]
    assert template["instructions"]["score_blinded"] is True
    case = template["cases"][0]
    assert case["candidates"] == [
        {
            "product_id": 2,
            "sku": "sku-2",
            "title": "Product 2",
            "product_class": "simple",
            "category": "Category",
            "brand": "Fixture",
            "label": None,
            "notes": "",
        },
        {
            "product_id": 3,
            "sku": "sku-3",
            "title": "Product 3",
            "product_class": "simple",
            "category": "Category",
            "brand": "Fixture",
            "label": None,
            "notes": "",
        },
        {
            "product_id": 4,
            "sku": "sku-4",
            "title": "Product 4",
            "product_class": "simple",
            "category": "Category",
            "brand": "Fixture",
            "label": None,
            "notes": "",
        },
    ]
    assert case["seed"] == {
        "product_id": 1,
        "sku": "sku-1",
        "title": "Product 1",
        "product_class": "simple",
        "category": "Category",
        "brand": "Fixture",
    }
    encoded = str(template)
    assert "min_score" not in encoded
    assert "exact_ids" not in encoded
    assert "radial_ids" not in encoded


def test_binding_requires_complete_labels_and_binds_exact_capture() -> None:
    capture = _capture()
    completed = prepare_judgment_template(capture)
    labels = {2: "substitute", 3: "acceptable", 4: "irrelevant"}
    for candidate in completed["cases"][0]["candidates"]:
        candidate["label"] = labels[candidate["product_id"]]

    bound = bind_completed_judgments(capture, completed)

    assert bound["schema_version"] == 2
    digest = bound["identity"]["judgment_set_sha256"]
    assert len(digest) == 64
    assert bound["merchant_judgments"]["judgment_set_sha256"] == digest
    assert bound["capture_provenance"]["technical_capture_sha256"] == completed["capture_sha256"]
    assert (
        bound["merchant_judgments"]["identity"]["technical_capture_sha256"]
        == completed["capture_sha256"]
    )
    assert judgment_labels(bound) == {"seed-a": {2: "substitute", 3: "acceptable", 4: "irrelevant"}}

    changed_capture = copy.deepcopy(capture)
    changed_capture["thresholds"][0]["cases"][0]["radial_ids"] = [2]
    with pytest.raises(ValueError, match="exact technical capture"):
        bind_completed_judgments(changed_capture, completed)


def test_binding_refuses_incomplete_or_invalid_labels() -> None:
    capture = _capture()
    incomplete = prepare_judgment_template(capture)
    incomplete["cases"][0]["candidates"][0]["label"] = "substitute"

    with pytest.raises(ValueError, match="one valid label"):
        bind_completed_judgments(capture, incomplete)

    invalid = prepare_judgment_template(capture)
    for candidate in invalid["cases"][0]["candidates"]:
        candidate["label"] = "substitute"
    invalid["cases"][0]["candidates"][0]["label"] = "maybe"
    with pytest.raises(ValueError, match="one valid label"):
        bind_completed_judgments(capture, invalid)


def test_binding_refuses_tampered_review_context() -> None:
    capture = _capture()
    completed = prepare_judgment_template(capture)
    for candidate in completed["cases"][0]["candidates"]:
        candidate["label"] = "substitute"
    completed["cases"][0]["candidates"][0]["title"] = "Misleading title"

    with pytest.raises(ValueError, match="review context"):
        bind_completed_judgments(capture, completed)


def test_bound_capture_detects_judgment_tampering() -> None:
    capture = _capture()
    completed = prepare_judgment_template(capture)
    for candidate in completed["cases"][0]["candidates"]:
        candidate["label"] = "substitute"
    bound = bind_completed_judgments(capture, completed)
    bound["merchant_judgments"]["identity"]["cases"][0]["judgments"][0]["label"] = "unsafe"

    with pytest.raises(ValueError, match="identity does not match"):
        judgment_labels(bound)


def _capture() -> dict[str, Any]:
    base_case = {
        "case_id": "seed-a",
        "seed_product_id": 1,
        "slices": ["head", "simple"],
        "exact_ids": [2, 3, 4],
        "exact_total": 3,
        "radial_latency_ms": [10.0, 11.0, 12.0, 13.0],
        "unsafe_ids": [],
    }
    return {
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
        "thresholds": [
            {"min_score": 0.92, "cases": [{**base_case, "radial_ids": [2, 3]}]},
            {"min_score": 0.88, "cases": [{**base_case, "radial_ids": [2, 3, 4]}]},
        ],
        "review_catalog": [
            {
                "product_id": product_id,
                "sku": f"sku-{product_id}",
                "title": f"Product {product_id}",
                "product_class": "simple",
                "category": "Category",
                "brand": "Fixture",
            }
            for product_id in (1, 2, 3, 4)
        ],
        "capture_provenance": {"fixture": True},
    }
