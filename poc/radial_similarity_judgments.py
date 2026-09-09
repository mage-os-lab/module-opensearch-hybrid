from __future__ import annotations

import copy
from typing import Any, cast

from poc.manifest import canonical_sha256

JUDGMENT_KIND = "mageos-opensearch-hybrid-radial-merchant-judgments"
JUDGMENT_LABELS = frozenset({"substitute", "acceptable", "irrelevant", "unsafe"})
REVIEW_CONTEXT_FIELDS = ("sku", "title", "product_class", "category", "brand")


def prepare_judgment_template(capture: dict[str, Any]) -> dict[str, Any]:
    cases = _capture_case_pool(capture)
    review_catalog = _review_catalog(capture, cases)
    return {
        "schema_version": 1,
        "kind": JUDGMENT_KIND,
        "capture_sha256": canonical_sha256(capture),
        "split": capture["split"],
        "instructions": {
            "labels": sorted(JUDGMENT_LABELS),
            "score_blinded": True,
            "complete_every_candidate": True,
        },
        "cases": [_judgment_case(case, review_catalog) for case in cases],
    }


def _judgment_case(
    case: dict[str, Any],
    review_catalog: dict[int, dict[str, str | int]],
) -> dict[str, Any]:
    output = {
        "case_id": case["case_id"],
        "seed_product_id": case["seed_product_id"],
        "slices": case["slices"],
        "candidates": [
            {
                **review_catalog.get(product_id, {"product_id": product_id}),
                "label": None,
                "notes": "",
            }
            for product_id in case["candidate_ids"]
        ],
    }
    seed = review_catalog.get(int(case["seed_product_id"]))
    if seed is not None:
        output["seed"] = seed
    return output


def bind_completed_judgments(
    capture: dict[str, Any],
    completed: dict[str, Any],
) -> dict[str, Any]:
    expected = prepare_judgment_template(capture)
    if completed.get("schema_version") != 1 or completed.get("kind") != JUDGMENT_KIND:
        raise ValueError("the merchant judgment document contract is invalid")
    if completed.get("capture_sha256") != expected["capture_sha256"]:
        raise ValueError("merchant judgments do not match the exact technical capture")
    if completed.get("split") != capture.get("split"):
        raise ValueError("merchant judgment split does not match the technical capture")
    normalized_cases = _normalize_completed_cases(expected, completed)
    identity = capture.get("identity")
    provenance = capture.get("capture_provenance")
    if not isinstance(identity, dict) or not isinstance(provenance, dict):
        raise ValueError("the technical capture identity or provenance is missing")
    judgment_identity = {
        "schema_version": 1,
        "technical_capture_sha256": expected["capture_sha256"],
        "split": capture["split"],
        "store_id": identity["store_id"],
        "use_case": identity["use_case"],
        "cases": normalized_cases,
    }
    judgment_set_sha256 = canonical_sha256(judgment_identity)

    bound = copy.deepcopy(capture)
    bound["schema_version"] = 2
    bound_identity = cast(dict[str, Any], bound["identity"])
    bound_identity["judgment_set_sha256"] = judgment_set_sha256
    bound["merchant_judgments"] = {
        "schema_version": 1,
        "kind": JUDGMENT_KIND,
        "judgment_set_sha256": judgment_set_sha256,
        "completed_document_sha256": canonical_sha256(completed),
        "identity": judgment_identity,
    }
    labels_by_case = {
        case["case_id"]: {
            judgment["product_id"]: judgment["label"] for judgment in case["judgments"]
        }
        for case in normalized_cases
    }
    for threshold in cast(list[dict[str, Any]], bound["thresholds"]):
        for case in cast(list[dict[str, Any]], threshold["cases"]):
            labels = labels_by_case[str(case["case_id"])]
            case["unsafe_ids"] = sorted(
                product_id for product_id, label in labels.items() if label == "unsafe"
            )
    bound_provenance = cast(dict[str, Any], bound["capture_provenance"])
    bound_provenance["technical_capture_sha256"] = expected["capture_sha256"]
    return bound


def judgment_labels(capture: dict[str, Any]) -> dict[str, dict[int, str]] | None:
    if capture.get("schema_version") == 1:
        return None
    merchant = capture.get("merchant_judgments")
    identity = capture.get("identity")
    provenance = capture.get("capture_provenance")
    if (
        not isinstance(merchant, dict)
        or not isinstance(identity, dict)
        or not isinstance(provenance, dict)
    ):
        raise ValueError("schema-2 calibration capture requires merchant judgments")
    judgment_identity = merchant.get("identity")
    if not isinstance(judgment_identity, dict):
        raise ValueError("merchant judgment identity is missing")
    digest = canonical_sha256(judgment_identity)
    if (
        merchant.get("schema_version") != 1
        or merchant.get("kind") != JUDGMENT_KIND
        or merchant.get("judgment_set_sha256") != digest
        or identity.get("judgment_set_sha256") != digest
        or judgment_identity.get("technical_capture_sha256")
        != provenance.get("technical_capture_sha256")
        or judgment_identity.get("split") != capture.get("split")
        or judgment_identity.get("store_id") != identity.get("store_id")
        or judgment_identity.get("use_case") != identity.get("use_case")
    ):
        raise ValueError("merchant judgment identity does not match the capture")
    cases = judgment_identity.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("merchant judgment cases are missing")
    labels_by_case: dict[str, dict[int, str]] = {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("merchant judgment case is invalid")
        case_id = case.get("case_id")
        judgments = case.get("judgments")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in labels_by_case
            or not isinstance(judgments, list)
        ):
            raise ValueError("merchant judgment case identity is invalid")
        labels: dict[int, str] = {}
        for judgment in judgments:
            if not isinstance(judgment, dict):
                raise ValueError("merchant judgment entry is invalid")
            product_id = judgment.get("product_id")
            label = judgment.get("label")
            if (
                not isinstance(product_id, int)
                or product_id <= 0
                or product_id in labels
                or label not in JUDGMENT_LABELS
            ):
                raise ValueError("merchant judgment label is invalid")
            labels[product_id] = str(label)
        labels_by_case[case_id] = labels
    _validate_capture_coverage(capture, labels_by_case)
    return labels_by_case


def _capture_case_pool(capture: dict[str, Any]) -> list[dict[str, Any]]:
    if capture.get("schema_version") != 1:
        raise ValueError("judgment preparation requires an unbound schema-1 technical capture")
    if capture.get("split") not in ("development", "holdout"):
        raise ValueError("the technical capture split is invalid")
    thresholds = capture.get("thresholds")
    if not isinstance(thresholds, list) or not thresholds:
        raise ValueError("the technical capture has no thresholds")
    cases_by_id: dict[str, dict[str, Any]] = {}
    case_order: list[str] = []
    for threshold in thresholds:
        if not isinstance(threshold, dict) or not isinstance(threshold.get("cases"), list):
            raise ValueError("the technical capture threshold is invalid")
        seen_threshold_cases: set[str] = set()
        for case in cast(list[Any], threshold["cases"]):
            if not isinstance(case, dict):
                raise ValueError("the technical capture case is invalid")
            case_id = case.get("case_id")
            seed_product_id = case.get("seed_product_id")
            slices = case.get("slices")
            radial_ids = case.get("radial_ids")
            if (
                not isinstance(case_id, str)
                or not case_id
                or case_id in seen_threshold_cases
                or not isinstance(seed_product_id, int)
                or seed_product_id <= 0
                or not isinstance(slices, list)
                or not slices
                or any(not isinstance(value, str) or not value for value in slices)
                or not isinstance(radial_ids, list)
                or any(not isinstance(value, int) or value <= 0 for value in radial_ids)
                or len(set(radial_ids)) != len(radial_ids)
            ):
                raise ValueError("the technical capture case contract is invalid")
            seen_threshold_cases.add(case_id)
            existing = cases_by_id.get(case_id)
            if existing is None:
                existing = {
                    "case_id": case_id,
                    "seed_product_id": seed_product_id,
                    "slices": sorted(set(cast(list[str], slices))),
                    "candidate_ids": set(),
                }
                cases_by_id[case_id] = existing
                case_order.append(case_id)
            elif existing["seed_product_id"] != seed_product_id or existing["slices"] != sorted(
                set(cast(list[str], slices))
            ):
                raise ValueError("the technical capture case changed between thresholds")
            cast(set[int], existing["candidate_ids"]).update(cast(list[int], radial_ids))
        if set(case_order) != seen_threshold_cases:
            raise ValueError("every threshold must contain the same technical capture cases")
    return [
        {**cases_by_id[case_id], "candidate_ids": sorted(cases_by_id[case_id]["candidate_ids"])}
        for case_id in case_order
    ]


def _normalize_completed_cases(
    expected: dict[str, Any],
    completed: dict[str, Any],
) -> list[dict[str, Any]]:
    completed_cases = completed.get("cases")
    if not isinstance(completed_cases, list):
        raise ValueError("merchant judgment cases are missing")
    completed_by_id: dict[str, dict[str, Any]] = {}
    for case in completed_cases:
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
            raise ValueError("merchant judgment case is invalid")
        case_id = str(case["case_id"])
        if case_id in completed_by_id:
            raise ValueError("merchant judgment case IDs must be unique")
        completed_by_id[case_id] = case
    expected_cases = cast(list[dict[str, Any]], expected["cases"])
    if set(completed_by_id) != {str(case["case_id"]) for case in expected_cases}:
        raise ValueError("merchant judgment cases do not match the technical capture")

    normalized: list[dict[str, Any]] = []
    for expected_case in expected_cases:
        case_id = str(expected_case["case_id"])
        case = completed_by_id[case_id]
        if (
            case.get("seed_product_id") != expected_case["seed_product_id"]
            or case.get("slices") != expected_case["slices"]
        ):
            raise ValueError("merchant judgment case metadata does not match the capture")
        if case.get("seed") != expected_case.get("seed"):
            raise ValueError("merchant judgment review context does not match the capture")
        candidates = case.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("merchant judgment candidates are missing")
        labels: dict[int, str] = {}
        expected_by_id = {
            int(candidate["product_id"]): candidate
            for candidate in cast(list[dict[str, Any]], expected_case["candidates"])
        }
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError("merchant judgment candidate is invalid")
            product_id = candidate.get("product_id")
            label = candidate.get("label")
            if (
                not isinstance(product_id, int)
                or product_id <= 0
                or product_id in labels
                or label not in JUDGMENT_LABELS
            ):
                raise ValueError("every merchant candidate requires one valid label")
            expected_candidate = expected_by_id.get(product_id)
            review_context = {
                key: value
                for key, value in candidate.items()
                if key not in ("label", "notes")
            }
            expected_context = {
                key: value
                for key, value in expected_candidate.items()
                if key not in ("label", "notes")
            } if expected_candidate is not None else None
            if review_context != expected_context:
                raise ValueError("merchant judgment review context does not match the capture")
            labels[product_id] = str(label)
        expected_ids = {
            int(candidate["product_id"])
            for candidate in cast(list[dict[str, Any]], expected_case["candidates"])
        }
        if set(labels) != expected_ids:
            raise ValueError("merchant judgment candidate coverage is incomplete")
        normalized.append(
            {
                "case_id": case_id,
                "seed_product_id": expected_case["seed_product_id"],
                "slices": expected_case["slices"],
                "judgments": [
                    {"product_id": product_id, "label": labels[product_id]}
                    for product_id in sorted(labels)
                ],
            }
        )
    return normalized


def _review_catalog(
    capture: dict[str, Any],
    cases: list[dict[str, Any]],
) -> dict[int, dict[str, str | int]]:
    raw_catalog = capture.get("review_catalog")
    if raw_catalog is None:
        return {}
    if not isinstance(raw_catalog, list):
        raise ValueError("the technical capture review catalog is invalid")
    catalog: dict[int, dict[str, str | int]] = {}
    for item in raw_catalog:
        if not isinstance(item, dict):
            raise ValueError("the technical capture review catalog is invalid")
        product_id = item.get("product_id")
        if (
            not isinstance(product_id, int)
            or isinstance(product_id, bool)
            or product_id <= 0
            or product_id in catalog
            or set(item) != {"product_id", *REVIEW_CONTEXT_FIELDS}
            or any(not isinstance(item.get(field), str) for field in REVIEW_CONTEXT_FIELDS)
        ):
            raise ValueError("the technical capture review catalog is invalid")
        catalog[product_id] = cast(dict[str, str | int], item)
    expected_ids = {
        int(case["seed_product_id"])
        for case in cases
    }
    for case in cases:
        expected_ids.update(cast(set[int], case["candidate_ids"]))
    if set(catalog) != expected_ids:
        raise ValueError("the technical capture review catalog coverage is incomplete")
    return catalog


def _validate_capture_coverage(
    capture: dict[str, Any],
    labels_by_case: dict[str, dict[int, str]],
) -> None:
    expected = prepare_judgment_template({**capture, "schema_version": 1})
    expected_by_case = {
        str(case["case_id"]): {
            int(candidate["product_id"])
            for candidate in cast(list[dict[str, Any]], case["candidates"])
        }
        for case in cast(list[dict[str, Any]], expected["cases"])
    }
    if set(labels_by_case) != set(expected_by_case):
        raise ValueError("merchant judgment cases do not cover the capture")
    for case_id, expected_ids in expected_by_case.items():
        if set(labels_by_case[case_id]) != expected_ids:
            raise ValueError("merchant judgment candidates do not cover the capture")
