from __future__ import annotations

import re
import statistics
from typing import Any, cast

from poc.latency import summarize_latency_ms
from poc.manifest import canonical_sha256
from poc.radial_similarity_capture import radial_capture_qualification
from poc.radial_similarity_judgments import judgment_labels


def evaluate_capture(capture: dict[str, Any]) -> dict[str, Any]:
    _validate_capture_header(capture)
    qualification = _validated_qualification(capture)
    merchant_labels = judgment_labels(capture)
    storefront_limit = int(capture["storefront_limit"])
    maximum_result_count = int(capture["maximum_result_count"])
    concurrency = int(capture["concurrency"])
    thresholds = cast(list[dict[str, Any]], capture["thresholds"])
    candidates = [
        _evaluate_candidate(
            threshold,
            storefront_limit=storefront_limit,
            maximum_result_count=maximum_result_count,
            concurrency=concurrency,
            merchant_labels=merchant_labels,
            qualification_eligible=bool(qualification["decision_eligible"]),
            merchant_qualification_eligible=bool(
                qualification["merchant_decision_eligible"]
            ),
        )
        for threshold in thresholds
    ]

    return {
        "schema_version": 2,
        "evaluation": "mageos-opensearch-hybrid-radial-calibration",
        "split": capture["split"],
        "identity": capture["identity"],
        "qualification": qualification,
        "capture_sha256": canonical_sha256(capture),
        "storefront_limit": storefront_limit,
        "maximum_result_count": maximum_result_count,
        "concurrency": concurrency,
        "candidates": candidates,
        "quality_evidence_complete": merchant_labels is not None,
        "technical_eligible_candidates": [
            candidate["min_score"] for candidate in candidates if candidate["technical_eligible"]
        ],
        "eligible_candidates": [
            candidate["min_score"] for candidate in candidates if candidate["eligible"]
        ],
        "accepted_profile": None,
        "requires_human_selection": True,
    }


def _evaluate_candidate(
    threshold: dict[str, Any],
    *,
    storefront_limit: int,
    maximum_result_count: int,
    concurrency: int,
    merchant_labels: dict[str, dict[int, str]] | None,
    qualification_eligible: bool,
    merchant_qualification_eligible: bool,
) -> dict[str, Any]:
    min_score = threshold.get("min_score")
    cases = threshold.get("cases")
    if (
        (not isinstance(min_score, int) and not isinstance(min_score, float))
        or not 0.0 < float(min_score) <= 1.0
        or not isinstance(cases, list)
        or not cases
    ):
        raise ValueError("every threshold requires a valid min_score and non-empty cases")
    recalls: list[float] = []
    ann_false_positives = 0
    returned = 0
    result_counts: list[float] = []
    latencies: list[float] = []
    slice_recalls: dict[str, list[float]] = {}
    unsafe_hit_count = 0
    merchant_precisions: list[float] = []
    merchant_recalls: list[float] = []
    merchant_gains: list[float] = []
    merchant_false_positives = 0
    merchant_returned = 0
    case_ids: set[str] = set()
    case_reports: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("every calibration case must be an object")
        case_id = case.get("case_id")
        slices = case.get("slices")
        exact_ids = _positive_unique_ids(case.get("exact_ids"), "exact_ids")
        radial_ids = _positive_unique_ids(case.get("radial_ids"), "radial_ids")
        unsafe_ids = _positive_unique_ids(case.get("unsafe_ids"), "unsafe_ids")
        exact_total = case.get("exact_total")
        case_latencies = case.get("radial_latency_ms")
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ValueError("calibration case IDs must be unique non-empty strings")
        case_ids.add(case_id)
        if (
            not isinstance(slices, list)
            or not slices
            or any(not isinstance(slice_name, str) or not slice_name for slice_name in slices)
        ):
            raise ValueError("every calibration case requires non-empty slice names")
        if not isinstance(exact_total, int) or exact_total < 0:
            raise ValueError("every calibration case requires an exact membership total")
        if exact_total != len(exact_ids):
            raise ValueError("the exact membership capture is truncated")
        if len(radial_ids) > maximum_result_count:
            raise ValueError("a radial capture exceeds the maximum result count")
        if (
            not isinstance(case_latencies, list)
            or len(case_latencies) < concurrency
            or any(
                (not isinstance(latency, int) and not isinstance(latency, float))
                or float(latency) < 0.0
                for latency in case_latencies
            )
        ):
            raise ValueError("every calibration case requires bounded concurrency latency samples")
        exact_at_limit = exact_ids[:storefront_limit]
        radial_at_limit = radial_ids[:storefront_limit]
        exact_at_limit_set = set(exact_at_limit)
        recall = (
            len(exact_at_limit_set.intersection(radial_at_limit)) / len(exact_at_limit_set)
            if exact_at_limit_set
            else 1.0
            if not radial_at_limit
            else 0.0
        )
        case_ann_false_positives = len(set(radial_ids).difference(exact_ids))
        case_unsafe_hits = len(set(radial_ids).intersection(unsafe_ids))
        merchant_report: dict[str, Any] | None = None
        if merchant_labels is not None:
            labels = merchant_labels[case_id]
            radial_at_limit_labels = [labels[product_id] for product_id in radial_at_limit]
            relevant_pool = {
                product_id
                for product_id, label in labels.items()
                if label in ("substitute", "acceptable")
            }
            relevant_returned = sum(
                label in ("substitute", "acceptable") for label in radial_at_limit_labels
            )
            merchant_precision = (
                relevant_returned / len(radial_at_limit) if radial_at_limit else 1.0
            )
            merchant_recall = (
                len(set(radial_at_limit).intersection(relevant_pool)) / len(relevant_pool)
                if relevant_pool
                else 1.0
            )
            gains = {"substitute": 2, "acceptable": 1, "irrelevant": 0, "unsafe": 0}
            maximum_gain = sum(
                sorted((gains[label] for label in labels.values()), reverse=True)[:storefront_limit]
            )
            returned_gain = sum(gains[label] for label in radial_at_limit_labels)
            normalized_gain = returned_gain / maximum_gain if maximum_gain else 1.0
            merchant_irrelevant = sum(label == "irrelevant" for label in radial_at_limit_labels)
            merchant_unsafe = sum(label == "unsafe" for label in radial_at_limit_labels)
            merchant_false_positives += merchant_irrelevant + merchant_unsafe
            merchant_returned += len(radial_at_limit)
            merchant_precisions.append(merchant_precision)
            merchant_recalls.append(merchant_recall)
            merchant_gains.append(normalized_gain)
            case_unsafe_hits = merchant_unsafe
            merchant_report = {
                "precision_at_limit": merchant_precision,
                "recall_against_judged_pool_at_limit": merchant_recall,
                "normalized_gain_at_limit": normalized_gain,
                "relevant_result_count": relevant_returned,
                "irrelevant_result_count": merchant_irrelevant,
                "unsafe_result_count": merchant_unsafe,
                "judged_pool_relevant_count": len(relevant_pool),
            }
        recalls.append(recall)
        returned += len(radial_ids)
        ann_false_positives += case_ann_false_positives
        unsafe_hit_count += case_unsafe_hits
        result_counts.append(float(len(radial_ids)))
        latencies.extend(float(latency) for latency in case_latencies)
        for slice_name in sorted(set(cast(list[str], slices))):
            slice_recalls.setdefault(slice_name, []).append(recall)
        case_reports.append(
            {
                "case_id": case_id,
                "ann_recall_at_limit": recall,
                "ann_false_positive_count": case_ann_false_positives,
                "unsafe_hit_count": case_unsafe_hits,
                "exact_result_count": len(exact_ids),
                "radial_result_count": len(radial_ids),
                "empty_result": len(radial_ids) == 0,
                "merchant_quality": merchant_report,
            }
        )
    mean_recall = statistics.fmean(recalls)
    mean_slice_recall = {
        slice_name: statistics.fmean(values) for slice_name, values in sorted(slice_recalls.items())
    }
    latency_summary = summarize_latency_ms(latencies)
    result_count_summary = summarize_latency_ms(result_counts)
    gates = {
        "mean_recall_at_least_0_97": mean_recall >= 0.97,
        "every_slice_recall_at_least_0_90": min(mean_slice_recall.values()) >= 0.90,
        "latency_p95_at_most_100_ms": concurrency == 4
        and float(latency_summary["p95_ms"]) <= 100.0,
        "bounded_result_count": int(result_count_summary["maximum_ms"]) <= maximum_result_count,
        "no_unsafe_hits": unsafe_hit_count == 0,
        "complete_merchant_judgments": merchant_labels is not None,
        "registered_capture_scope": qualification_eligible,
        "representative_merchant_catalog": merchant_qualification_eligible,
    }

    technical_gate_names = (
        "mean_recall_at_least_0_97",
        "every_slice_recall_at_least_0_90",
        "latency_p95_at_most_100_ms",
        "bounded_result_count",
        "registered_capture_scope",
    )
    technical_eligible = all(gates[name] for name in technical_gate_names)

    return {
        "min_score": float(min_score),
        "case_count": len(cases),
        "mean_ann_recall_at_limit": mean_recall,
        "minimum_slice_ann_recall": min(mean_slice_recall.values()),
        "slice_ann_recall": mean_slice_recall,
        "ann_false_positive_count": ann_false_positives,
        "ann_false_positive_rate": ann_false_positives / returned if returned else 0.0,
        "merchant_quality": {
            "mean_precision_at_limit": statistics.fmean(merchant_precisions),
            "mean_recall_against_judged_pool_at_limit": statistics.fmean(merchant_recalls),
            "mean_normalized_gain_at_limit": statistics.fmean(merchant_gains),
            "false_positive_count": merchant_false_positives,
            "false_positive_rate": merchant_false_positives / merchant_returned
            if merchant_returned
            else 0.0,
        }
        if merchant_labels is not None
        else None,
        "result_count": {
            "p50": result_count_summary["p50_ms"],
            "p95": result_count_summary["p95_ms"],
            "maximum": int(result_count_summary["maximum_ms"]),
        },
        "empty_result_rate": sum(report["empty_result"] for report in case_reports)
        / len(case_reports),
        "latency_p95_ms": latency_summary["p95_ms"],
        "unsafe_hit_count": unsafe_hit_count,
        "gates": gates,
        "technical_eligible": technical_eligible,
        "eligible": all(gates.values()),
        "cases": case_reports,
    }


def _validate_capture_header(capture: dict[str, Any]) -> None:
    identity = capture.get("identity")
    thresholds = capture.get("thresholds")
    if capture.get("schema_version") not in (1, 2):
        raise ValueError("the calibration capture schema version is unsupported")
    if capture.get("split") not in ("development", "holdout"):
        raise ValueError("the calibration split must be development or holdout")
    if (
        not isinstance(capture.get("storefront_limit"), int)
        or not isinstance(capture.get("maximum_result_count"), int)
        or not 0 < int(capture["storefront_limit"]) <= int(capture["maximum_result_count"]) <= 20
    ):
        raise ValueError("the calibration result bounds are invalid")
    if not isinstance(capture.get("concurrency"), int) or int(capture["concurrency"]) <= 0:
        raise ValueError("the calibration concurrency is invalid")
    if not isinstance(thresholds, list) or not thresholds:
        raise ValueError("the calibration capture requires threshold candidates")
    if capture["split"] == "holdout" and len(thresholds) != 1:
        raise ValueError("a holdout capture may verify only one frozen threshold")
    raw_min_scores = [
        threshold.get("min_score") if isinstance(threshold, dict) else None
        for threshold in thresholds
    ]
    if any(
        not isinstance(min_score, int) and not isinstance(min_score, float)
        for min_score in raw_min_scores
    ):
        raise ValueError("calibration threshold candidates require numeric min_score values")
    min_scores = [float(cast(int | float, min_score)) for min_score in raw_min_scores]
    if len(set(min_scores)) != len(min_scores):
        raise ValueError("calibration threshold candidates must be unique")
    if not isinstance(identity, dict):
        raise ValueError("the calibration capture identity is missing")
    required_strings = (
        "model_id",
        "model_revision",
        "similarity",
        "source_recipe_version",
        "use_case",
    )
    if any(
        not isinstance(identity.get(field), str) or not identity[field]
        for field in required_strings
    ):
        raise ValueError("the calibration identity is incomplete")
    if not isinstance(identity.get("dimension"), int) or int(identity["dimension"]) <= 0:
        raise ValueError("the calibration dimension is invalid")
    if not isinstance(identity.get("store_id"), int) or int(identity["store_id"]) <= 0:
        raise ValueError("the calibration store scope is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", str(identity.get("judgment_set_sha256", ""))) is None:
        raise ValueError("the calibration judgment-set identity is invalid")


def _validated_qualification(capture: dict[str, Any]) -> dict[str, Any]:
    qualification = capture.get("qualification")
    provenance = capture.get("capture_provenance")
    if not isinstance(qualification, dict) or not isinstance(provenance, dict):
        raise ValueError("the calibration qualification evidence is missing")
    expected = radial_capture_qualification(str(qualification.get("scope", "")), provenance)
    if qualification != expected:
        raise ValueError("the calibration qualification evidence is inconsistent")
    return expected


def _positive_unique_ids(value: Any, field: str) -> list[int]:
    if (
        not isinstance(value, list)
        or any(not isinstance(product_id, int) or product_id <= 0 for product_id in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{field} must contain unique positive product IDs")

    return value
