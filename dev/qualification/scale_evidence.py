"""Verify retained Phase 5 large-catalog characterization evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, cast

PHASE5_TARGETS = frozenset({100_000, 1_000_000})
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ScaleEvidenceError(ValueError):
    """The scale artifact does not satisfy the registered evidence contract."""


def verify_scale_evidence(
    path: Path,
    *,
    expected_target: int | None = None,
) -> dict[str, object]:
    """Validate one completed synthetic large-catalog characterization artifact."""
    if path.is_symlink() or not path.is_file():
        raise ScaleEvidenceError("scale evidence must be one regular file")
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScaleEvidenceError(f"scale evidence is unreadable: {error}") from error
    if not isinstance(evidence, dict):
        raise ScaleEvidenceError("scale evidence must be a JSON object")
    if evidence.get("schema_version") != 2 or evidence.get("status") != "passed":
        raise ScaleEvidenceError("scale evidence schema or completion status is invalid")
    if evidence.get("qualification_scope") != "phase5_scale":
        raise ScaleEvidenceError("scale evidence qualification scope is invalid")

    catalog = _object(evidence.get("catalog"), "catalog")
    target_products = _integer(catalog.get("target_products"), "target products", minimum=1)
    if target_products not in PHASE5_TARGETS:
        raise ScaleEvidenceError("Phase 5 target products must be 100000 or 1000000")
    if expected_target is not None and target_products != expected_target:
        raise ScaleEvidenceError("scale evidence target differs from the expected target")
    eligible_before = _integer(catalog.get("eligible_before"), "eligible before", minimum=0)
    created_products = _integer(catalog.get("created_products"), "created products", minimum=1)
    eligible_after = _integer(catalog.get("eligible_after"), "eligible after", minimum=1)
    fixture_products = _integer(catalog.get("fixture_products"), "fixture products", minimum=1)
    if (
        eligible_after != target_products
        or created_products != target_products - eligible_before
        or fixture_products != created_products
    ):
        raise ScaleEvidenceError("scale catalog counts are internally inconsistent")

    runtime = _object(evidence.get("runtime"), "runtime")
    if not str(runtime.get("mageos_version", "")).startswith("3.4."):
        raise ScaleEvidenceError("scale evidence must use Mage-OS 3.4")
    if (
        re.fullmatch(
            r"2\.4\.9(?:-p[0-9]+)?",
            str(runtime.get("magento_product_version", "")),
        )
        is None
    ):
        raise ScaleEvidenceError("scale evidence Magento product version is unsupported")
    if re.fullmatch(r"8\.[45]\.[0-9]+", str(runtime.get("php_version", ""))) is None:
        raise ScaleEvidenceError("scale evidence PHP version is unsupported")
    if re.fullmatch(r"[1-9][0-9]*[GM]", str(runtime.get("php_memory_limit", ""))) is None:
        raise ScaleEvidenceError("scale evidence PHP memory limit is invalid")
    if not str(runtime.get("opensearch_version", "")).startswith("3.8."):
        raise ScaleEvidenceError("scale evidence must use OpenSearch 3.8")
    _sha256(runtime.get("module_composer_sha256"), "module Composer digest")
    _integer(runtime.get("module_package_file_count"), "module package file count", minimum=1)
    _sha256(runtime.get("module_package_payload_sha256"), "module package payload digest")
    _sha256(runtime.get("module_package_archive_sha256"), "module package archive digest")
    if runtime.get("encoder_runtime") != "deterministic_hash_fake_only":
        raise ScaleEvidenceError("scale evidence encoder runtime is not the registered fake")

    generation = _object(evidence.get("generation"), "generation")
    generation_id = _integer(generation.get("generation_id"), "generation ID", minimum=1)
    processed_batches = _integer(
        generation.get("processed_batches"),
        "processed batches",
        minimum=1,
    )
    if (
        generation.get("state") != "READY"
        or generation.get("validation_ready") is not True
        or _integer(generation.get("coverage_total"), "coverage total", minimum=1)
        != target_products
        or _integer(generation.get("coverage_complete"), "coverage complete", minimum=0)
        != target_products
        or _integer(generation.get("coverage_failed"), "coverage failed", minimum=0) != 0
    ):
        raise ScaleEvidenceError("scale generation coverage or validation is incomplete")
    _sha256(generation.get("contract_digest"), "generation contract digest")
    _sha256(
        generation.get("result_contract_digest"),
        "generation result-contract digest",
    )

    timings = _object(evidence.get("timings_seconds"), "timings")
    _number(timings.get("product_insert"), "product insert time", minimum=0.0)
    _number(timings.get("native_reindex"), "native reindex time", minimum=0.0)
    _number(timings.get("hybrid_build"), "hybrid build time", minimum=0.001)

    graphql = _object(evidence.get("graphql"), "GraphQL")
    graphql_fixture_total = _integer(
        graphql.get("fixture_total"),
        "GraphQL fixture total",
        minimum=1,
    )
    if graphql_fixture_total != fixture_products:
        raise ScaleEvidenceError("GraphQL fixture total differs from the catalog fixture")
    for key in (
        "page_one_ms",
        "page_two_ms",
        "included_price_filter_ms",
        "excluded_price_filter_ms",
    ):
        _number(graphql.get(key), f"GraphQL {key}", minimum=0.0)
    for key in ("stable_pages", "price_filter", "price_aggregation"):
        if graphql.get(key) is not True:
            raise ScaleEvidenceError(f"GraphQL {key} check did not pass")

    process = _object(evidence.get("process"), "process")
    _integer(process.get("peak_memory_bytes"), "peak memory bytes", minimum=1)

    capacity = _object(evidence.get("capacity_guidance"), "capacity guidance")
    if (
        capacity.get("production_encoder_included") is not False
        or capacity.get("eligible") is not False
        or capacity.get("reason") != "synthetic_fixture_and_fake_encoder"
    ):
        raise ScaleEvidenceError(
            "synthetic fake-encoder evidence cannot be production capacity evidence"
        )

    return {
        "qualification_scope": "phase5_scale",
        "target_products": target_products,
        "fixture_products": fixture_products,
        "generation_id": generation_id,
        "processed_batches": processed_batches,
        "production_capacity_eligible": False,
    }


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ScaleEvidenceError(f"scale evidence {label} must be an object")

    return cast(dict[str, Any], value)


def _integer(value: object, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ScaleEvidenceError(f"scale evidence {label} is invalid")

    return value


def _number(value: object, label: str, *, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < minimum:
        raise ScaleEvidenceError(f"scale evidence {label} is invalid")

    return float(value)


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or HEX_SHA256.fullmatch(value) is None:
        raise ScaleEvidenceError(f"scale evidence {label} is invalid")

    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--expected-target", type=int, required=True)
    args = parser.parse_args()
    summary = verify_scale_evidence(
        args.evidence,
        expected_target=args.expected_target,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
