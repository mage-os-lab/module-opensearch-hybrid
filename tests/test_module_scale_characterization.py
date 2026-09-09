from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = ROOT
sys.path.insert(0, str(MODULE_ROOT))

from dev.qualification.scale_evidence import (  # noqa: E402
    ScaleEvidenceError,
    verify_scale_evidence,
)


def evidence(target_products: int = 100_000) -> dict[str, object]:
    eligible_before = 6
    fixture_products = target_products - eligible_before

    return {
        "schema_version": 2,
        "status": "passed",
        "qualification_scope": "phase5_scale",
        "catalog": {
            "target_products": target_products,
            "eligible_before": eligible_before,
            "created_products": fixture_products,
            "eligible_after": target_products,
            "fixture_products": fixture_products,
        },
        "runtime": {
            "mageos_version": "3.4.0",
            "magento_product_version": "2.4.9",
            "php_version": "8.4.25",
            "php_memory_limit": "12G",
            "opensearch_version": "3.8.0",
            "module_composer_sha256": "a" * 64,
            "module_package_file_count": 267,
            "module_package_payload_sha256": "d" * 64,
            "module_package_archive_sha256": "e" * 64,
            "encoder_runtime": "deterministic_hash_fake_only",
        },
        "generation": {
            "generation_id": 1,
            "state": "READY",
            "coverage_total": target_products,
            "coverage_complete": target_products,
            "coverage_failed": 0,
            "processed_batches": 1563,
            "validation_ready": True,
            "contract_digest": "b" * 64,
            "result_contract_digest": "c" * 64,
        },
        "timings_seconds": {
            "product_insert": 25.1,
            "native_reindex": 31.2,
            "hybrid_build": 75.3,
        },
        "graphql": {
            "fixture_total": fixture_products,
            "page_one_ms": 41.0,
            "page_two_ms": 39.0,
            "included_price_filter_ms": 44.0,
            "excluded_price_filter_ms": 38.0,
            "stable_pages": True,
            "price_filter": True,
            "price_aggregation": True,
        },
        "process": {"peak_memory_bytes": 67_108_864},
        "capacity_guidance": {
            "production_encoder_included": False,
            "eligible": False,
            "reason": "synthetic_fixture_and_fake_encoder",
        },
    }


def write_evidence(tmp_path: Path, value: dict[str, object]) -> Path:
    output = tmp_path / "scale-evidence.json"
    output.write_text(json.dumps(value), encoding="utf-8")

    return output


@pytest.mark.parametrize("target_products", [100_000, 1_000_000])
def test_verifies_phase5_scale_evidence(tmp_path: Path, target_products: int) -> None:
    summary = verify_scale_evidence(
        write_evidence(tmp_path, evidence(target_products)),
        expected_target=target_products,
    )

    assert summary == {
        "qualification_scope": "phase5_scale",
        "target_products": target_products,
        "fixture_products": target_products - 6,
        "generation_id": 1,
        "processed_batches": 1563,
        "production_capacity_eligible": False,
    }


def test_rejects_incomplete_or_internally_inconsistent_evidence(tmp_path: Path) -> None:
    value = evidence()
    value["status"] = "build_passed"
    with pytest.raises(ScaleEvidenceError, match="status"):
        verify_scale_evidence(write_evidence(tmp_path, value))

    value = evidence()
    value["generation"]["coverage_complete"] = 99_999  # type: ignore[index]
    with pytest.raises(ScaleEvidenceError, match="coverage"):
        verify_scale_evidence(write_evidence(tmp_path, value))

    value = evidence()
    runtime = value["runtime"]
    assert isinstance(runtime, dict)
    del runtime["module_package_payload_sha256"]
    with pytest.raises(ScaleEvidenceError, match="package payload"):
        verify_scale_evidence(write_evidence(tmp_path, value))


def test_rejects_smoke_size_or_fake_encoder_as_capacity_evidence(tmp_path: Path) -> None:
    value = evidence(10_000)
    with pytest.raises(ScaleEvidenceError, match="100000 or 1000000"):
        verify_scale_evidence(write_evidence(tmp_path, value))

    value = evidence()
    value["capacity_guidance"]["eligible"] = True  # type: ignore[index]
    with pytest.raises(ScaleEvidenceError, match="production capacity"):
        verify_scale_evidence(write_evidence(tmp_path, value))

    value = evidence(1_000_000)
    value["runtime"]["php_memory_limit"] = "unlimited"  # type: ignore[index]
    with pytest.raises(ScaleEvidenceError, match="PHP memory limit"):
        verify_scale_evidence(write_evidence(tmp_path, value))


def test_distinguishes_mageos_release_from_magento_product_version(tmp_path: Path) -> None:
    value = evidence()
    value["runtime"]["mageos_version"] = "2.4.9"  # type: ignore[index]
    with pytest.raises(ScaleEvidenceError, match="Mage-OS 3.4"):
        verify_scale_evidence(write_evidence(tmp_path, value))

    value = evidence()
    value["runtime"]["magento_product_version"] = "3.4.0"  # type: ignore[index]
    with pytest.raises(ScaleEvidenceError, match="Magento product version"):
        verify_scale_evidence(write_evidence(tmp_path, value))


def test_scale_runner_is_exactly_gated_and_emits_verified_evidence() -> None:
    installer_path = MODULE_ROOT / "dev/ci/install-scale-fixture.sh"
    installer = installer_path.read_text()
    runner = (MODULE_ROOT / "dev/ci/run-large-catalog.sh").read_text()
    build = (MODULE_ROOT / "dev/ci/assert-large-catalog-build.php").read_text()
    graphql = (MODULE_ROOT / "dev/ci/assert-large-catalog-graphql.php").read_text()
    compose = (MODULE_ROOT / "dev/ci/compose.yaml").read_text()
    workflow = (ROOT / ".github/workflows/module-opensearch-hybrid.yml").read_text()

    assert "set -euo pipefail" in installer
    assert "Refusing to overwrite existing scale fixture" in installer
    assert "mage-os/project-community-edition" in installer
    assert "mage-os/module-opensearch-hybrid:@dev" in installer
    assert "setup:install" in installer
    assert "setup:di:compile" in installer
    assert "assert-doctor.php" in installer
    assert "MAGEOS_SCALE_CONFIRMATION" in runner
    assert "MAGEOS_SCALE_EVIDENCE_PATH" in runner
    assert "MAGEOS_MODULE_PACKAGE_MANIFEST" in runner
    assert "scale_evidence.py" in runner
    assert "1_000_000" in build
    assert "scale-%d" in build
    assert "MAGEOS_MODULE_INSTALL_ROOT" in build
    assert "MAGEOS_MODULE_PACKAGE_MANIFEST" in build
    assert "InstalledVersions::getPrettyVersion('mage-os/product-community-edition')" in build
    assert "magento_product_version" in build
    assert "php_memory_limit" in build
    assert "build_passed" in build
    assert "synthetic_fixture_and_fake_encoder" in build
    assert "fixture_total" in graphql
    assert "($evidence['schema_version'] ?? null) !== 2" in graphql
    assert "$evidence['status'] = 'passed'" in graphql
    assert "MAGEOS_CI_MYSQL_TMPFS_SIZE:-1g" in compose
    assert "MAGEOS_CI_OPENSEARCH_TMPFS_SIZE:-1g" in compose
    assert "MAGEOS_CI_OPENSEARCH_JAVA_OPTS:--Xms512m -Xmx512m" in compose
    assert "MAGEOS_LARGE_CATALOG_PHP_MEMORY_LIMIT" in runner
    assert 'build_memory_limit="12G"' in runner
    assert 'php -d "memory_limit=${build_memory_limit}"' in runner
    assert "  pull_request:\n  push:\n" in workflow
    assert "run: make check" in workflow


def test_fresh_scale_fixture_accepts_empty_safe_radial_state() -> None:
    doctor = (MODULE_ROOT / "dev/ci/assert-doctor.php").read_text()

    assert "$radialDisabled = is_array($radialValues);" in doctor
    assert "foreach ($radialValues as $radialValue)" in doctor


def test_scale_shell_entrypoints_fail_closed_before_writes(tmp_path: Path) -> None:
    installer = MODULE_ROOT / "dev/ci/install-scale-fixture.sh"
    runner = MODULE_ROOT / "dev/ci/run-large-catalog.sh"
    fixture = tmp_path / "fixture"
    environment = os.environ | {
        "MAGEOS_FIXTURE_ROOT": str(fixture),
        "MAGEOS_LARGE_CATALOG_PRODUCTS": "10000",
    }

    result = subprocess.run(
        [installer],
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 2
    assert "must be exactly 100000 or 1000000" in result.stderr
    assert not fixture.exists()

    environment["MAGEOS_LARGE_CATALOG_PRODUCTS"] = "100000"
    environment["MAGEOS_SCALE_CONFIRMATION"] = "scale-100000"
    fixture.mkdir()
    result = subprocess.run(
        [installer],
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 2
    assert "Refusing to overwrite existing scale fixture" in result.stderr

    environment["MAGEOS_FIXTURE_ROOT"] = str(tmp_path / "new-fixture")
    environment["MAGEOS_LARGE_CATALOG_PHP_MEMORY_LIMIT"] = "0G"
    result = subprocess.run(
        [runner],
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 2
    assert "positive whole number" in result.stderr
    assert not (tmp_path / "new-fixture").exists()

    del environment["MAGEOS_LARGE_CATALOG_PHP_MEMORY_LIMIT"]
    result = subprocess.run(
        [runner],
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 2
    assert "package manifest" in result.stderr
    assert not (tmp_path / "new-fixture").exists()
