from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest

from poc.module_package import (
    ZIP_TIMESTAMP,
    build_module_package,
    load_manifest,
    verify_archive,
    verify_module_payload,
)

ROOT = Path(__file__).resolve().parents[1]


def test_package_is_reproducible_safe_and_verifiable(tmp_path: Path) -> None:
    module = _module_fixture(tmp_path / "module")
    cache = module / ".phpunit.cache"
    cache.mkdir()
    (cache / "test-results").write_text("host-local")
    sbom_vendor = module / "dev" / "sbom" / "tool" / "vendor"
    sbom_vendor.mkdir(parents=True)
    (sbom_vendor / "autoload.php").write_text("generated tool dependency\n")
    first_archive = tmp_path / "first" / "module.zip"
    first_manifest = tmp_path / "first" / "module.manifest.json"
    second_archive = tmp_path / "second" / "different-name.zip"
    second_manifest = tmp_path / "second" / "different-name.manifest.json"

    first = build_module_package(module, first_archive, first_manifest)
    second = build_module_package(module, second_archive, second_manifest)

    assert first == second
    assert first_archive.read_bytes() == second_archive.read_bytes()
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    assert first["payload"]["file_count"] == 3
    assert first["payload"]["root"] == "."
    assert [item["path"] for item in first["payload"]["files"]] == [
        "Model/Service.php",
        "composer.json",
        "registration.php",
    ]
    verify_module_payload(module, load_manifest(first_manifest))
    verify_archive(first_archive, first)
    with zipfile.ZipFile(first_archive) as archive:
        assert all(member.date_time == ZIP_TIMESTAMP for member in archive.infolist())
        assert archive.namelist() == sorted(archive.namelist())
        assert ".phpunit.cache/test-results" not in archive.namelist()


def test_payload_verification_detects_source_drift(tmp_path: Path) -> None:
    module = _module_fixture(tmp_path / "module")
    archive = tmp_path / "module.zip"
    manifest_path = tmp_path / "module.manifest.json"
    manifest = build_module_package(module, archive, manifest_path)
    (module / "registration.php").write_text("changed\n")

    with pytest.raises(ValueError, match="payload files do not match"):
        verify_module_payload(module, manifest)


def test_archive_verification_detects_artifact_drift(tmp_path: Path) -> None:
    module = _module_fixture(tmp_path / "module")
    archive = tmp_path / "module.zip"
    manifest_path = tmp_path / "module.manifest.json"
    manifest = build_module_package(module, archive, manifest_path)
    archive.write_bytes(archive.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="archive byte count"):
        verify_archive(archive, manifest)


def test_package_refuses_symbolic_links(tmp_path: Path) -> None:
    module = _module_fixture(tmp_path / "module")
    os.symlink(module / "registration.php", module / "linked.php")

    with pytest.raises(ValueError, match="symbolic link"):
        build_module_package(module, tmp_path / "module.zip", tmp_path / "manifest.json")


def test_package_refuses_parent_repository_runtime_reference(tmp_path: Path) -> None:
    module = _module_fixture(tmp_path / "module")
    (module / "Model/Service.php").write_text("<?php require __DIR__ . '/../../poc/manifest.py';\n")

    with pytest.raises(ValueError, match="parent evidence repository"):
        build_module_package(module, tmp_path / "module.zip", tmp_path / "manifest.json")


def test_package_allows_parent_repository_links_in_documentation(tmp_path: Path) -> None:
    module = _module_fixture(tmp_path / "module")
    docs = module / "docs"
    docs.mkdir()
    (docs / "qualification.md").write_text("See ../../results/evidence.json\n")

    build_module_package(module, tmp_path / "module.zip", tmp_path / "manifest.json")


def test_package_requires_exact_composer_identity_and_external_outputs(tmp_path: Path) -> None:
    module = _module_fixture(tmp_path / "module")
    composer_path = module / "composer.json"
    composer = json.loads(composer_path.read_text())
    composer["name"] = "example/wrong-module"
    composer_path.write_text(json.dumps(composer))

    with pytest.raises(ValueError, match="composer package name"):
        build_module_package(module, tmp_path / "module.zip", tmp_path / "manifest.json")

    composer["name"] = "mage-os/module-opensearch-hybrid"
    composer_path.write_text(json.dumps(composer))
    with pytest.raises(ValueError, match="outside the module root"):
        build_module_package(
            module,
            module / "Model/module.zip",
            tmp_path / "manifest.json",
        )


def test_module_package_and_manifest_are_retained_together_in_ci() -> None:
    workflow = (ROOT / ".github/workflows/module-opensearch-hybrid.yml").read_text()

    assert "name: mage-os-module-opensearch-hybrid-package" in workflow
    assert "${{ runner.temp }}/mage-os-module-opensearch-hybrid.zip" in workflow
    assert "${{ runner.temp }}/mage-os-module-opensearch-hybrid.manifest.json" in workflow
    assert "if-no-files-found: error" in workflow
    assert "retention-days: 30" in workflow


def test_module_distribution_contains_declared_license_and_trademark_notices() -> None:
    module = ROOT
    composer = json.loads((module / "composer.json").read_text())
    osl = (module / "LICENSE.txt").read_text()
    afl = (module / "LICENSE_AFL.txt").read_text()
    trademarks = (module / "TRADEMARKS.md").read_text()

    assert composer["license"] == ["OSL-3.0", "AFL-3.0"]
    assert osl.startswith('\nOpen Software License ("OSL") v. 3.0')
    assert "Licensed under the Open Software License version 3.0" in osl
    assert afl.startswith('\nAcademic Free License ("AFL") v. 3.0')
    assert "Licensed under the Academic Free License version 3.0" in afl
    assert "Mage-OS Association approval" in trademarks
    assert "OpenSearch Trademark Policy" in trademarks
    assert "does not imply endorsement" in trademarks


def _module_fixture(path: Path) -> Path:
    (path / "Model").mkdir(parents=True)
    (path / "composer.json").write_text(
        json.dumps(
            {
                "name": "mage-os/module-opensearch-hybrid",
                "type": "magento2-module",
            }
        )
    )
    (path / "registration.php").write_text("<?php // module registration\n")
    executable = path / "Model/Service.php"
    executable.write_text("<?php // runtime\n")
    executable.chmod(0o755)
    return path



def test_package_omits_contributor_tools_and_generated_data(tmp_path: Path) -> None:
    module = _module_fixture(tmp_path / "module")
    for name in (
        "services",
        "poc",
        "scripts",
        "tests",
        "config",
        "data",
        "results",
        "runs",
        "dist",
    ):
        directory = module / name
        directory.mkdir()
        (directory / "local.txt").write_text("contributor tooling or private generated output")
    (module / ".env").write_text("private local configuration")
    (module / "uv.lock").write_text("Python development dependencies")

    manifest = build_module_package(module, tmp_path / "module.zip", tmp_path / "manifest.json")

    assert [entry["path"] for entry in manifest["payload"]["files"]] == [
        "Model/Service.php",
        "composer.json",
        "registration.php",
    ]


def test_package_builds_into_excluded_dist_and_verifies_the_source(tmp_path: Path) -> None:
    module = _module_fixture(tmp_path / "module")
    archive = module / "dist/module.zip"
    manifest_path = module / "dist/manifest.json"

    manifest = build_module_package(module, archive, manifest_path)

    verify_module_payload(module, manifest)
    verify_archive(archive, manifest)
    assert manifest["payload"]["file_count"] == 3
