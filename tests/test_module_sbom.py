from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from dev.sbom.verify import (  # noqa: E402
    normalize_module_sbom,
    verify_resolved_module_sbom,
)

MODULE_REF = "pkg:composer/mage-os/module-opensearch-hybrid@dev-main"
FRAMEWORK_REF = "pkg:composer/mage-os/framework@3.4.2"
ROOT_REF = "pkg:composer/mage-os/project-community-edition@3.4.0"


def _sbom() -> dict[str, object]:
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "group": "mage-os",
                "name": "project-community-edition",
                "version": "3.4.0",
                "bom-ref": ROOT_REF,
                "purl": ROOT_REF,
            }
        },
        "components": [
            {
                "type": "library",
                "group": "mage-os",
                "name": "module-opensearch-hybrid",
                "version": "dev-main",
                "bom-ref": MODULE_REF,
                "purl": MODULE_REF,
            },
            {
                "type": "library",
                "group": "mage-os",
                "name": "framework",
                "version": "3.4.2",
                "bom-ref": FRAMEWORK_REF,
                "purl": FRAMEWORK_REF,
            },
        ],
        "dependencies": [
            {"ref": ROOT_REF, "dependsOn": [MODULE_REF]},
            {"ref": MODULE_REF, "dependsOn": [FRAMEWORK_REF]},
            {"ref": FRAMEWORK_REF},
        ],
    }


def _write(tmp_path: Path, sbom: dict[str, object]) -> Path:
    path = tmp_path / "bom.json"
    path.write_text(json.dumps(sbom), encoding="utf-8")
    return path


def test_verifies_a_resolved_module_dependency_graph(tmp_path: Path) -> None:
    summary = verify_resolved_module_sbom(_write(tmp_path, _sbom()))

    assert summary == {
        "component": "mage-os/module-opensearch-hybrid",
        "component_count": 2,
        "dependency_count": 3,
        "module_ref": MODULE_REF,
        "module_version": "dev-main",
        "spec_version": "1.6",
    }


def test_rejects_an_sbom_without_the_module(tmp_path: Path) -> None:
    sbom = _sbom()
    sbom["components"] = sbom["components"][1:]  # type: ignore[index]

    with pytest.raises(ValueError, match="module component"):
        verify_resolved_module_sbom(_write(tmp_path, sbom))


def test_rejects_unresolved_versions_and_a_disconnected_module(tmp_path: Path) -> None:
    sbom = _sbom()
    sbom["components"][1]["version"] = "3.4.*"  # type: ignore[index]

    with pytest.raises(ValueError, match="resolved version"):
        verify_resolved_module_sbom(_write(tmp_path, sbom))

    sbom = _sbom()
    sbom["dependencies"][0]["dependsOn"] = []  # type: ignore[index]

    with pytest.raises(ValueError, match="root dependency graph"):
        verify_resolved_module_sbom(_write(tmp_path, sbom))


def test_rejects_unknown_dependency_references(tmp_path: Path) -> None:
    sbom = _sbom()
    sbom["dependencies"][1]["dependsOn"] = ["pkg:composer/example/missing@1.0.0"]  # type: ignore[index]

    with pytest.raises(ValueError, match="unknown component"):
        verify_resolved_module_sbom(_write(tmp_path, sbom))


def test_normalizes_local_path_references_before_verification(tmp_path: Path) -> None:
    sbom = _sbom()
    module = sbom["components"][0]  # type: ignore[index]
    module["externalReferences"] = [
        {
            "type": "distribution",
            "url": "/home/runner/work/module-opensearch-hybrid/module-opensearch-hybrid",
        },
        {"type": "website", "url": "https://example.invalid/module"},
    ]
    source = _write(tmp_path, sbom)
    normalized = tmp_path / "normalized.json"

    with pytest.raises(ValueError, match="local filesystem reference"):
        verify_resolved_module_sbom(source)

    removed = normalize_module_sbom(source, normalized)
    first_bytes = normalized.read_bytes()
    second_removed = normalize_module_sbom(normalized, normalized)

    assert removed == 1
    assert second_removed == 0
    assert normalized.read_bytes() == first_bytes
    assert b"/home/runner" not in first_bytes
    assert b"https://example.invalid/module" in first_bytes
    verify_resolved_module_sbom(normalized)
