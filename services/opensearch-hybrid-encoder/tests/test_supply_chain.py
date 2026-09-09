from __future__ import annotations

import hashlib
import io
import json
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from mageos_opensearch_hybrid_encoder.supply_chain import (
    SupplyChainError,
    build_supply_chain_manifest,
    inspect_oci_archive,
    normalize_scanner_evidence,
    verify_supply_chain_manifest,
    write_supply_chain_manifest,
)

ENCODER = Path(__file__).resolve().parents[1]
TOOL_LOCK = ENCODER / "supply-chain-tools.json"
NOW = datetime(2026, 8, 28, 16, 0, tzinfo=UTC)


def test_supply_chain_manifest_binds_registry_free_image_and_security_evidence(
    tmp_path: Path,
) -> None:
    evidence = supply_chain_evidence(tmp_path)
    expected_digest = json.loads(evidence["sealed_identity"].read_text())["deployment"][
        "artifact_digest"
    ]

    manifest = build_supply_chain_manifest(**evidence, evaluated_at=NOW)
    image = cast(dict[str, object], manifest["image"])
    identity = cast(dict[str, object], manifest["identity"])
    sbom = cast(dict[str, object], manifest["sbom"])
    scan = cast(dict[str, object], manifest["vulnerability_scan"])

    assert manifest["schema_version"] == 1
    assert manifest["status"] == "passed"
    assert manifest["platform"] == "linux/amd64"
    assert image["manifest_digest"] == expected_digest
    assert image["archive_sha256"] == sha256(evidence["oci_archive"])
    assert identity["encoder_identity_digest"] == json.loads(
        evidence["sealed_identity"].read_text()
    )["encoder_identity_digest"]
    assert sbom["format"] == "cyclonedx-json"
    severity_counts = cast(dict[str, int], scan["severity_counts"])
    assert severity_counts["Low"] == 1
    assert scan["policy"] == {
        "fail_on": ["High", "Critical"],
        "ignored_matches_allowed": False,
        "maximum_database_age_hours": 24,
    }
    assert manifest["signature"] == {
        "binding": "external-detached",
        "required": True,
        "scheme": "sigstore-bundle",
    }
    assert inspect_oci_archive(evidence["oci_archive"])["manifest_digest"] == expected_digest

    output = tmp_path / "supply-chain.manifest.json"
    write_supply_chain_manifest(output, manifest)
    verify_supply_chain_manifest(output, **evidence, now=NOW)


def test_supply_chain_rejects_oci_digest_that_differs_from_sealed_identity(
    tmp_path: Path,
) -> None:
    evidence = supply_chain_evidence(tmp_path)
    identity = json.loads(evidence["sealed_identity"].read_text())
    identity["deployment"]["artifact_digest"] = "sha256:" + ("f" * 64)
    identity["identity_manifest"]["deployment"]["artifact_digest"] = "sha256:" + (
        "f" * 64
    )
    evidence["sealed_identity"].write_text(json.dumps(identity))

    with pytest.raises(SupplyChainError, match="sealed identity"):
        build_supply_chain_manifest(**evidence, evaluated_at=NOW)


@pytest.mark.parametrize("severity", ["High", "Critical"])
def test_supply_chain_rejects_release_blocking_vulnerability(
    tmp_path: Path,
    severity: str,
) -> None:
    evidence = supply_chain_evidence(tmp_path)
    scan = json.loads(evidence["grype_json"].read_text())
    scan["matches"][0]["vulnerability"]["severity"] = severity
    evidence["grype_json"].write_text(json.dumps(scan))

    with pytest.raises(SupplyChainError, match="release-blocking vulnerability"):
        build_supply_chain_manifest(**evidence, evaluated_at=NOW)


def test_supply_chain_rejects_ignored_findings_and_stale_database(tmp_path: Path) -> None:
    evidence = supply_chain_evidence(tmp_path)
    scan = json.loads(evidence["grype_json"].read_text())
    scan["ignoredMatches"] = [scan["matches"][0]]
    evidence["grype_json"].write_text(json.dumps(scan))

    with pytest.raises(SupplyChainError, match="ignored vulnerability"):
        build_supply_chain_manifest(**evidence, evaluated_at=NOW)

    evidence = supply_chain_evidence(tmp_path / "stale")
    scan = json.loads(evidence["grype_json"].read_text())
    scan["descriptor"]["db"]["status"]["built"] = (NOW - timedelta(hours=25)).isoformat()
    evidence["grype_json"].write_text(json.dumps(scan))

    with pytest.raises(SupplyChainError, match="database is stale"):
        build_supply_chain_manifest(**evidence, evaluated_at=NOW)


def test_supply_chain_rejects_unlocked_tool_version_and_evidence_drift(
    tmp_path: Path,
) -> None:
    evidence = supply_chain_evidence(tmp_path)
    sbom = json.loads(evidence["cyclonedx_json"].read_text())
    sbom["metadata"]["tools"]["components"][0]["version"] = "0.0.0"
    evidence["cyclonedx_json"].write_text(json.dumps(sbom))

    with pytest.raises(SupplyChainError, match="locked Syft version"):
        build_supply_chain_manifest(**evidence, evaluated_at=NOW)

    evidence = supply_chain_evidence(tmp_path / "drift")
    manifest = build_supply_chain_manifest(**evidence, evaluated_at=NOW)
    output = tmp_path / "drift.manifest.json"
    write_supply_chain_manifest(output, manifest)
    evidence["cyclonedx_json"].write_text(evidence["cyclonedx_json"].read_text() + "\n")

    with pytest.raises(SupplyChainError, match="evidence differs"):
        verify_supply_chain_manifest(output, **evidence, now=NOW)


def test_supply_chain_manifest_never_overwrites_retained_evidence(tmp_path: Path) -> None:
    output = tmp_path / "supply-chain.manifest.json"
    output.write_text("retained\n")

    with pytest.raises(SupplyChainError, match="already exists"):
        write_supply_chain_manifest(output, {"schema_version": 1})


def test_registry_neutral_supply_chain_flow_is_locked_and_documented() -> None:
    lock = json.loads(TOOL_LOCK.read_text())
    readme = (ENCODER / "README.md").read_text()

    assert lock["platform"] == "linux/amd64"
    assert lock["tools"]["syft"]["version"] == "1.51.1"
    assert lock["tools"]["grype"]["version"] == "0.118.0"
    assert lock["tools"]["cosign"]["version"] == "3.1.3"
    assert all(
        tool["url"].startswith("https://github.com/")
        and "ghcr.io" not in tool["url"]
        and len(tool["sha256"]) == 64
        for tool in lock["tools"].values()
    )
    assert "cyclonedx-json@1.6" in readme
    assert "--show-suppressed" in readme
    assert "supply_chain normalize" in readme
    assert "cosign sign-blob" in readme
    assert "No container registry is required" in readme


def test_scanner_evidence_normalizer_removes_host_paths_and_configuration(
    tmp_path: Path,
) -> None:
    evidence = supply_chain_evidence(tmp_path)
    syft = json.loads(evidence["syft_json"].read_text())
    syft["source"]["metadata"]["userInput"] = "/home/example/private/encoder.oci.tar"
    syft["descriptor"]["configuration"] = {
        "packages": {"golang": {"local-mod-cache-dir": "/home/example/go/pkg/mod"}}
    }
    evidence["syft_json"].write_text(json.dumps(syft))
    grype = json.loads(evidence["grype_json"].read_text())
    grype["source"]["target"]["userInput"] = "/home/example/private/encoder.oci.tar"
    grype["descriptor"]["configuration"] = {
        "db": {"cache-dir": "/home/example/Library/Caches/grype"},
        "show-suppressed": True,
    }
    grype["descriptor"]["db"]["status"]["path"] = "/home/example/grype.db"
    evidence["grype_json"].write_text(json.dumps(grype))

    with pytest.raises(SupplyChainError, match="normalized"):
        build_supply_chain_manifest(**evidence, evaluated_at=NOW)

    normalize_scanner_evidence(evidence["syft_json"], evidence["grype_json"])

    assert "/home/example" not in evidence["syft_json"].read_text()
    assert "/home/example" not in evidence["grype_json"].read_text()
    assert json.loads(evidence["syft_json"].read_text())["source"]["metadata"][
        "userInput"
    ] == "encoder.oci.tar"
    assert json.loads(evidence["grype_json"].read_text())["source"]["target"][
        "userInput"
    ] == "encoder.oci.tar"
    build_supply_chain_manifest(**evidence, evaluated_at=NOW)


def test_scanner_evidence_normalizer_requires_suppressed_findings(tmp_path: Path) -> None:
    evidence = supply_chain_evidence(tmp_path)
    grype = json.loads(evidence["grype_json"].read_text())
    grype["descriptor"]["configuration"] = {"show-suppressed": False}
    evidence["grype_json"].write_text(json.dumps(grype))

    with pytest.raises(SupplyChainError, match="show suppressed"):
        normalize_scanner_evidence(evidence["syft_json"], evidence["grype_json"])


def supply_chain_evidence(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    oci_archive = tmp_path / "encoder.oci.tar"
    image_digest = write_oci_archive(oci_archive)
    sealed_identity = tmp_path / "sealed-identity.json"
    identity_manifest = {"deployment": {"artifact_digest": image_digest}}
    identity_digest = hashlib.sha256(
        json.dumps(identity_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    sealed_identity.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "production_eligible": True,
                "architecture": "linux/amd64",
                "encoder_identity_digest": identity_digest,
                "deployment": {"artifact_digest": image_digest},
                "identity_manifest": identity_manifest,
            }
        )
    )
    syft_json = tmp_path / "encoder.syft.json"
    syft_json.write_text(
        json.dumps(
            {
                "artifacts": [{"name": "python", "version": "3.13.15"}],
                "source": {
                    "type": "image",
                    "name": "mageos-opensearch-hybrid-encoder",
                    "version": image_digest,
                    "metadata": {
                        "userInput": "encoder.oci.tar",
                        "manifestDigest": image_digest,
                    },
                },
                "descriptor": {"name": "syft", "version": "1.51.1"},
            }
        )
    )
    cyclonedx_json = tmp_path / "encoder.cdx.json"
    cyclonedx_json.write_text(
        json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.6",
                "version": 1,
                "metadata": {
                    "tools": {
                        "components": [
                            {"type": "application", "name": "syft", "version": "1.51.1"}
                        ]
                    },
                    "component": {
                        "type": "container",
                        "name": "mageos-opensearch-hybrid-encoder",
                        "version": image_digest,
                    },
                },
                "components": [{"type": "library", "name": "python", "version": "3.13.15"}],
            }
        )
    )
    grype_json = tmp_path / "encoder.grype.json"
    grype_json.write_text(
        json.dumps(
            {
                "matches": [
                    {
                        "vulnerability": {"id": "CVE-example", "severity": "Low"},
                        "artifact": {"name": "example", "version": "1.0"},
                    }
                ],
                "source": {
                    "type": "image",
                    "target": {
                        "userInput": "encoder.oci.tar",
                        "manifestDigest": image_digest,
                    },
                },
                "descriptor": {
                    "name": "grype",
                    "version": "0.118.0",
                    "configuration": {"show-suppressed": True},
                    "db": {
                        "status": {
                            "schemaVersion": "v6.1.9",
                            "from": (
                                "https://grype.anchore.io/databases/v6/"
                                "vulnerability-db.tar.zst?checksum=sha256%3A" + ("b" * 64)
                            ),
                            "built": (NOW - timedelta(hours=1)).isoformat(),
                            "valid": True,
                        },
                        "providers": {
                            "debian": {
                                "captured": (NOW - timedelta(hours=2)).isoformat(),
                                "input": "xxh64:example",
                            }
                        },
                    },
                },
            }
        )
    )
    return {
        "oci_archive": oci_archive,
        "sealed_identity": sealed_identity,
        "syft_json": syft_json,
        "cyclonedx_json": cyclonedx_json,
        "grype_json": grype_json,
        "tool_lock": TOOL_LOCK,
    }


def write_oci_archive(path: Path) -> str:
    config = json.dumps({"architecture": "amd64", "os": "linux"}, sort_keys=True).encode()
    layer_buffer = io.BytesIO()
    with tarfile.open(fileobj=layer_buffer, mode="w") as layer_archive:
        files = {
            "srv/encoder/release-marker.txt": b"qualified encoder layer\n",
            "usr/local/lib/python3.13/site-packages/example-1.0.dist-info/METADATA": (
                b"Metadata-Version: 2.4\nName: example\nVersion: 1.0\n"
            ),
        }
        for name, content in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mtime = 0
            layer_archive.addfile(info, io.BytesIO(content))
    layer = layer_buffer.getvalue()
    config_digest = digest(config)
    layer_digest = digest(layer)
    manifest = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {
                "mediaType": "application/vnd.oci.image.config.v1+json",
                "digest": config_digest,
                "size": len(config),
            },
            "layers": [
                {
                    "mediaType": "application/vnd.oci.image.layer.v1.tar",
                    "digest": layer_digest,
                    "size": len(layer),
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    manifest_digest = digest(manifest)
    index = json.dumps(
        {
            "schemaVersion": 2,
            "manifests": [
                {
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "digest": manifest_digest,
                    "size": len(manifest),
                    "platform": {"architecture": "amd64", "os": "linux"},
                }
            ],
        },
        sort_keys=True,
    ).encode()
    files = {
        "oci-layout": b'{"imageLayoutVersion":"1.0.0"}',
        "index.json": index,
        blob_path(config_digest): config,
        blob_path(layer_digest): layer,
        blob_path(manifest_digest): manifest,
    }
    with tarfile.open(path, "w") as archive:
        for name, content in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(content))
    return manifest_digest


def digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blob_path(value: str) -> str:
    return "blobs/sha256/" + value.removeprefix("sha256:")
