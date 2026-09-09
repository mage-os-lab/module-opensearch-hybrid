"""Bind a registry-neutral encoder OCI image to SBOM and vulnerability evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
import tempfile
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import IO, Any, cast
from urllib.parse import parse_qs, urlparse

SCHEMA_VERSION = 1
PLATFORM = "linux/amd64"
MAXIMUM_DATABASE_AGE = timedelta(hours=24)
MAXIMUM_CLOCK_SKEW = timedelta(minutes=5)
EXPECTED_COMPONENT_NAME = "mageos-opensearch-hybrid-encoder"
EXPECTED_TOOLS = ("syft", "grype", "cosign")
SEVERITIES = ("Unknown", "Negligible", "Low", "Medium", "High", "Critical")
BLOCKING_SEVERITIES = frozenset({"High", "Critical"})
DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
HEX_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GRYPE_DB_SCHEMA_PATTERN = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+$")
OCI_MANIFEST_TYPE = "application/vnd.oci.image.manifest.v1+json"
OCI_CONFIG_TYPE = "application/vnd.oci.image.config.v1+json"


class SupplyChainError(ValueError):
    """Release evidence cannot prove the exact production image is safe to publish."""


def build_supply_chain_manifest(
    *,
    oci_archive: Path,
    sealed_identity: Path,
    syft_json: Path,
    cyclonedx_json: Path,
    grype_json: Path,
    tool_lock: Path,
    evaluated_at: datetime | None = None,
) -> dict[str, object]:
    """Validate release evidence and build an unsigned canonical checksum manifest."""
    evaluation_time = _utc_time(evaluated_at or datetime.now(UTC), "evaluation time")
    tools = _load_tool_lock(tool_lock)
    image = inspect_oci_archive(oci_archive)
    image_digest = cast(str, image["manifest_digest"])
    identity = _inspect_identity(sealed_identity, image_digest)
    catalog = _inspect_syft_catalog(
        syft_json,
        expected_version=tools["syft"],
        image_digest=image_digest,
    )
    sbom = _inspect_cyclonedx(
        cyclonedx_json,
        expected_version=tools["syft"],
        image_digest=image_digest,
    )
    scan = _inspect_grype_scan(
        grype_json,
        expected_version=tools["grype"],
        image_digest=image_digest,
        evaluated_at=evaluation_time,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "platform": PLATFORM,
        "evaluated_at": _format_time(evaluation_time),
        "tool_lock": {
            "sha256": _sha256_file(tool_lock),
            "tools": tools,
        },
        "image": image,
        "identity": identity,
        "catalog": catalog,
        "sbom": sbom,
        "vulnerability_scan": scan,
        "signature": {
            "binding": "external-detached",
            "required": True,
            "scheme": "sigstore-bundle",
        },
    }


def verify_supply_chain_manifest(
    manifest_path: Path,
    *,
    oci_archive: Path,
    sealed_identity: Path,
    syft_json: Path,
    cyclonedx_json: Path,
    grype_json: Path,
    tool_lock: Path,
    now: datetime | None = None,
) -> None:
    """Rebuild a retained manifest from its inputs and reject any evidence drift."""
    retained = _load_json_object(manifest_path, "supply-chain manifest")
    if retained.get("schema_version") != SCHEMA_VERSION:
        raise SupplyChainError("supply-chain manifest schema is unsupported")
    evaluated_at = _parse_time(retained.get("evaluated_at"), "manifest evaluation time")
    verification_time = _utc_time(now or datetime.now(UTC), "verification time")
    age = verification_time - evaluated_at
    if age < -MAXIMUM_CLOCK_SKEW:
        raise SupplyChainError("supply-chain manifest evaluation time is in the future")
    if age > MAXIMUM_DATABASE_AGE:
        raise SupplyChainError("supply-chain manifest is stale")
    rebuilt = build_supply_chain_manifest(
        oci_archive=oci_archive,
        sealed_identity=sealed_identity,
        syft_json=syft_json,
        cyclonedx_json=cyclonedx_json,
        grype_json=grype_json,
        tool_lock=tool_lock,
        evaluated_at=evaluated_at,
    )
    if retained != rebuilt:
        raise SupplyChainError("supply-chain evidence differs from the retained manifest")


def write_supply_chain_manifest(path: Path, manifest: dict[str, object]) -> None:
    """Write canonical evidence without overwriting a retained release record."""
    if not path.is_absolute() or not path.parent.is_dir() or path.is_symlink():
        raise SupplyChainError("supply-chain output must be a new absolute path under a directory")
    try:
        with path.open("xb") as handle:
            handle.write(_canonical_json(manifest) + b"\n")
    except FileExistsError as error:
        raise SupplyChainError("supply-chain output already exists") from error


def normalize_scanner_evidence(syft_path: Path, grype_path: Path) -> None:
    """Remove host-local scanner configuration while preserving the security result."""
    syft = _load_json_object(syft_path, "Syft catalog")
    syft_descriptor = _object(syft.get("descriptor"), "Syft descriptor")
    syft_descriptor.pop("configuration", None)
    syft_source = _object(syft.get("source"), "Syft source")
    syft_metadata = _object(syft_source.get("metadata"), "Syft source metadata")
    syft_metadata["userInput"] = "encoder.oci.tar"

    grype = _load_json_object(grype_path, "Grype scan")
    grype_descriptor = _object(grype.get("descriptor"), "Grype descriptor")
    grype_configuration = _object(
        grype_descriptor.get("configuration"),
        "Grype configuration",
    )
    if grype_configuration.get("show-suppressed") is not True:
        raise SupplyChainError("Grype must show suppressed findings before normalization")
    grype_descriptor["configuration"] = {"show-suppressed": True}
    database = _object(grype_descriptor.get("db"), "Grype database")
    database_status = _object(database.get("status"), "Grype database status")
    database_status.pop("path", None)
    grype_source = _object(grype.get("source"), "Grype source")
    grype_target = _object(grype_source.get("target"), "Grype source target")
    grype_target["userInput"] = "encoder.oci.tar"

    _replace_with_canonical_json(syft_path, syft)
    _replace_with_canonical_json(grype_path, grype)


def inspect_oci_archive(path: Path) -> dict[str, object]:
    """Validate a single-platform OCI archive and return its immutable digest facts."""
    _require_regular_file(path, "OCI archive")
    try:
        with tarfile.open(path, "r:*") as archive:
            members = _safe_tar_members(archive)
            layout = _read_tar_json(archive, members, "oci-layout", "OCI layout")
            if layout != {"imageLayoutVersion": "1.0.0"}:
                raise SupplyChainError("OCI layout version is unsupported")
            index = _read_tar_json(archive, members, "index.json", "OCI index")
            if index.get("schemaVersion") != 2:
                raise SupplyChainError("OCI index schema is unsupported")
            descriptors = _objects(index.get("manifests"), "OCI index manifests")
            if len(descriptors) != 1:
                raise SupplyChainError("OCI archive must contain exactly one image manifest")
            descriptor = descriptors[0]
            if descriptor.get("mediaType") != OCI_MANIFEST_TYPE:
                raise SupplyChainError("OCI index must reference an OCI image manifest")
            _validate_platform(_object(descriptor.get("platform"), "OCI image platform"))
            manifest_bytes = _verified_blob_bytes(archive, members, descriptor, "OCI manifest")
            manifest = _load_json_bytes(manifest_bytes, "OCI manifest")
            if (
                manifest.get("schemaVersion") != 2
                or manifest.get("mediaType") != OCI_MANIFEST_TYPE
            ):
                raise SupplyChainError("OCI image manifest schema is unsupported")
            config_descriptor = _object(manifest.get("config"), "OCI image config")
            if config_descriptor.get("mediaType") != OCI_CONFIG_TYPE:
                raise SupplyChainError("OCI image config media type is unsupported")
            config_bytes = _verified_blob_bytes(
                archive,
                members,
                config_descriptor,
                "OCI image config",
            )
            config = _load_json_bytes(config_bytes, "OCI image config")
            _validate_platform(config)
            layers = _objects(manifest.get("layers"), "OCI image layers")
            if not layers:
                raise SupplyChainError("OCI image must contain at least one layer")
            for position, layer in enumerate(layers):
                _verify_blob(archive, members, layer, f"OCI image layer {position}")
    except (OSError, tarfile.TarError) as error:
        raise SupplyChainError(f"unable to read OCI archive: {error}") from error

    return {
        "format": "oci-archive",
        "archive_bytes": path.stat().st_size,
        "archive_sha256": _sha256_file(path),
        "manifest_digest": _digest(descriptor.get("digest"), "OCI manifest digest"),
        "config_digest": _digest(config_descriptor.get("digest"), "OCI config digest"),
        "layer_count": len(layers),
    }


def _inspect_identity(path: Path, image_digest: str) -> dict[str, object]:
    identity = _load_json_object(path, "sealed encoder identity")
    if (
        identity.get("schema_version") != 2
        or identity.get("production_eligible") is not True
        or identity.get("architecture") != PLATFORM
    ):
        raise SupplyChainError("sealed identity is not production-eligible Linux amd64 evidence")
    identity_digest = identity.get("encoder_identity_digest")
    if (
        not isinstance(identity_digest, str)
        or HEX_SHA256_PATTERN.fullmatch(identity_digest) is None
    ):
        raise SupplyChainError("sealed identity digest is invalid")
    manifest = _object(identity.get("identity_manifest"), "sealed identity source manifest")
    if hashlib.sha256(_canonical_json(manifest)).hexdigest() != identity_digest:
        raise SupplyChainError("sealed identity digest does not match its source manifest")
    deployment = _object(identity.get("deployment"), "sealed identity deployment")
    manifest_deployment = _object(
        manifest.get("deployment"),
        "sealed identity source deployment",
    )
    if (
        deployment.get("artifact_digest") != image_digest
        or manifest_deployment.get("artifact_digest") != image_digest
    ):
        raise SupplyChainError("sealed identity does not bind the supplied OCI image")
    return {
        "sealed_identity_bytes": path.stat().st_size,
        "sealed_identity_sha256": _sha256_file(path),
        "encoder_identity_digest": identity_digest,
        "deployment_digest": image_digest,
    }


def _inspect_syft_catalog(
    path: Path,
    *,
    expected_version: str,
    image_digest: str,
) -> dict[str, object]:
    value = _load_json_object(path, "Syft catalog")
    descriptor = _object(value.get("descriptor"), "Syft descriptor")
    if descriptor.get("name") != "syft" or descriptor.get("version") != expected_version:
        raise SupplyChainError("Syft catalog was not generated by the locked Syft version")
    if "configuration" in descriptor:
        raise SupplyChainError("Syft catalog must be normalized before retention")
    source = _object(value.get("source"), "Syft catalog source")
    metadata = _object(source.get("metadata"), "Syft catalog source metadata")
    if (
        source.get("type") != "image"
        or source.get("name") != EXPECTED_COMPONENT_NAME
        or source.get("version") != image_digest
        or metadata.get("manifestDigest") != image_digest
        or metadata.get("userInput") != "encoder.oci.tar"
    ):
        raise SupplyChainError("Syft catalog does not bind the supplied Linux amd64 image")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise SupplyChainError("Syft catalog contains no packages")
    return {
        "format": "syft-json",
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "package_count": len(artifacts),
        "tool": {"name": "syft", "version": expected_version},
    }


def _inspect_cyclonedx(
    path: Path,
    *,
    expected_version: str,
    image_digest: str,
) -> dict[str, object]:
    value = _load_json_object(path, "CycloneDX SBOM")
    if (
        value.get("bomFormat") != "CycloneDX"
        or value.get("specVersion") != "1.6"
        or value.get("version") != 1
    ):
        raise SupplyChainError("encoder SBOM must be CycloneDX 1.6")
    metadata = _object(value.get("metadata"), "CycloneDX metadata")
    tools = _object(metadata.get("tools"), "CycloneDX tools")
    tool_components = _objects(tools.get("components"), "CycloneDX tool components")
    if not any(
        component.get("name") == "syft" and component.get("version") == expected_version
        for component in tool_components
    ):
        raise SupplyChainError("CycloneDX SBOM was not generated by the locked Syft version")
    root = _object(metadata.get("component"), "CycloneDX root component")
    if (
        root.get("type") != "container"
        or root.get("name") != EXPECTED_COMPONENT_NAME
        or root.get("version") != image_digest
    ):
        raise SupplyChainError("CycloneDX SBOM does not bind the supplied encoder image")
    components = value.get("components")
    if not isinstance(components, list) or not components:
        raise SupplyChainError("CycloneDX SBOM contains no components")
    return {
        "format": "cyclonedx-json",
        "spec_version": "1.6",
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "component_count": len(components),
    }


def _inspect_grype_scan(
    path: Path,
    *,
    expected_version: str,
    image_digest: str,
    evaluated_at: datetime,
) -> dict[str, object]:
    value = _load_json_object(path, "Grype scan")
    descriptor = _object(value.get("descriptor"), "Grype descriptor")
    if descriptor.get("name") != "grype" or descriptor.get("version") != expected_version:
        raise SupplyChainError("scan was not generated by the locked Grype version")
    if descriptor.get("configuration") != {"show-suppressed": True}:
        raise SupplyChainError("Grype scan must be normalized before retention")
    source = _object(value.get("source"), "Grype source")
    target = _object(source.get("target"), "Grype source target")
    if (
        source.get("type") != "image"
        or target.get("manifestDigest") != image_digest
        or target.get("userInput") != "encoder.oci.tar"
    ):
        raise SupplyChainError("Grype scan does not bind the supplied encoder image")
    ignored = value.get("ignoredMatches")
    if ignored is None:
        ignored = []
    elif not isinstance(ignored, list):
        raise SupplyChainError("Grype ignored matches must be an array")
    if ignored:
        raise SupplyChainError("ignored vulnerability findings are not allowed for release")
    matches = _objects(value.get("matches"), "Grype vulnerability matches")
    counts: Counter[str] = Counter()
    blocking: list[str] = []
    for match in matches:
        vulnerability = _object(match.get("vulnerability"), "Grype vulnerability")
        identifier = vulnerability.get("id")
        severity = vulnerability.get("severity")
        if not isinstance(identifier, str) or not identifier or severity not in SEVERITIES:
            raise SupplyChainError("Grype vulnerability finding is incomplete")
        severity_text = severity
        counts[severity_text] += 1
        if severity_text in BLOCKING_SEVERITIES:
            blocking.append(identifier)
    if blocking:
        raise SupplyChainError(
            f"release-blocking vulnerability found: {sorted(blocking)[0]}"
        )
    database = _object(descriptor.get("db"), "Grype vulnerability database")
    database_status = _object(database.get("status"), "Grype database status")
    if "path" in database_status:
        raise SupplyChainError("Grype scan must be normalized before retention")
    if database_status.get("valid") is not True:
        raise SupplyChainError("Grype vulnerability database is not valid")
    built_at = _parse_time(database_status.get("built"), "Grype database build time")
    database_age = evaluated_at - built_at
    if database_age < -MAXIMUM_CLOCK_SKEW:
        raise SupplyChainError("Grype database build time is in the future")
    if database_age > MAXIMUM_DATABASE_AGE:
        raise SupplyChainError("Grype vulnerability database is stale")
    schema_version = database_status.get("schemaVersion")
    if (
        not isinstance(schema_version, str)
        or GRYPE_DB_SCHEMA_PATTERN.fullmatch(schema_version) is None
    ):
        raise SupplyChainError("Grype database schema version is invalid")
    source_url = database_status.get("from")
    if not isinstance(source_url, str):
        raise SupplyChainError("Grype database source is invalid")
    parsed_source = urlparse(source_url)
    checksums = parse_qs(parsed_source.query).get("checksum", [])
    if (
        parsed_source.scheme != "https"
        or parsed_source.hostname != "grype.anchore.io"
        or len(checksums) != 1
    ):
        raise SupplyChainError("Grype database source is invalid")
    checksum = _digest(checksums[0], "Grype database checksum")
    providers = _object(database.get("providers"), "Grype database providers")
    if not providers:
        raise SupplyChainError("Grype vulnerability database has no providers")
    severity_counts = {severity: counts[severity] for severity in SEVERITIES}
    return {
        "format": "grype-json",
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "match_count": len(matches),
        "severity_counts": severity_counts,
        "tool": {"name": "grype", "version": expected_version},
        "database": {
            "built_at": _format_time(built_at),
            "checksum": checksum,
            "schema_version": schema_version,
            "provider_count": len(providers),
        },
        "policy": {
            "fail_on": ["High", "Critical"],
            "ignored_matches_allowed": False,
            "maximum_database_age_hours": 24,
        },
        "status": "passed",
    }


def _load_tool_lock(path: Path) -> dict[str, str]:
    value = _load_json_object(path, "supply-chain tool lock")
    if value.get("schema_version") != 1 or value.get("platform") != PLATFORM:
        raise SupplyChainError("supply-chain tool lock schema or platform is unsupported")
    tools = _object(value.get("tools"), "supply-chain tools")
    if set(tools) != set(EXPECTED_TOOLS):
        raise SupplyChainError("supply-chain tool lock must contain Syft, Grype, and Cosign")
    versions: dict[str, str] = {}
    for name in EXPECTED_TOOLS:
        tool = _object(tools.get(name), f"{name} tool lock")
        if set(tool) != {"version", "artifact", "sha256", "url"}:
            raise SupplyChainError(f"{name} tool lock fields are invalid")
        version = tool.get("version")
        artifact = tool.get("artifact")
        digest = tool.get("sha256")
        url = tool.get("url")
        if (
            not isinstance(version, str)
            or not version
            or not isinstance(artifact, str)
            or not artifact
            or not isinstance(digest, str)
            or HEX_SHA256_PATTERN.fullmatch(digest) is None
            or not isinstance(url, str)
            or not url.startswith("https://github.com/")
            or not url.endswith("/" + artifact)
        ):
            raise SupplyChainError(f"{name} tool lock is invalid")
        versions[name] = version
    return versions


def _safe_tar_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or "\\" in member.name:
            raise SupplyChainError(f"OCI archive contains unsafe path: {member.name}")
        if member.name in members:
            raise SupplyChainError(f"OCI archive contains duplicate path: {member.name}")
        if member.isdir():
            continue
        if not member.isfile():
            raise SupplyChainError(f"OCI archive contains non-regular entry: {member.name}")
        members[member.name] = member
    return members


def _read_tar_json(
    archive: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    name: str,
    label: str,
) -> dict[str, object]:
    member = members.get(name)
    if member is None:
        raise SupplyChainError(f"{label} is missing from the OCI archive")
    if member.size > 10 * 1024 * 1024:
        raise SupplyChainError(f"{label} exceeds the metadata size limit")
    handle = archive.extractfile(member)
    if handle is None:
        raise SupplyChainError(f"unable to read {label}")
    return _load_json_bytes(handle.read(), label)


def _verified_blob_bytes(
    archive: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    descriptor: dict[str, object],
    label: str,
) -> bytes:
    size = _descriptor_size(descriptor, label)
    if size > 10 * 1024 * 1024:
        raise SupplyChainError(f"{label} exceeds the metadata size limit")
    member = _blob_member(members, descriptor, label)
    handle = archive.extractfile(member)
    if handle is None:
        raise SupplyChainError(f"unable to read {label}")
    value = handle.read()
    if len(value) != size or _digest_bytes(value) != descriptor["digest"]:
        raise SupplyChainError(f"{label} digest or size does not match its descriptor")
    return value


def _verify_blob(
    archive: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    descriptor: dict[str, object],
    label: str,
) -> None:
    member = _blob_member(members, descriptor, label)
    size = _descriptor_size(descriptor, label)
    if member.size != size:
        raise SupplyChainError(f"{label} size does not match its descriptor")
    handle = archive.extractfile(member)
    if handle is None:
        raise SupplyChainError(f"unable to read {label}")
    if _sha256_stream(handle) != cast(str, descriptor["digest"]).removeprefix("sha256:"):
        raise SupplyChainError(f"{label} digest does not match its descriptor")


def _blob_member(
    members: dict[str, tarfile.TarInfo],
    descriptor: dict[str, object],
    label: str,
) -> tarfile.TarInfo:
    digest = _digest(descriptor.get("digest"), f"{label} digest")
    name = "blobs/sha256/" + digest.removeprefix("sha256:")
    member = members.get(name)
    if member is None:
        raise SupplyChainError(f"{label} blob is missing")
    return member


def _descriptor_size(descriptor: dict[str, object], label: str) -> int:
    size = descriptor.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise SupplyChainError(f"{label} size is invalid")
    return size


def _validate_platform(value: dict[str, object]) -> None:
    if value.get("architecture") != "amd64" or value.get("os") != "linux":
        raise SupplyChainError("OCI image platform must be Linux amd64")


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    _require_regular_file(path, label)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SupplyChainError(f"unable to read {label}: {error}") from error
    return _object(value, label)


def _load_json_bytes(value: bytes, label: str) -> dict[str, object]:
    try:
        decoded = json.loads(
            value,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SupplyChainError(f"unable to read {label}: {error}") from error
    return _object(decoded, label)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise SupplyChainError(f"JSON contains duplicate key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SupplyChainError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _objects(value: object, label: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise SupplyChainError(f"{label} must be an array")
    return [_object(item, label) for item in value]


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or DIGEST_PATTERN.fullmatch(value) is None:
        raise SupplyChainError(f"{label} must be an immutable sha256 digest")
    return value


def _utc_time(value: datetime, label: str) -> datetime:
    if value.tzinfo is None:
        raise SupplyChainError(f"{label} must include a timezone")
    return value.astimezone(UTC)


def _parse_time(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise SupplyChainError(f"{label} must be an RFC 3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SupplyChainError(f"{label} must be an RFC 3339 timestamp") from error
    return _utc_time(parsed, label)


def _format_time(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise SupplyChainError(f"{label} must be a regular file")


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_stream(handle)


def _sha256_stream(handle: IO[bytes]) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _replace_with_canonical_json(path: Path, value: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(value) + b"\n")
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bind encoder OCI release security evidence")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect-image")
    inspect_parser.add_argument("--oci-archive", required=True, type=Path)
    normalize_parser = subparsers.add_parser("normalize")
    normalize_parser.add_argument("--syft-json", required=True, type=Path)
    normalize_parser.add_argument("--grype-json", required=True, type=Path)
    for command in ("build", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--oci-archive", required=True, type=Path)
        subparser.add_argument("--sealed-identity", required=True, type=Path)
        subparser.add_argument("--syft-json", required=True, type=Path)
        subparser.add_argument("--cyclonedx-json", required=True, type=Path)
        subparser.add_argument("--grype-json", required=True, type=Path)
        subparser.add_argument("--tool-lock", required=True, type=Path)
        subparser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "inspect-image":
        print(_canonical_json(inspect_oci_archive(args.oci_archive)).decode())
        return
    if args.command == "normalize":
        normalize_scanner_evidence(args.syft_json, args.grype_json)
        print("normalized scanner evidence for retention")
        return
    inputs = {
        "oci_archive": args.oci_archive,
        "sealed_identity": args.sealed_identity,
        "syft_json": args.syft_json,
        "cyclonedx_json": args.cyclonedx_json,
        "grype_json": args.grype_json,
        "tool_lock": args.tool_lock,
    }
    if args.command == "build":
        manifest = build_supply_chain_manifest(**inputs)
        write_supply_chain_manifest(args.manifest, manifest)
        print(f"wrote unsigned supply-chain manifest to {args.manifest}")
    else:
        verify_supply_chain_manifest(args.manifest, **inputs)
        print(f"verified supply-chain evidence for {args.oci_archive}")


if __name__ == "__main__":
    main()
