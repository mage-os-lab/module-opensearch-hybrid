#!/usr/bin/env python3
"""Validate an identity-addressed same-host deployment without printing secrets."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import stat
import sys
from pathlib import Path

NAMED_OCI_IMAGE_PATTERN = re.compile(
    r"^(?P<repository>[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?(?::[0-9]{1,5})?"
    r"(?:/[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?)+)@"
    r"(?P<digest>sha256:[0-9a-f]{64})$"
)
LOCAL_OCI_IMAGE_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
IDENTITY_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REQUIRED = {
    "ENCODER_API_TOKEN_HOST_PATH",
    "ENCODER_METRICS_TOKEN_HOST_PATH",
    "ENCODER_IDENTITY_MANIFEST_HOST_PATH",
    "ENCODER_IDENTITY_SHORT",
    "ENCODER_IMAGE",
    "ENCODER_DEPLOYMENT_DIGEST",
    "ENCODER_ROUTING_MODE",
    "ENCODER_PULL_POLICY",
    "ENCODER_RUN_GID",
    "ENCODER_RUN_UID",
}


class PreflightError(RuntimeError):
    """The release environment is incomplete or internally inconsistent."""


def load_environment(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text().splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise PreflightError(f"unable to read environment file: {error}") from error
    for number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in values or not key.replace("_", "").isalnum():
            raise PreflightError(f"invalid environment entry at line {number}")
        values[key] = value
    missing = sorted(key for key in REQUIRED if not values.get(key))
    if missing:
        raise PreflightError("missing required environment values: " + ", ".join(missing))
    return values


def canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate(environment: dict[str, str]) -> dict[str, str]:
    pull_policy = environment["ENCODER_PULL_POLICY"]
    if pull_policy not in {"always", "missing", "never"}:
        raise PreflightError("ENCODER_PULL_POLICY must be always, missing, or never")
    repository, image_digest = parse_oci_image(environment["ENCODER_IMAGE"], pull_policy)
    deployment_digest = environment["ENCODER_DEPLOYMENT_DIGEST"]
    if deployment_digest != image_digest:
        raise PreflightError("ENCODER_DEPLOYMENT_DIGEST does not match ENCODER_IMAGE")

    manifest_path = regular_absolute_file(
        environment["ENCODER_IDENTITY_MANIFEST_HOST_PATH"],
        "identity manifest",
    )
    try:
        manifest = json.loads(manifest_path.read_text(), object_pairs_hook=unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"unable to read identity manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise PreflightError("identity manifest must be a JSON object")
    identity_digest = canonical_digest(manifest)
    if IDENTITY_PATTERN.fullmatch(identity_digest) is None:
        raise PreflightError("identity manifest digest is invalid")
    expected_short = identity_digest[:12]
    if environment["ENCODER_IDENTITY_SHORT"] != expected_short:
        raise PreflightError("ENCODER_IDENTITY_SHORT does not match the manifest")
    if manifest.get("schema_version") != 2:
        raise PreflightError("identity manifest must use registry-neutral schema 2")
    deployment = manifest.get("deployment")
    if not isinstance(deployment, dict):
        raise PreflightError("identity manifest deployment record is missing")
    if set(deployment) != {"kind", "artifact_digest", "base_artifact_digest"}:
        raise PreflightError("identity manifest deployment record is invalid")
    if (
        deployment.get("kind") != "oci_image"
        or deployment.get("artifact_digest") != deployment_digest
        or LOCAL_OCI_IMAGE_PATTERN.fullmatch(str(deployment.get("base_artifact_digest", "")))
        is None
    ):
        raise PreflightError("identity manifest does not bind the selected OCI artifact")

    run_uid = numeric_id(environment["ENCODER_RUN_UID"], "ENCODER_RUN_UID")
    run_gid = numeric_id(environment["ENCODER_RUN_GID"], "ENCODER_RUN_GID")
    api_path, api_token = validate_token_file(
        environment["ENCODER_API_TOKEN_HOST_PATH"],
        "encoder API token",
        run_uid,
        run_gid,
    )
    metrics_path, metrics_token = validate_token_file(
        environment["ENCODER_METRICS_TOKEN_HOST_PATH"],
        "encoder metrics token",
        run_uid,
        run_gid,
    )
    if api_path == metrics_path or hmac.compare_digest(api_token, metrics_token):
        raise PreflightError("encoder API and metrics tokens must be separate and different")

    routing = environment["ENCODER_ROUTING_MODE"]
    result = {
        "identity_digest": identity_digest,
        "deployment_digest": deployment_digest,
        "image_reference": environment["ENCODER_IMAGE"],
        "pull_policy": pull_policy,
        "routing_mode": routing,
    }
    if repository is not None:
        result["image_repository"] = repository
    if routing == "docker-network":
        network_alias = environment.get("ENCODER_NETWORK_ALIAS", "")
        magento_network = environment.get("MAGENTO_DOCKER_NETWORK", "")
        if network_alias != f"mageos-hybrid-encoder-{expected_short}":
            raise PreflightError("ENCODER_NETWORK_ALIAS does not match the manifest")
        if not magento_network:
            raise PreflightError("MAGENTO_DOCKER_NETWORK is required for Docker routing")
        result["network_alias"] = network_alias
        result["magento_network"] = magento_network
    elif routing == "loopback":
        raw_port = environment.get("ENCODER_LOOPBACK_PORT", "")
        try:
            port = int(raw_port)
        except ValueError as error:
            raise PreflightError("ENCODER_LOOPBACK_PORT must be an integer") from error
        if port < 1024 or port > 65535:
            raise PreflightError("ENCODER_LOOPBACK_PORT must be between 1024 and 65535")
        result["loopback_endpoint"] = f"http://127.0.0.1:{port}"
    else:
        raise PreflightError("ENCODER_ROUTING_MODE must be docker-network or loopback")

    return result


def parse_oci_image(value: str, pull_policy: str) -> tuple[str | None, str]:
    if LOCAL_OCI_IMAGE_PATTERN.fullmatch(value) is not None:
        if pull_policy != "never":
            raise PreflightError("a local content-addressed image requires pull policy never")
        return None, value
    match = NAMED_OCI_IMAGE_PATTERN.fullmatch(value)
    if match is None:
        raise PreflightError(
            "ENCODER_IMAGE must be a digest-qualified OCI reference or local sha256 image ID"
        )
    return match.group("repository"), match.group("digest")


def regular_absolute_file(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise PreflightError(f"{label} must be an absolute regular non-symlink file")
    return path


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise PreflightError(f"identity manifest contains duplicate key: {key}")
        value[key] = item
    return value


def numeric_id(value: str, label: str) -> int:
    try:
        identifier = int(value)
    except ValueError as error:
        raise PreflightError(f"{label} must be an integer") from error
    if identifier < 1 or identifier > 2_147_483_647:
        raise PreflightError(f"{label} must identify a non-root user or group")
    return identifier


def validate_token_file(
    value: str,
    label: str,
    run_uid: int,
    run_gid: int,
) -> tuple[Path, str]:
    path = regular_absolute_file(value, label)
    details = path.stat()
    if details.st_uid != run_uid or details.st_gid != run_gid:
        raise PreflightError(f"{label} owner must match ENCODER_RUN_UID and GID")
    if stat.S_IMODE(details.st_mode) not in {0o400, 0o600}:
        raise PreflightError(f"{label} mode must be 0400 or 0600")
    try:
        token = path.read_text()
    except (OSError, UnicodeDecodeError) as error:
        raise PreflightError(f"unable to read {label}: {error}") from error
    token = token.removesuffix("\n").removesuffix("\r")
    size = len(token.encode())
    if (
        size < 32
        or size > 4096
        or any(ord(character) < 33 or ord(character) > 126 for character in token)
    ):
        raise PreflightError(f"{label} must be 32 to 4096 visible ASCII bytes")
    return path, token


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: preflight.py /absolute/path/to/encoder.env")
    try:
        result = validate(load_environment(Path(sys.argv[1])))
    except PreflightError as error:
        raise SystemExit(f"preflight failed: {error}") from error
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
