from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"
PRODUCTION_DOCKERFILE = DEPLOY.parent / "Dockerfile.production"
IMAGE_DIGEST = "sha256:" + ("a" * 64)
IMAGE_REPOSITORY = "registry.example/community/opensearch-hybrid-encoder"


def canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_production_dockerfile_defaults_to_the_qualified_base_digest() -> None:
    first_line = PRODUCTION_DOCKERFILE.read_text().splitlines()[0]

    assert first_line == (
        "ARG PYTHON_BASE_IMAGE=python:3.13.15-slim-trixie@"
        "sha256:16f75ad0fbc6c4883a8afd63b2d700c3cf68ccffc1aaeca5304ca0a3a908451f"
    )


def deployment_files(tmp_path: Path) -> tuple[Path, str]:
    manifest = {
        "schema_version": 2,
        "deployment": {
            "kind": "oci_image",
            "artifact_digest": IMAGE_DIGEST,
            "base_artifact_digest": "sha256:" + ("b" * 64),
        },
    }
    identity_digest = canonical_digest(manifest)
    release = tmp_path / identity_digest
    release.mkdir()
    manifest_path = release / "identity.json"
    manifest_path.write_text(json.dumps(manifest))
    token_path = tmp_path / "encoder-token"
    token_path.write_text(("d" * 64) + "\n")
    os.chmod(token_path, 0o600)
    metrics_token_path = tmp_path / "encoder-metrics-token"
    metrics_token_path.write_text(("m" * 64) + "\n")
    os.chmod(metrics_token_path, 0o600)
    environment_path = tmp_path / "encoder.env"
    environment_path.write_text(
        "\n".join(
            [
                f"ENCODER_IMAGE={IMAGE_REPOSITORY}@{IMAGE_DIGEST}",
                f"ENCODER_DEPLOYMENT_DIGEST={IMAGE_DIGEST}",
                "ENCODER_PULL_POLICY=always",
                f"ENCODER_IDENTITY_SHORT={identity_digest[:12]}",
                "ENCODER_ROUTING_MODE=docker-network",
                f"ENCODER_NETWORK_ALIAS=mageos-hybrid-encoder-{identity_digest[:12]}",
                f"ENCODER_IDENTITY_MANIFEST_HOST_PATH={manifest_path}",
                f"ENCODER_API_TOKEN_HOST_PATH={token_path}",
                f"ENCODER_METRICS_TOKEN_HOST_PATH={metrics_token_path}",
                f"ENCODER_RUN_UID={os.getuid()}",
                f"ENCODER_RUN_GID={os.getgid()}",
                "MAGENTO_DOCKER_NETWORK=magento_default",
            ]
        )
        + "\n"
    )
    return environment_path, identity_digest


def test_preflight_binds_manifest_image_alias_and_secret_without_disclosure(
    tmp_path: Path,
) -> None:
    environment_path, identity_digest = deployment_files(tmp_path)

    result = subprocess.run(
        [sys.executable, str(DEPLOY / "preflight.py"), str(environment_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["identity_digest"] == identity_digest
    assert output["deployment_digest"] == IMAGE_DIGEST
    assert output["image_repository"] == IMAGE_REPOSITORY
    assert output["routing_mode"] == "docker-network"
    assert "d" * 64 not in result.stdout
    assert "m" * 64 not in result.stdout


def test_preflight_rejects_shared_api_and_metrics_credentials(tmp_path: Path) -> None:
    environment_path, _ = deployment_files(tmp_path)
    environment = environment_path.read_text()
    api_path = next(
        line.partition("=")[2]
        for line in environment.splitlines()
        if line.startswith("ENCODER_API_TOKEN_HOST_PATH=")
    )
    environment = environment.replace(
        next(
            line
            for line in environment.splitlines()
            if line.startswith("ENCODER_METRICS_TOKEN_HOST_PATH=")
        ),
        f"ENCODER_METRICS_TOKEN_HOST_PATH={api_path}",
    )
    environment_path.write_text(environment)

    result = subprocess.run(
        [sys.executable, str(DEPLOY / "preflight.py"), str(environment_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "separate and different" in result.stderr


def test_preflight_rejects_deployment_digest_drift(tmp_path: Path) -> None:
    environment_path, _ = deployment_files(tmp_path)
    environment_path.write_text(
        environment_path.read_text().replace(
            f"ENCODER_DEPLOYMENT_DIGEST={IMAGE_DIGEST}",
            "ENCODER_DEPLOYMENT_DIGEST=sha256:" + ("c" * 64),
        )
    )

    result = subprocess.run(
        [sys.executable, str(DEPLOY / "preflight.py"), str(environment_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "does not match ENCODER_IMAGE" in result.stderr


def test_preflight_supports_local_content_addressed_image_without_registry(
    tmp_path: Path,
) -> None:
    environment_path, _ = deployment_files(tmp_path)
    environment = environment_path.read_text()
    environment = environment.replace(
        f"ENCODER_IMAGE={IMAGE_REPOSITORY}@{IMAGE_DIGEST}",
        f"ENCODER_IMAGE={IMAGE_DIGEST}",
    ).replace("ENCODER_PULL_POLICY=always", "ENCODER_PULL_POLICY=never")
    environment_path.write_text(environment)

    result = subprocess.run(
        [sys.executable, str(DEPLOY / "preflight.py"), str(environment_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["image_reference"] == IMAGE_DIGEST
    assert output["pull_policy"] == "never"


def test_preflight_rejects_mutable_image_tag(tmp_path: Path) -> None:
    environment_path, _ = deployment_files(tmp_path)
    environment_path.write_text(
        environment_path.read_text().replace(
            f"ENCODER_IMAGE={IMAGE_REPOSITORY}@{IMAGE_DIGEST}",
            f"ENCODER_IMAGE={IMAGE_REPOSITORY}:latest",
        )
    )

    result = subprocess.run(
        [sys.executable, str(DEPLOY / "preflight.py"), str(environment_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "digest-qualified OCI reference" in result.stderr


def test_preflight_supports_bare_metal_magento_through_loopback(tmp_path: Path) -> None:
    environment_path, _ = deployment_files(tmp_path)
    environment = environment_path.read_text()
    environment = environment.replace(
        "ENCODER_ROUTING_MODE=docker-network",
        "ENCODER_ROUTING_MODE=loopback",
    )
    environment += "ENCODER_LOOPBACK_PORT=18081\n"
    environment_path.write_text(environment)

    result = subprocess.run(
        [sys.executable, str(DEPLOY / "preflight.py"), str(environment_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["loopback_endpoint"] == "http://127.0.0.1:18081"
