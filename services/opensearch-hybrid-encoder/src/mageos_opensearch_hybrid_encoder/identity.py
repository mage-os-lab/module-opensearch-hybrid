"""Seal a production encoder identity from immutable local artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 2
API_VERSION = "v1"
PRODUCTION_ARCHITECTURE = "linux/amd64"
REQUIRED_ARTIFACT_ROLES = (
    "document_model",
    "query_model",
    "tokenizer",
    "model_config",
    "amd64_parity",
    "quality_guard",
    "self_retrieval",
    "reference_vectors",
    "latency_concurrency_four",
)
REQUIRED_EVIDENCE_ROLES = (
    "amd64_parity",
    "quality_guard",
    "self_retrieval",
    "reference_vectors",
    "latency_concurrency_four",
)
EXPECTED_MODEL = {
    "id": "Snowflake/snowflake-arctic-embed-m-v2.0",
    "revision": "95c2741480856aa9666782eb4afe11959938017f",
    "dimension": 256,
    "similarity": "cosine",
    "normalization": "l2_after_truncation",
    "truncate_to_dimension": 256,
    "max_sequence_length": 512,
}
EXPECTED_RECIPE = {
    "version": "mageos-v1",
    "query_prefix": "query: ",
    "query_template": "{query}",
    "document_prefix": "",
    "document_template": "{title}. {description}. {attributes}",
    "lexical_brand_attributes": ["manufacturer"],
    "semantic_feature_attributes": ["color", "material"],
}
MODEL_CONTRACT = {
    "id": EXPECTED_MODEL["id"],
    "revision": EXPECTED_MODEL["revision"],
    "dimension": EXPECTED_MODEL["dimension"],
    "similarity": EXPECTED_MODEL["similarity"],
    "query_prefix": EXPECTED_RECIPE["query_prefix"],
    "query_template": EXPECTED_RECIPE["query_template"],
    "document_template": EXPECTED_RECIPE["document_template"],
    "source_recipe_version": EXPECTED_RECIPE["version"],
    "lexical_brand_attributes": EXPECTED_RECIPE["lexical_brand_attributes"],
    "semantic_feature_attributes": EXPECTED_RECIPE["semantic_feature_attributes"],
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class IdentityError(ValueError):
    """The production identity cannot be proven from the supplied bytes."""


def seal_identity(manifest_path: Path, artifact_root: Path) -> dict[str, object]:
    """Validate and flatten an immutable production identity manifest."""
    manifest = _load_manifest(manifest_path)
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "api_version",
            "model",
            "recipe",
            "runtime",
            "deployment",
            "artifacts",
            "qualification",
        },
        "manifest",
    )
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["api_version"] != API_VERSION:
        raise IdentityError("identity schema or API version is not supported")
    model = _object(manifest["model"], "model")
    recipe = _object(manifest["recipe"], "recipe")
    if model != EXPECTED_MODEL or recipe != EXPECTED_RECIPE:
        raise IdentityError("model or recipe does not match the frozen Mage-OS contract")
    runtime = _validate_runtime(_object(manifest["runtime"], "runtime"))
    deployment = _validate_deployment(_object(manifest["deployment"], "deployment"))
    artifacts = _validate_artifacts(
        _object(manifest["artifacts"], "artifacts"),
        artifact_root,
    )
    if artifacts["query_model"] != artifacts["document_model"]:
        raise IdentityError("query and document roles must bind the same fp32 artifact")
    qualification = _validate_qualification(
        _object(manifest["qualification"], "qualification"),
        artifacts,
    )
    identity_digest = hashlib.sha256(_canonical_json(manifest)).hexdigest()

    return {
        "schema_version": SCHEMA_VERSION,
        "api_version": API_VERSION,
        "model_id": EXPECTED_MODEL["id"],
        "model_revision": EXPECTED_MODEL["revision"],
        "dimension": EXPECTED_MODEL["dimension"],
        "recipe_version": EXPECTED_RECIPE["version"],
        "model_contract_digest": hashlib.sha256(_canonical_json(MODEL_CONTRACT)).hexdigest(),
        "encoder_identity_digest": identity_digest,
        "production_eligible": True,
        "architecture": PRODUCTION_ARCHITECTURE,
        "runtime": runtime,
        "deployment": deployment,
        "artifacts": artifacts,
        "qualification": qualification,
        "identity_manifest": manifest,
    }


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(), object_pairs_hook=_unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IdentityError(f"unable to read identity manifest: {error}") from error
    return _object(value, "manifest")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise IdentityError(f"identity manifest contains duplicate key: {key}")
        value[key] = item
    return value


def _validate_runtime(runtime: dict[str, object]) -> dict[str, object]:
    _require_exact_keys(
        runtime,
        {
            "architecture",
            "implementation",
            "execution_provider",
            "query_precision",
            "document_precision",
            "query_batch_size",
            "max_batch_size",
            "versions",
        },
        "runtime",
    )
    if runtime["architecture"] != PRODUCTION_ARCHITECTURE:
        raise IdentityError("production identity requires the linux/amd64 runtime")
    expected = {
        "execution_provider": "CPUExecutionProvider",
        "query_precision": "fp32",
        "document_precision": "fp32",
        "query_batch_size": 1,
        "max_batch_size": 64,
    }
    if any(runtime[key] != value for key, value in expected.items()):
        raise IdentityError("runtime precision, provider, or batching is not the frozen contract")
    if not _bounded_text(runtime["implementation"], 128):
        raise IdentityError("runtime implementation must be a bounded non-empty string")
    versions = _object(runtime["versions"], "runtime versions")
    _require_exact_keys(versions, {"python", "numpy", "onnxruntime", "tokenizers"}, "versions")
    if not all(_bounded_text(value, 64) for value in versions.values()):
        raise IdentityError("runtime versions must be bounded non-empty strings")

    return runtime


def _validate_deployment(deployment: dict[str, object]) -> dict[str, object]:
    _require_exact_keys(
        deployment,
        {"kind", "artifact_digest", "base_artifact_digest"},
        "deployment",
    )
    if deployment["kind"] != "oci_image":
        raise IdentityError("deployment kind must be oci_image")
    for key in ("artifact_digest", "base_artifact_digest"):
        value = deployment[key]
        if not isinstance(value, str) or ARTIFACT_DIGEST_PATTERN.fullmatch(value) is None:
            raise IdentityError(f"deployment {key} must be an immutable sha256 digest")

    return deployment


def _validate_artifacts(
    artifacts: dict[str, object], artifact_root: Path
) -> dict[str, dict[str, object]]:
    if set(artifacts) != set(REQUIRED_ARTIFACT_ROLES):
        raise IdentityError("artifact roles do not match the frozen production identity")
    resolved_root = artifact_root.resolve()
    verified: dict[str, dict[str, object]] = {}
    for role in sorted(artifacts):
        record = _object(artifacts[role], f"artifact {role}")
        _require_exact_keys(record, {"path", "sha256", "size"}, f"artifact {role}")
        relative = record["path"]
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise IdentityError(f"artifact {role} must use a relative path")
        pure_path = PurePosixPath(relative)
        if pure_path.is_absolute() or ".." in pure_path.parts:
            raise IdentityError(f"artifact {role} must use a relative path without traversal")
        expected_hash = record["sha256"]
        expected_size = record["size"]
        if not isinstance(expected_hash, str) or SHA256_PATTERN.fullmatch(expected_hash) is None:
            raise IdentityError(f"artifact {role} has an invalid sha256")
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size <= 0
        ):
            raise IdentityError(f"artifact {role} has an invalid size")
        candidate = artifact_root.joinpath(*pure_path.parts)
        resolved = candidate.resolve()
        if (
            not resolved.is_relative_to(resolved_root)
            or candidate.is_symlink()
            or not candidate.is_file()
        ):
            raise IdentityError(f"artifact {role} is not a regular file under the artifact root")
        actual = candidate.read_bytes()
        if len(actual) != expected_size or hashlib.sha256(actual).hexdigest() != expected_hash:
            raise IdentityError(f"artifact {role} bytes do not match the sealed identity")
        verified[role] = {"path": relative, "sha256": expected_hash, "size": expected_size}

    return verified


def _validate_qualification(
    qualification: dict[str, object], artifacts: dict[str, dict[str, object]]
) -> dict[str, object]:
    _require_exact_keys(qualification, {"status", "evidence_artifact_roles"}, "qualification")
    roles = qualification["evidence_artifact_roles"]
    if qualification["status"] != "qualified" or roles != list(REQUIRED_EVIDENCE_ROLES):
        raise IdentityError("qualification evidence is incomplete or not qualified")
    if not all(role in artifacts for role in REQUIRED_EVIDENCE_ROLES):
        raise IdentityError("qualification evidence artifacts are missing")

    return qualification


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise IdentityError(f"{label} must be a JSON object")
    return value


def _require_exact_keys(value: dict[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise IdentityError(f"{label} fields do not match the frozen identity schema")


def _bounded_text(value: object, maximum: int) -> bool:
    return isinstance(value, str) and bool(value) and len(value.encode()) <= maximum


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def write_sealed_identity(path: Path, identity: dict[str, object]) -> None:
    """Write reviewable sealed output without overwriting retained evidence."""
    if not path.is_absolute() or not path.parent.is_dir() or path.is_symlink():
        raise IdentityError("sealed identity output must be a new absolute path under a directory")
    try:
        with path.open("x") as handle:
            json.dump(identity, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise IdentityError("sealed identity output already exists") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and seal an amd64 encoder identity")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    identity = seal_identity(args.manifest, args.artifact_root)
    if args.output is None:
        print(json.dumps(identity, sort_keys=True))
    else:
        write_sealed_identity(args.output, identity)
        print(f"wrote sealed encoder identity to {args.output}")


if __name__ == "__main__":
    main()
