"""Build a sealed-identity manifest from a fully qualified release artifact tree."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeGuard, cast

from .identity import (
    EXPECTED_MODEL,
    EXPECTED_RECIPE,
    REQUIRED_ARTIFACT_ROLES,
    REQUIRED_EVIDENCE_ROLES,
)
from .qualification import artifact_facts

DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
EXPECTED_RUNTIME_VERSIONS = {
    "python": "3.13.15",
    "numpy": "2.5.2",
    "onnxruntime": "1.28.0",
    "tokenizers": "0.22.2",
}
ARTIFACT_PATHS = {
    "document_model": "models/model.onnx",
    "query_model": "models/model.onnx",
    "tokenizer": "tokenizer/tokenizer.json",
    "model_config": "model/config.json",
    "amd64_parity": "evidence/amd64-parity.json",
    "quality_guard": "evidence/quality-guard.json",
    "self_retrieval": "evidence/self-retrieval.json",
    "reference_vectors": "evidence/reference-vectors.json",
    "latency_concurrency_four": "evidence/latency-concurrency-four.json",
}


class ReleaseError(RuntimeError):
    """A release tree cannot prove the frozen production identity contract."""


def build_manifest(
    artifact_root: Path,
    *,
    deployment_digest: str,
    base_artifact_digest: str,
) -> dict[str, object]:
    """Validate every release byte and return the registry-neutral schema 2 manifest."""
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise ReleaseError("artifact root must be a regular directory")
    for label, value in {
        "deployment digest": deployment_digest,
        "base artifact digest": base_artifact_digest,
    }.items():
        if DIGEST_PATTERN.fullmatch(value) is None:
            raise ReleaseError(f"{label} must be an immutable sha256 digest")
    if tuple(ARTIFACT_PATHS) != REQUIRED_ARTIFACT_ROLES:
        raise ReleaseError("release artifact path registry differs from the identity contract")

    records = {
        role: _artifact_record(artifact_root, relative)
        for role, relative in ARTIFACT_PATHS.items()
    }
    _validate_model_config(artifact_root / ARTIFACT_PATHS["model_config"])
    bound_runtime_artifacts = {
        role: {"sha256": records[role]["sha256"], "size": records[role]["size"]}
        for role in ("document_model", "query_model", "tokenizer")
    }
    evidence = {
        role: _read_json_object(artifact_root / ARTIFACT_PATHS[role], role)
        for role in REQUIRED_EVIDENCE_ROLES
    }
    _validate_reference(evidence["reference_vectors"], bound_runtime_artifacts)
    versions = _validate_amd64_parity(
        evidence["amd64_parity"],
        bound_runtime_artifacts,
    )
    _validate_quality_guard(evidence["quality_guard"])
    _validate_self_retrieval(evidence["self_retrieval"])
    _validate_latency(
        evidence["latency_concurrency_four"],
        bound_runtime_artifacts,
        versions,
    )

    return {
        "schema_version": 2,
        "api_version": "v1",
        "model": dict(EXPECTED_MODEL),
        "recipe": dict(EXPECTED_RECIPE),
        "runtime": {
            "architecture": "linux/amd64",
            "implementation": "mageos_onnxruntime_tokenizers_v1",
            "execution_provider": "CPUExecutionProvider",
            "query_precision": "fp32",
            "document_precision": "fp32",
            "query_batch_size": 1,
            "max_batch_size": 64,
            "versions": versions,
        },
        "deployment": {
            "kind": "oci_image",
            "artifact_digest": deployment_digest,
            "base_artifact_digest": base_artifact_digest,
        },
        "artifacts": records,
        "qualification": {
            "status": "qualified",
            "evidence_artifact_roles": list(REQUIRED_EVIDENCE_ROLES),
        },
    }


def _artifact_record(artifact_root: Path, relative: str) -> dict[str, object]:
    path = artifact_root / relative
    try:
        facts = artifact_facts(path)
    except RuntimeError as error:
        raise ReleaseError(f"release artifact is invalid: {relative}") from error
    return {"path": relative, **facts}


def _validate_model_config(path: Path) -> None:
    value = _read_json_object(path, "model_config")
    if (
        value.get("architectures") != ["GteModel"]
        or value.get("hidden_size") != 768
        or value.get("model_type") != "gte"
    ):
        raise ReleaseError("model config does not match the frozen Arctic architecture")


def _read_json_object(path: Path, role: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant: {value}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ReleaseError(f"{role} evidence must be strict JSON") from error
    if not isinstance(value, dict):
        raise ReleaseError(f"{role} evidence must be a JSON object")
    return cast(dict[str, Any], value)


def _validate_reference(
    evidence: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, object]],
) -> None:
    if (
        evidence.get("artifact_type") != "reference_vectors"
        or evidence.get("status") != "captured"
        or evidence.get("decision_eligible") is not False
        or evidence.get("runtime_versions") != EXPECTED_RUNTIME_VERSIONS
        or evidence.get("artifacts") != artifacts
    ):
        raise ReleaseError("reference vector evidence or artifact binding is invalid")


def _validate_amd64_parity(
    evidence: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, str]:
    versions = evidence.get("runtime_versions")
    if (
        evidence.get("artifact_type") != "amd64_parity"
        or evidence.get("status") != "passed"
        or evidence.get("decision_eligible") is not True
        or evidence.get("platform") != {"system": "Linux", "architecture": "amd64"}
        or evidence.get("artifacts") != artifacts
        or versions != EXPECTED_RUNTIME_VERSIONS
    ):
        raise ReleaseError("amd64 parity evidence or artifact binding is invalid")
    return dict(EXPECTED_RUNTIME_VERSIONS)


def _validate_quality_guard(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("schema_version") != 2
        or evidence.get("status") != "passed"
        or evidence.get("source_inputs_eligible") is not True
        or evidence.get("quality_evidence_eligible_for_decision") is not True
        or not _matches_evidence_model(evidence.get("model"))
    ):
        raise ReleaseError("quality guard evidence is not decision eligible")
    fp32_ndcg = evidence.get("fp32_ndcg@10")
    if (
        not _finite_number(fp32_ndcg)
        or not 0.0 <= float(fp32_ndcg) <= 1.0
        or evidence.get("selection_surface") != "WANDS development split"
        or evidence.get("held_out_test_metrics_used") is not False
    ):
        raise ReleaseError("fp32 quality evidence is incomplete or invalid")


def _validate_self_retrieval(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("schema_version") != 2
        or evidence.get("artifact_type") != "self_retrieval"
        or evidence.get("status") != "passed"
        or evidence.get("quality_evidence_eligible_for_decision") is not True
        or not _matches_evidence_model(evidence.get("model"))
    ):
        raise ReleaseError("self retrieval evidence is not decision eligible")
    fp32 = evidence.get("fp32")
    minimum = evidence.get("minimum_top1_rate")
    if not isinstance(fp32, dict):
        raise ReleaseError("fp32 self retrieval evidence is incomplete or invalid")
    queries = fp32.get("queries")
    hits = fp32.get("top1_hits")
    rate = fp32.get("top1_rate")
    if (
        not isinstance(queries, int)
        or isinstance(queries, bool)
        or queries <= 0
        or not isinstance(hits, int)
        or isinstance(hits, bool)
        or not 0 <= hits <= queries
        or not _finite_number(rate)
        or not _finite_number(minimum)
        or not 0.0 < float(minimum) <= 1.0
        or not math.isclose(float(rate), hits / queries, abs_tol=1e-12)
        or float(rate) < float(minimum)
    ):
        raise ReleaseError("fp32 self retrieval evidence is incomplete or invalid")


def _finite_number(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _matches_evidence_model(value: object) -> bool:
    return (
        isinstance(value, dict)
        and value.get("hf_id") == EXPECTED_MODEL["id"]
        and value.get("revision") == EXPECTED_MODEL["revision"]
    )


def _validate_latency(
    evidence: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, object]],
    versions: Mapping[str, str],
) -> None:
    summary = evidence.get("summary_ms")
    p95 = summary.get("p95") if isinstance(summary, dict) else None
    if (
        evidence.get("artifact_type") != "latency_concurrency_four"
        or evidence.get("status") != "passed"
        or evidence.get("decision_eligible") is not True
        or evidence.get("platform") != {"system": "Linux", "architecture": "amd64"}
        or evidence.get("runtime_versions") != versions
        or evidence.get("artifacts") != artifacts
        or evidence.get("concurrency") != 4
        or evidence.get("batch_size") != 1
        or evidence.get("p95_budget_ms") != 200.0
        or not isinstance(p95, int | float)
        or not math.isfinite(p95)
        or p95 > 200.0
    ):
        raise ReleaseError("concurrency-four latency evidence or artifact binding is invalid")


def _write_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    if not path.is_absolute() or not path.parent.is_dir() or path.is_symlink():
        raise ReleaseError("manifest output must be a new absolute path under a directory")
    try:
        with path.open("x") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise ReleaseError("manifest output already exists") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a qualified encoder identity manifest")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--deployment-digest", required=True)
    parser.add_argument("--base-artifact-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(
        args.artifact_root,
        deployment_digest=args.deployment_digest,
        base_artifact_digest=args.base_artifact_digest,
    )
    _write_manifest(args.output, manifest)
    print(f"wrote qualified identity manifest to {args.output}")


if __name__ == "__main__":
    main()
