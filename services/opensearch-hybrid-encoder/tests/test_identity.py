import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from mageos_opensearch_hybrid_encoder.identity import (
    IdentityError,
    seal_identity,
    write_sealed_identity,
)


def artifact(root: Path, role: str, content: bytes) -> dict[str, object]:
    path = root / role
    path.write_bytes(content)
    return {
        "path": role,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
    }


def manifest(root: Path) -> dict[str, object]:
    model = artifact(root, "model", b"shared_fp32_model")
    artifacts = {
        "document_model": model,
        "query_model": model,
        **{
            role: artifact(root, role, role.encode())
            for role in (
                "tokenizer",
                "model_config",
                "amd64_parity",
                "quality_guard",
                "self_retrieval",
                "reference_vectors",
                "latency_concurrency_four",
            )
        },
    }
    return {
        "schema_version": 2,
        "api_version": "v1",
        "model": {
            "id": "Snowflake/snowflake-arctic-embed-m-v2.0",
            "revision": "95c2741480856aa9666782eb4afe11959938017f",
            "dimension": 256,
            "similarity": "cosine",
            "normalization": "l2_after_truncation",
            "truncate_to_dimension": 256,
            "max_sequence_length": 512,
        },
        "recipe": {
            "version": "mageos-v1",
            "query_prefix": "query: ",
            "query_template": "{query}",
            "document_prefix": "",
            "document_template": "{title}. {description}. {attributes}",
            "lexical_brand_attributes": ["manufacturer"],
            "semantic_feature_attributes": ["color", "material"],
        },
        "runtime": {
            "architecture": "linux/amd64",
            "implementation": "mageos_onnxruntime_tokenizers_v1",
            "execution_provider": "CPUExecutionProvider",
            "query_precision": "fp32",
            "document_precision": "fp32",
            "query_batch_size": 1,
            "max_batch_size": 64,
            "versions": {
                "python": "3.13.7",
                "numpy": "2.5.2",
                "onnxruntime": "1.22.1",
                "tokenizers": "0.22.2",
            },
        },
        "deployment": {
            "kind": "oci_image",
            "artifact_digest": "sha256:" + ("a" * 64),
            "base_artifact_digest": "sha256:" + ("b" * 64),
        },
        "artifacts": artifacts,
        "qualification": {
            "status": "qualified",
            "evidence_artifact_roles": [
                "amd64_parity",
                "quality_guard",
                "self_retrieval",
                "reference_vectors",
                "latency_concurrency_four",
            ],
        },
    }


def test_sealed_identity_binds_contract_runtime_artifacts_and_evidence(tmp_path: Path) -> None:
    value = manifest(tmp_path)
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(value))

    identity = seal_identity(path, tmp_path)

    assert identity["schema_version"] == 2
    assert identity["api_version"] == "v1"
    assert identity["production_eligible"] is True
    assert identity["architecture"] == "linux/amd64"
    assert identity["model_contract_digest"] == (
        "f8af800a6f6d690737cd6d0968e8d56985647abb8cf738e8913e062191f6c1e0"
    )
    assert len(str(identity["encoder_identity_digest"])) == 64
    artifacts = cast(dict[str, dict[str, object]], identity["artifacts"])
    assert artifacts["query_model"] == artifacts["document_model"]
    assert artifacts["query_model"]["sha256"] == hashlib.sha256(
        b"shared_fp32_model"
    ).hexdigest()
    deployment = cast(dict[str, object], identity["deployment"])
    assert deployment["kind"] == "oci_image"
    assert "image" not in deployment


def test_sealed_identity_rejects_changed_artifact_bytes(tmp_path: Path) -> None:
    value = manifest(tmp_path)
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(value))
    (tmp_path / "model").write_bytes(b"substituted")

    with pytest.raises(IdentityError, match="model"):
        seal_identity(path, tmp_path)


def test_sealed_identity_rejects_distinct_query_and_document_models(tmp_path: Path) -> None:
    value = manifest(tmp_path)
    artifacts = cast(dict[str, dict[str, object]], value["artifacts"])
    artifacts["query_model"] = artifact(tmp_path, "query", b"different")
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(value))

    with pytest.raises(IdentityError, match="same fp32 artifact"):
        seal_identity(path, tmp_path)


def test_sealed_identity_rejects_arm64_or_incomplete_qualification(tmp_path: Path) -> None:
    value = manifest(tmp_path)
    runtime = value["runtime"]
    assert isinstance(runtime, dict)
    runtime["architecture"] = "linux/arm64"
    qualification = value["qualification"]
    assert isinstance(qualification, dict)
    qualification["evidence_artifact_roles"] = ["amd64_parity"]
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(value))

    with pytest.raises(IdentityError, match="linux/amd64"):
        seal_identity(path, tmp_path)


def test_sealed_identity_rejects_artifact_path_traversal(tmp_path: Path) -> None:
    value = manifest(tmp_path)
    artifacts = value["artifacts"]
    assert isinstance(artifacts, dict)
    query_model = artifacts["query_model"]
    assert isinstance(query_model, dict)
    query_model["path"] = "../query_model"
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(value))

    with pytest.raises(IdentityError, match="relative path"):
        seal_identity(path, tmp_path)


def test_sealed_identity_rejects_registry_name_in_canonical_deployment(tmp_path: Path) -> None:
    value = manifest(tmp_path)
    deployment = value["deployment"]
    assert isinstance(deployment, dict)
    deployment["image"] = "ghcr.io/example/opensearch-hybrid-encoder"
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(value))

    with pytest.raises(IdentityError, match="deployment fields"):
        seal_identity(path, tmp_path)


def test_sealed_identity_output_is_exclusive_and_reviewable(tmp_path: Path) -> None:
    output = tmp_path / "sealed-identity.json"
    value = {"encoder_identity_digest": "a" * 64, "production_eligible": True}

    write_sealed_identity(output, value)

    assert json.loads(output.read_text()) == value
    with pytest.raises(IdentityError, match="already exists"):
        write_sealed_identity(output, value)
