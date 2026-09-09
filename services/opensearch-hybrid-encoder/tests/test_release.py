from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from mageos_opensearch_hybrid_encoder.identity import seal_identity
from mageos_opensearch_hybrid_encoder.qualification import artifact_facts
from mageos_opensearch_hybrid_encoder.release import ReleaseError, build_manifest


def release_artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    files = {
        "models/model.onnx": b"shared-fp32-model",
        "tokenizer/tokenizer.json": b"tokenizer",
        "model/config.json": json.dumps(
            {
                "architectures": ["GteModel"],
                "hidden_size": 768,
                "model_type": "gte",
            }
        ).encode(),
    }
    for relative, contents in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    bound = {
        "document_model": artifact_facts(root / "models/model.onnx"),
        "query_model": artifact_facts(root / "models/model.onnx"),
        "tokenizer": artifact_facts(root / "tokenizer/tokenizer.json"),
    }
    evidence = {
        "evidence/amd64-parity.json": {
            "artifact_type": "amd64_parity",
            "status": "passed",
            "decision_eligible": True,
            "platform": {"system": "Linux", "architecture": "amd64"},
            "runtime_versions": runtime_versions(),
            "artifacts": bound,
        },
        "evidence/quality-guard.json": {
            "schema_version": 2,
            "status": "passed",
            "source_inputs_eligible": True,
            "quality_evidence_eligible_for_decision": True,
            "model": frozen_evidence_model(),
            "fp32_ndcg@10": 0.78,
            "selection_surface": "WANDS development split",
            "held_out_test_metrics_used": False,
        },
        "evidence/self-retrieval.json": {
            "schema_version": 2,
            "artifact_type": "self_retrieval",
            "status": "passed",
            "quality_evidence_eligible_for_decision": True,
            "model": frozen_evidence_model(),
            "minimum_top1_rate": 0.8,
            "fp32": {"queries": 1000, "top1_hits": 948, "top1_rate": 0.948},
        },
        "evidence/reference-vectors.json": {
            "artifact_type": "reference_vectors",
            "status": "captured",
            "decision_eligible": False,
            "runtime_versions": runtime_versions(),
            "artifacts": bound,
        },
        "evidence/latency-concurrency-four.json": {
            "artifact_type": "latency_concurrency_four",
            "status": "passed",
            "decision_eligible": True,
            "platform": {"system": "Linux", "architecture": "amd64"},
            "runtime_versions": runtime_versions(),
            "artifacts": bound,
            "concurrency": 4,
            "batch_size": 1,
            "summary_ms": {"p95": 100.0},
            "p95_budget_ms": 200.0,
        },
    }
    for relative, value in evidence.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    return root


def runtime_versions() -> dict[str, str]:
    return {
        "python": "3.13.15",
        "numpy": "2.5.2",
        "onnxruntime": "1.28.0",
        "tokenizers": "0.22.2",
    }


def frozen_evidence_model() -> dict[str, str]:
    return {
        "hf_id": "Snowflake/snowflake-arctic-embed-m-v2.0",
        "revision": "95c2741480856aa9666782eb4afe11959938017f",
    }


def test_release_manifest_binds_canonical_artifacts_without_registry_name(
    tmp_path: Path,
) -> None:
    root = release_artifacts(tmp_path)

    manifest = build_manifest(
        root,
        deployment_digest="sha256:" + ("a" * 64),
        base_artifact_digest="sha256:" + ("b" * 64),
    )

    assert manifest["schema_version"] == 2
    deployment = cast(dict[str, str], manifest["deployment"])
    assert deployment == {
        "kind": "oci_image",
        "artifact_digest": "sha256:" + ("a" * 64),
        "base_artifact_digest": "sha256:" + ("b" * 64),
    }
    assert "image" not in deployment
    runtime = cast(dict[str, object], manifest["runtime"])
    assert runtime["versions"] == runtime_versions()
    artifacts = cast(dict[str, dict[str, object]], manifest["artifacts"])
    assert artifacts["query_model"]["path"] == "models/model.onnx"
    assert artifacts["query_model"] == artifacts["document_model"]
    assert artifacts["amd64_parity"]["path"] == "evidence/amd64-parity.json"
    manifest_path = tmp_path / "identity.json"
    manifest_path.write_text(json.dumps(manifest))
    sealed = seal_identity(manifest_path, root)
    assert sealed["production_eligible"] is True


def test_release_manifest_rejects_failed_or_mismatched_evidence(tmp_path: Path) -> None:
    root = release_artifacts(tmp_path)
    quality = root / "evidence/quality-guard.json"
    value = json.loads(quality.read_text())
    value["status"] = "failed"
    quality.write_text(json.dumps(value))

    with pytest.raises(ReleaseError, match="quality guard"):
        build_manifest(
            root,
            deployment_digest="sha256:" + ("a" * 64),
            base_artifact_digest="sha256:" + ("b" * 64),
        )

    root = release_artifacts(tmp_path / "second")
    parity = root / "evidence/amd64-parity.json"
    value = json.loads(parity.read_text())
    value["artifacts"]["query_model"]["sha256"] = "c" * 64
    parity.write_text(json.dumps(value))

    with pytest.raises(ReleaseError, match="artifact binding"):
        build_manifest(
            root,
            deployment_digest="sha256:" + ("a" * 64),
            base_artifact_digest="sha256:" + ("b" * 64),
        )


def test_release_manifest_requires_selected_fp32_quality_evidence(tmp_path: Path) -> None:
    root = release_artifacts(tmp_path)
    quality = root / "evidence/quality-guard.json"
    value = json.loads(quality.read_text())
    del value["fp32_ndcg@10"]
    quality.write_text(json.dumps(value))

    with pytest.raises(ReleaseError, match="fp32 quality"):
        build_manifest(
            root,
            deployment_digest="sha256:" + ("a" * 64),
            base_artifact_digest="sha256:" + ("b" * 64),
        )

    root = release_artifacts(tmp_path / "second")
    self_retrieval = root / "evidence/self-retrieval.json"
    value = json.loads(self_retrieval.read_text())
    del value["fp32"]
    self_retrieval.write_text(json.dumps(value))

    with pytest.raises(ReleaseError, match="fp32 self retrieval"):
        build_manifest(
            root,
            deployment_digest="sha256:" + ("a" * 64),
            base_artifact_digest="sha256:" + ("b" * 64),
        )
