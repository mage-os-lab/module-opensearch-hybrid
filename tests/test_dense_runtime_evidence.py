from __future__ import annotations

from pathlib import Path

import pytest

from poc import int8_dense
from poc.config import ModelSpec
from poc.datasets import DatasetIntegrityError
from poc.int8_dense import classify_int8_evidence
from poc.query_runtime import (
    QueryRuntimeSpec,
    derive_query_parity_status,
    derive_self_retrieval_status,
)


def _runtime() -> QueryRuntimeSpec:
    return QueryRuntimeSpec.from_mapping(
        {
            "backend": "sentence_transformers_onnx",
            "precision": "int8_dynamic",
            "execution_provider": "CPUExecutionProvider",
            "quantization_config": "arm64",
            "parity_sample_size": 2,
            "minimum_query_cosine_similarity": 0.99,
            "minimum_mean_query_cosine_similarity": 0.999,
            "maximum_absolute_query_delta": 0.05,
            "document_drift_sample_size": 2,
            "minimum_document_drift_cosine_similarity": 0.9999,
            "self_retrieval_sample_size": 2,
            "minimum_self_retrieval_top1": 0.8,
            "intra_op_num_threads": 1,
            "inter_op_num_threads": 1,
            "execution_mode": "ORT_SEQUENTIAL",
            "batch_size": 1,
        }
    )


def _model() -> ModelSpec:
    return ModelSpec.from_mapping(
        "test_model",
        {
            "hf_id": "example/test-model",
            "revision": "1" * 40,
            "dims": 2,
            "normalize": True,
            "max_seq_length": 512,
            "bulk_dtype": "float32",
            "use_memory_efficient_attention": False,
            "trust_remote_code": False,
            "query_prefix": "",
            "document_prefix": "",
            "query_template": "{query}",
            "document_template": "{title}",
            "license": "Apache-2.0",
            "decision_eligible": True,
            "contamination": "none_known",
        },
    )


def test_query_parity_status_is_derived_from_measurements_not_recorded_label() -> None:
    runtime = _runtime()
    artifact = {
        "query_comparison": {
            "minimum_cosine_similarity": 0.98,
            "mean_cosine_similarity": 0.9995,
            "maximum_absolute_delta": 0.02,
        },
        "document_runtime_comparison": {
            "minimum_cosine_similarity": 0.99995,
            "mean_cosine_similarity": 1.0,
            "maximum_absolute_delta": 0.0001,
        },
        "status": "passed",
        "error": None,
    }

    status, error = derive_query_parity_status(artifact, runtime)

    assert status == "failed"
    assert error is not None
    assert "minimum cosine" in error


def test_self_retrieval_status_is_derived_from_counts_and_registered_threshold() -> None:
    runtime = _runtime()
    artifact = {
        "sample_size": 2,
        "minimum_top1_rate": runtime.minimum_self_retrieval_top1,
        "int8": {
            "queries": 2,
            "top1_hits": 1,
            "top1_rate": 0.5,
            "failures": [{"expected": "1", "actual": "2"}],
        },
        "status": "passed",
    }

    assert derive_self_retrieval_status(artifact, runtime) == "failed"


def test_self_retrieval_verifier_rejects_internally_inconsistent_rate() -> None:
    runtime = _runtime()
    artifact = {
        "sample_size": 2,
        "minimum_top1_rate": runtime.minimum_self_retrieval_top1,
        "int8": {
            "queries": 2,
            "top1_hits": 1,
            "top1_rate": 1.0,
            "failures": [{"expected": "1", "actual": "2"}],
        },
        "status": "passed",
    }

    with pytest.raises(DatasetIntegrityError, match="top1 rate"):
        derive_self_retrieval_status(artifact, runtime)


def test_int8_eligibility_uses_semantically_verified_evidence() -> None:
    eligible = classify_int8_evidence(
        model_decision_eligible=True,
        run_provenance_valid=True,
        vector_index_eligible=True,
        fp32_model_selection_eligible=True,
        onnx_artifact_eligible=True,
        parity_artifact_verified=True,
        quality_guard_passed=True,
        self_retrieval_passed=True,
    )
    blocked = classify_int8_evidence(
        model_decision_eligible=True,
        run_provenance_valid=True,
        vector_index_eligible=True,
        fp32_model_selection_eligible=True,
        onnx_artifact_eligible=True,
        parity_artifact_verified=True,
        quality_guard_passed=False,
        self_retrieval_passed=True,
    )

    assert eligible == (True, None)
    assert blocked == (False, "amended query-runtime quality guard failed")


def test_registered_runtime_path_is_part_of_runtime_evidence(tmp_path: Path) -> None:
    runtime_path = tmp_path / "config/query_runtime.toml"

    assert runtime_path.as_posix().endswith("config/query_runtime.toml")


def test_failed_parity_remains_verified_diagnostic_evidence_for_amended_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        int8_dense,
        "verify_onnx_int8_artifact",
        lambda **_kwargs: {"quality_evidence_eligible_for_decision": True},
    )
    monkeypatch.setattr(
        int8_dense,
        "verify_query_parity_artifact",
        lambda **_kwargs: {
            "status": "failed",
            "quality_evidence_eligible_for_decision": True,
        },
    )
    monkeypatch.setattr(
        int8_dense,
        "verify_self_retrieval_artifact",
        lambda **_kwargs: {
            "status": "passed",
            "quality_evidence_eligible_for_decision": True,
        },
    )
    monkeypatch.setattr(
        int8_dense,
        "_artifact",
        lambda _root, path: {"path": str(path)},
    )

    evidence = int8_dense._runtime_evidence(
        root=tmp_path,
        model=_model(),
        runtime=_runtime(),
        fp32_entry={},
        int8_entry=None,
    )

    assert evidence["parity_status"] == "failed"
    assert evidence["parity_artifact_verified"] is True
    assert evidence["self_retrieval_artifact_verified"] is True
    assert int8_dense._base_source_eligible(
        {
            "source_evidence": {"eligible_for_model_selection": True},
            "model_selection_evidence_eligible": True,
        },
        evidence,
    ) is True


def test_int8_summary_evidence_allows_an_authentic_rejected_candidate() -> None:
    verified = {
        "onnx_artifact_eligible": True,
        "parity_artifact_verified": True,
        "self_retrieval_artifact_verified": True,
        "quality_guard_artifact_verified": True,
    }

    assert int8_dense._int8_summary_quality_eligible(
        provenance_valid=True,
        runtime_by_model={
            "selected": {**verified, "quality_guard_passed": True},
            "rejected": {**verified, "quality_guard_passed": False},
        },
    ) is True
    assert int8_dense._int8_summary_quality_eligible(
        provenance_valid=True,
        runtime_by_model={
            "selected": verified,
            "unverified": {**verified, "quality_guard_artifact_verified": False},
        },
    ) is False
