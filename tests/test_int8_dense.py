from __future__ import annotations

from poc.int8_dense import classify_int8_evidence


def test_int8_evidence_uses_semantically_verified_runtime_inputs() -> None:
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
