from __future__ import annotations

import pytest

from poc.evidence import DecisionEvidenceError, assert_decision_eligible


def test_exploratory_run_cannot_enter_decision_table() -> None:
    with pytest.raises(DecisionEvidenceError, match="not eligible"):
        assert_decision_eligible(
            {
                "eligible_for_decision": False,
                "encoder_runtime": "deterministic_hash_smoke_only",
            }
        )


def test_production_parity_run_can_enter_decision_table() -> None:
    assert_decision_eligible({"eligible_for_decision": True, "encoder_runtime": "onnx_int8_cpu"})
