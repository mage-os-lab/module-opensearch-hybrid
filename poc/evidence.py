from __future__ import annotations

from collections.abc import Mapping


class DecisionEvidenceError(ValueError):
    """Raised when an exploratory artifact is used in the go/no-go decision table."""


def assert_decision_eligible(manifest: Mapping[str, object]) -> None:
    if manifest.get("eligible_for_decision") is not True:
        runtime = manifest.get("encoder_runtime", "unknown")
        raise DecisionEvidenceError(
            f"run is not eligible for decision evidence (encoder_runtime={runtime!r})"
        )
