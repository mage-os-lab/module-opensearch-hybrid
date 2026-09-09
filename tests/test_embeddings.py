from __future__ import annotations

import math

from poc.embeddings import DeterministicHashEncoder


def test_smoke_encoder_is_deterministic_normalized_and_ineligible() -> None:
    encoder = DeterministicHashEncoder(dims=32)
    first = encoder.encode("red waterproof trail running shoe")
    second = encoder.encode("red waterproof trail running shoe")
    assert first == second
    assert math.isclose(math.sqrt(sum(value * value for value in first)), 1.0)
    assert not encoder.eligible_for_decision
    assert len(encoder.fingerprint) == 64
