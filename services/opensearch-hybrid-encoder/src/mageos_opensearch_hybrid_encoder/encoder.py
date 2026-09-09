"""Deterministic vectors for integration tests, never production relevance."""

import hashlib
import math

MODEL_ID = "mageos/deterministic-fake-v1"
MODEL_REVISION = "adc77e05a39323ce3777428e5b75c135503fe4d344670fa1ff1bb9f6acffd8a8"
DIMENSION = 256
RUNTIME = "deterministic_hash_fake_only"
RECIPE_VERSION = "mageos-v1"


def encode(text: str) -> list[float]:
    """Match the module's deterministic PHP integration-test encoder."""
    values: list[float] = []
    sum_squares = 0.0
    for index in range(DIMENSION):
        digest = hashlib.sha256(f"{text}:{index}".encode()).digest()
        unsigned = int.from_bytes(digest[:2], byteorder="big", signed=False)
        value = (float(unsigned) / 65535.0 * 2.0) - 1.0
        values.append(value)
        sum_squares += value * value
    if sum_squares == 0.0:
        raise RuntimeError("the deterministic vector has zero magnitude")
    magnitude = math.sqrt(sum_squares)
    return [value / magnitude for value in values]
