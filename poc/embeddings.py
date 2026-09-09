from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from itertools import pairwise

_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class DeterministicHashEncoder:
    """A deterministic smoke-only encoder that must never enter relevance evidence."""

    dims: int = 32
    identifier: str = "opensearch-hybrid-smoke-hash"
    revision: str = "v1"
    eligible_for_decision: bool = False

    def __post_init__(self) -> None:
        if self.dims < 8:
            raise ValueError("smoke encoder requires at least 8 dimensions")

    @property
    def fingerprint(self) -> str:
        material = f"{self.identifier}:{self.revision}:dims={self.dims}:tokenizer=ascii-v1"
        return hashlib.sha256(material.encode()).hexdigest()

    def encode(self, text: str) -> list[float]:
        tokens = _TOKEN.findall(text.lower())
        if not tokens:
            raise ValueError("cannot encode text without alphanumeric tokens")

        bigrams = (f"{left}_{right}" for left, right in pairwise(tokens))
        features = [*tokens, *bigrams]
        vector = [0.0] * self.dims
        for feature in features:
            digest = hashlib.sha256(feature.encode()).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dims
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            raise ValueError("smoke encoder produced a zero vector")
        return [value / norm for value in vector]

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        return [self.encode(text) for text in texts]
