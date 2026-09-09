from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from poc.config import ConfigError


@dataclass(frozen=True, slots=True)
class ThreeWayCandidate:
    name: str
    weights: tuple[float, float, float]

    @property
    def lexical_weight(self) -> float:
        return self.weights[0]

    @property
    def dense_weight(self) -> float:
        return self.weights[1]

    @property
    def sparse_weight(self) -> float:
        return self.weights[2]


@dataclass(frozen=True, slots=True)
class ThreeWaySpec:
    index_name: str
    dense_model: str
    candidates: tuple[ThreeWayCandidate, ...]


def load_three_way_spec(path: Path) -> ThreeWaySpec:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    fusion = raw.get("fusion")
    candidates = fusion.get("candidates") if isinstance(fusion, dict) else None
    if raw.get("schema_version") != 1 or not isinstance(candidates, list):
        raise ConfigError("three-way fusion configuration is missing")
    parsed: list[ThreeWayCandidate] = []
    for raw_candidate in candidates:
        if not isinstance(raw_candidate, dict):
            raise ConfigError("three-way fusion candidate must be a table")
        name = raw_candidate.get("name")
        weights = raw_candidate.get("weights")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(weights, list)
            or len(weights) != 3
            or not all(
                isinstance(weight, int | float) and not isinstance(weight, bool)
                for weight in weights
            )
        ):
            raise ConfigError("three-way fusion candidate is invalid")
        normalized = (
            float(weights[0]),
            float(weights[1]),
            float(weights[2]),
        )
        if any(weight < 0.0 for weight in normalized) or abs(sum(normalized) - 1.0) > 1e-12:
            raise ConfigError("three-way fusion weights must be non-negative and sum to one")
        parsed.append(ThreeWayCandidate(name, normalized))
    index_name = raw.get("index_name")
    dense_model = raw.get("dense_model")
    if not isinstance(index_name, str) or not index_name:
        raise ConfigError("three-way index name must not be blank")
    if not isinstance(dense_model, str) or not dense_model:
        raise ConfigError("three-way dense model must not be blank")
    if not parsed or len({candidate.name for candidate in parsed}) != len(parsed):
        raise ConfigError("three-way candidates must have unique names")
    if len({candidate.weights for candidate in parsed}) != len(parsed):
        raise ConfigError("three-way candidate weights must be unique")
    return ThreeWaySpec(index_name, dense_model, tuple(parsed))
