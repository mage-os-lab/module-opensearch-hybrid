from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from poc.experiments import load_bm25_experiments
from poc.manifest import canonical_sha256


@pytest.fixture
def bm25_selection() -> dict[str, Any]:
    """Synthetic selection identity for unit tests, without saved benchmark results."""
    experiments = load_bm25_experiments(
        Path(__file__).resolve().parents[1] / "config/experiments.toml"
    )
    profile = cast(
        dict[str, Any], json.loads(json.dumps(experiments.tuning_candidates[0].to_dict()))
    )
    return {
        "schema_version": 2,
        "dataset": "WANDS",
        "selected_profile": profile,
        "selected_profile_sha256": canonical_sha256(profile),
    }
