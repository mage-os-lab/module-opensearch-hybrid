from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from poc.datasets import (
    DatasetFileSpec,
    DatasetIntegrityError,
    download_verified_file,
    load_trec_product_search_config,
    load_wands_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_wands_registry_pins_source_files() -> None:
    spec = load_wands_config(ROOT / "config/datasets.toml")
    assert spec.revision == "3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5"
    assert spec.expected_products == 42_994
    assert spec.expected_unique_pairs == 231_873
    assert set(spec.files) == {"product", "query", "label"}
    assert all(len(file.sha256) == 64 for file in spec.files.values())


def test_trec_product_search_registry_pins_official_sources() -> None:
    spec = load_trec_product_search_config(ROOT / "config/datasets.toml")

    assert spec.revision == "2d39a0a369995216d464c32c631633af64770ce0"
    assert spec.expected_products == 1_118_658
    assert spec.expected_queries == 117
    assert spec.expected_judged_queries == 80
    assert spec.expected_judgments == 33_020
    assert set(spec.files) == {"corpus", "queries", "qrels"}
    assert spec.files["corpus"].bytes == 568_098_976


def test_download_is_cached_and_requires_repair_for_invalid_existing_file(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    source.write_bytes(b"pinned bytes\n")
    payload_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    spec = DatasetFileSpec(
        name="fixture",
        filename="fixture.csv",
        url=source.as_uri(),
        sha256=payload_hash,
        bytes=source.stat().st_size,
    )
    destination = tmp_path / "download" / "fixture.csv"

    assert download_verified_file(destination, spec, repair=False) == "downloaded"
    assert download_verified_file(destination, spec, repair=False) == "cached"

    destination.write_text("corrupt")
    with pytest.raises(DatasetIntegrityError, match="refusing to replace"):
        download_verified_file(destination, spec, repair=False)
    assert download_verified_file(destination, spec, repair=True) == "downloaded"
    assert destination.read_bytes() == source.read_bytes()
