from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from poc.datasets import (
    DatasetFileSpec,
    DatasetIntegrityError,
    WandsDatasetSpec,
    file_facts,
)
from poc.trec import read_qrels
from poc.wands import (
    DUPLICATE_POLICY,
    LABEL_COLUMNS,
    PRODUCT_COLUMNS,
    QUERY_COLUMNS,
    aggregate_relevance,
    prepare_wands,
    split_query_ids,
    verify_prepared_wands,
)


def test_relevance_aggregation_uses_majority_then_conservative_tie() -> None:
    assert aggregate_relevance([2]) == 2
    assert aggregate_relevance([2, 1]) == 1
    assert aggregate_relevance([0, 1]) == 0
    assert aggregate_relevance([2, 1, 1]) == 1


def test_query_split_is_exact_balanced_and_stable() -> None:
    queries = {str(index): {"query": f"q{index}", "query_class": "c"} for index in range(8)}
    first = split_query_ids(queries, "seed")
    second = split_query_ids(dict(reversed(list(queries.items()))), "seed")
    assert first == second
    assert list(first.values()).count("dev") == 4
    assert list(first.values()).count("test") == 4


def test_prepare_and_verify_wands_fixture(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    prepared = tmp_path / "prepared"
    raw.mkdir()
    _write_tsv(
        raw / "product.csv",
        PRODUCT_COLUMNS,
        [
            [
                "0",
                "red chair",
                "Chairs",
                "Furniture / Chairs",
                "soft chair",
                "color:red",
                "1",
                "4.5",
                "1",
            ],
            ["1", "oak desk", "Desks", "Furniture / Desks", "", "material:oak", "", "", ""],
            [
                "2",
                "steel bottle",
                "Bottles",
                "Kitchen / Bottles",
                "cold bottle",
                "material:steel",
                "2",
                "4",
                "2",
            ],
        ],
    )
    _write_tsv(
        raw / "query.csv",
        QUERY_COLUMNS,
        [
            ["0", "red chair", "Chairs"],
            ["1", "oak desk", "Desks"],
            ["2", "steel bottle", "Bottles"],
            ["3", "soft seat", ""],
        ],
    )
    _write_tsv(
        raw / "label.csv",
        LABEL_COLUMNS,
        [
            ["0", "0", "0", "Exact"],
            ["1", "0", "0", "Partial"],
            ["2", "1", "1", "Partial"],
            ["3", "2", "2", "Irrelevant"],
            ["4", "3", "0", "Exact"],
        ],
    )
    spec = _fixture_spec(raw)

    first_manifest = prepare_wands(spec, raw, prepared)
    first_facts = file_facts(prepared / "manifest.json")
    summary = verify_prepared_wands(spec, raw, prepared)
    second_manifest = prepare_wands(spec, raw, prepared)

    assert first_manifest == second_manifest
    assert file_facts(prepared / "manifest.json") == first_facts
    assert summary == {
        "products": 3,
        "queries": 4,
        "qrels": 4,
        "dev_queries": 2,
        "test_queries": 2,
    }
    qrels = {
        (record.query_id, record.document_id): record.relevance
        for record in read_qrels(prepared / "qrels.all.trec")
    }
    assert qrels[("0", "0")] == 1
    assert sum(1 for _ in (prepared / "label-conflicts.jsonl").open()) == 1
    assert first_manifest["missing_query_fields"] == {"query_class": 1}


def test_verify_prepared_wands_rejects_rehashed_query_text_drift(
    tmp_path: Path,
) -> None:
    raw, prepared, spec = _prepared_fixture(tmp_path)

    for name in ("queries.all.jsonl", "queries.dev.jsonl", "queries.test.jsonl"):
        path = prepared / name
        records = [json.loads(line) for line in path.read_text().splitlines()]
        changed = False
        for record in records:
            if record["query_id"] == "0":
                record["query"] = "tampered current query"
                changed = True
        if changed:
            path.write_text(
                "".join(
                    json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
                    for record in records
                )
            )
    _refresh_output_facts(
        prepared,
        ("queries.all.jsonl", "queries.dev.jsonl", "queries.test.jsonl"),
    )

    with pytest.raises(DatasetIntegrityError, match="prepared query content"):
        verify_prepared_wands(spec, raw, prepared)


def test_verify_prepared_wands_rejects_rehashed_product_drift(
    tmp_path: Path,
) -> None:
    raw, prepared, spec = _prepared_fixture(tmp_path)
    path = prepared / "products.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["title"] = "tampered product"
    path.write_text(
        "".join(
            json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
            for record in records
        )
    )
    _refresh_output_facts(prepared, ("products.jsonl",))

    with pytest.raises(DatasetIntegrityError, match="products.jsonl differs"):
        verify_prepared_wands(spec, raw, prepared)


def test_verify_prepared_wands_rejects_rehashed_qrels_drift(
    tmp_path: Path,
) -> None:
    raw, prepared, spec = _prepared_fixture(tmp_path)
    all_path = prepared / "qrels.all.trec"
    first = all_path.read_text().splitlines()[0].split()
    query_id = first[0]
    replacement_gain = "2" if first[3] != "2" else "1"
    changed_names = ["qrels.all.trec"]
    for name in ("qrels.all.trec", "qrels.dev.trec", "qrels.test.trec"):
        path = prepared / name
        lines = path.read_text().splitlines()
        changed = False
        for index, line in enumerate(lines):
            parts = line.split()
            if parts[:3] == first[:3]:
                parts[3] = replacement_gain
                lines[index] = " ".join(parts)
                changed = True
        if changed:
            path.write_text("\n".join(lines) + "\n")
            if name not in changed_names:
                changed_names.append(name)
    assert query_id
    _refresh_output_facts(prepared, tuple(changed_names))

    with pytest.raises(DatasetIntegrityError, match="qrels.all.trec differs"):
        verify_prepared_wands(spec, raw, prepared)


def test_verify_prepared_wands_rejects_rehashed_conflict_audit_drift(
    tmp_path: Path,
) -> None:
    raw, prepared, spec = _prepared_fixture(tmp_path)
    path = prepared / "label-conflicts.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["selected_gain"] = 2
    path.write_text(
        "".join(
            json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
            for record in records
        )
    )
    _refresh_output_facts(prepared, ("label-conflicts.jsonl",))

    with pytest.raises(DatasetIntegrityError, match="label-conflicts.jsonl differs"):
        verify_prepared_wands(spec, raw, prepared)


def test_verify_prepared_wands_allows_provenance_metadata_drift(
    tmp_path: Path,
) -> None:
    raw, prepared, spec = _prepared_fixture(tmp_path)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["benchmark_provenance"]["code_revision"] = {
        "git_commit": "f" * 40,
        "source_dirty": False,
        "source_tree_sha256": "e" * 64,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    assert verify_prepared_wands(spec, raw, prepared) == {
        "products": 3,
        "queries": 4,
        "qrels": 4,
        "dev_queries": 2,
        "test_queries": 2,
    }


def test_verify_prepared_wands_rejects_manifest_content_drift(
    tmp_path: Path,
) -> None:
    raw, prepared, spec = _prepared_fixture(tmp_path)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["annotations"]["primary_gain_mapping"] = "tampered"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(DatasetIntegrityError, match="manifest.json differs"):
        verify_prepared_wands(spec, raw, prepared)


def test_verify_prepared_wands_rejects_manifest_numeric_type_drift(
    tmp_path: Path,
) -> None:
    raw, prepared, spec = _prepared_fixture(tmp_path)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["raw_counts"]["products"] = 3.0
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(DatasetIntegrityError, match="manifest.json differs"):
        verify_prepared_wands(spec, raw, prepared)


@pytest.mark.parametrize("invalid_provenance", [None, "not-a-json-object"])
def test_verify_prepared_wands_rejects_invalid_provenance_metadata(
    tmp_path: Path,
    invalid_provenance: object,
) -> None:
    raw, prepared, spec = _prepared_fixture(tmp_path)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if invalid_provenance is None:
        manifest.pop("benchmark_provenance")
    else:
        manifest["benchmark_provenance"] = invalid_provenance
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(DatasetIntegrityError, match="benchmark provenance"):
        verify_prepared_wands(spec, raw, prepared)


def _prepared_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, WandsDatasetSpec]:
    raw = tmp_path / "raw"
    prepared = tmp_path / "prepared"
    raw.mkdir()
    _write_tsv(
        raw / "product.csv",
        PRODUCT_COLUMNS,
        [
            ["0", "red chair", "Chairs", "Furniture", "soft", "red", "1", "4", "1"],
            ["1", "oak desk", "Desks", "Furniture", "oak", "wood", "1", "4", "1"],
            ["2", "steel bottle", "Bottles", "Kitchen", "cold", "steel", "1", "4", "1"],
        ],
    )
    _write_tsv(
        raw / "query.csv",
        QUERY_COLUMNS,
        [
            ["0", "red chair", "Chairs"],
            ["1", "oak desk", "Desks"],
            ["2", "steel bottle", "Bottles"],
            ["3", "soft seat", ""],
        ],
    )
    _write_tsv(
        raw / "label.csv",
        LABEL_COLUMNS,
        [
            ["0", "0", "0", "Exact"],
            ["1", "0", "0", "Partial"],
            ["2", "1", "1", "Partial"],
            ["3", "2", "2", "Irrelevant"],
            ["4", "3", "0", "Exact"],
        ],
    )
    spec = _fixture_spec(raw)
    prepare_wands(spec, raw, prepared)
    return raw, prepared, spec


def _refresh_output_facts(prepared: Path, names: tuple[str, ...]) -> None:
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in names:
        facts = file_facts(prepared / name)
        manifest["outputs"][name]["sha256"] = facts.sha256
        manifest["outputs"][name]["bytes"] = facts.bytes
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _write_tsv(path: Path, columns: tuple[str, ...], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def _fixture_spec(raw: Path) -> WandsDatasetSpec:
    files = {
        name: DatasetFileSpec(
            name=name,
            filename=f"{name}.csv",
            url=f"https://example.test/{name}.csv",
            sha256=file_facts(raw / f"{name}.csv").sha256,
            bytes=file_facts(raw / f"{name}.csv").bytes,
        )
        for name in ("product", "query", "label")
    }
    return WandsDatasetSpec(
        source="https://github.com/wayfair/WANDS",
        revision="a" * 40,
        expected_products=3,
        expected_queries=4,
        expected_judgments=5,
        expected_unique_pairs=4,
        expected_duplicate_pairs=1,
        expected_conflicting_pairs=1,
        license="MIT",
        primary_gain_mapping="Exact=2,Partial=1,Irrelevant=0",
        duplicate_policy=DUPLICATE_POLICY,
        split_seed="fixture-v1",
        files=files,
    )
