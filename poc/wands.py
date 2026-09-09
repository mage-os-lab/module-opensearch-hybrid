from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from poc.datasets import (
    DatasetIntegrityError,
    WandsDatasetSpec,
    file_facts,
    verify_wands_raw,
)
from poc.manifest import canonical_sha256, read_json, write_json
from poc.trec import QrelRecord, identifier_sort_key, read_qrels, write_qrels

PRODUCT_COLUMNS = (
    "product_id",
    "product_name",
    "product_class",
    "category hierarchy",
    "product_description",
    "product_features",
    "rating_count",
    "average_rating",
    "review_count",
)
QUERY_COLUMNS = ("query_id", "query", "query_class")
LABEL_COLUMNS = ("id", "query_id", "product_id", "label")
LABEL_TO_GAIN = {"Exact": 2, "Partial": 1, "Irrelevant": 0}
GAIN_TO_LABEL = {gain: label for label, gain in LABEL_TO_GAIN.items()}
SPLIT_ALGORITHM = "sha256_rank_half_v1"
DUPLICATE_POLICY = "majority_vote_then_lower_gain_on_tie"
_QUERY_TOKEN = re.compile(r"[\w]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class QueryRecord:
    query_id: str
    query: str
    query_class: str
    split: str
    token_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OutputFacts:
    sha256: str
    bytes: int
    records: int


@dataclass(frozen=True, slots=True)
class LabelAudit:
    raw_judgments: int
    unique_pairs: int
    duplicate_pairs: int
    duplicate_rows: int
    conflicting_pairs: int
    distribution: dict[str, int]
    pair_gains: dict[tuple[str, str], list[int]]


def prepare_wands(
    spec: WandsDatasetSpec,
    raw_directory: Path,
    prepared_directory: Path,
) -> dict[str, object]:
    raw_facts = verify_wands_raw(spec, raw_directory)
    if spec.primary_gain_mapping != "Exact=2,Partial=1,Irrelevant=0":
        raise DatasetIntegrityError("unsupported WANDS primary gain mapping")
    if spec.duplicate_policy != DUPLICATE_POLICY:
        raise DatasetIntegrityError("unsupported WANDS duplicate-label policy")

    queries = _read_queries(raw_directory / spec.files["query"].filename)
    _assert_count("queries", len(queries), spec.expected_queries)
    missing_query_fields = {
        "query_class": sum(not values["query_class"] for values in queries.values())
    }
    split_by_query = split_query_ids(queries, spec.split_seed)
    query_records = [
        QueryRecord(
            query_id=query_id,
            query=values["query"],
            query_class=values["query_class"],
            split=split_by_query[query_id],
            token_count=len(_QUERY_TOKEN.findall(values["query"])),
        )
        for query_id, values in queries.items()
    ]
    query_records.sort(key=lambda record: identifier_sort_key(record.query_id))

    product_path = raw_directory / spec.files["product"].filename
    product_ids, missing_fields = _audit_products(product_path)
    _assert_count("products", len(product_ids), spec.expected_products)

    label_audit = _audit_labels(
        raw_directory / spec.files["label"].filename,
        product_ids=product_ids,
        query_ids=set(queries),
    )
    _assert_count("judgments", label_audit.raw_judgments, spec.expected_judgments)
    _assert_count(
        "unique query-product pairs", label_audit.unique_pairs, spec.expected_unique_pairs
    )
    _assert_count(
        "duplicate query-product pairs", label_audit.duplicate_pairs, spec.expected_duplicate_pairs
    )
    _assert_count(
        "conflicting query-product pairs",
        label_audit.conflicting_pairs,
        spec.expected_conflicting_pairs,
    )

    prepared_directory.mkdir(parents=True, exist_ok=True)
    output_counts: dict[str, int] = {}
    output_counts["products.jsonl"] = _write_jsonl(
        prepared_directory / "products.jsonl",
        _prepared_products(product_path),
    )
    output_counts["queries.all.jsonl"] = _write_jsonl(
        prepared_directory / "queries.all.jsonl",
        (record.to_dict() for record in query_records),
    )
    for split in ("dev", "test"):
        output_counts[f"queries.{split}.jsonl"] = _write_jsonl(
            prepared_directory / f"queries.{split}.jsonl",
            (record.to_dict() for record in query_records if record.split == split),
        )

    qrels = [
        QrelRecord(
            query_id=query_id,
            document_id=product_id,
            relevance=aggregate_relevance(gains),
        )
        for (query_id, product_id), gains in label_audit.pair_gains.items()
    ]
    write_qrels(prepared_directory / "qrels.all.trec", qrels)
    output_counts["qrels.all.trec"] = len(qrels)
    for split in ("dev", "test"):
        split_qrels = [record for record in qrels if split_by_query[record.query_id] == split]
        write_qrels(prepared_directory / f"qrels.{split}.trec", split_qrels)
        output_counts[f"qrels.{split}.trec"] = len(split_qrels)

    conflict_rows = _conflict_rows(label_audit.pair_gains)
    output_counts["label-conflicts.jsonl"] = _write_jsonl(
        prepared_directory / "label-conflicts.jsonl",
        conflict_rows,
    )

    output_facts = {
        name: asdict(_output_facts(prepared_directory / name, records))
        for name, records in sorted(output_counts.items())
    }
    dev_ids = [record.query_id for record in query_records if record.split == "dev"]
    test_ids = [record.query_id for record in query_records if record.split == "test"]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset": "WANDS",
        "source": spec.source,
        "source_revision": spec.revision,
        "license": spec.license,
        "raw_files": {name: asdict(facts) for name, facts in sorted(raw_facts.items())},
        "raw_counts": {
            "products": len(product_ids),
            "queries": len(queries),
            "judgments": label_audit.raw_judgments,
        },
        "annotations": {
            "unique_pairs": label_audit.unique_pairs,
            "duplicate_pairs": label_audit.duplicate_pairs,
            "duplicate_rows": label_audit.duplicate_rows,
            "conflicting_pairs": label_audit.conflicting_pairs,
            "raw_label_distribution": label_audit.distribution,
            "primary_gain_mapping": spec.primary_gain_mapping,
            "duplicate_policy": spec.duplicate_policy,
        },
        "split": {
            "algorithm": SPLIT_ALGORITHM,
            "seed": spec.split_seed,
            "dev_queries": len(dev_ids),
            "test_queries": len(test_ids),
            "dev_query_ids_sha256": canonical_sha256(dev_ids),
            "test_query_ids_sha256": canonical_sha256(test_ids),
        },
        "missing_product_fields": missing_fields,
        "missing_query_fields": missing_query_fields,
        "field_notes": {
            "brand": "WANDS has no general brand field; prepared records use an empty string",
            "category": "copied from the source column named 'category hierarchy'",
        },
        "outputs": output_facts,
    }
    write_json(prepared_directory / "manifest.json", manifest)
    return manifest


def verify_prepared_wands(
    spec: WandsDatasetSpec,
    raw_directory: Path,
    prepared_directory: Path,
) -> dict[str, object]:
    raw_facts = verify_wands_raw(spec, raw_directory)
    manifest = cast(dict[str, Any], read_json(prepared_directory / "manifest.json"))
    if manifest.get("schema_version") != 1 or manifest.get("dataset") != "WANDS":
        raise DatasetIntegrityError("prepared WANDS manifest has the wrong schema or dataset")
    if manifest.get("source_revision") != spec.revision:
        raise DatasetIntegrityError("prepared WANDS source revision differs from configuration")
    expected_raw = {name: asdict(facts) for name, facts in sorted(raw_facts.items())}
    if manifest.get("raw_files") != expected_raw:
        raise DatasetIntegrityError(
            "prepared WANDS raw-file facts differ from current source files"
        )

    raw_outputs = manifest.get("outputs")
    if not isinstance(raw_outputs, dict):
        raise DatasetIntegrityError("prepared WANDS manifest has no output inventory")
    for name, raw_expected in raw_outputs.items():
        if not isinstance(name, str) or not isinstance(raw_expected, dict):
            raise DatasetIntegrityError("invalid prepared WANDS output inventory")
        path = prepared_directory / name
        actual = _output_facts(path, _line_count(path))
        expected = OutputFacts(
            sha256=str(raw_expected.get("sha256")),
            bytes=int(raw_expected.get("bytes", -1)),
            records=int(raw_expected.get("records", -1)),
        )
        if actual != expected:
            raise DatasetIntegrityError(f"prepared output {name} differs from its manifest")

    raw_queries = _read_queries(raw_directory / spec.files["query"].filename)
    _assert_count("queries", len(raw_queries), spec.expected_queries)
    expected_splits = split_query_ids(raw_queries, spec.split_seed)
    expected_query_records = [
        QueryRecord(
            query_id=query_id,
            query=values["query"],
            query_class=values["query_class"],
            split=expected_splits[query_id],
            token_count=len(_QUERY_TOKEN.findall(values["query"])),
        )
        for query_id, values in raw_queries.items()
    ]
    expected_query_records.sort(key=lambda record: identifier_sort_key(record.query_id))
    query_splits = _verify_query_outputs(
        prepared_directory,
        expected_records=expected_query_records,
    )
    qrels = read_qrels(prepared_directory / "qrels.all.trec")
    if len(qrels) != spec.expected_unique_pairs:
        raise DatasetIntegrityError("prepared all-qrels count differs from configuration")
    if {record.query_id for record in qrels} != set(query_splits):
        raise DatasetIntegrityError("prepared all-qrels do not cover exactly the prepared queries")
    if any(record.relevance not in LABEL_TO_GAIN.values() for record in qrels):
        raise DatasetIntegrityError("prepared all-qrels contain an unsupported relevance gain")
    all_pairs = {(record.query_id, record.document_id) for record in qrels}
    split_pairs: dict[str, set[tuple[str, str]]] = {}
    for split in ("dev", "test"):
        split_qrels = read_qrels(prepared_directory / f"qrels.{split}.trec")
        if any(query_splits.get(record.query_id) != split for record in split_qrels):
            raise DatasetIntegrityError(
                f"prepared {split} qrels contain a query from another split"
            )
        split_pairs[split] = {(record.query_id, record.document_id) for record in split_qrels}
    if split_pairs["dev"] & split_pairs["test"]:
        raise DatasetIntegrityError("prepared dev and test qrels overlap")
    if split_pairs["dev"] | split_pairs["test"] != all_pairs:
        raise DatasetIntegrityError("prepared dev and test qrels do not partition all qrels")

    product_count = _verify_product_jsonl(prepared_directory / "products.jsonl")
    if product_count != spec.expected_products:
        raise DatasetIntegrityError("prepared product count differs from configuration")
    _verify_deterministic_rebuild(
        spec,
        raw_directory=raw_directory,
        prepared_directory=prepared_directory,
    )
    return {
        "products": product_count,
        "queries": len(query_splits),
        "qrels": len(qrels),
        "dev_queries": sum(split == "dev" for split in query_splits.values()),
        "test_queries": sum(split == "test" for split in query_splits.values()),
    }


def _verify_deterministic_rebuild(
    spec: WandsDatasetSpec,
    *,
    raw_directory: Path,
    prepared_directory: Path,
) -> None:
    bound_outputs = (
        "products.jsonl",
        "queries.all.jsonl",
        "queries.dev.jsonl",
        "queries.test.jsonl",
        "qrels.all.trec",
        "qrels.dev.trec",
        "qrels.test.trec",
        "label-conflicts.jsonl",
    )
    with tempfile.TemporaryDirectory(prefix="opensearch-hybrid-wands-rebuild-") as temporary:
        rebuilt = Path(temporary) / "wands"
        prepare_wands(spec, raw_directory, rebuilt)
        for name in bound_outputs:
            if file_facts(prepared_directory / name) != file_facts(rebuilt / name):
                raise DatasetIntegrityError(
                    f"prepared {name} differs from deterministic raw-source reconstruction"
                )
        if canonical_sha256(
            _deterministic_manifest_content(prepared_directory / "manifest.json")
        ) != canonical_sha256(
            _deterministic_manifest_content(rebuilt / "manifest.json")
        ):
            raise DatasetIntegrityError(
                "prepared manifest.json differs from deterministic raw-source reconstruction"
            )


def _deterministic_manifest_content(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise DatasetIntegrityError("prepared WANDS manifest must be a JSON object")
    content = dict(value)
    provenance = content.pop("benchmark_provenance", None)
    if not isinstance(provenance, dict):
        raise DatasetIntegrityError(
            "prepared WANDS manifest benchmark provenance must be a JSON object"
        )
    return content


def split_query_ids(queries: dict[str, dict[str, str]], seed: str) -> dict[str, str]:
    if len(queries) < 2:
        raise DatasetIntegrityError("WANDS split requires at least two queries")
    ranked = sorted(
        queries,
        key=lambda query_id: (
            hashlib.sha256(f"{seed}\0{query_id}".encode()).hexdigest(),
            identifier_sort_key(query_id),
        ),
    )
    dev_count = len(ranked) // 2
    dev_ids = set(ranked[:dev_count])
    return {query_id: "dev" if query_id in dev_ids else "test" for query_id in queries}


def aggregate_relevance(gains: list[int]) -> int:
    if not gains:
        raise DatasetIntegrityError("cannot aggregate an empty relevance list")
    counts = Counter(gains)
    highest_vote_count = max(counts.values())
    return min(gain for gain, count in counts.items() if count == highest_vote_count)


def _read_queries(path: Path) -> dict[str, dict[str, str]]:
    queries: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(_read_tsv(path, QUERY_COLUMNS), start=2):
        query_id = _required_value(row, "query_id", path, row_number)
        if query_id in queries:
            raise DatasetIntegrityError(f"duplicate query_id {query_id!r} in {path}")
        queries[query_id] = {
            "query": _required_value(row, "query", path, row_number),
            "query_class": row["query_class"].strip(),
        }
    return queries


def _audit_products(path: Path) -> tuple[set[str], dict[str, int]]:
    product_ids: set[str] = set()
    missing = Counter[str]()
    optional_fields = PRODUCT_COLUMNS[2:]
    for row_number, row in enumerate(_read_tsv(path, PRODUCT_COLUMNS), start=2):
        product_id = _required_value(row, "product_id", path, row_number)
        _required_value(row, "product_name", path, row_number)
        if product_id in product_ids:
            raise DatasetIntegrityError(f"duplicate product_id {product_id!r} in {path}")
        product_ids.add(product_id)
        for field in optional_fields:
            if not row[field].strip():
                missing[field] += 1
        _optional_float(row["rating_count"], path, row_number, "rating_count")
        _optional_float(row["average_rating"], path, row_number, "average_rating")
        _optional_float(row["review_count"], path, row_number, "review_count")
    return product_ids, dict(sorted(missing.items()))


def _prepared_products(path: Path) -> Iterator[dict[str, object]]:
    for row_number, row in enumerate(_read_tsv(path, PRODUCT_COLUMNS), start=2):
        yield {
            "average_rating": _optional_float(
                row["average_rating"], path, row_number, "average_rating"
            ),
            "brand": "",
            "category": row["category hierarchy"].strip(),
            "description": row["product_description"].strip(),
            "features": row["product_features"].strip(),
            "product_class": row["product_class"].strip(),
            "product_id": row["product_id"].strip(),
            "rating_count": _optional_float(row["rating_count"], path, row_number, "rating_count"),
            "review_count": _optional_float(row["review_count"], path, row_number, "review_count"),
            "title": row["product_name"].strip(),
        }


def _audit_labels(path: Path, *, product_ids: set[str], query_ids: set[str]) -> LabelAudit:
    annotation_ids: set[str] = set()
    pair_gains: dict[tuple[str, str], list[int]] = defaultdict(list)
    distribution = Counter[str]()
    for row_number, row in enumerate(_read_tsv(path, LABEL_COLUMNS), start=2):
        annotation_id = _required_value(row, "id", path, row_number)
        query_id = _required_value(row, "query_id", path, row_number)
        product_id = _required_value(row, "product_id", path, row_number)
        label = _required_value(row, "label", path, row_number)
        if annotation_id in annotation_ids:
            raise DatasetIntegrityError(f"duplicate annotation id {annotation_id!r} in {path}")
        if query_id not in query_ids:
            raise DatasetIntegrityError(f"orphan query_id {query_id!r} in {path}")
        if product_id not in product_ids:
            raise DatasetIntegrityError(f"orphan product_id {product_id!r} in {path}")
        if label not in LABEL_TO_GAIN:
            raise DatasetIntegrityError(f"unknown WANDS label {label!r} in {path}:{row_number}")
        annotation_ids.add(annotation_id)
        pair_gains[(query_id, product_id)].append(LABEL_TO_GAIN[label])
        distribution[label] += 1

    duplicate_pairs = sum(len(gains) > 1 for gains in pair_gains.values())
    conflicting_pairs = sum(len(set(gains)) > 1 for gains in pair_gains.values())
    return LabelAudit(
        raw_judgments=len(annotation_ids),
        unique_pairs=len(pair_gains),
        duplicate_pairs=duplicate_pairs,
        duplicate_rows=len(annotation_ids) - len(pair_gains),
        conflicting_pairs=conflicting_pairs,
        distribution=dict(sorted(distribution.items())),
        pair_gains=dict(pair_gains),
    )


def _conflict_rows(pair_gains: dict[tuple[str, str], list[int]]) -> Iterator[dict[str, object]]:
    for query_id, product_id in sorted(
        (pair for pair, gains in pair_gains.items() if len(set(gains)) > 1),
        key=lambda pair: (identifier_sort_key(pair[0]), identifier_sort_key(pair[1])),
    ):
        gains = pair_gains[(query_id, product_id)]
        yield {
            "chosen_gain": aggregate_relevance(gains),
            "labels": [GAIN_TO_LABEL[gain] for gain in gains],
            "product_id": product_id,
            "query_id": query_id,
        }


def _read_tsv(path: Path, expected_columns: tuple[str, ...]) -> Iterator[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != expected_columns:
            raise DatasetIntegrityError(
                f"unexpected columns in {path}: {reader.fieldnames!r}; "
                f"expected {expected_columns!r}"
            )
        for row_number, raw_row in enumerate(reader, start=2):
            if None in raw_row or any(value is None for value in raw_row.values()):
                raise DatasetIntegrityError(f"malformed tabular row in {path}:{row_number}")
            yield cast(dict[str, str], raw_row)


def _required_value(row: dict[str, str], field: str, path: Path, row_number: int) -> str:
    value = row[field].strip()
    if not value:
        raise DatasetIntegrityError(f"empty {field} in {path}:{row_number}")
    return value


def _optional_float(value: str, path: Path, row_number: int, field: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        result = float(stripped)
    except ValueError as exc:
        raise DatasetIntegrityError(f"invalid {field} in {path}:{row_number}") from exc
    if not math.isfinite(result):
        raise DatasetIntegrityError(f"non-finite {field} in {path}:{row_number}")
    return result


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> int:
    temporary = path.with_name(f".{path.name}.tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                    + "\n"
                )
                count += 1
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return count


def _output_facts(path: Path, records: int) -> OutputFacts:
    facts = file_facts(path)
    return OutputFacts(sha256=facts.sha256, bytes=facts.bytes, records=records)


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1024 * 1024), b""))


def _verify_query_outputs(
    directory: Path,
    *,
    expected_records: list[QueryRecord],
) -> dict[str, str]:
    all_records = [
        cast(dict[str, Any], json.loads(line))
        for line in (directory / "queries.all.jsonl").read_text().splitlines()
    ]
    expected_all = [record.to_dict() for record in expected_records]
    if all_records != expected_all:
        raise DatasetIntegrityError(
            "prepared query content differs from the pinned raw WANDS source"
        )
    query_splits: dict[str, str] = {}
    for line_number, record in enumerate(all_records, start=1):
        query_id = str(record.get("query_id", ""))
        split = str(record.get("split", ""))
        if not query_id or split not in {"dev", "test"} or query_id in query_splits:
            raise DatasetIntegrityError(f"invalid query record at line {line_number}")
        query_splits[query_id] = split
    if len(query_splits) != len(expected_records):
        raise DatasetIntegrityError("prepared query count differs from configuration")
    expected_dev_count = len(expected_records) // 2
    if sum(split == "dev" for split in query_splits.values()) != expected_dev_count:
        raise DatasetIntegrityError("prepared dev query split is not the registered half")
    for split in ("dev", "test"):
        split_records = [
            cast(dict[str, Any], json.loads(line))
            for line in (directory / f"queries.{split}.jsonl").read_text().splitlines()
        ]
        expected_split_records = [
            record.to_dict() for record in expected_records if record.split == split
        ]
        if split_records != expected_split_records:
            raise DatasetIntegrityError(
                f"prepared query content differs for the {split} split"
            )
        ids = {str(record.get("query_id", "")) for record in split_records}
        expected_ids = {query_id for query_id, value in query_splits.items() if value == split}
        if ids != expected_ids or len(ids) != len(split_records):
            raise DatasetIntegrityError(f"prepared {split} query file differs from all queries")
    return query_splits


def _verify_product_jsonl(path: Path) -> int:
    product_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = cast(dict[str, Any], json.loads(line))
            product_id = str(record.get("product_id", ""))
            if not product_id or product_id in product_ids:
                raise DatasetIntegrityError(f"invalid product at {path}:{line_number}")
            product_ids.add(product_id)
    return len(product_ids)


def _assert_count(name: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise DatasetIntegrityError(f"WANDS {name} count is {actual}; expected {expected}")
