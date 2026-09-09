from __future__ import annotations

import csv
import gzip
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from poc.datasets import (
    DatasetIntegrityError,
    TrecProductSearchDatasetSpec,
    file_facts,
    verify_trec_product_search_raw,
)
from poc.manifest import read_json, write_json
from poc.trec import QrelRecord, identifier_sort_key, write_qrels


@dataclass(frozen=True, slots=True)
class TrecAlignment:
    products: int
    queries: int
    judged_queries: int
    judgments: int
    unique_judged_products: int
    orphan_judgments: int
    orphan_products: int
    orphan_rate: float


def prepare_trec_product_search(
    spec: TrecProductSearchDatasetSpec,
    raw_directory: Path,
    prepared_directory: Path,
    result_path: Path,
    *,
    maximum_orphan_rate: float,
) -> dict[str, object]:
    raw_facts = verify_trec_product_search_raw(spec, raw_directory)
    corpus_path = raw_directory / spec.files["corpus"].filename
    query_path = raw_directory / spec.files["queries"].filename
    qrels_path = raw_directory / spec.files["qrels"].filename
    alignment = inspect_trec_product_search_alignment(
        corpus_path,
        query_path,
        qrels_path,
        maximum_orphan_rate=maximum_orphan_rate,
    )
    expected = {
        "products": spec.expected_products,
        "queries": spec.expected_queries,
        "judged_queries": spec.expected_judged_queries,
        "judgments": spec.expected_judgments,
        "unique_judged_products": spec.expected_unique_judged_products,
    }
    actual = {
        "products": alignment.products,
        "queries": alignment.queries,
        "judged_queries": alignment.judged_queries,
        "judgments": alignment.judgments,
        "unique_judged_products": alignment.unique_judged_products,
    }
    if actual != expected:
        raise DatasetIntegrityError(
            f"TREC Product Search prepared counts differ: expected={expected}, actual={actual}"
        )
    queries = load_trec_queries(query_path)
    qrels = load_trec_qrels(qrels_path, set(queries))
    prepared_directory.mkdir(parents=True, exist_ok=True)
    prepared_queries = prepared_directory / "queries.test.jsonl"
    temporary_queries = prepared_queries.with_name(f".{prepared_queries.name}.tmp")
    with temporary_queries.open("w", encoding="utf-8", newline="\n") as handle:
        for query_id in sorted(queries, key=identifier_sort_key):
            handle.write(
                json.dumps(
                    {"query_id": query_id, "query": queries[query_id], "split": "test"},
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
    temporary_queries.replace(prepared_queries)
    prepared_qrels = prepared_directory / "qrels.test.trec"
    negative_mapped_to_zero = sum(1 for _, _, relevance in qrels if relevance < 0)
    write_qrels(
        prepared_qrels,
        [
            QrelRecord(query_id, document_id, max(0, relevance))
            for query_id, document_id, relevance in qrels
        ],
    )
    project_root = result_path.parents[2]
    result: dict[str, object] = {
        "schema_version": 1,
        "dataset": "TREC Product Search 2024",
        "source_revision": spec.revision,
        "license": spec.license,
        "query_authority": spec.query_authority,
        "raw_files": {
            name: {
                "path": str((raw_directory / file_spec.filename).relative_to(project_root)),
                "sha256": raw_facts[name].sha256,
                "bytes": raw_facts[name].bytes,
            }
            for name, file_spec in spec.files.items()
        },
        "counts": actual,
        "orphan_judgments": alignment.orphan_judgments,
        "orphan_products": alignment.orphan_products,
        "orphan_rate": alignment.orphan_rate,
        "maximum_orphan_rate": maximum_orphan_rate,
        "negative_relevance_mapped_to_zero": negative_mapped_to_zero,
        "prepared_queries": {
            "path": str(prepared_queries.relative_to(project_root)),
            "sha256": file_facts(prepared_queries).sha256,
            "bytes": file_facts(prepared_queries).bytes,
        },
        "prepared_qrels": {
            "path": str(prepared_qrels.relative_to(project_root)),
            "sha256": file_facts(prepared_qrels).sha256,
            "bytes": file_facts(prepared_qrels).bytes,
        },
        "status": "passed",
    }
    write_json(result_path, result)
    return result


def verify_prepared_trec_product_search(
    spec: TrecProductSearchDatasetSpec,
    raw_directory: Path,
    prepared_directory: Path,
    result_path: Path,
    *,
    maximum_orphan_rate: float,
) -> dict[str, Any]:
    recorded = cast(dict[str, Any], read_json(result_path))
    raw_facts = verify_trec_product_search_raw(spec, raw_directory)
    alignment = inspect_trec_product_search_alignment(
        raw_directory / spec.files["corpus"].filename,
        raw_directory / spec.files["queries"].filename,
        raw_directory / spec.files["qrels"].filename,
        maximum_orphan_rate=maximum_orphan_rate,
    )
    expected_counts = {
        "products": spec.expected_products,
        "queries": spec.expected_queries,
        "judged_queries": spec.expected_judged_queries,
        "judgments": spec.expected_judgments,
        "unique_judged_products": spec.expected_unique_judged_products,
    }
    if recorded.get("counts") != expected_counts:
        raise DatasetIntegrityError("recorded TREC prepared counts differ")
    for key in ("orphan_judgments", "orphan_products", "orphan_rate"):
        if recorded.get(key) != getattr(alignment, key):
            raise DatasetIntegrityError(f"recorded TREC {key} differs")
    raw_files = cast(dict[str, dict[str, Any]], recorded["raw_files"])
    for name, facts in raw_facts.items():
        if raw_files[name].get("sha256") != facts.sha256:
            raise DatasetIntegrityError(f"recorded TREC raw hash differs: {name}")
        if raw_files[name].get("bytes") != facts.bytes:
            raise DatasetIntegrityError(f"recorded TREC raw size differs: {name}")
    project_root = result_path.parents[2]
    expected_prepared_paths = {
        "prepared_queries": prepared_directory / "queries.test.jsonl",
        "prepared_qrels": prepared_directory / "qrels.test.trec",
    }
    for key, path in expected_prepared_paths.items():
        artifact = cast(dict[str, Any], recorded[key])
        facts = file_facts(path)
        if (
            artifact.get("path") != str(path.relative_to(project_root))
            or facts.sha256 != artifact.get("sha256")
            or facts.bytes != artifact.get("bytes")
        ):
            raise DatasetIntegrityError(f"recorded TREC {key} artifact differs")
    queries, qrels, negative_mapped_to_zero = verify_trec_prepared_content(
        raw_query_path=raw_directory / spec.files["queries"].filename,
        raw_qrels_path=raw_directory / spec.files["qrels"].filename,
        prepared_query_path=expected_prepared_paths["prepared_queries"],
        prepared_qrels_path=expected_prepared_paths["prepared_qrels"],
    )
    if len(queries) != spec.expected_queries or len(qrels) != spec.expected_judgments:
        raise DatasetIntegrityError("prepared TREC qrel count differs")
    if (
        recorded.get("schema_version") != 1
        or recorded.get("dataset") != "TREC Product Search 2024"
        or recorded.get("source_revision") != spec.revision
        or recorded.get("license") != spec.license
        or recorded.get("query_authority") != spec.query_authority
        or recorded.get("maximum_orphan_rate") != maximum_orphan_rate
        or recorded.get("negative_relevance_mapped_to_zero")
        != negative_mapped_to_zero
        or recorded.get("status") != "passed"
    ):
        raise DatasetIntegrityError("TREC preparation status is not passed")
    return recorded


def verify_trec_prepared_content(
    *,
    raw_query_path: Path,
    raw_qrels_path: Path,
    prepared_query_path: Path,
    prepared_qrels_path: Path,
) -> tuple[dict[str, str], list[tuple[str, str, int]], int]:
    queries = load_trec_queries(raw_query_path)
    expected_queries = "".join(
        json.dumps(
            {"query_id": query_id, "query": queries[query_id], "split": "test"},
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
        for query_id in sorted(queries, key=identifier_sort_key)
    )
    if prepared_query_path.read_text() != expected_queries:
        raise DatasetIntegrityError(
            "prepared query content differs from registered TREC topics"
        )
    qrels = load_trec_qrels(raw_qrels_path, set(queries))
    normalized_qrels = sorted(
        (
            QrelRecord(query_id, document_id, max(0, relevance))
            for query_id, document_id, relevance in qrels
        ),
        key=lambda record: (
            identifier_sort_key(record.query_id),
            identifier_sort_key(record.document_id),
        ),
    )
    expected_qrels = "".join(record.line() for record in normalized_qrels)
    if prepared_qrels_path.read_text() != expected_qrels:
        raise DatasetIntegrityError(
            "prepared qrel content differs from registered TREC judgments"
        )
    negative_mapped_to_zero = sum(
        1 for _, _, relevance in qrels if relevance < 0
    )
    return queries, qrels, negative_mapped_to_zero


def iter_trec_products(path: Path) -> Iterator[tuple[str, dict[str, str]]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if path.name.endswith(".trec.gz"):
                fields = line.rstrip("\n").split("\t", maxsplit=2)
                if len(fields) != 3 or not fields[0]:
                    raise DatasetIntegrityError(
                        f"invalid TREC collection record at line {line_number}"
                    )
                yield fields[0], {"title": fields[1], "description": fields[2]}
                continue
            try:
                raw = cast(dict[str, Any], json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise DatasetIntegrityError(
                    f"invalid TREC corpus JSON at line {line_number}"
                ) from exc
            document_id = str(raw.get("docid", ""))
            title = raw.get("title")
            text = raw.get("text")
            if not document_id or not isinstance(title, str) or not isinstance(text, str):
                raise DatasetIntegrityError(
                    f"invalid TREC corpus record at line {line_number}"
                )
            yield document_id, {"title": title, "description": text}


def load_trec_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["id", "query"]:
            raise DatasetIntegrityError("TREC topic header differs from id/query")
        for row in reader:
            query_id = str(row.get("id", "")).strip()
            query = str(row.get("query", "")).strip()
            if not query_id or not query:
                raise DatasetIntegrityError("TREC topic contains an empty ID or query")
            if query_id in queries:
                raise DatasetIntegrityError(f"duplicate TREC topic ID {query_id}")
            queries[query_id] = query
    if not queries:
        raise DatasetIntegrityError("TREC topic file contains no queries")
    return queries


def load_trec_qrels(path: Path, query_ids: set[str]) -> list[tuple[str, str, int]]:
    qrels: list[tuple[str, str, int]] = []
    pairs: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.split()
            if len(parts) != 4 or parts[1] != "0":
                raise DatasetIntegrityError(f"invalid TREC qrel line {line_number}")
            query_id, document_id = parts[0], parts[2]
            if query_id not in query_ids:
                raise DatasetIntegrityError(
                    f"TREC qrel references unknown topic {query_id}"
                )
            pair = (query_id, document_id)
            if pair in pairs:
                raise DatasetIntegrityError(
                    f"duplicate TREC qrel pair {query_id}/{document_id}"
                )
            pairs.add(pair)
            try:
                relevance = int(parts[3])
            except ValueError as exc:
                raise DatasetIntegrityError(
                    f"invalid TREC relevance at line {line_number}"
                ) from exc
            if not -1 <= relevance <= 3:
                raise DatasetIntegrityError(
                    f"TREC relevance is outside [-1, 3] at line {line_number}"
                )
            qrels.append((query_id, document_id, relevance))
    if not qrels:
        raise DatasetIntegrityError("TREC qrel file contains no judgments")
    return qrels


def inspect_trec_product_search_alignment(
    corpus_path: Path,
    query_path: Path,
    qrels_path: Path,
    *,
    maximum_orphan_rate: float,
) -> TrecAlignment:
    if not 0.0 <= maximum_orphan_rate <= 1.0:
        raise ValueError("maximum orphan rate must be in [0, 1]")
    queries = load_trec_queries(query_path)
    qrels = load_trec_qrels(qrels_path, set(queries))
    judged_products = {document_id for _, document_id, _ in qrels}
    found_judged_products: set[str] = set()
    seen_document_ids: set[int] = set()
    products = 0
    for document_id, _ in iter_trec_products(corpus_path):
        products += 1
        try:
            numeric_id = int(document_id)
        except ValueError as exc:
            raise DatasetIntegrityError(
                f"TREC corpus document ID is not numeric: {document_id}"
            ) from exc
        if numeric_id in seen_document_ids:
            raise DatasetIntegrityError(f"duplicate TREC corpus document ID {numeric_id}")
        seen_document_ids.add(numeric_id)
        if document_id in judged_products:
            found_judged_products.add(document_id)
    orphan_products = judged_products - found_judged_products
    orphan_judgments = sum(
        1 for _, document_id, _ in qrels if document_id in orphan_products
    )
    orphan_rate = orphan_judgments / len(qrels)
    if orphan_rate > maximum_orphan_rate:
        raise DatasetIntegrityError(
            f"TREC qrel orphan rate {orphan_rate:.6f} exceeds "
            f"{maximum_orphan_rate:.6f}"
        )
    return TrecAlignment(
        products=products,
        queries=len(queries),
        judged_queries=len({query_id for query_id, _, _ in qrels}),
        judgments=len(qrels),
        unique_judged_products=len(judged_products),
        orphan_judgments=orphan_judgments,
        orphan_products=len(orphan_products),
        orphan_rate=orphan_rate,
    )
