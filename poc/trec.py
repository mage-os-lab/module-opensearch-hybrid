from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


def identifier_sort_key(value: str) -> tuple[int, int | str]:
    if value.isdecimal():
        return (0, int(value))
    return (1, value)


@dataclass(frozen=True, slots=True)
class RunRecord:
    query_id: str
    document_id: str
    rank: int
    score: float
    tag: str

    def line(self) -> str:
        return f"{self.query_id} Q0 {self.document_id} {self.rank} {self.score:.12g} {self.tag}\n"


@dataclass(frozen=True, slots=True)
class QrelRecord:
    query_id: str
    document_id: str
    relevance: int

    def line(self) -> str:
        return f"{self.query_id} 0 {self.document_id} {self.relevance}\n"


def write_run(path: Path, records: list[RunRecord]) -> None:
    ordered = sorted(records, key=lambda record: (record.query_id, record.rank))
    _validate(ordered)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("".join(record.line() for record in ordered))
    temporary.replace(path)


def read_run(path: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.split()
        if len(fields) != 6 or fields[1] != "Q0":
            raise ValueError(f"invalid TREC run line {line_number} in {path}")
        records.append(
            RunRecord(
                query_id=fields[0],
                document_id=fields[2],
                rank=int(fields[3]),
                score=float(fields[4]),
                tag=fields[5],
            )
        )
    _validate(records)
    return records


def write_qrels(path: Path, records: list[QrelRecord]) -> None:
    ordered = sorted(
        records,
        key=lambda record: (
            identifier_sort_key(record.query_id),
            identifier_sort_key(record.document_id),
        ),
    )
    _validate_qrels(ordered)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("".join(record.line() for record in ordered))
    temporary.replace(path)


def read_qrels(path: Path) -> list[QrelRecord]:
    records: list[QrelRecord] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.split()
        if len(fields) != 4 or fields[1] != "0":
            raise ValueError(f"invalid TREC qrels line {line_number} in {path}")
        records.append(
            QrelRecord(
                query_id=fields[0],
                document_id=fields[2],
                relevance=int(fields[3]),
            )
        )
    _validate_qrels(records)
    return records


def _validate(records: list[RunRecord]) -> None:
    ranks: dict[str, list[int]] = defaultdict(list)
    tags: set[str] = set()
    documents: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.rank <= 0:
            raise ValueError("TREC ranks are one-based")
        if record.document_id in documents[record.query_id]:
            raise ValueError(
                f"duplicate document {record.document_id!r} for query {record.query_id!r}"
            )
        documents[record.query_id].add(record.document_id)
        ranks[record.query_id].append(record.rank)
        tags.add(record.tag)
    for query_id, query_ranks in ranks.items():
        if query_ranks != list(range(1, len(query_ranks) + 1)):
            raise ValueError(f"non-contiguous ranks for query {query_id!r}")
    if len(tags) > 1:
        raise ValueError("a run file must contain exactly one run tag")


def _validate_qrels(records: list[QrelRecord]) -> None:
    pairs: set[tuple[str, str]] = set()
    for record in records:
        pair = (record.query_id, record.document_id)
        if pair in pairs:
            raise ValueError(f"duplicate qrel for query/document pair {pair!r}")
        if record.relevance < 0:
            raise ValueError("qrel relevance must be non-negative")
        pairs.add(pair)
