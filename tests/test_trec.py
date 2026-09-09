from __future__ import annotations

from pathlib import Path

from poc.trec import QrelRecord, RunRecord, read_qrels, read_run, write_qrels, write_run


def test_trec_run_is_sorted_and_byte_stable(tmp_path: Path) -> None:
    path = tmp_path / "run.trec"
    records = [
        RunRecord("q2", "d2", 1, 0.5, "test"),
        RunRecord("q1", "d1", 1, 1.0, "test"),
    ]
    write_run(path, records)
    first = path.read_bytes()
    write_run(path, records)
    assert path.read_bytes() == first
    assert read_run(path) == [records[1], records[0]]


def test_qrels_are_naturally_sorted_and_byte_stable(tmp_path: Path) -> None:
    path = tmp_path / "qrels.trec"
    records = [
        QrelRecord("10", "2", 0),
        QrelRecord("2", "11", 2),
        QrelRecord("2", "3", 1),
    ]
    write_qrels(path, records)
    first = path.read_bytes()
    write_qrels(path, records)
    assert path.read_bytes() == first
    assert read_qrels(path) == [records[2], records[1], records[0]]
