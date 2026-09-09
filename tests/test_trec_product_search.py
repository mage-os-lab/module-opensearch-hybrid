from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from poc.datasets import DatasetIntegrityError
from poc.trec_product_search import (
    inspect_trec_product_search_alignment,
    verify_trec_prepared_content,
)


def _write_fixture(root: Path, *, qrel_document: str) -> tuple[Path, Path, Path]:
    corpus = root / "corpus.jsonl.gz"
    with gzip.open(corpus, "wt", encoding="utf-8", newline="\n") as handle:
        handle.write('{"docid":1,"title":"First","text":"Description"}\n')
        handle.write('{"docid":2,"title":"Second","text":"Other"}\n')
    queries = root / "queries.tsv"
    queries.write_text("id\tquery\n10\tfirst product\n", encoding="utf-8")
    qrels = root / "qrels.txt"
    qrels.write_text(f"10 0 {qrel_document} 3\n", encoding="utf-8")
    return corpus, queries, qrels


def test_trec_alignment_reports_zero_orphans_for_matching_ids(tmp_path: Path) -> None:
    corpus, queries, qrels = _write_fixture(tmp_path, qrel_document="1")

    result = inspect_trec_product_search_alignment(
        corpus,
        queries,
        qrels,
        maximum_orphan_rate=0.01,
    )

    assert result.products == 2
    assert result.queries == 1
    assert result.judged_queries == 1
    assert result.judgments == 1
    assert result.orphan_judgments == 0
    assert result.orphan_rate == 0.0


def test_trec_alignment_aborts_above_registered_orphan_rate(tmp_path: Path) -> None:
    corpus, queries, qrels = _write_fixture(tmp_path, qrel_document="3")

    with pytest.raises(DatasetIntegrityError, match="orphan rate"):
        inspect_trec_product_search_alignment(
            corpus,
            queries,
            qrels,
            maximum_orphan_rate=0.01,
        )


def test_trec_alignment_accepts_shuffled_unique_document_ids(tmp_path: Path) -> None:
    corpus, queries, qrels = _write_fixture(tmp_path, qrel_document="1")
    with gzip.open(corpus, "wt", encoding="utf-8", newline="\n") as handle:
        handle.write('{"docid":2,"title":"Second","text":"Other"}\n')
        handle.write('{"docid":1,"title":"First","text":"Description"}\n')

    result = inspect_trec_product_search_alignment(
        corpus,
        queries,
        qrels,
        maximum_orphan_rate=0.01,
    )

    assert result.products == 2
    assert result.orphan_rate == 0.0


def test_trec_prepared_content_is_reconstructed_from_registered_raw(
    tmp_path: Path,
) -> None:
    _, queries, qrels = _write_fixture(tmp_path, qrel_document="1")
    prepared_queries = tmp_path / "queries.test.jsonl"
    prepared_queries.write_text(
        '{"query":"first product","query_id":"10","split":"test"}\n'
    )
    prepared_qrels = tmp_path / "qrels.test.trec"
    prepared_qrels.write_text("10 0 1 3\n")

    verify_trec_prepared_content(
        raw_query_path=queries,
        raw_qrels_path=qrels,
        prepared_query_path=prepared_queries,
        prepared_qrels_path=prepared_qrels,
    )

    prepared_queries.write_text(
        '{"query":"easy replacement","query_id":"10","split":"test"}\n'
    )
    with pytest.raises(DatasetIntegrityError, match="prepared query content"):
        verify_trec_prepared_content(
            raw_query_path=queries,
            raw_qrels_path=qrels,
            prepared_query_path=prepared_queries,
            prepared_qrels_path=prepared_qrels,
        )
