from __future__ import annotations

import gzip
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

import poc.manifest as manifest_module
import poc.trec_indexing as indexing_module
from poc.datasets import DatasetIntegrityError, TrecProductSearchDatasetSpec
from poc.provenance import ProvenanceError
from poc.trec_indexing import (
    TrecIndexSpec,
    build_trec_lexical_index,
    trec_index_quality_eligible,
    trec_lexical_index_definition,
    trec_opensearch_url,
    verify_trec_lexical_index,
)

ROOT = Path(__file__).resolve().parents[1]


def test_trec_client_ignores_the_wands_generic_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENSEARCH_HYBRID_OS_URL", "http://wrong-wands:9201")
    monkeypatch.delenv("OPENSEARCH_HYBRID_TREC_OS_URL", raising=False)
    assert trec_opensearch_url() == "http://127.0.0.1:9219"
    monkeypatch.setenv("OPENSEARCH_HYBRID_TREC_OS_URL", "http://127.0.0.1:9219/")
    assert trec_opensearch_url() == "http://127.0.0.1:9219"
    monkeypatch.setenv("OPENSEARCH_HYBRID_TREC_OS_URL", "http://127.0.0.1:9201")
    with pytest.raises(ProvenanceError, match="registered benchmark profile"):
        trec_opensearch_url()


def test_trec_lexical_index_preserves_frozen_wands_text_analyzers() -> None:
    definition = trec_lexical_index_definition(
        TrecIndexSpec(name="trec", shards=1, replicas=0)
    )

    properties = definition["mappings"]["properties"]
    assert properties["title"]["fields"]["magento"]["analyzer"] == "magento_default"
    assert properties["description"]["fields"]["prefix"]["analyzer"] == "prefix_search"
    assert definition["settings"]["index"]["refresh_interval"] == "-1"
    assert "embedding_arctic_embed_m_v2" not in properties


def test_trec_index_quality_rejects_missing_start_provenance() -> None:
    assert not trec_index_quality_eligible(ROOT, {})


def test_trec_lexical_index_binds_every_live_source_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = tmp_path / "data/raw/trec-product-search-2024/collection.trec.gz"
    corpus.parent.mkdir(parents=True)
    with gzip.open(corpus, "wt", encoding="utf-8") as handle:
        handle.write("1\tOne\tFirst description\n")
        handle.write("2\tTwo\tSecond description\n")
    preparation = tmp_path / "results/trec-product-search/preparation.json"
    preparation.parent.mkdir(parents=True)
    preparation.write_text("{}\n")
    manifest_path = tmp_path / "results/trec-product-search/index-manifest.json"
    dataset = TrecProductSearchDatasetSpec(
        source="https://example.test/trec",
        revision="a" * 40,
        expected_products=2,
        expected_queries=1,
        expected_judged_queries=1,
        expected_judgments=1,
        expected_unique_judged_products=1,
        license="fixture",
        primary_gain_mapping="fixture",
        query_authority="fixture",
        files={},
    )
    index = TrecIndexSpec(name="trec-index", shards=1, replicas=0)
    provenance = {"code_revision": {"git_commit": "b" * 40}}
    monkeypatch.setattr(
        indexing_module,
        "collect_manifest_provenance",
        lambda *args, **kwargs: provenance,
    )
    monkeypatch.setattr(
        manifest_module,
        "collect_manifest_provenance",
        lambda *args, **kwargs: provenance,
    )
    monkeypatch.setattr(
        indexing_module,
        "verify_prepared_trec_product_search",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        indexing_module,
        "verify_decision_provenance",
        lambda *args, **kwargs: True,
    )
    client = FakeTrecIndexClient(index=index.name, document_count=2)

    manifest = build_trec_lexical_index(
        cast(Any, client),
        dataset_spec=dataset,
        index_spec=index,
        corpus_path=corpus,
        preparation_path=preparation,
        manifest_path=manifest_path,
        expected_version="2.19.6",
        maximum_orphan_rate=0.05,
    )

    assert cast(dict[str, Any], manifest["index_content"])["documents"] == 2
    assert verify_trec_lexical_index(
        cast(Any, client),
        dataset_spec=dataset,
        index_spec=index,
        corpus_path=corpus,
        preparation_path=preparation,
        manifest_path=manifest_path,
        expected_version="2.19.6",
        maximum_orphan_rate=0.05,
    ) == manifest

    client.documents["2"]["description"] = "altered"
    with pytest.raises(DatasetIntegrityError, match="source differs for document 2"):
        verify_trec_lexical_index(
            cast(Any, client),
            dataset_spec=dataset,
            index_spec=index,
            corpus_path=corpus,
            preparation_path=preparation,
            manifest_path=manifest_path,
            expected_version="2.19.6",
            maximum_orphan_rate=0.05,
        )


class FakeTrecIndexClient:
    def __init__(self, *, index: str, document_count: int) -> None:
        self.index = index
        self.document_count = document_count
        self.definition: dict[str, Any] | None = None
        self.documents: dict[str, dict[str, Any]] = {}
        self.write_block = False

    def wait_until_ready(self, *, expected_version: str) -> str:
        return expected_version

    def delete_index(self, index: str) -> None:
        assert index == self.index

    def create_index(self, index: str, definition: object) -> None:
        assert index == self.index
        self.definition = cast(dict[str, Any], definition)

    def bulk_index(
        self,
        index: str,
        documents: Any,
        *,
        batch_size: int,
    ) -> None:
        del batch_size
        assert index == self.index
        self.documents = {
            document_id: deepcopy(dict(document))
            for document_id, document in documents
        }

    def refresh(self, index: str) -> None:
        assert index == self.index

    def force_merge(self, index: str) -> None:
        assert index == self.index

    def update_index_settings(self, index: str, settings: object) -> None:
        assert index == self.index
        if settings == {"index": {"blocks": {"write": True}}}:
            self.write_block = True

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: object | None = None,
        params: object | None = None,
    ) -> dict[str, object]:
        if path == f"/{self.index}/_count":
            return {"count": self.document_count}
        if path == f"/{self.index}/_mget":
            assert method == "POST"
            assert isinstance(json_body, dict)
            return {
                "docs": [
                    {
                        "_id": document_id,
                        "found": document_id in self.documents,
                        "_source": deepcopy(self.documents.get(document_id)),
                    }
                    for document_id in cast(list[str], json_body["ids"])
                ]
            }
        if path == f"/{self.index}/_stats/docs,segments":
            return {
                "_all": {
                    "primaries": {
                        "docs": {"count": self.document_count, "deleted": 0},
                        "segments": {"count": 1},
                    }
                }
            }
        if path == f"/{self.index}/_mapping":
            assert self.definition is not None
            return {self.index: {"mappings": self.definition["mappings"]}}
        if path == f"/{self.index}/_settings":
            assert params == {"flat_settings": "false"}
            assert self.definition is not None
            settings = cast(dict[str, Any], self.definition["settings"])
            live_index = deepcopy(cast(dict[str, Any], settings["index"]))
            live_index["refresh_interval"] = "1s"
            if self.write_block:
                live_index["blocks"] = {"write": "true"}
            live_index["analysis"] = deepcopy(settings["analysis"])
            return {
                self.index: {
                    "settings": {
                        "index": {
                            "uuid": "trec-index-uuid",
                            "creation_date": "1",
                            **live_index,
                        }
                    }
                }
            }
        if path == f"/{self.index}":
            return {
                self.index: {
                    "settings": {"index": {"uuid": "trec-index-uuid"}}
                }
            }
        raise AssertionError((method, path))
