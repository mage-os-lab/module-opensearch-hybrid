from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from poc.datasets import DatasetIntegrityError
from poc.index_evidence import (
    INDEX_WRITE_BLOCK_EVIDENCE,
    content_addressed_pipeline_id,
    lock_live_index_for_decisions,
    verify_live_index_settings,
    verify_live_search_pipeline,
)


class FakeClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response

    def request(self, *_: object, **__: object) -> dict[str, Any]:
        return self.response


class FakeMutableClient:
    def __init__(self) -> None:
        self.settings: dict[str, Any] = {
            "products": {
                "settings": {
                    "index": {
                        "number_of_shards": "1",
                        "number_of_replicas": "0",
                        "refresh_interval": "1s",
                    }
                }
            }
        }
        self.pipelines: dict[str, dict[str, Any]] = {}
        self.updates: list[tuple[str, dict[str, Any]]] = []

    def update_index_settings(
        self, index: str, settings: Mapping[str, Any]
    ) -> None:
        self.updates.append((index, dict(settings)))
        self.settings[index]["settings"]["index"]["blocks"] = {
            "write": "true"
        }

    def request(self, method: str, path: str, **_: object) -> dict[str, Any]:
        if method == "GET" and path == "/products/_settings":
            return self.settings
        if method == "GET" and path.startswith("/_search/pipeline/"):
            pipeline_id = path.rsplit("/", 1)[-1]
            return {pipeline_id: self.pipelines[pipeline_id]}
        raise AssertionError((method, path))


def test_live_index_settings_bind_registered_analysis() -> None:
    definition = {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "-1",
            },
            "analysis": {
                "analyzer": {
                    "product_text": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase"],
                    }
                }
            },
        }
    }
    live: dict[str, Any] = {
        "products": {
            "settings": {
                "index": {
                    "creation_date": "1",
                    "knn.derived_source": {"enabled": "true"},
                    "merge": {
                        "policy": {
                            "floor_segment": "2mb",
                            "max_merge_at_once": "10",
                        }
                    },
                    "provided_name": "products",
                    "replication": {"type": "DOCUMENT"},
                    "uuid": "generated",
                    "version": {"created": "1"},
                    "knn": "true",
                    "number_of_shards": "1",
                    "number_of_replicas": "0",
                    "refresh_interval": "1s",
                    "blocks": {"write": "true"},
                    "analysis": {
                        "analyzer": {
                            "product_text": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase"],
                            }
                        }
                    },
                }
            }
        }
    }
    client = FakeClient(live)

    verify_live_index_settings(
        client,
        index="products",
        definition=definition,
    )

    live["products"]["settings"]["index"]["analysis"]["analyzer"][
        "product_text"
    ]["filter"] = ["lowercase", "stemmer"]
    with pytest.raises(DatasetIntegrityError, match="settings or analysis"):
        verify_live_index_settings(
            client,
            index="products",
            definition=definition,
        )


def test_live_index_settings_require_write_block() -> None:
    definition = {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "-1",
            }
        }
    }
    live = {
        "products": {
            "settings": {
                "index": {
                    "number_of_shards": "1",
                    "number_of_replicas": "0",
                    "refresh_interval": "1s",
                }
            }
        }
    }

    with pytest.raises(DatasetIntegrityError, match="write block"):
        verify_live_index_settings(
            FakeClient(live),
            index="products",
            definition=definition,
        )


def test_lock_live_index_sets_and_verifies_write_block() -> None:
    client = FakeMutableClient()

    evidence = lock_live_index_for_decisions(client, index="products")

    assert client.updates == [
        ("products", {"index": {"blocks": {"write": True}}})
    ]
    assert evidence == INDEX_WRITE_BLOCK_EVIDENCE


def test_pipeline_identity_and_live_verifier_bind_exact_content() -> None:
    definition = {
        "phase_results_processors": [
            {"normalization-processor": {"normalization": {"technique": "min_max"}}}
        ]
    }
    pipeline_id = content_addressed_pipeline_id("opensearch-hybrid-test", definition)
    client = FakeMutableClient()
    client.pipelines[pipeline_id] = definition

    assert pipeline_id.startswith("opensearch-hybrid-test-sha256-")
    verify_live_search_pipeline(
        client,
        pipeline_id=pipeline_id,
        expected_definition=definition,
    )

    client.pipelines[pipeline_id] = {"phase_results_processors": []}
    with pytest.raises(DatasetIntegrityError, match="search pipeline differs"):
        verify_live_search_pipeline(
            client,
            pipeline_id=pipeline_id,
            expected_definition=definition,
        )
