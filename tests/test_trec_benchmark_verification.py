from __future__ import annotations

import json
from pathlib import Path

import pytest

from poc.datasets import DatasetIntegrityError, file_facts
from poc.experiments import load_bm25_experiments
from poc.manifest import canonical_sha256
from poc.trec_benchmark import (
    verify_registered_trec_bm25_inputs,
    verify_trec_artifact_bindings,
    verify_trec_manifest_metadata,
    verify_trec_method_attribution,
    verify_trec_registered_artifact,
)


def test_trec_artifact_verification_binds_index_and_preparation(tmp_path: Path) -> None:
    index_manifest = tmp_path / "index-manifest.json"
    preparation = tmp_path / "preparation.json"
    index_manifest.write_text('{"index": "one"}\n')
    preparation.write_text('{"dataset": "one"}\n')
    manifest = {
        "index_manifest_sha256": file_facts(index_manifest).sha256,
        "preparation_sha256": file_facts(preparation).sha256,
    }

    verify_trec_artifact_bindings(
        manifest,
        index_manifest_path=index_manifest,
        preparation_path=preparation,
    )

    index_manifest.write_text('{"index": "changed"}\n')
    with pytest.raises(DatasetIntegrityError, match="index manifest"):
        verify_trec_artifact_bindings(
            manifest,
            index_manifest_path=index_manifest,
            preparation_path=preparation,
        )


def test_trec_method_attribution_rejects_tampered_configuration() -> None:
    config: dict[str, object] = {
        "mode": "document_only",
        "fields": ("title", "description"),
        "weight": 0.3,
    }
    stored_config = json.loads(json.dumps(config))
    manifest = {
        "method": "neural-sparse",
        "declared_variable": "registered_retrieval_arm",
        "method_config": stored_config,
        "method_config_sha256": canonical_sha256(config),
    }

    verify_trec_method_attribution(
        manifest,
        expected_method="neural-sparse",
        expected_declared_variable="registered_retrieval_arm",
        expected_config=config,
    )

    manifest["method_config"] = {**config, "weight": 0.5}
    with pytest.raises(DatasetIntegrityError, match="method attribution"):
        verify_trec_method_attribution(
            manifest,
            expected_method="neural-sparse",
            expected_declared_variable="registered_retrieval_arm",
            expected_config=config,
        )


def test_trec_registered_artifact_rejects_an_alternate_path(tmp_path: Path) -> None:
    registered = tmp_path / "registered.jsonl"
    alternate = tmp_path / "alternate.jsonl"
    registered.write_text('{"query": "registered"}\n')
    alternate.write_text('{"query": "alternate"}\n')
    manifest = {
        "query_file": {
            "path": alternate.name,
            "sha256": file_facts(alternate).sha256,
        }
    }

    with pytest.raises(DatasetIntegrityError, match="registered query file"):
        verify_trec_registered_artifact(
            manifest,
            key="query_file",
            root=tmp_path,
            expected_path=registered,
        )


def test_trec_manifest_metadata_binds_run_identity() -> None:
    manifest = {
        "schema_version": 2,
        "dataset": "TREC Product Search 2024",
        "split": "test",
        "decision_scope": "held_out_transfer",
        "opensearch_version": "2.19.6",
        "tag": "registered-tag",
        "run_file": {
            "path": "runs/registered.trec",
            "sha256": "a" * 64,
            "records": 2,
        },
    }

    verify_trec_manifest_metadata(
        manifest,
        expected_version="2.19.6",
        expected_run_path="runs/registered.trec",
        record_tags=["registered-tag", "registered-tag"],
    )

    manifest["dataset"] = "another dataset"
    with pytest.raises(DatasetIntegrityError, match="metadata"):
        verify_trec_manifest_metadata(
            manifest,
            expected_version="2.19.6",
            expected_run_path="runs/registered.trec",
            record_tags=["registered-tag", "registered-tag"],
        )


def test_trec_bm25_requires_registered_index_and_experiments() -> None:
    root = Path(__file__).resolve().parents[1]
    experiments = load_bm25_experiments(root / "config/experiments.toml")

    with pytest.raises(DatasetIntegrityError, match="registered configuration"):
        verify_registered_trec_bm25_inputs(
            root=root,
            index="unregistered-index",
            experiments=experiments,
        )
