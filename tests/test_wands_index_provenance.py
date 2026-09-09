from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import poc.indexing as indexing_module
import poc.manifest as manifest_module
from poc.config import ModelSpec
from poc.datasets import DatasetIntegrityError, WandsDatasetSpec, file_facts
from poc.indexing import (
    WandsIndexSpec,
    build_wands_lexical_index,
    build_wands_vector_index,
    verify_wands_lexical_index,
)
from poc.os_client import OpenSearchClient
from poc.provenance import BenchmarkProfile, collect_manifest_provenance, load_benchmark_profile

INDEX = "fixture-wands"


def _fixture_live_environment(profile: BenchmarkProfile) -> dict[str, Any]:
    container_id = "a" * 64
    runtime_contract: dict[str, object] = {}
    return {
        "profile_id": profile.profile_id,
        "compose_project": profile.compose_project,
        "compose_service": profile.compose_service,
        "compose_files": list(profile.compose_files),
        "container_id": container_id,
        "image": profile.container_image,
        "image_id": "sha256:" + "b" * 64,
        "cpu_limit": profile.cpu_limit,
        "memory_limit_bytes": profile.memory_limit_bytes,
        "architecture": "x86_64",
        "latency_architecture_eligible": True,
        "limits_verified": True,
        "runtime": {
            "repository_digest": (
                "opensearchproject/opensearch@" + profile.container_image_id
            ),
            "platform_manifest_digest": "sha256:" + "c" * 64,
            "platform_os": "linux",
            "contract": runtime_contract,
            "contract_sha256": hashlib.sha256(b"{}").hexdigest(),
        },
        "opensearch": {
            "url": profile.opensearch_url,
            "version": profile.opensearch_version,
            "node_name": container_id[:12],
            "plugins": [
                {"component": component, "version": profile.plugin_version}
                for component in profile.plugin_components
            ],
        },
    }


@pytest.fixture(autouse=True)
def _mock_live_benchmark_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "poc.provenance._collect_live_environment_facts",
        lambda *, root, profile: _fixture_live_environment(profile),
    )


class FakeIndexClient:
    def __init__(self) -> None:
        self.definition: dict[str, Any] | None = None
        self.documents: dict[str, dict[str, Any]] = {}
        self.tamper_analysis = False
        self.write_block = False

    def wait_until_ready(self, *, expected_version: str) -> str:
        assert expected_version == "3.8.0"
        return expected_version

    def delete_index(self, index: str) -> None:
        assert index == INDEX

    def create_index(self, index: str, definition: dict[str, Any]) -> None:
        assert index == INDEX
        self.definition = definition

    def bulk_index(
        self,
        index: str,
        documents: Iterable[tuple[str, dict[str, Any]]],
        *,
        batch_size: int = 500,
    ) -> None:
        del batch_size
        assert index == INDEX
        self.documents = {
            document_id: copy.deepcopy(document) for document_id, document in documents
        }

    def refresh(self, index: str) -> None:
        assert index == INDEX

    def force_merge(self, index: str) -> None:
        assert index == INDEX

    def update_index_settings(self, index: str, settings: dict[str, Any]) -> None:
        assert index == INDEX
        if settings == {"index": {"refresh_interval": "1s"}}:
            return
        assert settings == {"index": {"blocks": {"write": True}}}
        self.write_block = True

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: object | None = None,
        params: dict[str, object] | None = None,
    ) -> object:
        if path == f"/{INDEX}/_mget":
            assert method == "POST"
            assert isinstance(json_body, dict)
            return {
                "docs": [
                    {
                        "_id": document_id,
                        "found": document_id in self.documents,
                        "_source": copy.deepcopy(self.documents.get(document_id)),
                    }
                    for document_id in cast(list[str], json_body["ids"])
                ]
            }
        assert method == "GET"
        if path == f"/{INDEX}":
            return {INDEX: {"settings": {"index": {"uuid": "index-uuid"}}}}
        if path == f"/{INDEX}/_stats/docs,segments":
            return {
                "_all": {
                    "primaries": {
                        "docs": {"count": len(self.documents), "deleted": 0},
                        "segments": {"count": 1},
                    }
                }
            }
        if path == f"/{INDEX}/_mapping":
            assert self.definition is not None
            return {INDEX: {"mappings": self.definition["mappings"]}}
        if path == f"/{INDEX}/_count":
            return {"count": len(self.documents)}
        if path == f"/{INDEX}/_settings":
            assert params == {"flat_settings": "false"}
            assert self.definition is not None
            settings = copy.deepcopy(self.definition["settings"]["index"])
            settings["refresh_interval"] = "1s"
            if self.write_block:
                settings["blocks"] = {"write": "true"}
            settings["analysis"] = copy.deepcopy(self.definition["settings"]["analysis"])
            if self.tamper_analysis:
                settings["analysis"]["analyzer"]["product_text"]["filter"] = [
                    "lowercase",
                    "default_stemmer",
                ]
            return {
                INDEX: {
                    "settings": {
                        "index": {
                            "uuid": "index-uuid",
                            "creation_date": "1",
                            **settings,
                        },
                    }
                }
            }
        raise AssertionError(path)


@pytest.fixture(autouse=True)
def _verified_preparation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        indexing_module,
        "verify_prepared_wands",
        lambda *args, **kwargs: {
            "products": 1,
            "queries": 1,
            "qrels": 1,
            "dev_queries": 0,
            "test_queries": 1,
        },
    )


def test_wands_index_uses_one_start_provenance_and_verifies_derived_eligibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_root(tmp_path)
    snapshots: list[dict[str, Any]] = []

    def capture(root: Path) -> dict[str, Any]:
        provenance = collect_manifest_provenance(root)
        snapshots.append(provenance)
        return provenance

    monkeypatch.setattr(indexing_module, "collect_manifest_provenance", capture)

    def reject_implicit_capture(root: Path) -> dict[str, Any]:
        raise AssertionError(f"unexpected implicit provenance capture for {root}")

    monkeypatch.setattr(
        manifest_module,
        "collect_manifest_provenance",
        reject_implicit_capture,
    )
    client = _client()
    manifest = build_wands_lexical_index(
        client,
        dataset_spec=_dataset(),
        index_spec=_index_spec(),
        prepared_directory=prepared,
        manifest_path=tmp_path / "results/wands/index-manifest.json",
    )

    assert len(snapshots) == 1
    assert manifest["schema_version"] == 2
    assert manifest["benchmark_provenance"] == json.loads(json.dumps(snapshots[0]))
    assert manifest["quality_evidence_eligible_for_decision"] is True

    verified = verify_wands_lexical_index(
        client,
        dataset_spec=_dataset(),
        index_spec=_index_spec(),
        prepared_directory=prepared,
        manifest_path=tmp_path / "results/wands/index-manifest.json",
    )
    assert verified["quality_evidence_eligible_for_decision"] is True

    cast(FakeIndexClient, client).documents["1"]["title"] = "altered"
    with pytest.raises(DatasetIntegrityError, match="source differs for document 1"):
        verify_wands_lexical_index(
            client,
            dataset_spec=_dataset(),
            index_spec=_index_spec(),
            prepared_directory=prepared,
            manifest_path=tmp_path / "results/wands/index-manifest.json",
        )


@pytest.mark.parametrize(
    "invalid_case",
    ("null_commit", "dirty_source", "missing_environment_binding"),
)
def test_wands_index_quality_fails_closed_for_invalid_start_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_case: str,
) -> None:
    prepared = _prepare_root(tmp_path)
    provenance = collect_manifest_provenance(tmp_path)
    if invalid_case == "null_commit":
        cast(dict[str, Any], provenance["code_revision"])["git_commit"] = None
    elif invalid_case == "dirty_source":
        cast(dict[str, Any], provenance["code_revision"])["source_dirty"] = True
    else:
        provenance.pop("benchmark_environment")
    monkeypatch.setattr(
        indexing_module,
        "collect_manifest_provenance",
        lambda root: provenance,
    )
    client = _client()
    manifest = build_wands_lexical_index(
        client,
        dataset_spec=_dataset(),
        index_spec=_index_spec(),
        prepared_directory=prepared,
        manifest_path=tmp_path / "results/wands/index-manifest.json",
    )

    assert manifest["quality_evidence_eligible_for_decision"] is False
    verified = verify_wands_lexical_index(
        client,
        dataset_spec=_dataset(),
        index_spec=_index_spec(),
        prepared_directory=prepared,
        manifest_path=tmp_path / "results/wands/index-manifest.json",
    )
    assert verified["quality_evidence_eligible_for_decision"] is False


def test_wands_index_verifier_rejects_promoted_eligibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_root(tmp_path)
    provenance = collect_manifest_provenance(tmp_path)
    cast(dict[str, Any], provenance["code_revision"])["git_commit"] = None
    monkeypatch.setattr(
        indexing_module,
        "collect_manifest_provenance",
        lambda root: provenance,
    )
    client = _client()
    manifest_path = tmp_path / "results/wands/index-manifest.json"
    manifest = build_wands_lexical_index(
        client,
        dataset_spec=_dataset(),
        index_spec=_index_spec(),
        prepared_directory=prepared,
        manifest_path=manifest_path,
    )
    manifest["quality_evidence_eligible_for_decision"] = True
    _write_json(manifest_path, manifest)

    with pytest.raises(DatasetIntegrityError, match="eligibility"):
        verify_wands_lexical_index(
            client,
            dataset_spec=_dataset(),
            index_spec=_index_spec(),
            prepared_directory=prepared,
            manifest_path=manifest_path,
        )


def test_wands_index_fails_closed_when_source_changes_after_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_root(tmp_path)

    def capture_then_change_source(root: Path) -> dict[str, Any]:
        provenance = collect_manifest_provenance(root)
        (root / "poc/fixture.py").write_text("VALUE = 2\n")
        return provenance

    monkeypatch.setattr(
        indexing_module,
        "collect_manifest_provenance",
        capture_then_change_source,
    )
    client = _client()
    manifest_path = tmp_path / "results/wands/index-manifest.json"
    manifest = build_wands_lexical_index(
        client,
        dataset_spec=_dataset(),
        index_spec=_index_spec(),
        prepared_directory=prepared,
        manifest_path=manifest_path,
    )

    assert manifest["quality_evidence_eligible_for_decision"] is False
    verified = verify_wands_lexical_index(
        client,
        dataset_spec=_dataset(),
        index_spec=_index_spec(),
        prepared_directory=prepared,
        manifest_path=manifest_path,
    )
    assert verified["quality_evidence_eligible_for_decision"] is False


def test_wands_index_verifier_rejects_changed_live_analysis(
    tmp_path: Path,
) -> None:
    prepared = _prepare_root(tmp_path)
    client = cast(FakeIndexClient, _client())
    manifest_path = tmp_path / "results/wands/index-manifest.json"
    build_wands_lexical_index(
        cast(OpenSearchClient, client),
        dataset_spec=_dataset(),
        index_spec=_index_spec(),
        prepared_directory=prepared,
        manifest_path=manifest_path,
    )
    client.tamper_analysis = True

    with pytest.raises(DatasetIntegrityError, match="settings or analysis"):
        verify_wands_lexical_index(
            cast(OpenSearchClient, client),
            dataset_spec=_dataset(),
            index_spec=_index_spec(),
            prepared_directory=prepared,
            manifest_path=manifest_path,
        )


def test_wands_index_verifier_binds_all_prepared_artifacts(
    tmp_path: Path,
) -> None:
    prepared = _prepare_root(tmp_path)
    client = _client()
    manifest_path = tmp_path / "results/wands/index-manifest.json"
    build_wands_lexical_index(
        client,
        dataset_spec=_dataset(),
        index_spec=_index_spec(),
        prepared_directory=prepared,
        manifest_path=manifest_path,
    )
    (prepared / "qrels.test.trec").write_text("1 0 1 2\n")

    with pytest.raises(DatasetIntegrityError, match="preparation evidence"):
        verify_wands_lexical_index(
            client,
            dataset_spec=_dataset(),
            index_spec=_index_spec(),
            prepared_directory=prepared,
            manifest_path=manifest_path,
        )


def test_wands_vector_index_binds_embedding_cache_and_start_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_root(tmp_path)
    model = _model()
    embedding_manifest_path = tmp_path / f"results/wands/embeddings/{model.name}.manifest.json"
    _write_json(
        embedding_manifest_path,
        {
            "model_artifact_sha256": "a" * 64,
            "encoder_runtime": "fixture",
        },
    )
    monkeypatch.setattr(indexing_module, "verify_embedding_cache", lambda **kwargs: {})
    monkeypatch.setattr(
        indexing_module,
        "load_cached_vectors",
        lambda **kwargs: (["1"], np.asarray([[1.0, 0.0]], dtype=np.float32)),
    )
    snapshots: list[dict[str, Any]] = []

    def capture(root: Path) -> dict[str, Any]:
        provenance = collect_manifest_provenance(root)
        snapshots.append(provenance)
        return provenance

    monkeypatch.setattr(indexing_module, "collect_manifest_provenance", capture)
    client = _client()
    manifest_path = tmp_path / "results/wands/index-manifest.json"
    manifest = build_wands_vector_index(
        client,
        dataset_spec=_dataset(),
        index_spec=_index_spec(),
        models=(model,),
        root=tmp_path,
        prepared_directory=prepared,
        manifest_path=manifest_path,
    )

    assert len(snapshots) == 1
    assert manifest["schema_version"] == 2
    assert manifest["quality_evidence_eligible_for_decision"] is True
    vector = cast(list[dict[str, Any]], manifest["vector_models"])[0]
    assert vector["embedding_manifest"]["sha256"] == file_facts(embedding_manifest_path).sha256

    verified = verify_wands_lexical_index(
        client,
        dataset_spec=_dataset(),
        index_spec=_index_spec(),
        prepared_directory=prepared,
        manifest_path=manifest_path,
    )
    assert verified["vector_models"] == [model.name]

    cast(FakeIndexClient, client).documents["1"]["embedding_fixture_model"] = [0.25, 0.75]
    with pytest.raises(DatasetIntegrityError, match="source differs for document 1"):
        verify_wands_lexical_index(
            client,
            dataset_spec=_dataset(),
            index_spec=_index_spec(),
            prepared_directory=prepared,
            manifest_path=manifest_path,
        )


def _prepare_root(root: Path) -> Path:
    profile_path = root / "config/benchmark.toml"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(
        "\n".join(
            (
                "schema_version = 2",
                'profile_id = "fixture"',
                "requires_live_verification = true",
                "cpu_limit = 8",
                "memory_limit_bytes = 17179869184",
                'compose_project = "fixture-project"',
                'compose_service = "opensearch"',
                'compose_files = ["docker-compose.yml"]',
                'container_image = "opensearchproject/opensearch:3.8.0"',
                'container_image_id = "sha256:' + "b" * 64 + '"',
                'opensearch_url = "http://127.0.0.1:9201"',
                'opensearch_version = "3.8.0"',
                'plugin_version = "3.8.0.0"',
                'plugin_components = ["opensearch-knn"]',
                'latency_required_architecture = "x86_64"',
                "",
            )
        )
    )
    (root / "config/datasets.toml").write_text("schema_version = 1\n")
    source_path = root / "poc/fixture.py"
    source_path.parent.mkdir()
    source_path.write_text("VALUE = 1\n")
    profile = load_benchmark_profile(profile_path)
    _write_json(
        root / "results/environment/benchmark-profile.json",
        {
            "schema_version": 2,
            "verified_at": "2026-08-22T12:00:00+00:00",
            **_fixture_live_environment(profile),
        },
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "add",
            "config/benchmark.toml",
            "config/datasets.toml",
            "poc/fixture.py",
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    prepared = root / "data/prepared/wands"
    prepared.mkdir(parents=True)
    products_path = prepared / "products.jsonl"
    products_path.write_text(json.dumps({"product_id": "1", "title": "shoe"}) + "\n")
    for name in ("queries.all.jsonl", "queries.dev.jsonl", "queries.test.jsonl"):
        (prepared / name).write_text(
            json.dumps({"query_id": "1", "query": "shoe", "split": "test"}) + "\n"
        )
    for name in ("qrels.all.trec", "qrels.dev.trec", "qrels.test.trec"):
        (prepared / name).write_text("1 0 1 1\n")
    _write_json(
        prepared / "manifest.json",
        {
            "outputs": {
                "products.jsonl": {
                    "sha256": file_facts(products_path).sha256,
                    "records": 1,
                }
            }
        },
    )
    return prepared


def _dataset() -> WandsDatasetSpec:
    return WandsDatasetSpec(
        source="fixture",
        revision="a" * 40,
        expected_products=1,
        expected_queries=1,
        expected_judgments=1,
        expected_unique_pairs=1,
        expected_duplicate_pairs=0,
        expected_conflicting_pairs=0,
        license="fixture",
        primary_gain_mapping="fixture",
        duplicate_policy="fixture",
        split_seed="fixture",
        files={},
    )


def _index_spec() -> WandsIndexSpec:
    return WandsIndexSpec(name=INDEX, shards=1, replicas=0)


def _model() -> ModelSpec:
    return ModelSpec(
        name="fixture_model",
        hf_id="fixture/model",
        revision="b" * 40,
        dims=2,
        normalize=True,
        max_seq_length=32,
        bulk_dtype="float32",
        use_memory_efficient_attention=False,
        trust_remote_code=False,
        query_prefix="query: ",
        document_prefix="passage: ",
        query_template="{query}",
        document_template="{title} {description} {attributes}",
        license="fixture",
        decision_eligible=True,
        contamination="none",
    )


def _client() -> OpenSearchClient:
    return cast(OpenSearchClient, FakeIndexClient())


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
