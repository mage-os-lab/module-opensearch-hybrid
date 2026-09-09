from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from poc.config import ModelSpec, load_model_registry
from poc.datasets import DatasetIntegrityError, file_facts
from poc.experiments import BM25Profile
from poc.hybrid import (
    hybrid_pipeline_id,
    verify_hybrid_artifact,
    write_hybrid_bundle,
)
from poc.manifest import canonical_sha256
from poc.provenance import collect_manifest_provenance
from poc.search import build_normalization_pipeline
from poc.trec import QrelRecord, RunRecord, write_qrels

ROOT = Path(__file__).resolve().parents[1]
INDEX = "fixture-wands"


class FakeBackend:
    device = "cpu"
    runtime = "fixture-int8-onnx"
    model_artifact_sha256 = "b" * 64

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        return np.ones((len(texts), 2), dtype=np.float32)


class FakeClient:
    def __init__(self) -> None:
        self.pipeline_id = ""
        self.pipeline: dict[str, Any] = {}

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: object | None = None,
        params: dict[str, object] | None = None,
    ) -> object:
        del json_body, params
        assert method == "GET"
        if path == "/":
            return {"version": {"number": "3.8.0"}}
        if path == f"/{INDEX}":
            return {INDEX: {"settings": {"index": {"uuid": "fixture-uuid"}}}}
        if path == f"/{INDEX}/_stats/docs,segments":
            return {
                "_all": {
                    "primaries": {
                        "docs": {"count": 1, "deleted": 0},
                        "segments": {"count": 1},
                    }
                }
            }
        if path == f"/_search/pipeline/{self.pipeline_id}":
            return {self.pipeline_id: self.pipeline}
        raise AssertionError(path)


def test_hybrid_run_schema_v2_derives_eligibility_and_rejects_bool_strings(
    tmp_path: Path,
) -> None:
    model, profile, client, provenance = _prepare_root(tmp_path)
    backend = FakeBackend()
    result = write_hybrid_bundle(
        client,  # type: ignore[arg-type]
        root=tmp_path,
        index=INDEX,
        model=model,
        backend=backend,
        profile=profile,
        lexical_weight=0.3,
        query_path=tmp_path / "data/prepared/wands/queries.test.jsonl",
        qrels_path=tmp_path / "data/prepared/wands/qrels.test.trec",
        split="test",
        tag="fixture-hybrid-test",
        records=[
            RunRecord(
                query_id="1",
                document_id="d1",
                rank=1,
                score=100.0,
                tag="fixture-hybrid-test",
            )
        ],
        query_vectors_hash="a" * 64,
        encoding_seconds=0.1,
        eligible_for_decision=True,
        ineligibility_reason=None,
        decision_scope="held_out",
        benchmark_provenance=provenance,
        upstream_evidence={"schema_version": 2, "eligible_for_decision": False},
        selected_model_eligible=True,
    )
    entry = result.selection_entry(tmp_path)
    manifest = _read_json(result.manifest_path)

    assert manifest["schema_version"] == 2
    assert manifest["benchmark_provenance"] == provenance
    assert manifest["eligible_for_decision"] is False
    assert manifest["ineligibility_reason"] == (
        "run has no valid clean committed start provenance; "
        "hybrid upstream evidence is not decision eligible"
    )
    verified = verify_hybrid_artifact(
        client,  # type: ignore[arg-type]
        root=tmp_path,
        index=INDEX,
        model=model,
        backend_artifact_sha256=backend.model_artifact_sha256,
        entry=entry,
        expected_lexical_weight=0.3,
        expected_split="test",
        expected_decision_scope="held_out",
        expected_benchmark_provenance=provenance,
        expected_upstream_evidence={"schema_version": 2, "eligible_for_decision": False},
        expected_selected_model_eligible=True,
    )
    assert verified == manifest

    manifest["eligible_for_decision"] = "false"
    _write_json(result.manifest_path, manifest)
    entry["manifest_sha256"] = file_facts(result.manifest_path).sha256
    with pytest.raises(DatasetIntegrityError, match="eligibility"):
        verify_hybrid_artifact(
            client,  # type: ignore[arg-type]
            root=tmp_path,
            index=INDEX,
            model=model,
            backend_artifact_sha256=backend.model_artifact_sha256,
            entry=entry,
            expected_lexical_weight=0.3,
            expected_split="test",
            expected_decision_scope="held_out",
            expected_benchmark_provenance=provenance,
            expected_upstream_evidence={
                "schema_version": 2,
                "eligible_for_decision": False,
            },
            expected_selected_model_eligible=True,
        )


def _prepare_root(
    root: Path,
) -> tuple[ModelSpec, BM25Profile, FakeClient, dict[str, Any]]:
    for name in ("benchmark.toml", "models.toml", "query_runtime.toml"):
        destination = root / "config" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "config" / name, destination)
    environment = root / "results/environment/benchmark-profile.json"
    environment.parent.mkdir(parents=True, exist_ok=True)
    environment.write_text("{}\n")
    model = load_model_registry(root / "config/models.toml")["arctic_embed_m_v2"]
    profile = BM25Profile.from_mapping(
        {
            "name": "fixture",
            "fields": ["title^8", "description"],
            "query_type": "best_fields",
            "operator": "or",
            "tie_breaker": 0.1,
        }
    )
    prepared = root / "data/prepared/wands"
    prepared.mkdir(parents=True, exist_ok=True)
    (prepared / "queries.test.jsonl").write_text(
        '{"query_id":"1","query":"table","split":"test"}\n'
    )
    write_qrels(
        prepared / "qrels.test.trec",
        [QrelRecord(query_id="1", document_id="d1", relevance=2)],
    )
    index_manifest = root / "results/wands/index-manifest.json"
    _write_json(index_manifest, {"index_definition_sha256": "c" * 64})
    model_manifest = root / f"results/wands/embeddings/{model.name}.manifest.json"
    _write_json(
        model_manifest,
        {
            "model_artifact_sha256": "d" * 64,
            "encoder_runtime": "fixture-document-encoder",
        },
    )
    bm25_selection = root / "results/wands/bm25-selection.json"
    _write_json(
        bm25_selection,
        {
            "selected_profile": profile.to_dict(),
            "selected_profile_sha256": canonical_sha256(profile.to_dict()),
        },
    )
    provenance = json.loads(json.dumps(collect_manifest_provenance(root)))
    client = FakeClient()
    client.pipeline_id = hybrid_pipeline_id(model, 0.3)
    client.pipeline = build_normalization_pipeline(lexical_weight=0.3)
    return model, profile, client, provenance


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
