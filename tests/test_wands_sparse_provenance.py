from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import poc.manifest as manifest_module
import poc.sparse_benchmark as sparse_module
from poc.config import WANDS_INT8_MODEL_NAMES
from poc.datasets import DatasetIntegrityError, file_facts
from poc.experiments import BM25Experiments, BM25Profile
from poc.index_evidence import INDEX_WRITE_BLOCK_EVIDENCE
from poc.indexing import wands_completion_provenance_evidence
from poc.manifest import canonical_sha256, read_json
from poc.neural_sparse import (
    NeuralSparseSpec,
    build_sparse_index_definition,
    build_sparse_ingest_pipeline,
)
from poc.os_client import OpenSearchClient
from poc.provenance import BenchmarkProfile, collect_manifest_provenance, load_benchmark_profile
from poc.sparse_benchmark import run_wands_sparse_benchmark

INDEX = "fixture-wands-sparse"


def _fixture_live_environment(profile: BenchmarkProfile) -> dict[str, Any]:
    container_id = "a" * 64
    empty_contract_hash = hashlib.sha256(b"{}").hexdigest()
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
                "opensearchproject/opensearch@sha256:" + "b" * 64
            ),
            "platform_manifest_digest": "sha256:" + "c" * 64,
            "platform_os": "linux",
            "contract": {},
            "contract_sha256": empty_contract_hash,
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


@dataclass(frozen=True)
class FakeCrossCheck:
    def to_dict(self) -> dict[str, object]:
        return {"status": "passed"}


class FakeSparseClient:
    def __init__(self) -> None:
        self.pipelines: dict[str, dict[str, Any]] = {}
        self.tamper_mapping = False

    def search(
        self,
        index: str,
        body: dict[str, Any],
        *,
        pipeline: str | None = None,
        request_cache: bool | None = None,
    ) -> dict[str, Any]:
        query = body.get("query")
        if isinstance(query, dict) and "bool" in query:
            assert query["bool"] == {
                "must": [
                    {
                        "neural_sparse": {
                            _spec().embedding_field: {
                                "query_text": "shoe",
                                "analyzer": _spec().query_analyzer,
                            }
                        }
                    }
                ],
                "filter": [{"ids": {"values": ["d1"]}}],
            }
        del request_cache
        assert index == INDEX
        if pipeline is not None:
            assert pipeline in self.pipelines
        return {"hits": {"hits": [{"_id": "d1", "_score": 1.0}]}}

    def put_search_pipeline(self, pipeline_id: str, definition: dict[str, Any]) -> None:
        self.pipelines[pipeline_id] = definition

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: object | None = None,
        params: dict[str, object] | None = None,
    ) -> object:
        del params
        if method == "GET" and path == f"/{INDEX}/_source/d1":
            return {"product_id": "d1", "title": "shoe"}
        if method == "POST" and path == "/_analyze":
            assert json_body == {"analyzer": _spec().query_analyzer, "text": "shoe"}
            return {"tokens": [{"token": "shoe"}]}
        assert method == "GET"
        if path == f"/{INDEX}":
            return {INDEX: {"settings": {"index": {"uuid": "sparse-index-uuid"}}}}
        if path == f"/{INDEX}/_stats/docs,segments":
            return {
                "_all": {
                    "primaries": {
                        "docs": {"count": 1, "deleted": 0},
                        "segments": {"count": 1},
                    }
                }
            }
        if path == f"/{INDEX}/_mapping":
            definition = build_sparse_index_definition(_spec(), use_default_pipeline=False)
            if self.tamper_mapping:
                definition["mappings"]["dynamic"] = True
            return {INDEX: {"mappings": definition["mappings"]}}
        if path == f"/{INDEX}/_settings":
            definition = build_sparse_index_definition(_spec(), use_default_pipeline=False)
            settings = copy.deepcopy(definition["settings"]["index"])
            settings["refresh_interval"] = "1s"
            settings["blocks"] = {"write": True}
            settings["analysis"] = copy.deepcopy(definition["settings"]["analysis"])
            return {
                INDEX: {
                    "settings": {
                        "index": {
                            "uuid": "sparse-index-uuid",
                            "creation_date": "1",
                            **settings,
                        }
                    }
                }
            }
        if path == f"/_ingest/pipeline/{_spec().ingest_pipeline}":
            return {
                _spec().ingest_pipeline: build_sparse_ingest_pipeline(_spec(), model_id="model-1")
            }
        if path.startswith("/_search/pipeline/"):
            pipeline_id = path.rsplit("/", 1)[-1]
            return {pipeline_id: self.pipelines[pipeline_id]}
        raise AssertionError(path)


def test_wands_sparse_uses_one_start_snapshot_and_strict_upstream_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_root(tmp_path)
    snapshots: list[dict[str, Any]] = []

    def capture(root: Path) -> dict[str, Any]:
        provenance = collect_manifest_provenance(root)
        snapshots.append(provenance)
        return provenance

    monkeypatch.setattr(sparse_module, "collect_manifest_provenance", capture)
    _patch_dependencies(monkeypatch)

    def reject_implicit_capture(root: Path) -> dict[str, Any]:
        raise AssertionError(f"unexpected implicit provenance capture for {root}")

    monkeypatch.setattr(
        manifest_module,
        "collect_manifest_provenance",
        reject_implicit_capture,
    )
    client = _client()
    selection = run_wands_sparse_benchmark(
        client,
        root=tmp_path,
        spec=_spec(),
        experiments=_experiments(),
    )

    assert len(snapshots) == 1
    persisted_provenance = json.loads(json.dumps(snapshots[0]))
    assert selection["schema_version"] == 2
    assert selection["benchmark_provenance"] == persisted_provenance
    assert selection["quality_evidence_eligible_for_decision"] is True
    artifacts = cast(dict[str, dict[str, Any]], selection["artifacts"])
    for entry in artifacts.values():
        manifest = _read_json(tmp_path / str(entry["manifest"]))
        assert manifest["schema_version"] == 2
        assert manifest["benchmark_provenance"] == persisted_provenance
        assert manifest["eligible_for_decision"] is (manifest["decision_scope"] == "held_out")

    verified = sparse_module.verify_wands_sparse_summary(
        client,
        root=tmp_path,
        spec=_spec(),
        experiments=_experiments(),
    )
    assert verified == selection


@pytest.mark.parametrize("invalid_case", ("null_commit", "index", "bm25"))
def test_wands_sparse_quality_fails_closed_for_invalid_start_or_upstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_case: str,
) -> None:
    _prepare_root(tmp_path)
    provenance = collect_manifest_provenance(tmp_path)
    if invalid_case == "null_commit":
        cast(dict[str, Any], provenance["code_revision"])["git_commit"] = None
    elif invalid_case == "index":
        path = tmp_path / "results/wands/neural-sparse/index-manifest.json"
        manifest = _read_json(path)
        manifest["quality_evidence_eligible_for_decision"] = False
        _write_json(path, manifest)
        _refresh_tokenizer_binding(tmp_path)
    else:
        path = tmp_path / "results/wands/bm25-selection.json"
        manifest = _read_json(path)
        manifest["quality_evidence_eligible_for_decision"] = False
        _write_json(path, manifest)
    monkeypatch.setattr(
        sparse_module,
        "collect_manifest_provenance",
        lambda root: provenance,
    )
    _patch_dependencies(monkeypatch)

    client = _client()
    selection = run_wands_sparse_benchmark(
        client,
        root=tmp_path,
        spec=_spec(),
        experiments=_experiments(),
    )

    assert selection["quality_evidence_eligible_for_decision"] is False
    artifacts = cast(dict[str, dict[str, Any]], selection["artifacts"])
    for key in ("sparse/test", "hybrid/test/selected"):
        manifest = _read_json(tmp_path / str(artifacts[key]["manifest"]))
        assert manifest["eligible_for_decision"] is False

    verified = sparse_module.verify_wands_sparse_summary(
        client,
        root=tmp_path,
        spec=_spec(),
        experiments=_experiments(),
    )
    assert verified["quality_evidence_eligible_for_decision"] is False


def test_wands_sparse_verifier_rejects_eligibility_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_root(tmp_path)
    provenance = collect_manifest_provenance(tmp_path)
    cast(dict[str, Any], provenance["code_revision"])["git_commit"] = None
    monkeypatch.setattr(
        sparse_module,
        "collect_manifest_provenance",
        lambda root: provenance,
    )
    _patch_dependencies(monkeypatch)
    client = _client()
    selection = run_wands_sparse_benchmark(
        client,
        root=tmp_path,
        spec=_spec(),
        experiments=_experiments(),
    )
    selection["quality_evidence_eligible_for_decision"] = True
    _write_json(tmp_path / "results/wands/neural-sparse/summary.json", selection)

    with pytest.raises(DatasetIntegrityError, match="eligibility"):
        sparse_module.verify_wands_sparse_summary(
            client,
            root=tmp_path,
            spec=_spec(),
            experiments=_experiments(),
        )


def test_wands_sparse_fails_closed_for_live_index_semantic_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_root(tmp_path)
    _patch_dependencies(monkeypatch)
    client = cast(FakeSparseClient, _client())
    client.tamper_mapping = True

    selection = run_wands_sparse_benchmark(
        cast(OpenSearchClient, client),
        root=tmp_path,
        spec=_spec(),
        experiments=_experiments(),
    )

    assert selection["quality_evidence_eligible_for_decision"] is False
    verified = sparse_module.verify_wands_sparse_summary(
        cast(OpenSearchClient, client),
        root=tmp_path,
        spec=_spec(),
        experiments=_experiments(),
    )
    assert verified["quality_evidence_eligible_for_decision"] is False


def test_wands_sparse_verifier_rejects_partial_artifact_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_root(tmp_path)
    _patch_dependencies(monkeypatch)
    client = _client()
    selection = run_wands_sparse_benchmark(
        client,
        root=tmp_path,
        spec=_spec(),
        experiments=_experiments(),
    )
    cast(dict[str, Any], selection["artifacts"]).pop("sparse/test")
    _write_json(tmp_path / "results/wands/neural-sparse/summary.json", selection)

    with pytest.raises(DatasetIntegrityError, match="artifact set"):
        sparse_module.verify_wands_sparse_summary(
            client,
            root=tmp_path,
            spec=_spec(),
            experiments=_experiments(),
        )


def test_dense_comparison_claims_require_semantic_verification_and_eligibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_json(
        tmp_path / "results/wands/hybrid-summary.json",
        {"eligible_for_decision": True},
    )
    _write_json(
        tmp_path / "results/wands/rrf-summary.json",
        {"eligible_for_decision": False},
    )
    models = {name: object() for name in WANDS_INT8_MODEL_NAMES}
    calls: list[str] = []
    monkeypatch.setattr(sparse_module, "load_model_registry", lambda path: models)
    monkeypatch.setattr(
        sparse_module,
        "load_query_runtime_spec",
        lambda path: object(),
    )
    def verify_minmax(*args: object, **kwargs: object) -> dict[str, Any]:
        calls.append("minmax")
        return cast(
            dict[str, Any],
            read_json(tmp_path / "results/wands/hybrid-summary.json"),
        )

    def verify_rrf(*args: object, **kwargs: object) -> dict[str, Any]:
        calls.append("rrf")
        return cast(
            dict[str, Any],
            read_json(tmp_path / "results/wands/rrf-summary.json"),
        )

    monkeypatch.setattr(
        sparse_module,
        "verify_wands_hybrid_summary",
        verify_minmax,
    )
    monkeypatch.setattr(
        sparse_module,
        "verify_wands_rrf_summary",
        verify_rrf,
    )

    _, _, evidence = sparse_module._verified_dense_comparisons(
        _client(),
        root=tmp_path,
        spec=_spec(),
        experiments=_experiments(),
    )

    assert calls == ["minmax", "rrf"]
    assert evidence["dense_minmax_semantically_verified"] is True
    assert evidence["dense_rrf_semantically_verified"] is True
    assert evidence["dense_minmax_eligible_for_decision"] is True
    assert evidence["dense_rrf_eligible_for_decision"] is False
    assert evidence["eligible_for_decision"] is False


def _prepare_root(root: Path) -> None:
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
    source_path = root / "poc/fixture.py"
    source_path.parent.mkdir()
    source_path.write_text("VALUE = 1\n")
    benchmark_profile = load_benchmark_profile(profile_path)
    _write_json(
        root / "results/environment/benchmark-profile.json",
        {
            "schema_version": 2,
            "verified_at": "2026-08-22T12:00:00+00:00",
            **_fixture_live_environment(benchmark_profile),
        },
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "add", "config/benchmark.toml", "poc/fixture.py"],
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
    for split in ("dev", "test"):
        (prepared / f"queries.{split}.jsonl").write_text(
            json.dumps({"query_id": "q1", "query": "shoe", "split": split}) + "\n"
        )
        (prepared / f"qrels.{split}.trec").write_text("q1 0 d1 1\n")

    products_path = prepared / "products.jsonl"
    products_path.write_text(json.dumps({"product_id": "d1", "title": "shoe"}) + "\n")
    model_path = root / "results/wands/neural-sparse/model-deployment.json"
    _write_json(model_path, {"model_id": "model-1"})
    embeddings_path = root / "data/cache/neural-sparse/wands-doc-v3-distill.jsonl"
    embeddings_path.parent.mkdir(parents=True)
    embeddings_path.write_text(json.dumps({"product_id": "d1", "embedding": {"shoe": 1.0}}) + "\n")
    precompute_path = root / "results/wands/neural-sparse/precompute-manifest.json"
    _write_json(precompute_path, {"status": "passed"})

    provenance = json.loads(json.dumps(collect_manifest_provenance(root)))
    _, completion_revision = wands_completion_provenance_evidence(root, provenance)
    spec = _spec()
    definition = build_sparse_index_definition(spec, use_default_pipeline=False)
    pipeline = build_sparse_ingest_pipeline(spec, model_id="model-1")
    index_manifest_path = root / "results/wands/neural-sparse/index-manifest.json"
    _write_json(
        index_manifest_path,
        {
            "schema_version": 2,
            "dataset": "WANDS",
            "index": {
                "name": INDEX,
                "uuid": "sparse-index-uuid",
                "document_count": 1,
                "segment_count": 1,
                "deleted_document_count": 0,
            },
            "index_definition_sha256": canonical_sha256(definition),
            "index_write_block": INDEX_WRITE_BLOCK_EVIDENCE,
            "index_content": _content_evidence(),
            "ingest_pipeline_sha256": canonical_sha256(pipeline),
            "model_id": "model-1",
            "model_artifact_sha256": file_facts(model_path).sha256,
            "prepared_products_sha256": file_facts(products_path).sha256,
            "precomputed_embeddings_sha256": file_facts(embeddings_path).sha256,
            "precompute_manifest_sha256": file_facts(precompute_path).sha256,
            "precompute_evidence": _precompute_evidence(),
            "preparation_evidence": _preparation_evidence(),
            "query_mode": "doc_only_builtin_analyzer",
            "query_analyzer": spec.query_analyzer,
            "benchmark_provenance": provenance,
            "completion_code_revision": completion_revision,
            "quality_evidence_eligible_for_decision": True,
            "latency_evidence_eligible_for_decision": False,
        },
    )
    _write_json(
        root / "results/wands/neural-sparse/tokenizer-inspection.json",
        {
            "schema_version": 2,
            "document_id": "d1",
            "artifact_sparse_term_count": 1,
            "analyzer_terms": ["shoe"],
            "overlap": ["shoe"],
            "probe_mode": "id_filtered_positive_match_with_top5_diagnostics",
            "expected_document_matched": True,
            "probe_hit_ids": ["d1"],
            "status": "passed",
            "index_manifest": _artifact(root, index_manifest_path),
            "quality_evidence_eligible_for_decision": True,
        },
    )
    profile = _profile("tuned")
    _write_json(
        root / "results/wands/bm25-selection.json",
        {
            "schema_version": 2,
            "dataset": "WANDS",
            "selected_profile": profile.to_dict(),
            "held_out_test_metrics": {
                "tuned": {"ndcg@10": 1.0},
                "competent_integrator": {"ndcg@10": 1.0},
            },
            "benchmark_provenance": provenance,
            "completion_code_revision": completion_revision,
            "quality_evidence_eligible_for_decision": True,
        },
    )
    _write_json(
        root / "results/wands/hybrid-summary.json",
        {"held_out_test_metrics": {"ndcg@10": 1.0}},
    )
    _write_json(
        root / "results/wands/rrf-summary.json",
        {"held_out_test_metrics": {"ndcg@10": 1.0}},
    )


def _patch_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sparse_module,
        "verify_live_index_content",
        lambda *args, **kwargs: _content_evidence(),
    )
    monkeypatch.setattr(
        sparse_module,
        "verify_wands_sparse_precompute",
        lambda **kwargs: _precompute_evidence(),
    )
    monkeypatch.setattr(
        sparse_module,
        "cross_check_run",
        lambda *args, **kwargs: FakeCrossCheck(),
    )
    monkeypatch.setattr(
        sparse_module,
        "verify_wands_bm25_selection",
        lambda client, *, root, index, experiments: cast(
            dict[str, Any], read_json(root / "results/wands/bm25-selection.json")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        sparse_module,
        "load_wands_config",
        lambda path: SimpleNamespace(expected_products=1),
    )
    monkeypatch.setattr(
        sparse_module,
        "verify_wands_preparation_evidence",
        lambda **kwargs: _preparation_evidence(),
    )
    monkeypatch.setattr(
        sparse_module,
        "_verified_dense_comparisons",
        lambda client, *, root, spec, experiments: (
            cast(
                dict[str, Any],
                read_json(root / "results/wands/hybrid-summary.json"),
            ),
            cast(
                dict[str, Any],
                read_json(root / "results/wands/rrf-summary.json"),
            ),
            {
                "dense_minmax_summary": _artifact(root, root / "results/wands/hybrid-summary.json"),
                "dense_rrf_summary": _artifact(root, root / "results/wands/rrf-summary.json"),
                "dense_minmax_semantically_verified": True,
                "dense_rrf_semantically_verified": True,
                "dense_minmax_eligible_for_decision": True,
                "dense_rrf_eligible_for_decision": True,
                "eligible_for_decision": True,
            },
        ),
    )


def _refresh_tokenizer_binding(root: Path) -> None:
    index_path = root / "results/wands/neural-sparse/index-manifest.json"
    inspection_path = root / "results/wands/neural-sparse/tokenizer-inspection.json"
    inspection = _read_json(inspection_path)
    inspection["index_manifest"] = _artifact(root, index_path)
    _write_json(inspection_path, inspection)


def _spec() -> NeuralSparseSpec:
    return NeuralSparseSpec(
        model_name="fixture-model",
        model_version="1.0.0",
        model_format="TORCH_SCRIPT",
        query_analyzer="bert-uncased",
        package_url="https://artifacts.opensearch.org/fixture.zip",
        package_sha256="c" * 64,
        package_bytes=3,
        model_content_sha256="a" * 64,
        model_content_bytes=11,
        tokenizer_content_sha256="b" * 64,
        tokenizer_content_bytes=12,
        index_name=INDEX,
        ingest_pipeline="fixture-ingest",
        text_field="sparse_text",
        embedding_field="sparse_embedding",
        prune_type="max_ratio",
        prune_ratio=0.1,
        inference_batch_size=1,
        bulk_request_size=1,
    )


def _experiments() -> BM25Experiments:
    profile = _profile("tuned")
    competent = _profile("competent_integrator")
    return BM25Experiments(
        naive=_profile("naive"),
        magento_emulation=profile,
        competent_integrator=competent,
        tuning_candidates=(profile,),
        hybrid_weight_count=1,
        lexical_weight_grid=(0.3,),
        rrf_rank_constants=(60,),
        fusion_normalization="min_max",
        fusion_combination="arithmetic_mean",
        primary_metric="ndcg@10",
        minimum_paired_delta=0.03,
        alpha=0.05,
        multiple_testing_correction="holm",
        maximum_trec_orphan_rate=0.05,
        minimum_judged_at_10=0.5,
        minimum_ann_recall_at_100=0.98,
        ann_ef_search_grid=(100,),
    )


def _profile(name: str) -> BM25Profile:
    return BM25Profile(
        name=name,
        fields=("title",),
        query_type="best_fields",
        operator="or",
        tie_breaker=0.0,
    )


def _client() -> OpenSearchClient:
    return cast(OpenSearchClient, FakeSparseClient())


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }


def _preparation_evidence() -> dict[str, object]:
    return {
        "dataset_registry": {"path": "config/datasets.toml"},
        "prepared_files": {"products.jsonl": "fixture"},
        "semantic_verification": {"products": 1},
        "eligible_for_decision": True,
    }


def _precompute_evidence() -> dict[str, object]:
    return {
        "schema_version": 2,
        "eligible_for_decision": True,
    }


def _content_evidence() -> dict[str, object]:
    return {
        "algorithm": "sha256-canonical-opensearch-source-v1",
        "document_order": "registered_exact_id_sequence",
        "page_size": 500,
        "documents": 1,
        "float32_fields": ["sparse_embedding"],
        "sha256": "a" * 64,
    }
