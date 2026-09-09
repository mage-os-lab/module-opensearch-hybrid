from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

import poc.bm25 as bm25_module
import poc.manifest as manifest_module
from poc.bm25 import run_wands_bm25_benchmark
from poc.datasets import DatasetIntegrityError, file_facts
from poc.evaluation import evaluate_run
from poc.experiments import BM25Experiments, BM25Profile
from poc.os_client import OpenSearchClient
from poc.provenance import BenchmarkProfile, collect_manifest_provenance, load_benchmark_profile
from poc.trec import RunRecord, write_run

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
    monkeypatch.setattr(
        bm25_module,
        "_verify_registered_wands_lexical_index",
        lambda client, *, root, index: {
            "index": index,
            "quality_evidence_eligible_for_decision": True,
        },
    )


class FakeOpenSearchClient:
    def search(
        self,
        index: str,
        body: dict[str, Any],
        *,
        pipeline: str | None = None,
        request_cache: bool | None = None,
    ) -> dict[str, Any]:
        del body, pipeline, request_cache
        assert index == INDEX
        return {"hits": {"hits": [{"_id": "d1", "_score": 1.0}]}}

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
            return {INDEX: {"settings": {"index": {"uuid": "index-uuid"}}}}
        if path == f"/{INDEX}/_stats/docs,segments":
            return {
                "_all": {
                    "primaries": {
                        "docs": {"count": 1, "deleted": 0},
                        "segments": {"count": 1},
                    }
                }
            }
        raise AssertionError(path)


def test_bm25_benchmark_captures_one_start_provenance_for_all_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_root(tmp_path, index_quality_eligible=True)
    snapshots: list[dict[str, Any]] = []

    def capture(root: Path) -> dict[str, Any]:
        snapshot = collect_manifest_provenance(root)
        snapshots.append(snapshot)
        return snapshot

    monkeypatch.setattr(bm25_module, "collect_manifest_provenance", capture)

    def reject_implicit_capture(root: Path) -> dict[str, Any]:
        raise AssertionError(f"unexpected implicit provenance capture for {root}")

    monkeypatch.setattr(
        manifest_module,
        "collect_manifest_provenance",
        reject_implicit_capture,
    )

    selection = run_wands_bm25_benchmark(
        _client(),
        root=tmp_path,
        index=INDEX,
        experiments=_experiments(),
    )

    assert len(snapshots) == 1
    assert selection["schema_version"] == 2
    persisted_provenance = json.loads(json.dumps(snapshots[0]))
    assert selection["benchmark_provenance"] == persisted_provenance
    assert selection["quality_evidence_eligible_for_decision"] is True
    assert set(cast(list[str], selection["required_source_manifest_keys"])) == {
        "tuning/magento/dev",
        "tuning/candidate/dev",
        "magento/dev",
        "tuned/dev",
        "naive/test",
        "magento/test",
        "tuned/test",
        "competent_integrator/test",
    }
    artifacts = cast(dict[str, dict[str, Any]], selection["artifacts"])
    for entry in artifacts.values():
        manifest = _read_json(tmp_path / str(entry["manifest"]))
        assert manifest["schema_version"] == 2
        assert manifest["benchmark_provenance"] == persisted_provenance
        assert manifest["eligible_for_decision"] is (manifest["decision_scope"] == "held_out")

    verified = bm25_module.verify_wands_bm25_selection(
        _client(),
        root=tmp_path,
        index=INDEX,
        experiments=_experiments(),
    )
    assert verified == selection


@pytest.mark.parametrize(
    "invalid_case",
    (
        "null_commit",
        "dirty_source",
        "missing_profile_binding",
        "malformed_environment_binding",
        "index",
        "index_schema",
    ),
)
def test_bm25_held_out_and_selection_eligibility_require_bound_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_case: str,
) -> None:
    _prepare_root(tmp_path, index_quality_eligible=invalid_case != "index")
    if invalid_case == "index_schema":
        index_manifest_path = tmp_path / "results/wands/index-manifest.json"
        index_manifest = _read_json(index_manifest_path)
        index_manifest["schema_version"] = 1
        _write_json(index_manifest_path, index_manifest)
    captured = collect_manifest_provenance(tmp_path)
    if invalid_case == "null_commit":
        cast(dict[str, Any], captured["code_revision"])["git_commit"] = None
    elif invalid_case == "dirty_source":
        cast(dict[str, Any], captured["code_revision"])["source_dirty"] = True
    elif invalid_case == "missing_profile_binding":
        captured.pop("benchmark_profile_sha256")
    elif invalid_case == "malformed_environment_binding":
        cast(dict[str, Any], captured["benchmark_environment"])["sha256"] = "bad"
    monkeypatch.setattr(
        bm25_module,
        "collect_manifest_provenance",
        lambda root: captured,
    )

    selection = run_wands_bm25_benchmark(
        _client(),
        root=tmp_path,
        index=INDEX,
        experiments=_experiments(),
    )

    assert selection["quality_evidence_eligible_for_decision"] is False
    artifacts = cast(dict[str, dict[str, Any]], selection["artifacts"])
    for key in (
        "naive/test",
        "magento/test",
        "tuned/test",
        "competent_integrator/test",
    ):
        manifest = _read_json(tmp_path / str(artifacts[key]["manifest"]))
        assert manifest["eligible_for_decision"] is False
        assert manifest["ineligibility_reason"]

    verified = bm25_module.verify_wands_bm25_selection(
        _client(),
        root=tmp_path,
        index=INDEX,
        experiments=_experiments(),
    )
    assert verified["quality_evidence_eligible_for_decision"] is False


def test_bm25_selection_verifier_recomputes_source_provenance(
    tmp_path: Path,
) -> None:
    _prepare_root(tmp_path, index_quality_eligible=True)
    selection = run_wands_bm25_benchmark(
        _client(),
        root=tmp_path,
        index=INDEX,
        experiments=_experiments(),
    )
    artifacts = cast(dict[str, dict[str, Any]], selection["artifacts"])
    manifest_path = tmp_path / str(artifacts["tuned/test"]["manifest"])
    manifest = _read_json(manifest_path)
    cast(
        dict[str, Any],
        cast(dict[str, Any], manifest["benchmark_provenance"])["code_revision"],
    )["source_dirty"] = True
    _write_json(manifest_path, manifest)
    bindings = cast(dict[str, dict[str, Any]], selection["source_manifest_bindings"])
    bindings["tuned/test"] = _artifact(tmp_path, manifest_path)
    _write_json(tmp_path / "results/wands/bm25-selection.json", selection)

    with pytest.raises(DatasetIntegrityError, match="provenance"):
        bm25_module.verify_wands_bm25_selection(
            _client(),
            root=tmp_path,
            index=INDEX,
            experiments=_experiments(),
        )


def test_bm25_live_replay_rejects_self_consistent_nonselected_dev_tamper(
    tmp_path: Path,
) -> None:
    _prepare_root(tmp_path, index_quality_eligible=True)
    selection = run_wands_bm25_benchmark(
        _client(),
        root=tmp_path,
        index=INDEX,
        experiments=_experiments(),
    )
    artifacts = cast(dict[str, dict[str, Any]], selection["artifacts"])
    key = "tuning/candidate/dev"
    entry = artifacts[key]
    run_path = tmp_path / str(entry["run"])
    manifest_path = tmp_path / str(entry["manifest"])
    metrics_path = tmp_path / str(entry["metrics_file"])
    manifest = _read_json(manifest_path)
    tag = str(manifest["tag"])
    write_run(
        run_path,
        [
            RunRecord(
                query_id="q1",
                document_id="tampered",
                rank=1,
                score=1.0,
                tag=tag,
            )
        ],
    )
    evaluation = evaluate_run(
        tmp_path / "data/prepared/wands/qrels.dev.trec",
        run_path,
    )
    metrics = _read_json(metrics_path)
    metrics["metrics"] = evaluation.metrics
    metrics["per_query"] = evaluation.per_query
    _write_json(metrics_path, metrics)
    cast(dict[str, Any], manifest["run_file"])["sha256"] = file_facts(
        run_path
    ).sha256
    _write_json(manifest_path, manifest)
    entry["metrics"] = evaluation.metrics
    cast(dict[str, Any], selection["candidate_dev_metrics"])[
        "candidate"
    ] = evaluation.metrics
    cast(dict[str, dict[str, Any]], selection["source_manifest_bindings"])[
        key
    ] = _artifact(tmp_path, manifest_path)
    _write_json(tmp_path / "results/wands/bm25-selection.json", selection)

    with pytest.raises(DatasetIntegrityError, match="tuning/candidate/dev"):
        bm25_module.verify_wands_bm25_selection(
            _client(),
            root=tmp_path,
            index=INDEX,
            experiments=_experiments(),
        )


def test_bm25_eligibility_fails_closed_when_source_changes_after_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_root(tmp_path, index_quality_eligible=True)

    def capture_then_change_source(root: Path) -> dict[str, Any]:
        provenance = collect_manifest_provenance(root)
        (root / "poc/fixture.py").write_text("VALUE = 2\n")
        return provenance

    monkeypatch.setattr(
        bm25_module,
        "collect_manifest_provenance",
        capture_then_change_source,
    )

    selection = run_wands_bm25_benchmark(
        _client(),
        root=tmp_path,
        index=INDEX,
        experiments=_experiments(),
    )

    assert selection["benchmark_provenance_valid"] is False
    assert selection["quality_evidence_eligible_for_decision"] is False
    artifacts = cast(dict[str, dict[str, Any]], selection["artifacts"])
    for key in (
        "naive/test",
        "magento/test",
        "tuned/test",
        "competent_integrator/test",
    ):
        manifest = _read_json(tmp_path / str(artifacts[key]["manifest"]))
        assert manifest["benchmark_provenance_valid"] is False
        assert manifest["eligible_for_decision"] is False

    verified = bm25_module.verify_wands_bm25_selection(
        _client(),
        root=tmp_path,
        index=INDEX,
        experiments=_experiments(),
    )
    assert verified["quality_evidence_eligible_for_decision"] is False


def _prepare_root(root: Path, *, index_quality_eligible: bool) -> None:
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
    environment_path = root / "results/environment/benchmark-profile.json"
    profile = load_benchmark_profile(profile_path)
    _write_json(
        environment_path,
        {
            "schema_version": 2,
            "verified_at": "2026-08-22T12:00:00+00:00",
            **_fixture_live_environment(profile),
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
    provenance = collect_manifest_provenance(root)
    _write_json(
        root / "results/wands/index-manifest.json",
        {
            "schema_version": 2,
            "dataset": "WANDS",
            "index": {
                "name": INDEX,
                "uuid": "index-uuid",
                "document_count": 1,
                "segment_count": 1,
                "deleted_document_count": 0,
            },
            "index_definition_sha256": "d" * 64,
            "quality_evidence_eligible_for_decision": index_quality_eligible,
            "benchmark_provenance": provenance,
        },
    )
    prepared = root / "data/prepared/wands"
    prepared.mkdir(parents=True)
    for split in ("dev", "test"):
        (prepared / f"queries.{split}.jsonl").write_text(
            json.dumps({"query_id": "q1", "query": "shoe", "split": split}) + "\n"
        )
        (prepared / f"qrels.{split}.trec").write_text("q1 0 d1 1\n")


def _experiments() -> BM25Experiments:
    naive = _profile("naive")
    magento = _profile("magento")
    candidate = _profile("candidate")
    competent = BM25Profile(
        name="competent_integrator",
        fields=("title",),
        query_type="best_fields",
        operator="or",
        tie_breaker=0.0,
        strategy="competent_integrator",
        minimum_should_match="1",
        phrase_fields=("title",),
        prefix_fields=("title",),
    )
    return BM25Experiments(
        naive=naive,
        magento_emulation=magento,
        competent_integrator=competent,
        tuning_candidates=(magento, candidate),
        hybrid_weight_count=2,
        lexical_weight_grid=(0.3, 0.7),
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
    return cast(OpenSearchClient, FakeOpenSearchClient())


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
