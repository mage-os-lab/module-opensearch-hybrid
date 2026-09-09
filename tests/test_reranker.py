from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
import pytest

import poc.reranker as reranker_module
from poc.datasets import DatasetIntegrityError, file_facts
from poc.provenance import collect_manifest_provenance
from poc.reranker import (
    RerankerSpec,
    WandsRerankerEvidenceContext,
    load_reranker_spec,
    prepare_wands_reranker,
    rerank_records,
    select_onnx_file,
    verify_wands_reranker_benchmark,
)
from poc.trec import RunRecord

ROOT = Path(__file__).resolve().parents[1]


class _Scores:
    runtime = "test"
    artifact_sha256 = "abc"

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        assert len(pairs) == 3
        return np.asarray([0.1, 0.9, 0.9], dtype=np.float32)


class _PreparedScores:
    runtime = "test-runtime"
    artifact_sha256 = "runtime-hash"
    runtime_components: ClassVar[dict[str, dict[str, int | str]]] = {}

    def __init__(self, **_: object) -> None:
        self.onnx_path = Path(__file__)

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.asarray(
            [
                (sum(query.encode()) + sum(document.encode())) / 10_000.0
                for query, document in pairs
            ],
            dtype=np.float32,
        )


def test_reranker_spec_pins_architecture_specific_official_int8_files() -> None:
    spec = load_reranker_spec(ROOT / "config/reranker.toml")

    assert spec.repo_id == "cross-encoder/ettin-reranker-32m-v1"
    assert len(spec.revision) == 40
    assert spec.candidate_depth == 50
    assert spec.result_depth == 100
    assert select_onnx_file(spec, "arm64") == "onnx/model_qint8_arm64.onnx"
    assert select_onnx_file(spec, "x86_64") == "onnx/model_quint8_avx2.onnx"


def test_reranker_reorders_only_candidate_depth_with_source_rank_tie_break() -> None:
    spec = RerankerSpec(
        repo_id="cross-encoder/example",
        revision="a" * 40,
        license="apache-2.0",
        max_length=512,
        arm64_onnx_file="onnx/arm64.onnx",
        x86_64_onnx_file="onnx/avx2.onnx",
        arm64_runtime_artifacts_sha256="b" * 64,
        x86_64_runtime_artifacts_sha256="c" * 64,
        candidate_depth=3,
        result_depth=4,
        batch_size=3,
        source_arms=("dense_minmax",),
    )
    source = [
        RunRecord("q1", "a", 1, 4.0, "source"),
        RunRecord("q1", "b", 2, 3.0, "source"),
        RunRecord("q1", "c", 3, 2.0, "source"),
        RunRecord("q1", "d", 4, 1.0, "source"),
    ]

    records, samples = rerank_records(
        source,
        queries={"q1": "query"},
        documents={"a": "A", "b": "B", "c": "C", "d": "D"},
        backend=_Scores(),
        spec=spec,
        tag="reranked",
    )

    assert [record.document_id for record in records] == ["b", "c", "a", "d"]
    assert [record.rank for record in records] == [1, 2, 3, 4]
    assert [record.score for record in records] == [4.0, 3.0, 2.0, 1.0]
    assert len(samples) == 1 and samples[0] >= 0


def test_reranker_accepts_partial_source_pools_and_missing_queries() -> None:
    spec = RerankerSpec(
        repo_id="cross-encoder/example",
        revision="a" * 40,
        license="apache-2.0",
        max_length=512,
        arm64_onnx_file="onnx/arm64.onnx",
        x86_64_onnx_file="onnx/avx2.onnx",
        arm64_runtime_artifacts_sha256="b" * 64,
        x86_64_runtime_artifacts_sha256="c" * 64,
        candidate_depth=3,
        result_depth=4,
        batch_size=3,
        source_arms=("tuned_bm25",),
    )
    source = [
        RunRecord("q1", "a", 1, 2.0, "source"),
        RunRecord("q1", "b", 2, 1.0, "source"),
    ]

    records, samples = rerank_records(
        source,
        queries={"q1": "query", "q2": "no matches"},
        documents={"a": "A", "b": "B"},
        backend=_PreparedScores(),
        spec=spec,
        tag="reranked",
    )

    assert [record.document_id for record in records] == ["b", "a"]
    assert [record.rank for record in records] == [1, 2]
    assert [record.score for record in records] == [4.0, 3.0]
    assert len(samples) == 1 and samples[0] >= 0


def test_reranker_rejects_source_pool_larger_than_registered_depth() -> None:
    spec = RerankerSpec(
        repo_id="cross-encoder/example",
        revision="a" * 40,
        license="apache-2.0",
        max_length=512,
        arm64_onnx_file="onnx/arm64.onnx",
        x86_64_onnx_file="onnx/avx2.onnx",
        arm64_runtime_artifacts_sha256="b" * 64,
        x86_64_runtime_artifacts_sha256="c" * 64,
        candidate_depth=3,
        result_depth=4,
        batch_size=3,
        source_arms=("tuned_bm25",),
    )
    source = [
        RunRecord("q1", str(rank), rank, float(6 - rank), "source")
        for rank in range(1, 6)
    ]

    with pytest.raises(DatasetIntegrityError, match="exceeds registered depth"):
        rerank_records(
            source,
            queries={"q1": "query"},
            documents={str(rank): str(rank) for rank in range(1, 6)},
            backend=_Scores(),
            spec=spec,
            tag="reranked",
        )


def test_reranker_rejects_foreign_source_query() -> None:
    spec = _fixture_spec(source_arms=("tuned_bm25",))

    with pytest.raises(DatasetIntegrityError, match="unknown query"):
        rerank_records(
            [RunRecord("foreign", "p1", 1, 1.0, "source")],
            queries={"q1": "query"},
            documents={"p1": "document"},
            backend=_PreparedScores(),
            spec=spec,
            tag="reranked",
        )


def test_reranker_latency_samples_only_source_present_queries(tmp_path: Path) -> None:
    source_path = tmp_path / "source.trec"
    source_path.write_text("q1 Q0 p1 1 1.0 source\n")

    samples = reranker_module._standalone_latency_samples(
        _PreparedScores(),
        source_path=source_path,
        queries={"q1": "query", "q2": "no matches"},
        documents={"p1": "document"},
        spec=_fixture_spec(source_arms=("dense_minmax",)),
    )

    assert len(samples) == 1
    assert samples[0] >= 0


def test_reranker_score_cache_requires_live_deterministic_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _fixture_spec(source_arms=("tuned_bm25",))
    backend = _PreparedScores()
    pairs = [("one", "document one"), ("two", "document two")]
    scores = {
        pair: float(score)
        for pair, score in zip(pairs, backend.score(pairs), strict=True)
    }
    selected_facts = file_facts(Path(__file__))
    selected: dict[str, int | str] = {
        "sha256": selected_facts.sha256,
        "bytes": selected_facts.bytes,
    }
    monkeypatch.setattr(reranker_module, "EttinOnnxBackend", _PreparedScores)

    evidence = reranker_module._verify_reranker_score_replay(
        root=tmp_path,
        spec=spec,
        scores=scores,
        runtime_components={},
        runtime_artifact_sha256="runtime-hash",
        selected_onnx_artifact=selected,
    )
    assert evidence["status"] == "passed"

    tampered = dict(scores)
    tampered[pairs[0]] += 0.01
    with pytest.raises(DatasetIntegrityError, match="score replay"):
        reranker_module._verify_reranker_score_replay(
            root=tmp_path,
            spec=spec,
            scores=tampered,
            runtime_components={},
            runtime_artifact_sha256="runtime-hash",
            selected_onnx_artifact=selected,
        )


def test_reranker_preparation_derives_quality_and_latency_eligibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _fixture_spec(source_arms=("tuned_bm25", "dense_minmax"))
    (tmp_path / "config").mkdir()
    (tmp_path / "config/benchmark.toml").write_text(
        "\n".join(
            (
                "schema_version = 2",
                'profile_id = "test-profile"',
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
    environment_path = tmp_path / "results/environment/benchmark-profile.json"
    _write_json(
        environment_path,
        {
            "schema_version": 1,
            "profile_id": "test-profile",
            "cpu_limit": 8,
            "memory_limit_bytes": 17179869184,
            "architecture": "x86_64",
            "latency_architecture_eligible": True,
            "limits_verified": False,
        },
    )
    _initialize_source_repository(tmp_path)
    source_paths: dict[str, dict[str, Path]] = {}
    for arm in spec.source_arms:
        source_paths[arm] = {}
        for split in ("dev", "test"):
            run_path = tmp_path / f"runs/{arm}-{split}.trec"
            run_path.parent.mkdir(parents=True, exist_ok=True)
            run_path.write_text("q1 Q0 p1 1 1.0 source\n")
            source_provenance = _fixture_provenance(tmp_path)
            if arm == "dense_minmax" and split == "test":
                cast(dict[str, Any], source_provenance["code_revision"])["source_dirty"] = True
            _write_json(
                run_path.with_suffix(".manifest.json"),
                {
                    "eligible_for_decision": True,
                    "run_file": {
                        "path": str(run_path.relative_to(tmp_path)),
                        "sha256": file_facts(run_path).sha256,
                    },
                    "benchmark_provenance": source_provenance,
                },
            )
            source_paths[arm][split] = run_path
    _write_json(
        tmp_path / "results/wands/bm25-selection.json",
        {
            "quality_evidence_eligible_for_decision": True,
            "artifacts": {
                f"tuned/{split}": {
                    "run": str(source_paths["tuned_bm25"][split].relative_to(tmp_path)),
                    "manifest": str(
                        source_paths["tuned_bm25"][split]
                        .with_suffix(".manifest.json")
                        .relative_to(tmp_path)
                    ),
                }
                for split in ("dev", "test")
            },
        },
    )
    _write_json(
        tmp_path / "results/wands/hybrid-summary.json",
        {
            "selected_lexical_weight": 0.3,
            "eligible_for_decision": False,
            "artifacts": {
                "dev/lw030": {
                    "run": str(source_paths["dense_minmax"]["dev"].relative_to(tmp_path)),
                    "manifest": str(
                        source_paths["dense_minmax"]["dev"]
                        .with_suffix(".manifest.json")
                        .relative_to(tmp_path)
                    ),
                },
                "test/selected": {
                    "run": str(source_paths["dense_minmax"]["test"].relative_to(tmp_path)),
                    "manifest": str(
                        source_paths["dense_minmax"]["test"]
                        .with_suffix(".manifest.json")
                        .relative_to(tmp_path)
                    ),
                },
            },
        },
    )

    monkeypatch.setattr(reranker_module, "EttinOnnxBackend", _PreparedScores)
    selected_facts = file_facts(Path(__file__))
    monkeypatch.setattr(
        reranker_module,
        "_reranker_runtime_artifacts",
        lambda root, spec: (
            {},
            "runtime-hash",
            {"sha256": selected_facts.sha256, "bytes": selected_facts.bytes},
        ),
    )
    monkeypatch.setattr(
        reranker_module,
        "_queries_by_split",
        lambda root: {"dev": {"q1": "query"}, "test": {"q1": "query"}},
    )
    monkeypatch.setattr(
        reranker_module,
        "load_product_documents",
        lambda path: {"p1": "document"},
    )
    monkeypatch.setattr(
        reranker_module,
        "_verify_reranker_source_summaries",
        lambda **kwargs: {
            "bm25": json.loads((tmp_path / "results/wands/bm25-selection.json").read_text()),
            "hybrid": json.loads((tmp_path / "results/wands/hybrid-summary.json").read_text()),
        },
    )
    monkeypatch.setattr(
        reranker_module,
        "read_run",
        lambda path: [RunRecord("q1", "p1", 1, 1.0, "source")],
    )
    completion_revisions: list[dict[str, Any]] = []

    def completion_drift(
        root: Path,
        provenance: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        del root
        revision = dict(cast(dict[str, Any], provenance["code_revision"]))
        revision["git_commit"] = "b" * 40
        completion_revisions.append(revision)
        return False, revision

    monkeypatch.setattr(
        reranker_module,
        "wands_completion_provenance_evidence",
        completion_drift,
    )
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")

    manifest = prepare_wands_reranker(
        root=tmp_path,
        spec=spec,
        evidence_context=_evidence_context(),
        local_files_only=True,
    )

    assert manifest["quality_evidence_eligible_for_decision"] is False
    assert manifest["preparation_provenance_valid"] is False
    assert manifest["completion_code_revision"] == completion_revisions[0]
    assert "clean current source revision" in str(manifest["quality_ineligibility_reason"])
    assert manifest["standalone_latency_decision_eligible"] is False
    assert manifest["standalone_latency_ineligibility_reason"] == (
        "benchmark environment resource limits are not verified"
    )
    assert manifest["benchmark_environment"] == {
        "path": "results/environment/benchmark-profile.json",
        "sha256": file_facts(environment_path).sha256,
        "bytes": file_facts(environment_path).bytes,
    }
    provenance = cast(dict[str, Any], manifest["benchmark_provenance"])
    generator_host = cast(dict[str, Any], provenance["generator_host"])
    assert generator_host["architecture"] == "x86_64"


def test_reranker_summary_verifier_rejects_wrong_preparation_manifest_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _fixture_spec(source_arms=())
    preparation_path = tmp_path / "results/wands/reranker/precompute-manifest.json"
    cache_path = tmp_path / "data/cache/reranker/scores.jsonl"
    _write_json(
        preparation_path,
        {
            "pair_score_cache": {
                "path": str(cache_path.relative_to(tmp_path)),
            },
            "runtime_artifact_sha256": "runtime",
        },
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text('{"document":"d","query":"q","score":1.0}\n')
    _write_json(
        tmp_path / "results/wands/reranker/summary.json",
        {
            "artifacts": {},
            "preparation_manifest": {
                "path": str(preparation_path.relative_to(tmp_path)),
                "sha256": "0" * 64,
                "bytes": file_facts(preparation_path).bytes,
            },
        },
    )
    monkeypatch.setattr(
        reranker_module,
        "verify_wands_reranker_preparation",
        lambda **kwargs: json.loads(preparation_path.read_text()),
    )
    monkeypatch.setattr(
        reranker_module,
        "_queries_by_split",
        lambda root: {"dev": {}, "test": {}},
    )
    monkeypatch.setattr(reranker_module, "load_product_documents", lambda path: {})
    monkeypatch.setattr(reranker_module, "reranker_source_run_paths", lambda root: {})

    with pytest.raises(
        DatasetIntegrityError,
        match="preparation manifest",
    ):
        verify_wands_reranker_benchmark(
            root=tmp_path,
            spec=spec,
            evidence_context=_evidence_context(),
        )


def test_reranker_latency_requires_clean_start_commit() -> None:
    environment = {
        "profile_id": "test-profile",
        "cpu_limit": 8,
        "memory_limit_bytes": 17179869184,
        "architecture": "x86_64",
        "latency_architecture_eligible": True,
        "limits_verified": True,
    }

    eligible, reason = reranker_module._reranker_latency_eligibility(
        profile_id="test-profile",
        cpu_limit=8,
        memory_limit_bytes=17179869184,
        required_architecture="x86_64",
        environment=environment,
        host_architecture="x86_64",
        code_revision={"git_commit": None, "source_dirty": False},
        preparation_provenance_valid=True,
    )

    assert eligible is False
    assert reason == "latency run did not start from a clean committed source revision"

    eligible, reason = reranker_module._reranker_latency_eligibility(
        profile_id="test-profile",
        cpu_limit=8,
        memory_limit_bytes=17179869184,
        required_architecture="x86_64",
        environment=environment,
        host_architecture="x86_64",
        code_revision={"git_commit": "a" * 40, "source_dirty": False},
        preparation_provenance_valid=True,
    )

    assert eligible is False
    assert reason == "reranker host-process resource limits are not independently verified"


def test_reranker_quality_requires_valid_preparation_start_provenance() -> None:
    source_evidence = {
        "tuned_bm25": {"test": {"eligible_for_decision": True}},
        "dense_minmax": {"test": {"eligible_for_decision": True}},
    }

    eligible, reason = reranker_module._reranker_quality_eligibility(
        source_evidence=source_evidence,
        source_arms=("tuned_bm25", "dense_minmax"),
        model_cache_inputs_verified=True,
        preparation_provenance_valid=False,
    )

    assert eligible is False
    assert reason == ("reranker preparation did not retain one clean current source revision")

    eligible, reason = reranker_module._reranker_quality_eligibility(
        source_evidence=source_evidence,
        source_arms=("tuned_bm25", "dense_minmax"),
        model_cache_inputs_verified=True,
        preparation_provenance_valid=True,
    )

    assert eligible is True
    assert reason is None


def test_reranker_summary_claims_recompute_eligibility_and_deltas() -> None:
    metrics = {
        "tuned_bm25": {"test": {"ndcg@10": 0.72}},
        "dense_minmax": {"test": {"ndcg@10": 0.81}},
    }
    preparation = {
        "source_evidence": {"bound": True},
        "quality_evidence_eligible_for_decision": True,
        "quality_ineligibility_reason": None,
        "standalone_latency_decision_eligible": False,
        "standalone_latency_ineligibility_reason": "wrong architecture",
    }
    summary = {
        "reranked_metrics": metrics,
        "source_evidence": {"bound": True},
        "quality_evidence_eligible_for_decision": True,
        "quality_ineligibility_reason": None,
        "latency_evidence_eligible_for_decision": False,
        "latency_ineligibility_reason": "wrong architecture",
        "source_arm_metrics": {
            "tuned_bm25_test_ndcg@10": 0.67,
            "dense_minmax_test_ndcg@10": 0.78,
        },
        "bm25_reranker_minus_tuned_bm25_ndcg@10": 0.72 - 0.67,
        "bm25_reranker_minus_dense_minmax_ndcg@10": 0.72 - 0.78,
        "dense_reranker_minus_dense_minmax_ndcg@10": 99.0,
    }

    with pytest.raises(
        DatasetIntegrityError,
        match="dense_reranker_minus_dense_minmax",
    ):
        reranker_module._verify_reranker_summary_claims(
            summary=summary,
            preparation=preparation,
            source_arms=("tuned_bm25", "dense_minmax"),
            recomputed_metrics=metrics,
            test_manifest_eligibility=[True, True],
            tuned_test=0.67,
            dense_test=0.78,
        )

    summary["dense_reranker_minus_dense_minmax_ndcg@10"] = 0.03
    summary["quality_evidence_eligible_for_decision"] = False
    with pytest.raises(
        DatasetIntegrityError,
        match="summary quality eligibility",
    ):
        reranker_module._verify_reranker_summary_claims(
            summary=summary,
            preparation=preparation,
            source_arms=("tuned_bm25", "dense_minmax"),
            recomputed_metrics=metrics,
            test_manifest_eligibility=[True, True],
            tuned_test=0.67,
            dense_test=0.78,
        )


def test_reranker_source_gate_propagates_semantic_summary_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_path = tmp_path / "results/wands/bm25-selection.json"
    _write_json(
        summary_path,
        {
            "quality_evidence_eligible_for_decision": True,
            "selected_profile_sha256": "0" * 64,
        },
    )
    rejected: list[Path] = []

    def reject_rehashed_summary(*args: object, **kwargs: object) -> dict[str, Any]:
        del args, kwargs
        rejected.append(summary_path)
        raise DatasetIntegrityError("BM25 selection fields do not reproduce from registered inputs")

    monkeypatch.setattr(
        reranker_module,
        "verify_wands_bm25_selection",
        reject_rehashed_summary,
    )
    monkeypatch.setattr(
        reranker_module,
        "verify_wands_hybrid_summary",
        lambda *args, **kwargs: pytest.fail("hybrid verification must not continue"),
    )

    with pytest.raises(DatasetIntegrityError, match="do not reproduce"):
        reranker_module._verify_reranker_source_summaries(
            root=tmp_path,
            context=_evidence_context(),
        )

    assert rejected == [summary_path]


def _fixture_spec(*, source_arms: tuple[str, ...]) -> RerankerSpec:
    return RerankerSpec(
        repo_id="cross-encoder/example",
        revision="a" * 40,
        license="apache-2.0",
        max_length=512,
        arm64_onnx_file="onnx/arm64.onnx",
        x86_64_onnx_file="onnx/avx2.onnx",
        arm64_runtime_artifacts_sha256="b" * 64,
        x86_64_runtime_artifacts_sha256="c" * 64,
        candidate_depth=1,
        result_depth=1,
        batch_size=1,
        source_arms=source_arms,
    )


def _evidence_context() -> WandsRerankerEvidenceContext:
    return WandsRerankerEvidenceContext(
        client=cast(Any, object()),
        index="fixture-index",
        models=(),
        runtime=cast(Any, object()),
        experiments=cast(Any, object()),
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _fixture_provenance(root: Path) -> dict[str, object]:
    return cast(dict[str, object], collect_manifest_provenance(root))


def _initialize_source_repository(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "add", "config/benchmark.toml"],
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
