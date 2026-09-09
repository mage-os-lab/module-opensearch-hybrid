from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import poc.alternatives as alternatives_module
from poc.alternatives import WandsAlternativesEvidenceContext
from poc.datasets import DatasetIntegrityError, file_facts
from poc.provenance import collect_manifest_provenance


def test_alternatives_derive_each_family_eligibility_from_exact_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_benchmark_context(tmp_path)
    _write_source(
        tmp_path,
        summary_path="results/wands/bm25-selection.json",
        summary_eligibility=("quality_evidence_eligible_for_decision", True),
        entries={"tuned/test": ("tuned", True)},
    )
    _write_source(
        tmp_path,
        summary_path="results/wands/hybrid-summary.json",
        summary_eligibility=("eligible_for_decision", True),
        entries={"test/selected": ("dense", True)},
        extra={"selected_lexical_weight": 0.3},
    )
    _write_source(
        tmp_path,
        summary_path="results/wands/neural-sparse/summary.json",
        summary_eligibility=("quality_evidence_eligible_for_decision", False),
        entries={"hybrid/test/selected": ("sparse", False)},
    )
    _write_source(
        tmp_path,
        summary_path="results/wands/three-way/summary.json",
        summary_eligibility=("eligible_for_decision", True),
        entries={"test/selected": ("three-way", True)},
    )
    _write_source(
        tmp_path,
        summary_path="results/wands/reranker/summary.json",
        summary_eligibility=("quality_evidence_eligible_for_decision", True),
        entries={
            "tuned_bm25/test": ("bm25-reranker", True),
            "dense_minmax/test": ("dense-reranker", True),
        },
    )
    qrels = tmp_path / "data/prepared/wands/qrels.test.trec"
    qrels.parent.mkdir(parents=True, exist_ok=True)
    qrels.write_text("q 0 d 1\n")
    observed: list[bool] = []

    def compare_run_family(**kwargs: Any) -> dict[str, object]:
        observed.append(bool(kwargs["eligible_for_decision"]))
        return {"eligible_for_decision": kwargs["eligible_for_decision"]}

    monkeypatch.setattr(
        alternatives_module,
        "compare_run_family",
        compare_run_family,
    )
    summary_paths = {
        "bm25": tmp_path / "results/wands/bm25-selection.json",
        "dense": tmp_path / "results/wands/hybrid-summary.json",
        "sparse": tmp_path / "results/wands/neural-sparse/summary.json",
        "three_way": tmp_path / "results/wands/three-way/summary.json",
        "reranker": tmp_path / "results/wands/reranker/summary.json",
    }
    monkeypatch.setattr(
        alternatives_module,
        "_verify_alternative_source_summaries",
        lambda **kwargs: {
            name: json.loads(path.read_text()) for name, path in summary_paths.items()
        },
    )

    result = alternatives_module._compute_wands_alternatives_significance(
        root=tmp_path,
        evidence_context=_evidence_context(),
        alpha=0.05,
        minimum_delta=0.03,
    )

    assert observed == [False, True, False, True]
    evidence = cast(dict[str, dict[str, Any]], result["source_evidence"])
    assert evidence["sparse_minmax"]["eligible_for_decision"] is False
    assert evidence["dense_minmax"]["eligible_for_decision"] is True


def test_alternatives_reject_rehashed_semantically_invalid_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_path = tmp_path / "results/wands/bm25-selection.json"
    _write_json(
        summary_path,
        {
            "quality_evidence_eligible_for_decision": True,
            "selected_profile_sha256": "0" * 64,
            "self_rehashed_after_edit": file_facts(summary_path).sha256
            if summary_path.exists()
            else "0" * 64,
        },
    )
    edited = cast(dict[str, object], json.loads(summary_path.read_text()))
    edited["self_rehashed_after_edit"] = file_facts(summary_path).sha256
    _write_json(summary_path, edited)
    rejected_hashes: list[str] = []

    def reject_semantic_edit(*args: object, **kwargs: object) -> dict[str, Any]:
        del args, kwargs
        rejected_hashes.append(file_facts(summary_path).sha256)
        raise DatasetIntegrityError("BM25 selection fields do not reproduce from registered inputs")

    monkeypatch.setattr(
        alternatives_module,
        "verify_wands_bm25_selection",
        reject_semantic_edit,
    )

    with pytest.raises(DatasetIntegrityError, match="do not reproduce"):
        alternatives_module._verify_alternative_source_summaries(
            root=tmp_path,
            context=_evidence_context(),
        )

    assert rejected_hashes == [file_facts(summary_path).sha256]


def _write_source(
    root: Path,
    *,
    summary_path: str,
    summary_eligibility: tuple[str, bool] | None,
    entries: dict[str, tuple[str, bool]],
    extra: dict[str, object] | None = None,
) -> None:
    artifacts: dict[str, dict[str, object]] = {}
    for key, (name, eligible) in entries.items():
        run_path = root / f"runs/{name}.trec"
        manifest_path = root / f"runs/{name}.manifest.json"
        run_path.parent.mkdir(parents=True, exist_ok=True)
        run_path.write_text("q Q0 d 1 1.0 fixture\n")
        _write_json(
            manifest_path,
            {
                "eligible_for_decision": eligible,
                "run_file": {
                    "path": str(run_path.relative_to(root)),
                    "sha256": file_facts(run_path).sha256,
                },
                "benchmark_provenance": _fixture_provenance(root),
            },
        )
        artifacts[key] = {
            "run": str(run_path.relative_to(root)),
            "manifest": str(manifest_path.relative_to(root)),
        }
    summary: dict[str, object] = {"artifacts": artifacts, **(extra or {})}
    if summary_eligibility is not None:
        key, value = summary_eligibility
        summary[key] = value
    _write_json(root / summary_path, summary)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _write_benchmark_context(root: Path) -> None:
    profile_path = root / "config/benchmark.toml"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
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
    source_path = root / "poc/fixture.py"
    source_path.parent.mkdir()
    source_path.write_text("VALUE = 1\n")
    _write_json(
        root / "results/environment/benchmark-profile.json",
        {"schema_version": 1, "profile_id": "test-profile"},
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


def _fixture_provenance(root: Path) -> dict[str, object]:
    return cast(dict[str, object], collect_manifest_provenance(root))


def _evidence_context() -> WandsAlternativesEvidenceContext:
    reranker = SimpleNamespace(
        client=object(),
        index="fixture-index",
        models=(),
        runtime=object(),
        experiments=object(),
    )
    return cast(
        WandsAlternativesEvidenceContext,
        SimpleNamespace(
            reranker=reranker,
            sparse=object(),
            three_way=object(),
            three_way_model=object(),
            reranker_spec=object(),
        ),
    )
