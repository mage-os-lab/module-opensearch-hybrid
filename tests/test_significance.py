from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

import poc.significance as significance
from poc.config import WANDS_INT8_MODEL_NAMES, load_model_registry
from poc.datasets import DatasetIntegrityError
from poc.experiments import load_bm25_experiments
from poc.indexing import load_wands_index_config
from poc.neural_sparse import load_neural_sparse_spec
from poc.query_runtime import load_query_runtime_spec
from poc.significance import (
    apply_family_eligibility,
    collect_verified_wands_source_summaries,
    derive_family_eligibility,
    holm_adjusted_p_values,
    hybrid_candidate_run_paths,
    sparse_candidate_run_paths,
    verify_registered_wands_evidence_inputs,
)

ROOT = Path(__file__).resolve().parents[1]


def test_holm_correction_is_order_independent_and_monotone() -> None:
    adjusted = holm_adjusted_p_values({"third": 0.04, "first": 0.01, "second": 0.03})

    assert adjusted == {"third": 0.06, "first": 0.03, "second": 0.06}


def test_hybrid_significance_family_contains_minmax_and_rrf() -> None:
    minmax = {
        "model": "arctic_embed_m_v2",
        "selected_lexical_weight": 0.3,
        "artifacts": {"test/selected": {"run": "runs/minmax.trec"}},
    }
    rrf = {
        "model": "arctic_embed_m_v2",
        "selected_rank_constant": 20,
        "artifacts": {"test/selected": {"run": "runs/rrf.trec"}},
    }

    assert hybrid_candidate_run_paths(Path("/bench"), minmax, rrf) == {
        "hybrid_arctic_embed_m_v2_minmax_lw030": Path("/bench/runs/minmax.trec"),
        "hybrid_arctic_embed_m_v2_rrf_k020": Path("/bench/runs/rrf.trec"),
    }


def test_sparse_significance_family_contains_sparse_only_and_hybrid() -> None:
    summary = {
        "selected_lexical_weight": 0.3,
        "artifacts": {
            "sparse/test": {"run": "runs/sparse.trec"},
            "hybrid/test/selected": {"run": "runs/sparse-hybrid.trec"},
        },
    }

    assert sparse_candidate_run_paths(Path("/bench"), summary) == {
        "neural_sparse_doc_only": Path("/bench/runs/sparse.trec"),
        "bm25_neural_sparse_minmax_lw030": Path("/bench/runs/sparse-hybrid.trec"),
    }


def test_family_eligibility_requires_literal_true_for_every_input() -> None:
    assert derive_family_eligibility(
        benchmark_provenance_valid=True,
        source_eligibilities=(True, True, True),
    )
    assert not derive_family_eligibility(
        benchmark_provenance_valid=1,
        source_eligibilities=(True, True, True),
    )
    assert not derive_family_eligibility(
        benchmark_provenance_valid=True,
        source_eligibilities=(True, "true", True),
    )


def test_registered_wands_evidence_inputs_reject_caller_threshold_override() -> None:
    experiments = load_bm25_experiments(ROOT / "config/experiments.toml")
    registry = load_model_registry(ROOT / "config/models.toml")
    models = tuple(registry[name] for name in WANDS_INT8_MODEL_NAMES)
    runtime = load_query_runtime_spec(ROOT / "config/query_runtime.toml")
    sparse = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")
    index = load_wands_index_config(ROOT / "config/indexes.toml").name

    with pytest.raises(DatasetIntegrityError, match="experiment configuration"):
        verify_registered_wands_evidence_inputs(
            root=ROOT,
            index=index,
            models=models,
            runtime=runtime,
            sparse=sparse,
            experiments=replace(experiments, alpha=0.01),
            metric=experiments.primary_metric,
            alpha=0.01,
            minimum_delta=experiments.minimum_paired_delta,
            minimum_judged_at_10=experiments.minimum_judged_at_10,
        )

    with pytest.raises(DatasetIntegrityError, match="thresholds differ"):
        verify_registered_wands_evidence_inputs(
            root=ROOT,
            index=index,
            models=models,
            runtime=runtime,
            sparse=sparse,
            experiments=experiments,
            metric=experiments.primary_metric,
            alpha=0.01,
            minimum_delta=experiments.minimum_paired_delta,
            minimum_judged_at_10=experiments.minimum_judged_at_10,
        )

    with pytest.raises(DatasetIntegrityError, match="thresholds differ"):
        verify_registered_wands_evidence_inputs(
            root=ROOT,
            index=index,
            models=models,
            runtime=runtime,
            sparse=sparse,
            experiments=experiments,
            metric=experiments.primary_metric,
            alpha=experiments.alpha,
            minimum_delta=experiments.minimum_paired_delta,
            minimum_judged_at_10=0.5,
        )


def test_live_source_collection_calls_every_semantic_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_verifier(name: str) -> Any:
        def verify(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            calls.append(name)
            return {"source": name}

        return verify

    monkeypatch.setattr(
        significance,
        "verify_wands_bm25_selection",
        fake_verifier("bm25"),
    )
    monkeypatch.setattr(
        significance,
        "verify_wands_hybrid_summary",
        fake_verifier("hybrid"),
    )
    monkeypatch.setattr(
        significance,
        "verify_wands_rrf_summary",
        fake_verifier("rrf"),
    )
    monkeypatch.setattr(
        significance,
        "verify_wands_sparse_summary",
        fake_verifier("sparse"),
    )

    result = collect_verified_wands_source_summaries(
        cast(Any, object()),
        root=Path("/bench"),
        index="wands",
        models=cast(Any, ()),
        runtime=cast(Any, object()),
        sparse=cast(Any, object()),
        experiments=cast(Any, object()),
    )

    assert calls == ["bm25", "hybrid", "rrf", "sparse"]
    assert set(result) == {"bm25", "hybrid", "rrf", "sparse"}


def test_apply_family_eligibility_treats_only_literal_true_as_gate_support() -> None:
    comparison: dict[str, Any] = {
        "eligible_for_decision": False,
        "comparisons": {
            "candidate": {
                "supports_quality_significance_gate": 1,
                "eligible_for_decision": False,
                "interpretation": "provisional_only_ineligible_runtime",
            }
        },
    }

    apply_family_eligibility(comparison, True)

    candidate = cast(dict[str, Any], comparison["comparisons"])["candidate"]
    assert candidate["eligible_for_decision"] is True
    assert candidate["interpretation"] == "does_not_pass_quality_significance_gate"


def test_source_evidence_rejects_summary_object_that_no_longer_matches_disk(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "results/wands/bm25-selection.json"
    run_path = tmp_path / "runs/source.trec"
    manifest_path = tmp_path / "runs/source.manifest.json"
    metrics_path = tmp_path / "results/wands/source.metrics.json"
    for path, payload in (
        (summary_path, '{"changed": true}\n'),
        (run_path, "q1 Q0 d1 1 1.0 source\n"),
        (manifest_path, '{"eligible_for_decision": true}\n'),
        (metrics_path, "{}\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
    verified_summary = {
        "quality_evidence_eligible_for_decision": True,
        "artifacts": {
            "tuned/test": {
                "run": "runs/source.trec",
                "manifest": "runs/source.manifest.json",
                "metrics_file": "results/wands/source.metrics.json",
            }
        },
    }

    with pytest.raises(DatasetIntegrityError, match="changed after verification"):
        significance._source_evidence(
            tmp_path,
            summary_path=summary_path,
            summary=verified_summary,
            artifact_key="tuned/test",
            summary_eligibility_key="quality_evidence_eligible_for_decision",
            semantic_verifier="verify_wands_bm25_selection",
        )
