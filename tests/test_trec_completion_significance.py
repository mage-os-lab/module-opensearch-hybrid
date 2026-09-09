from __future__ import annotations

from poc.trec_benchmark import (
    derive_trec_evidence_eligibility,
    trec_revision_chain_matches,
)
from poc.trec_sparse_benchmark import (
    _assemble_trec_sparse_significance,
    derive_trec_significance_family_eligibility,
)


def test_trec_evidence_eligibility_rejects_completion_and_upstream_drift() -> None:
    upstream = {"eligible_for_decision": True, "artifact": {"sha256": "a" * 64}}

    assert derive_trec_evidence_eligibility(
        provenance_valid=True,
        start_upstream=upstream,
        completion_upstream=upstream,
        source_eligibilities=(True, True),
    )
    assert not derive_trec_evidence_eligibility(
        provenance_valid=True,
        start_upstream=upstream,
        completion_upstream={**upstream, "artifact": {"sha256": "b" * 64}},
        source_eligibilities=(True, True),
    )
    assert not derive_trec_evidence_eligibility(
        provenance_valid="True",
        start_upstream=upstream,
        completion_upstream=upstream,
        source_eligibilities=(True, True),
    )
    assert not derive_trec_evidence_eligibility(
        provenance_valid=True,
        start_upstream={**upstream, "eligible_for_decision": "True"},
        completion_upstream={**upstream, "eligible_for_decision": "True"},
        source_eligibilities=(True, True),
    )


def test_trec_revision_chain_requires_exact_current_completion() -> None:
    revision = {
        "git_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_dirty": False,
    }

    assert trec_revision_chain_matches(
        start_revision=revision,
        completion_revision=revision,
        current_revision=revision,
    )
    assert not trec_revision_chain_matches(
        start_revision=revision,
        completion_revision={**revision, "source_tree_sha256": "c" * 64},
        current_revision=revision,
    )
    assert not trec_revision_chain_matches(
        start_revision=revision,
        completion_revision=revision,
        current_revision={**revision, "source_dirty": "False"},
    )


def test_trec_significance_family_cannot_promote_string_eligibility() -> None:
    assert derive_trec_significance_family_eligibility(
        provenance_valid=True,
        upstream_unchanged=True,
        source_eligibilities=(True, True, True),
    )
    assert not derive_trec_significance_family_eligibility(
        provenance_valid="True",
        upstream_unchanged=True,
        source_eligibilities=(True, True, True),
    )
    assert not derive_trec_significance_family_eligibility(
        provenance_valid=True,
        upstream_unchanged="True",
        source_eligibilities=(True, True, True),
    )
    assert not derive_trec_significance_family_eligibility(
        provenance_valid=True,
        upstream_unchanged=True,
        source_eligibilities=(True, "True", True),
    )


def test_trec_significance_assembly_rejects_promotion_and_source_tamper() -> None:
    comparisons = {
        name: {
            "schema_version": 1,
            "dataset": "TREC Product Search 2024",
            "split": "held_out_test",
            "eligible_for_decision": False,
            "comparisons": {
                "candidate": {
                    "supports_quality_significance_gate": True,
                    "eligible_for_decision": False,
                    "interpretation": "provisional_only_ineligible_runtime",
                }
            },
        }
        for name in ("primary", "post_review_challenge")
    }
    source_names = (
        "bm25_tuned",
        "bm25_competent_integrator",
        "sparse_only",
        "sparse_hybrid",
    )
    sources = {
        name: {
            "eligible_for_decision": True,
            "files": {"run": {"sha256": name}},
        }
        for name in source_names
    }
    revision = {
        "git_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_dirty": False,
    }
    qrels = {"path": "qrels", "sha256": "c" * 64, "bytes": 1}

    eligible = _assemble_trec_sparse_significance(
        comparisons=comparisons,
        source_evidence=sources,
        completion_source_evidence=sources,
        qrels_artifact=qrels,
        completion_qrels_artifact=qrels,
        registered_inputs={"decision_thresholds": {}},
        benchmark_provenance={"code_revision": revision},
        provenance_valid=True,
        completion_revision=revision,
    )
    assert eligible["schema_version"] == 2
    assert eligible["eligible_for_decision"] is True
    assert eligible["source_evidence_unchanged_at_completion"] is True

    promoted = {name: dict(value) for name, value in sources.items()}
    promoted["sparse_hybrid"]["eligible_for_decision"] = "True"
    promotion_result = _assemble_trec_sparse_significance(
        comparisons=comparisons,
        source_evidence=promoted,
        completion_source_evidence=promoted,
        qrels_artifact=qrels,
        completion_qrels_artifact=qrels,
        registered_inputs={"decision_thresholds": {}},
        benchmark_provenance={"code_revision": revision},
        provenance_valid=True,
        completion_revision=revision,
    )
    assert promotion_result["eligible_for_decision"] is False

    completion_sources = {name: dict(value) for name, value in sources.items()}
    completion_sources["sparse_only"] = {
        **completion_sources["sparse_only"],
        "files": {"run": {"sha256": "tampered"}},
    }
    tampered = _assemble_trec_sparse_significance(
        comparisons=comparisons,
        source_evidence=sources,
        completion_source_evidence=completion_sources,
        qrels_artifact=qrels,
        completion_qrels_artifact=qrels,
        registered_inputs={"decision_thresholds": {}},
        benchmark_provenance={"code_revision": revision},
        provenance_valid=True,
        completion_revision=revision,
    )
    assert tampered["eligible_for_decision"] is False
    assert tampered["source_evidence_unchanged_at_completion"] is False
