from __future__ import annotations

from poc.robustness import (
    _assemble_wands_robustness_report,
    derive_robustness_decision_eligibility,
    map_qrels,
    rerank_within_judged,
)
from poc.trec import QrelRecord, RunRecord


def test_mapping_sensitivity_preserves_primary_and_builds_two_binary_views() -> None:
    qrels = [
        QrelRecord("q1", "exact", 2),
        QrelRecord("q1", "related", 1),
        QrelRecord("q1", "irrelevant", 0),
    ]

    assert map_qrels(qrels, mapping="primary", maximum_gain=2) == qrels
    assert map_qrels(qrels, mapping="strict_binary", maximum_gain=2) == [
        QrelRecord("q1", "exact", 1),
        QrelRecord("q1", "related", 0),
        QrelRecord("q1", "irrelevant", 0),
    ]
    assert map_qrels(qrels, mapping="broad_binary", maximum_gain=2) == [
        QrelRecord("q1", "exact", 1),
        QrelRecord("q1", "related", 1),
        QrelRecord("q1", "irrelevant", 0),
    ]


def test_judged_only_view_filters_and_reassigns_contiguous_ranks() -> None:
    qrels = [
        QrelRecord("q1", "a", 2),
        QrelRecord("q1", "c", 1),
        QrelRecord("q2", "z", 2),
    ]
    run = [
        RunRecord("q1", "a", 1, 3.0, "candidate"),
        RunRecord("q1", "b", 2, 2.0, "candidate"),
        RunRecord("q1", "c", 3, 1.0, "candidate"),
        RunRecord("q2", "x", 1, 2.0, "candidate"),
        RunRecord("q2", "z", 2, 1.0, "candidate"),
    ]

    filtered = rerank_within_judged(qrels, run)

    assert filtered == [
        RunRecord("q1", "a", 1, 3.0, "candidate"),
        RunRecord("q1", "c", 2, 1.0, "candidate"),
        RunRecord("q2", "z", 1, 1.0, "candidate"),
    ]


def test_mapping_rejects_a_declared_maximum_absent_from_qrels() -> None:
    qrels = [QrelRecord("q1", "a", 1)]

    try:
        map_qrels(qrels, mapping="strict_binary", maximum_gain=2)
    except ValueError as error:
        assert "maximum gain" in str(error)
    else:
        raise AssertionError("missing maximum gain should fail")


def test_robustness_decision_eligibility_is_separate_from_numerical_result() -> None:
    assert derive_robustness_decision_eligibility(
        benchmark_provenance_valid=True,
        source_family_eligible=True,
        judged_coverage_valid=True,
    )
    assert not derive_robustness_decision_eligibility(
        benchmark_provenance_valid=True,
        source_family_eligible=True,
        judged_coverage_valid=1,
    )
    assert not derive_robustness_decision_eligibility(
        benchmark_provenance_valid=True,
        source_family_eligible=False,
        judged_coverage_valid=True,
    )


def test_numerically_robust_report_stays_ineligible_when_a_source_is_ineligible() -> None:
    analysis = {
        "schema_version": 2,
        "all_headline_runs_meet_judged_at_10_floor": True,
        "headline_quality_conclusion_holds_across_all_mappings_and_views": True,
        "mappings": {
            "primary": {
                "full_corpus": {
                    "significance": {
                        "eligible_for_decision": False,
                        "comparisons": {
                            "candidate": {
                                "supports_quality_significance_gate": True,
                            }
                        },
                    }
                }
            }
        },
    }
    source_evidence = {
        "bm25_tuned": {"eligible_for_decision": True},
        "dense_minmax": {"eligible_for_decision": True},
        "dense_rrf": {"eligible_for_decision": True},
        "sparse_minmax": {"eligible_for_decision": False},
    }

    result = _assemble_wands_robustness_report(
        analysis=analysis,
        source_evidence=source_evidence,
        registered_inputs={},
        benchmark_provenance={},
        benchmark_provenance_valid=True,
        completion_revision={},
    )

    assert result["numerical_robustness_conclusion_holds"] is True
    assert result["eligible_for_decision"] is False
    assert result["decision_quality_conclusion_supported"] is False
    assert (
        result["decision_interpretation"]
        == "numerically_robust_but_ineligible_for_decision"
    )
