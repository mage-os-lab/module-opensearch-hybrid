from __future__ import annotations

from pathlib import Path

from poc.experiments import load_bm25_experiments
from poc.search import build_bm25_query

ROOT = Path(__file__).resolve().parents[1]


def test_bm25_tuning_budget_matches_hybrid_and_preserves_magento_clause() -> None:
    experiments = load_bm25_experiments(ROOT / "config/experiments.toml")
    assert len(experiments.tuning_candidates) == experiments.hybrid_weight_count == 5
    assert experiments.lexical_weight_grid == (0.2, 0.3, 0.4, 0.5, 0.6)
    assert experiments.rrf_rank_constants == (1, 5, 10, 20, 60)
    assert len(experiments.rrf_rank_constants) == len(experiments.tuning_candidates)
    assert experiments.fusion_normalization == "min_max"
    assert experiments.fusion_combination == "arithmetic_mean"
    assert experiments.primary_metric == "ndcg@10"
    assert experiments.minimum_paired_delta == 0.02
    assert experiments.alpha == 0.05
    assert experiments.multiple_testing_correction == "holm"
    assert experiments.maximum_trec_orphan_rate == 0.01
    assert experiments.minimum_judged_at_10 == 0.8
    assert experiments.minimum_ann_recall_at_100 == 0.98
    assert experiments.ann_ef_search_grid == (100, 200, 400, 800, 1600)
    assert experiments.tuning_candidates[0] == experiments.magento_emulation
    assert experiments.magento_emulation.query("coffee table") == build_bm25_query("coffee table")


def test_competent_integrator_arm_uses_stemming_fuzziness_phrase_and_prefix() -> None:
    experiments = load_bm25_experiments(ROOT / "config/experiments.toml")

    assert experiments.competent_integrator.query("running shoe") == {
        "bool": {
            "should": [
                {
                    "multi_match": {
                        "query": "running shoe",
                        "fields": [
                            "title.magento^6",
                            "product_class.magento^4",
                            "category.magento^4",
                            "brand.magento^2",
                            "description.magento",
                            "features.magento^2",
                        ],
                        "type": "best_fields",
                        "operator": "or",
                        "tie_breaker": 0.1,
                        "minimum_should_match": "2<75%",
                        "fuzziness": "AUTO",
                        "prefix_length": 1,
                    }
                },
                {
                    "multi_match": {
                        "query": "running shoe",
                        "fields": ["title.magento^8", "brand.magento^2"],
                        "type": "phrase",
                        "slop": 1,
                        "boost": 2.0,
                    }
                },
                {
                    "multi_match": {
                        "query": "running shoe",
                        "fields": ["title.prefix^4", "brand.prefix^2"],
                        "type": "phrase_prefix",
                        "max_expansions": 50,
                    }
                },
            ],
            "minimum_should_match": 1,
        }
    }
