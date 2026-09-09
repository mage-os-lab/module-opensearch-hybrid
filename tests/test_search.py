from __future__ import annotations

from poc.search import (
    build_exact_hybrid_request,
    build_exact_knn_request,
    build_hybrid_request,
    build_normalization_pipeline,
    build_rrf_pipeline,
    build_standalone_bm25_request,
    build_standalone_knn_request,
    build_two_clause_hybrid_request,
    build_weighted_normalization_pipeline,
    hybrid_lexical_clause,
)


def test_ann_request_sets_explicit_ef_search() -> None:
    request = build_standalone_knn_request(
        [1.0, 0.0], vector_field="embedding", k=100, ef_search=400
    )

    assert request["query"]["knn"]["embedding"]["method_parameters"] == {
        "ef_search": 400
    }


def test_weighted_normalization_pipeline_requires_a_normalized_weight_vector() -> None:
    pipeline = build_weighted_normalization_pipeline(
        (0.3, 0.5, 0.2),
        description="BM25 plus dense plus sparse",
    )

    parameters = pipeline["phase_results_processors"][0][
        "normalization-processor"
    ]["combination"]["parameters"]
    assert parameters["weights"] == [0.3, 0.5, 0.2]

    try:
        build_weighted_normalization_pipeline((0.3, 0.5, 0.3), description="bad")
    except ValueError as error:
        assert "sum" in str(error)
    else:
        raise AssertionError("non-normalized weights should fail")


def test_hybrid_reuses_byte_equivalent_bm25_clause() -> None:
    standalone = build_standalone_bm25_request("trail running shoe")
    hybrid = build_hybrid_request(
        "trail running shoe", [1.0, 0.0], vector_field="embedding", pagination_depth=100
    )
    assert standalone["query"] == hybrid_lexical_clause(hybrid)


def test_pipeline_weights_sum_to_one() -> None:
    pipeline = build_normalization_pipeline(lexical_weight=0.3)
    processor = pipeline["phase_results_processors"][0]["normalization-processor"]
    assert processor["combination"]["parameters"]["weights"] == [0.3, 0.7]


def test_rrf_pipeline_uses_native_score_ranker_processor() -> None:
    assert build_rrf_pipeline(rank_constant=20) == {
        "description": "OpenSearch Hybrid BM25+dense reciprocal rank fusion",
        "phase_results_processors": [
            {
                "score-ranker-processor": {
                    "combination": {
                        "technique": "rrf",
                        "rank_constant": 20,
                    }
                }
            }
        ],
    }


def test_rrf_pipeline_rejects_invalid_rank_constant() -> None:
    for value in (0, -1, True):
        try:
            build_rrf_pipeline(rank_constant=value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid RRF rank constant {value!r}")


def test_exact_knn_request_uses_registered_opensearch_score_script() -> None:
    request = build_exact_knn_request([1.0, 0.0], vector_field="embedding", k=100)
    assert request["track_scores"] is True
    assert request["sort"] == [
        {"_score": {"order": "desc"}},
        {"product_id": {"order": "asc"}},
    ]
    script_score = request["query"]["script_score"]
    assert script_score["query"] == {"match_all": {}}
    assert script_score["script"] == {
        "source": "knn_score",
        "lang": "knn",
        "params": {
            "field": "embedding",
            "query_value": [1.0, 0.0],
            "space_type": "cosinesimil",
        },
    }


def test_exact_hybrid_request_reuses_lexical_clause_and_exact_dense_script() -> None:
    lexical = {
        "multi_match": {
            "query": "coffee table",
            "fields": ["title^8", "description"],
            "type": "best_fields",
            "operator": "or",
            "tie_breaker": 0.1,
        }
    }

    request = build_exact_hybrid_request(
        lexical,
        [1.0, 0.0],
        vector_field="embedding",
        size=100,
        pagination_depth=100,
    )

    hybrid = request["query"]["hybrid"]
    assert hybrid["queries"][0] == lexical
    assert hybrid["queries"][0] is not lexical
    assert (
        hybrid["queries"][1]
        == build_exact_knn_request([1.0, 0.0], vector_field="embedding", k=100)["query"]
    )
    assert hybrid["pagination_depth"] == 100
    assert request["size"] == 100
    assert request["_source"] is False


def test_two_clause_hybrid_request_preserves_sparse_query() -> None:
    lexical = {"match": {"title": "coffee table"}}
    sparse = {"neural_sparse": {"embedding": {"query_text": "coffee table"}}}

    request = build_two_clause_hybrid_request(
        lexical,
        sparse,
        size=100,
        pagination_depth=100,
    )

    assert request["query"]["hybrid"]["queries"] == [lexical, sparse]
    assert request["query"]["hybrid"]["queries"][0] is not lexical
    assert request["query"]["hybrid"]["queries"][1] is not sparse
