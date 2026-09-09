from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

BM25_FIELDS = (
    "title^5",
    "product_class^3",
    "category^3",
    "brand^2",
    "description",
    "features",
)


def build_bm25_query(query: str) -> dict[str, Any]:
    return build_configured_bm25_query(
        query,
        fields=BM25_FIELDS,
        query_type="best_fields",
        operator="or",
        tie_breaker=0.1,
    )


def build_configured_bm25_query(
    query: str,
    *,
    fields: tuple[str, ...],
    query_type: str,
    operator: str,
    tie_breaker: float,
) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("query must not be blank")
    return {
        "multi_match": {
            "query": query,
            "fields": list(fields),
            "type": query_type,
            "operator": operator,
            "tie_breaker": tie_breaker,
        }
    }


def build_integrator_bm25_query(
    query: str,
    *,
    fields: tuple[str, ...],
    operator: str,
    tie_breaker: float,
    minimum_should_match: str,
    phrase_fields: tuple[str, ...],
    prefix_fields: tuple[str, ...],
) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("query must not be blank")
    return {
        "bool": {
            "should": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": list(fields),
                        "type": "best_fields",
                        "operator": operator,
                        "tie_breaker": tie_breaker,
                        "minimum_should_match": minimum_should_match,
                        "fuzziness": "AUTO",
                        "prefix_length": 1,
                    }
                },
                {
                    "multi_match": {
                        "query": query,
                        "fields": list(phrase_fields),
                        "type": "phrase",
                        "slop": 1,
                        "boost": 2.0,
                    }
                },
                {
                    "multi_match": {
                        "query": query,
                        "fields": list(prefix_fields),
                        "type": "phrase_prefix",
                        "max_expansions": 50,
                    }
                },
            ],
            "minimum_should_match": 1,
        }
    }


def build_knn_query(vector: list[float], *, vector_field: str, k: int) -> dict[str, Any]:
    if not vector:
        raise ValueError("query vector must not be empty")
    if k <= 0:
        raise ValueError("k must be positive")
    return {"knn": {vector_field: {"vector": vector, "k": k}}}


def build_standalone_bm25_request(query: str, *, size: int = 10) -> dict[str, Any]:
    return {"size": size, "query": build_bm25_query(query)}


def build_standalone_knn_request(
    vector: list[float],
    *,
    vector_field: str,
    k: int = 10,
    ef_search: int | None = None,
) -> dict[str, Any]:
    query = build_knn_query(vector, vector_field=vector_field, k=k)
    if ef_search is not None:
        if isinstance(ef_search, bool) or ef_search < k:
            raise ValueError("ANN ef_search must be an integer at least as large as k")
        query["knn"][vector_field]["method_parameters"] = {
            "ef_search": ef_search
        }
    return {
        "size": k,
        "_source": {"excludes": [vector_field]},
        "query": query,
    }


def build_exact_knn_request(
    vector: list[float],
    *,
    vector_field: str,
    k: int = 100,
) -> dict[str, Any]:
    if not vector:
        raise ValueError("query vector must not be empty")
    if k <= 0:
        raise ValueError("k must be positive")
    return {
        "size": k,
        "_source": False,
        "track_scores": True,
        "sort": [
            {"_score": {"order": "desc"}},
            {"product_id": {"order": "asc"}},
        ],
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "knn_score",
                    "lang": "knn",
                    "params": {
                        "field": vector_field,
                        "query_value": vector,
                        "space_type": "cosinesimil",
                    },
                },
            }
        },
    }


def build_exact_hybrid_request(
    lexical_query: dict[str, Any],
    vector: list[float],
    *,
    vector_field: str,
    size: int = 100,
    pagination_depth: int = 100,
) -> dict[str, Any]:
    if not lexical_query:
        raise ValueError("hybrid lexical query must not be empty")
    if size <= 0:
        raise ValueError("hybrid result size must be positive")
    if pagination_depth < size:
        raise ValueError("hybrid pagination depth must be at least the result size")
    exact_dense = build_exact_knn_request(
        vector,
        vector_field=vector_field,
        k=pagination_depth,
    )["query"]
    return {
        "size": size,
        "_source": False,
        "track_scores": True,
        "query": {
            "hybrid": {
                "pagination_depth": pagination_depth,
                "queries": [deepcopy(lexical_query), exact_dense],
            }
        },
    }


def build_two_clause_hybrid_request(
    first_query: dict[str, Any],
    second_query: dict[str, Any],
    *,
    size: int = 100,
    pagination_depth: int = 100,
) -> dict[str, Any]:
    if not first_query or not second_query:
        raise ValueError("hybrid query clauses must not be empty")
    if size <= 0 or pagination_depth < size:
        raise ValueError("hybrid pagination depth must cover the requested result size")
    return {
        "size": size,
        "_source": False,
        "track_scores": True,
        "query": {
            "hybrid": {
                "pagination_depth": pagination_depth,
                "queries": [deepcopy(first_query), deepcopy(second_query)],
            }
        },
    }


def build_hybrid_request(
    query: str,
    vector: list[float],
    *,
    vector_field: str,
    k: int = 10,
    pagination_depth: int = 100,
) -> dict[str, Any]:
    lexical = build_bm25_query(query)
    dense = build_knn_query(vector, vector_field=vector_field, k=max(k, pagination_depth))
    return {
        "size": k,
        "_source": {"excludes": [vector_field]},
        "query": {
            "hybrid": {
                "pagination_depth": pagination_depth,
                "queries": [deepcopy(lexical), dense],
            }
        },
    }


def build_normalization_pipeline(*, lexical_weight: float = 0.5) -> dict[str, Any]:
    if not 0.0 <= lexical_weight <= 1.0:
        raise ValueError("lexical weight must be in [0, 1]")
    return build_weighted_normalization_pipeline(
        (lexical_weight, 1.0 - lexical_weight),
        description="OpenSearch Hybrid BM25+dense min_max fusion",
    )


def build_weighted_normalization_pipeline(
    weights: Sequence[float],
    *,
    description: str,
) -> dict[str, Any]:
    normalized = [float(weight) for weight in weights]
    if (
        len(normalized) < 2
        or not description
        or any(not math.isfinite(weight) or weight < 0.0 for weight in normalized)
        or not math.isclose(sum(normalized), 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("fusion weights must be non-negative and sum to one")
    return {
        "description": description,
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {"weights": normalized},
                    },
                }
            }
        ],
    }


def build_rrf_pipeline(*, rank_constant: int = 60) -> dict[str, Any]:
    if isinstance(rank_constant, bool) or not isinstance(rank_constant, int):
        raise ValueError("RRF rank constant must be an integer")
    if rank_constant < 1:
        raise ValueError("RRF rank constant must be at least 1")
    return {
        "description": "OpenSearch Hybrid BM25+dense reciprocal rank fusion",
        "phase_results_processors": [
            {
                "score-ranker-processor": {
                    "combination": {
                        "technique": "rrf",
                        "rank_constant": rank_constant,
                    }
                }
            }
        ],
    }


def hybrid_lexical_clause(request: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(request["query"]["hybrid"]["queries"][0])
