from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from poc.bm25 import (
    load_prepared_queries,
    verify_wands_bm25_selection,
)
from poc.datasets import (
    DatasetIntegrityError,
    file_facts,
    load_wands_config,
)
from poc.experiments import (
    BM25Profile,
    load_bm25_experiments,
)
from poc.indexing import (
    load_wands_index_config,
    verify_wands_lexical_index,
    wands_completion_provenance_evidence,
    wands_recorded_provenance_valid,
)
from poc.manifest import (
    canonical_sha256,
    collect_index_facts,
    read_json,
    write_json,
)
from poc.neural_sparse import (
    NeuralSparseSpec,
    build_sparse_query,
    load_neural_sparse_spec,
)
from poc.os_client import OpenSearchClient
from poc.provenance import collect_manifest_provenance, require_registered_opensearch_client
from poc.search import build_normalization_pipeline, build_two_clause_hybrid_request
from poc.sku_slice import load_sku_slice_spec
from poc.sku_slice_indexing import verify_sku_slice_index

PIPELINE_ID = "opensearch-hybrid-wands-facet-semantics-lw030-v1"
FACET_FIELDS = ("brand", "category", "product_class")
FACET_LEXICAL_WEIGHT = 0.3
FACET_QUERY_COUNT = 3
FACET_SCHEMA_VERSION = 2
FACET_DATASET = "WANDS held-out queries on synthetic SKU proxy index"
FACET_METHOD = "hybrid_ranking_with_separate_lexical_aggregation_spot_check"


def _json_dict(value: dict[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(json.dumps(value)))


def build_facet_aggregations() -> dict[str, Any]:
    return {
        field: {"terms": {"field": f"{field}.facet", "size": 20}}
        for field in FACET_FIELDS
    }


def build_facet_requests(
    text: str,
    *,
    profile: BM25Profile,
    sparse: NeuralSparseSpec,
    active_filter: tuple[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    aggregations = build_facet_aggregations()
    hybrid = build_two_clause_hybrid_request(
        _apply_catalog_filter(profile.query(text), active_filter),
        _apply_catalog_filter(build_sparse_query(sparse, text), active_filter),
        size=10,
        pagination_depth=100,
    )
    hybrid["track_total_hits"] = True
    hybrid["aggs"] = aggregations
    lexical = {
        "size": 0,
        "track_total_hits": True,
        "query": _apply_catalog_filter(profile.query(text), active_filter),
        "aggs": aggregations,
    }
    return hybrid, lexical


def facet_method_config(
    *,
    profile: BM25Profile,
    sparse: NeuralSparseSpec,
) -> dict[str, Any]:
    pipeline = build_normalization_pipeline(
        lexical_weight=FACET_LEXICAL_WEIGHT
    )
    return _json_dict(
        {
            "bm25_profile": profile.to_dict(),
            "neural_sparse_spec": asdict(sparse),
            "pipeline_id": PIPELINE_ID,
            "pipeline_definition": pipeline,
            "pipeline_sha256": canonical_sha256(pipeline),
            "lexical_weight": FACET_LEXICAL_WEIGHT,
            "result_size": 10,
            "pagination_depth": 100,
            "request_cache": False,
            "facet_fields": list(FACET_FIELDS),
            "facet_aggregations": build_facet_aggregations(),
            "query_selection_count": FACET_QUERY_COUNT,
            "count_strategy": "separate_lexical_aggregation_leg",
            "count_equivalence_claimed": False,
        }
    )


def facet_evidence_status(
    *,
    provenance_valid: object,
    upstream_eligible: object,
    upstream_unchanged: object,
) -> dict[str, bool]:
    return {
        "diagnostic_reproducible": all(
            value is True
            for value in (
                provenance_valid,
                upstream_eligible,
                upstream_unchanged,
            )
        ),
        "quality_evidence_eligible_for_decision": False,
        "latency_evidence_eligible_for_decision": False,
    }


def verify_registered_facet_configuration(
    *,
    root: Path,
    index: str,
    profile: BM25Profile,
    sparse: NeuralSparseSpec,
) -> dict[str, Any]:
    dataset = load_wands_config(root / "config/datasets.toml")
    experiments = load_bm25_experiments(root / "config/experiments.toml")
    wands_index = load_wands_index_config(root / "config/indexes.toml")
    registered_sparse = load_neural_sparse_spec(
        root / "config/neural_sparse.toml"
    )
    sku = load_sku_slice_spec(root / "config/sku_slice.toml")
    selection_path = root / "results/wands/bm25-selection.json"
    selection = cast(dict[str, Any], read_json(selection_path))
    selected_profile = selection.get("selected_profile")
    selected_profile_sha256 = canonical_sha256(selected_profile)
    if (
        index != sku.index_name
        or sparse != registered_sparse
        or sku.lexical_weight != FACET_LEXICAL_WEIGHT
        or selection.get("schema_version") != 2
        or selection.get("dataset") != "WANDS"
        or not isinstance(selected_profile, Mapping)
        or _json_dict(profile.to_dict()) != dict(selected_profile)
        or selection.get("selected_profile_sha256")
        != selected_profile_sha256
    ):
        raise DatasetIntegrityError("facet registered configuration differs")
    return _json_dict(
        {
            "dataset": asdict(dataset),
            "wands_index": asdict(wands_index),
            "sku_slice": asdict(sku),
            "neural_sparse": asdict(registered_sparse),
            "bm25_experiments": asdict(experiments),
            "selected_bm25_profile": profile.to_dict(),
            "selected_bm25_profile_sha256": selected_profile_sha256,
            "bm25_selection": _artifact(root, selection_path),
            "config_artifacts": {
                name: _artifact(root, root / f"config/{filename}")
                for name, filename in (
                    ("datasets", "datasets.toml"),
                    ("experiments", "experiments.toml"),
                    ("indexes", "indexes.toml"),
                    ("neural_sparse", "neural_sparse.toml"),
                    ("sku_slice", "sku_slice.toml"),
                )
            },
        }
    )


def run_facet_semantics(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    profile: BM25Profile,
    sparse: NeuralSparseSpec,
) -> dict[str, object]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    benchmark_provenance = collect_manifest_provenance(
        root,
        profile_path=root / "config/benchmark.toml",
        environment_path=root / "results/environment/benchmark-profile.json",
    )
    client.wait_until_ready(expected_version="3.8.0")
    registered_configuration = verify_registered_facet_configuration(
        root=root,
        index=index,
        profile=profile,
        sparse=sparse,
    )
    upstream_evidence = _facet_upstream_evidence(
        client,
        root=root,
        index=index,
        profile=profile,
        sparse=sparse,
    )
    method_config = facet_method_config(profile=profile, sparse=sparse)
    pipeline = cast(dict[str, Any], method_config["pipeline_definition"])
    client.put_search_pipeline(PIPELINE_ID, pipeline)
    _verify_live_facet_pipeline(client, pipeline)
    observations = _execute_facet_observations(
        client,
        root=root,
        index=index,
        profile=profile,
        sparse=sparse,
    )
    completion_upstream_evidence = _facet_upstream_evidence(
        client,
        root=root,
        index=index,
        profile=profile,
        sparse=sparse,
    )
    upstream_unchanged = upstream_evidence == completion_upstream_evidence
    provenance_valid, completion_revision = wands_completion_provenance_evidence(
        root,
        benchmark_provenance,
    )
    _verify_live_facet_pipeline(client, pipeline)
    result = _facet_result(
        client,
        index=index,
        method_config=method_config,
        registered_configuration=registered_configuration,
        observations=observations,
        upstream_evidence=upstream_evidence,
        completion_upstream_evidence=completion_upstream_evidence,
        upstream_unchanged=upstream_unchanged,
        benchmark_provenance=benchmark_provenance,
        completion_revision=completion_revision,
        provenance_valid=provenance_valid,
    )
    write_json(root / "results/wands/facet-semantics.json", result)
    return result


def verify_facet_semantics(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    profile: BM25Profile,
    sparse: NeuralSparseSpec,
) -> dict[str, Any]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    client.wait_until_ready(expected_version="3.8.0")
    path = root / "results/wands/facet-semantics.json"
    recorded = cast(dict[str, Any], read_json(path))
    registered_configuration = verify_registered_facet_configuration(
        root=root,
        index=index,
        profile=profile,
        sparse=sparse,
    )
    upstream_evidence = _facet_upstream_evidence(
        client,
        root=root,
        index=index,
        profile=profile,
        sparse=sparse,
    )
    if (
        recorded.get("upstream_evidence") != upstream_evidence
        or recorded.get("completion_upstream_evidence") != upstream_evidence
        or recorded.get("upstream_evidence_unchanged") is not True
    ):
        raise DatasetIntegrityError("facet upstream evidence differs")
    if not isinstance(recorded.get("benchmark_provenance"), Mapping) or not isinstance(
        recorded.get("completion_code_revision"), Mapping
    ):
        raise DatasetIntegrityError("facet provenance evidence is missing")
    provenance_valid = wands_recorded_provenance_valid(
        root,
        provenance=recorded.get("benchmark_provenance"),
        completion_revision=recorded.get("completion_code_revision"),
    )
    method_config = facet_method_config(profile=profile, sparse=sparse)
    _verify_live_facet_pipeline(
        client,
        cast(dict[str, Any], method_config["pipeline_definition"]),
    )
    observations = _execute_facet_observations(
        client,
        root=root,
        index=index,
        profile=profile,
        sparse=sparse,
    )
    expected = _facet_result(
        client,
        index=index,
        method_config=method_config,
        registered_configuration=registered_configuration,
        observations=observations,
        upstream_evidence=upstream_evidence,
        completion_upstream_evidence=upstream_evidence,
        upstream_unchanged=True,
        benchmark_provenance=cast(
            dict[str, Any], recorded.get("benchmark_provenance")
        ),
        completion_revision=cast(
            dict[str, Any], recorded.get("completion_code_revision")
        ),
        provenance_valid=provenance_valid,
    )
    if recorded != expected:
        raise DatasetIntegrityError("facet semantics artifact does not reproduce")
    return recorded


def _execute_facet_observations(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    profile: BM25Profile,
    sparse: NeuralSparseSpec,
) -> dict[str, object]:
    query_path = root / "data/prepared/wands/queries.test.jsonl"
    queries = load_prepared_queries(query_path, expected_split="test")
    selected = sorted(queries.items())[:FACET_QUERY_COUNT]
    global_request = {
        "size": 0,
        "track_total_hits": True,
        "query": {"match_all": {}},
        "aggs": build_facet_aggregations(),
    }
    global_counts = _count_view(
        client.search(index, global_request, request_cache=False)
    )
    active_filter = _top_category_filter(global_counts)
    examples: list[dict[str, object]] = []
    for query_id, text in selected:
        hybrid_request, lexical_request = build_facet_requests(
            text,
            profile=profile,
            sparse=sparse,
        )
        hybrid = client.search(
            index,
            hybrid_request,
            pipeline=PIPELINE_ID,
            request_cache=False,
        )
        lexical = client.search(index, lexical_request, request_cache=False)
        hybrid_counts = _count_view(hybrid)
        lexical_counts = _count_view(lexical)
        filtered_hybrid_request, filtered_lexical_request = build_facet_requests(
            text,
            profile=profile,
            sparse=sparse,
            active_filter=active_filter,
        )
        filtered_hybrid = client.search(
            index,
            filtered_hybrid_request,
            pipeline=PIPELINE_ID,
            request_cache=False,
        )
        filtered_lexical = client.search(
            index,
            filtered_lexical_request,
            request_cache=False,
        )
        examples.append(
            {
                "query_id": query_id,
                "query": text,
                "request_sha256": {
                    "hybrid": canonical_sha256(hybrid_request),
                    "separate_lexical_aggregation_leg": canonical_sha256(
                        lexical_request
                    ),
                    "filtered_hybrid": canonical_sha256(
                        filtered_hybrid_request
                    ),
                    "filtered_separate_lexical_aggregation_leg": canonical_sha256(
                        filtered_lexical_request
                    ),
                },
                "unfiltered": {
                    "hybrid": hybrid_counts,
                    "separate_lexical_aggregation_leg": lexical_counts,
                },
                "active_filter": {
                    "field": active_filter[0],
                    "value": active_filter[1],
                },
                "filtered": {
                    "hybrid": _count_view(filtered_hybrid),
                    "separate_lexical_aggregation_leg": _count_view(filtered_lexical),
                },
            }
        )
    return {
        "queries": {
            "source": _artifact(root, query_path),
            "selection": "first three held-out query IDs in lexical sort order",
            "selected": [
                {"query_id": query_id, "query": text}
                for query_id, text in selected
            ],
        },
        "global_count_request_sha256": canonical_sha256(global_request),
        "active_filter_selection": {
            "source": "top non-empty global category bucket",
            "field": active_filter[0],
            "value": active_filter[1],
        },
        "examples": examples,
    }


def _facet_result(
    client: OpenSearchClient,
    *,
    index: str,
    method_config: dict[str, object],
    registered_configuration: dict[str, object],
    observations: dict[str, object],
    upstream_evidence: dict[str, object],
    completion_upstream_evidence: dict[str, object],
    upstream_unchanged: bool,
    benchmark_provenance: dict[str, Any],
    completion_revision: dict[str, Any],
    provenance_valid: bool,
) -> dict[str, object]:
    evidence_status = facet_evidence_status(
        provenance_valid=provenance_valid,
        upstream_eligible=upstream_evidence.get("eligible_for_diagnostic"),
        upstream_unchanged=upstream_unchanged,
    )
    diagnostic_reasons: list[str] = []
    if not provenance_valid:
        diagnostic_reasons.append(
            "facet diagnostic has no valid clean committed start provenance"
        )
    if upstream_evidence.get("eligible_for_diagnostic") is not True:
        diagnostic_reasons.append("facet semantic upstream verification is ineligible")
    if not upstream_unchanged:
        diagnostic_reasons.append("facet upstream evidence changed during execution")
    return {
        "schema_version": FACET_SCHEMA_VERSION,
        "dataset": FACET_DATASET,
        "method": FACET_METHOD,
        "classification": "count_semantics_spot_check_not_relevance_evidence",
        "active_catalog_filter_tested": True,
        "counts_and_total_hits_excluded_from_relevance_evaluation": True,
        "count_equivalence_claimed": False,
        "quality_evidence_eligible_for_decision": evidence_status[
            "quality_evidence_eligible_for_decision"
        ],
        "latency_evidence_eligible_for_decision": evidence_status[
            "latency_evidence_eligible_for_decision"
        ],
        "diagnostic_reproducible": evidence_status["diagnostic_reproducible"],
        "diagnostic_ineligibility_reasons": diagnostic_reasons,
        "index": asdict(collect_index_facts(client, index)),
        "registered_configuration": registered_configuration,
        "method_config": method_config,
        "method_config_sha256": canonical_sha256(method_config),
        "pipeline": {
            "id": PIPELINE_ID,
            "definition_sha256": method_config["pipeline_sha256"],
            "live_definition_verified": True,
        },
        **observations,
        "upstream_evidence": upstream_evidence,
        "completion_upstream_evidence": completion_upstream_evidence,
        "upstream_evidence_unchanged": upstream_unchanged,
        "benchmark_provenance": benchmark_provenance,
        "completion_code_revision": completion_revision,
        "benchmark_provenance_valid": provenance_valid,
        "module_answer": {
            "initial_strategy": (
                "issue a separate lexical aggregation request with the same active "
                "catalog filters while hybrid retrieval controls product ordering"
            ),
            "known_tradeoff": (
                "semantic-only candidates can appear in the ranked products without "
                "contributing to lexical facet counts"
            ),
            "graduation_requirement": (
                "validate the separate aggregation leg against Magento layered-navigation "
                "filter semantics on a real catalog before core graduation"
            ),
        },
    }


def _facet_upstream_evidence(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    profile: BM25Profile,
    sparse: NeuralSparseSpec,
) -> dict[str, object]:
    dataset = load_wands_config(root / "config/datasets.toml")
    experiments = load_bm25_experiments(root / "config/experiments.toml")
    wands_index_spec = load_wands_index_config(root / "config/indexes.toml")
    sku_spec = load_sku_slice_spec(root / "config/sku_slice.toml")
    wands_index = verify_wands_lexical_index(
        client,
        dataset_spec=dataset,
        index_spec=wands_index_spec,
        prepared_directory=root / "data/prepared/wands",
        manifest_path=root / "results/wands/index-manifest.json",
    )
    bm25 = verify_wands_bm25_selection(
        client,
        root=root,
        index=wands_index_spec.name,
        experiments=experiments,
    )
    if bm25.get("selected_profile") != _json_dict(profile.to_dict()):
        raise DatasetIntegrityError("facet BM25 upstream profile differs")
    sku_index = verify_sku_slice_index(
        client,
        root=root,
        dataset_spec=dataset,
        sparse_spec=sparse,
        sku_spec=sku_spec,
        products_path=root / "data/prepared/wands/products.jsonl",
        embeddings_path=(
            root / "data/cache/neural-sparse/wands-doc-v3-distill.jsonl"
        ),
        preparation_path=root / "results/wands/sku-slice/preparation.json",
        precompute_manifest_path=(
            root / "results/wands/neural-sparse/precompute-manifest.json"
        ),
        manifest_path=root / "results/wands/sku-slice/index-manifest.json",
    )
    if index != sku_spec.index_name:
        raise DatasetIntegrityError("facet live index differs from registered SKU index")
    return {
        "wands_preparation_manifest": _artifact(
            root, root / "data/prepared/wands/manifest.json"
        ),
        "wands_index_manifest": _artifact(
            root, root / "results/wands/index-manifest.json"
        ),
        "bm25_selection": _artifact(
            root, root / "results/wands/bm25-selection.json"
        ),
        "sku_preparation": _artifact(
            root, root / "results/wands/sku-slice/preparation.json"
        ),
        "sku_index_manifest": _artifact(
            root, root / "results/wands/sku-slice/index-manifest.json"
        ),
        "held_out_query_file": _artifact(
            root, root / "data/prepared/wands/queries.test.jsonl"
        ),
        "wands_index_quality_evidence_eligible_for_decision": wands_index.get(
            "quality_evidence_eligible_for_decision"
        )
        is True,
        "bm25_quality_evidence_eligible_for_decision": bm25.get(
            "quality_evidence_eligible_for_decision"
        )
        is True,
        "sku_index_quality_evidence_eligible_for_decision": sku_index.get(
            "quality_evidence_eligible_for_decision"
        )
        is True,
        "semantic_verification_passed": True,
        "eligible_for_diagnostic": True,
    }


def _verify_live_facet_pipeline(
    client: OpenSearchClient,
    expected: dict[str, Any],
) -> None:
    response = cast(
        dict[str, Any],
        client.request("GET", f"/_search/pipeline/{PIPELINE_ID}"),
    )
    if response.get(PIPELINE_ID) != expected:
        raise DatasetIntegrityError("live facet search pipeline differs")


def _apply_catalog_filter(
    query: dict[str, Any], active_filter: tuple[str, str] | None
) -> dict[str, Any]:
    if active_filter is None:
        return query
    field, value = active_filter
    if field not in FACET_FIELDS or not value:
        raise ValueError("facet active filter is invalid")
    return {
        "bool": {
            "must": [query],
            "filter": [{"term": {f"{field}.facet": value}}],
        }
    }


def _top_category_filter(counts: dict[str, object]) -> tuple[str, str]:
    facets = cast(dict[str, Any], counts["facets"])
    buckets = cast(list[dict[str, Any]], facets["category"])
    for bucket in buckets:
        value = str(bucket.get("key", ""))
        if value:
            return "category", value
    raise DatasetIntegrityError("facet response has no category for filter spot-check")


def _count_view(response: dict[str, Any]) -> dict[str, object]:
    hits = cast(dict[str, Any], response.get("hits", {}))
    total = hits.get("total")
    if not isinstance(total, dict):
        raise DatasetIntegrityError("facet response has no exact total-hits object")
    aggregations = response.get("aggregations")
    if not isinstance(aggregations, dict):
        raise DatasetIntegrityError("facet response has no aggregations")
    facets: dict[str, object] = {}
    for field in FACET_FIELDS:
        aggregation = aggregations.get(field)
        buckets = aggregation.get("buckets") if isinstance(aggregation, dict) else None
        if not isinstance(buckets, list):
            raise DatasetIntegrityError(f"facet response has no {field} buckets")
        facets[field] = [
            {"key": str(bucket["key"]), "doc_count": int(bucket["doc_count"])}
            for bucket in buckets
            if isinstance(bucket, dict)
        ]
    return {
        "total_hits": {
            "value": int(total.get("value", -1)),
            "relation": str(total.get("relation", "")),
        },
        "facets": facets,
    }


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }
