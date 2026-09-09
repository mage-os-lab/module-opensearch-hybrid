from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from poc.datasets import DatasetIntegrityError, file_facts, load_wands_config
from poc.evaluation import evaluate_run
from poc.evaluation_crosscheck import cross_check_run
from poc.hybrid import hybrid_trec_score_for_rank
from poc.index_evidence import (
    content_addressed_pipeline_id,
    verify_live_search_pipeline,
)
from poc.indexing import (
    wands_completion_provenance_evidence,
    wands_recorded_provenance_valid,
)
from poc.manifest import canonical_sha256, collect_index_facts, read_json, write_json
from poc.neural_sparse import (
    NeuralSparseSpec,
    build_sparse_query,
    load_neural_sparse_spec,
)
from poc.os_client import OpenSearchClient
from poc.provenance import (
    collect_manifest_provenance,
    require_registered_opensearch_client,
    verify_decision_provenance,
)
from poc.search import build_normalization_pipeline, build_two_clause_hybrid_request
from poc.sku_slice import (
    SkuSliceSpec,
    assess_known_item_regression,
    build_known_item_lexical_query,
    known_item_route,
    load_sku_queries,
    load_sku_slice_spec,
)
from poc.sku_slice_indexing import verify_sku_slice_index
from poc.trec import RunRecord, identifier_sort_key, read_run, write_run

RESULT_SIZE = 10
PAGINATION_DEPTH = 100
PIPELINE_DEFINITION = build_normalization_pipeline(lexical_weight=0.3)
PIPELINE_ID = content_addressed_pipeline_id(
    "opensearch-hybrid-wands-sku-sparse-minmax-lw030-v1",
    PIPELINE_DEFINITION,
)


def run_sku_slice_benchmark(
    client: OpenSearchClient,
    *,
    root: Path,
    sparse_spec: NeuralSparseSpec,
    sku_spec: SkuSliceSpec,
) -> dict[str, object]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    verify_registered_sku_benchmark_inputs(
        root=root,
        sparse_spec=sparse_spec,
        sku_spec=sku_spec,
    )
    benchmark_provenance = collect_manifest_provenance(
        root,
        profile_path=root / "config/benchmark.toml",
        environment_path=root / "results/environment/benchmark-profile.json",
    )
    upstream_evidence = _sku_upstream_evidence(
        client,
        root=root,
        sparse_spec=sparse_spec,
        sku_spec=sku_spec,
    )
    query_path = root / "data/prepared/wands/queries.sku-slice.jsonl"
    qrels_path = root / "data/prepared/wands/qrels.sku-slice.trec"
    queries = load_sku_queries(query_path)
    baseline_tag = "wands-synthetic-sku-lexical"
    baseline_records = execute_known_item_lexical(
        client,
        index=sku_spec.index_name,
        queries=queries,
        tag=baseline_tag,
    )
    pipeline = build_normalization_pipeline(
        lexical_weight=sku_spec.lexical_weight
    )
    client.put_search_pipeline(PIPELINE_ID, pipeline)
    verify_live_search_pipeline(
        client,
        pipeline_id=PIPELINE_ID,
        expected_definition=pipeline,
    )
    raw_hybrid_tag = "wands-synthetic-sku-bm25-sparse-minmax-lw030"
    raw_hybrid_records = execute_known_item_sparse_hybrid(
        client,
        index=sku_spec.index_name,
        queries=queries,
        sparse_spec=sparse_spec,
        sku_spec=sku_spec,
        tag=raw_hybrid_tag,
    )
    guarded_tag = "wands-synthetic-sku-guarded-bm25-sparse-minmax-lw030"
    guarded_records, routing = execute_guarded_known_item_sparse_hybrid(
        client,
        index=sku_spec.index_name,
        queries=queries,
        sparse_spec=sparse_spec,
        sku_spec=sku_spec,
        tag=guarded_tag,
    )
    completion_upstream_evidence = _sku_upstream_evidence(
        client,
        root=root,
        sparse_spec=sparse_spec,
        sku_spec=sku_spec,
    )
    provenance_valid, completion_code_revision = (
        wands_completion_provenance_evidence(root, benchmark_provenance)
    )
    evidence_eligible, evidence_reasons = derive_sku_benchmark_eligibility(
        provenance_valid=provenance_valid,
        start_upstream_eligible=upstream_evidence.get("eligible_for_decision"),
        completion_upstream_eligible=completion_upstream_evidence.get(
            "eligible_for_decision"
        ),
        upstream_unchanged=(
            upstream_evidence == completion_upstream_evidence
        ),
    )
    method_configs = _sku_method_configs(
        sparse_spec=sparse_spec,
        sku_spec=sku_spec,
        pipeline=pipeline,
        routing=routing,
    )
    bundle_inputs = {
        "baseline": (
            baseline_tag,
            "known_item_lexical",
            method_configs["baseline"],
            baseline_records,
        ),
        "unprotected_hybrid": (
            raw_hybrid_tag,
            "known_item_bm25_neural_sparse_unprotected_diagnostic",
            method_configs["unprotected_hybrid"],
            raw_hybrid_records,
        ),
        "guarded_hybrid": (
            guarded_tag,
            "known_item_guarded_bm25_neural_sparse",
            method_configs["guarded_hybrid"],
            guarded_records,
        ),
    }
    artifacts: dict[str, dict[str, object]] = {}
    for name, (tag, method, method_config, records) in bundle_inputs.items():
        artifacts[name] = write_sku_bundle(
            client,
            root=root,
            sku_spec=sku_spec,
            query_path=query_path,
            qrels_path=qrels_path,
            tag=tag,
            method=method,
            method_config=method_config,
            records=records,
            benchmark_provenance=benchmark_provenance,
            completion_code_revision=completion_code_revision,
            provenance_valid=provenance_valid,
            upstream_evidence=upstream_evidence,
            completion_upstream_evidence=completion_upstream_evidence,
            evidence_eligible=evidence_eligible,
        )
    baseline_scores = reciprocal_rank_scores(baseline_records, queries)
    raw_hybrid_scores = reciprocal_rank_scores(raw_hybrid_records, queries)
    guarded_scores = reciprocal_rank_scores(guarded_records, queries)
    assessments = {
        "overall": assess_known_item_regression(
            baseline=baseline_scores,
            candidate=guarded_scores,
            maximum_mrr_regression=sku_spec.maximum_mrr_regression,
            alpha=sku_spec.alpha,
        )
    }
    for kind in ("synthetic_sku", "exact_title"):
        query_ids = {
            query_id for query_id, query in queries.items() if query["kind"] == kind
        }
        assessments[kind] = assess_known_item_regression(
            baseline={query_id: baseline_scores[query_id] for query_id in query_ids},
            candidate={query_id: guarded_scores[query_id] for query_id in query_ids},
            maximum_mrr_regression=sku_spec.maximum_mrr_regression,
            alpha=sku_spec.alpha,
        )
    unprotected_assessment = assess_known_item_regression(
        baseline=baseline_scores,
        candidate=raw_hybrid_scores,
        maximum_mrr_regression=sku_spec.maximum_mrr_regression,
        alpha=sku_spec.alpha,
    )
    quality_eligible = evidence_eligible and all(
        entry.get("eligible_as_synthetic_known_item_regression_evidence")
        is True
        for entry in artifacts.values()
    )
    gate_assessments_pass = all(
        assessment.get("passes_no_significant_regression_gate") is True
        for assessment in assessments.values()
    )
    summary: dict[str, object] = {
        "schema_version": 2,
        "dataset": sku_spec.dataset,
        "provenance": "deterministic synthetic SKU and exact-title proxy",
        "eligible_as_authentic_merchant_sku_evidence": False,
        "eligible_as_synthetic_known_item_regression_evidence": quality_eligible,
        "quality_evidence_eligible_for_decision": quality_eligible,
        "latency_evidence_eligible_for_decision": False,
        "benchmark_provenance": benchmark_provenance,
        "benchmark_provenance_valid": provenance_valid,
        "completion_code_revision": completion_code_revision,
        "upstream_evidence": upstream_evidence,
        "completion_upstream_evidence": completion_upstream_evidence,
        "upstream_evidence_unchanged": (
            upstream_evidence == completion_upstream_evidence
        ),
        "quality_ineligibility_reasons": evidence_reasons,
        "query_counts": {
            "synthetic_sku": sku_spec.sku_queries,
            "exact_title": sku_spec.exact_title_queries,
            "total": sku_spec.query_count,
        },
        "baseline_metrics": artifacts["baseline"]["metrics"],
        "unprotected_hybrid_metrics": artifacts["unprotected_hybrid"]["metrics"],
        "guarded_hybrid_metrics": artifacts["guarded_hybrid"]["metrics"],
        "unprotected_hybrid_assessment": unprotected_assessment,
        "guarded_routing": routing,
        "no_regression_assessment": assessments,
        "passes_registered_synthetic_gate": (
            quality_eligible and gate_assessments_pass
        ),
        "artifacts": artifacts,
    }
    write_json(root / "results/wands/sku-slice/summary.json", summary)
    return summary


def verify_registered_sku_benchmark_inputs(
    *,
    root: Path,
    sparse_spec: NeuralSparseSpec,
    sku_spec: SkuSliceSpec,
) -> None:
    if sparse_spec != load_neural_sparse_spec(
        root / "config/neural_sparse.toml"
    ) or sku_spec != load_sku_slice_spec(root / "config/sku_slice.toml"):
        raise DatasetIntegrityError(
            "synthetic SKU benchmark registered configuration differs"
        )


def _sku_upstream_evidence(
    client: OpenSearchClient,
    *,
    root: Path,
    sparse_spec: NeuralSparseSpec,
    sku_spec: SkuSliceSpec,
) -> dict[str, object]:
    dataset_spec = load_wands_config(root / "config/datasets.toml")
    index_manifest_path = root / "results/wands/sku-slice/index-manifest.json"
    index_manifest = verify_sku_slice_index(
        client,
        root=root,
        dataset_spec=dataset_spec,
        sparse_spec=sparse_spec,
        sku_spec=sku_spec,
        products_path=root / "data/prepared/wands/products.jsonl",
        embeddings_path=(
            root / "data/cache/neural-sparse/wands-doc-v3-distill.jsonl"
        ),
        preparation_path=root / "results/wands/sku-slice/preparation.json",
        precompute_manifest_path=(
            root / "results/wands/neural-sparse/precompute-manifest.json"
        ),
        manifest_path=index_manifest_path,
    )
    return {
        "index_manifest": _artifact(root, index_manifest_path),
        "preparation": _artifact(
            root, root / "results/wands/sku-slice/preparation.json"
        ),
        "sparse_config": _artifact(root, root / "config/neural_sparse.toml"),
        "sku_config": _artifact(root, root / "config/sku_slice.toml"),
        "eligible_for_decision": (
            index_manifest.get("quality_evidence_eligible_for_decision") is True
        ),
    }


def derive_sku_benchmark_eligibility(
    *,
    provenance_valid: object,
    start_upstream_eligible: object,
    completion_upstream_eligible: object,
    upstream_unchanged: object,
) -> tuple[bool, list[str]]:
    checks = (
        (
            provenance_valid is True,
            "synthetic SKU benchmark provenance is not decision eligible",
        ),
        (
            start_upstream_eligible is True,
            "synthetic SKU start upstream evidence is not decision eligible",
        ),
        (
            completion_upstream_eligible is True,
            "synthetic SKU completion upstream evidence is not decision eligible",
        ),
        (
            upstream_unchanged is True,
            "synthetic SKU upstream evidence changed during the benchmark",
        ),
    )
    reasons = [reason for passed, reason in checks if not passed]
    return not reasons, reasons


def _sku_method_configs(
    *,
    sparse_spec: NeuralSparseSpec,
    sku_spec: SkuSliceSpec,
    pipeline: dict[str, Any],
    routing: dict[str, int],
) -> dict[str, dict[str, object]]:
    lexical: dict[str, object] = {
        "query": "sku_exact_plus_title_phrase_and_multimatch"
    }
    fusion = {
        "normalization": "min_max",
        "lexical_weight": sku_spec.lexical_weight,
        "sparse_weight": 1.0 - sku_spec.lexical_weight,
        "pipeline_id": PIPELINE_ID,
        "pipeline_sha256": canonical_sha256(pipeline),
    }
    sparse = {
        **lexical,
        "sparse_model": sparse_spec.model_name,
        "query_analyzer": sparse_spec.query_analyzer,
        "fusion": fusion,
    }
    return {
        "baseline": lexical,
        "unprotected_hybrid": sparse,
        "guarded_hybrid": {
            **sparse,
            "routing": {
                "sku_shape": "lexical_only",
                "lexical_top_title_exact_match": "lexical_only",
                "otherwise": "bm25_neural_sparse_minmax",
            },
            "observed_routing": routing,
        },
    }


def execute_known_item_lexical(
    client: OpenSearchClient,
    *,
    index: str,
    queries: dict[str, dict[str, str]],
    tag: str,
) -> list[RunRecord]:
    records: list[RunRecord] = []
    for query_id in sorted(queries, key=identifier_sort_key):
        response = client.search(
            index,
            {
                "size": RESULT_SIZE,
                "_source": False,
                "track_scores": True,
                "sort": [
                    {"_score": {"order": "desc"}},
                    {"product_id": {"order": "asc"}},
                ],
                "query": build_known_item_lexical_query(
                    queries[query_id]["query"]
                ),
            },
        )
        records.extend(_response_records(response, query_id=query_id, tag=tag))
    return records


def execute_known_item_sparse_hybrid(
    client: OpenSearchClient,
    *,
    index: str,
    queries: dict[str, dict[str, str]],
    sparse_spec: NeuralSparseSpec,
    sku_spec: SkuSliceSpec,
    tag: str,
) -> list[RunRecord]:
    pipeline = build_normalization_pipeline(
        lexical_weight=sku_spec.lexical_weight
    )
    verify_live_search_pipeline(
        client,
        pipeline_id=PIPELINE_ID,
        expected_definition=pipeline,
    )
    records: list[RunRecord] = []
    for query_id in sorted(queries, key=identifier_sort_key):
        text = queries[query_id]["query"]
        request = build_two_clause_hybrid_request(
            build_known_item_lexical_query(text),
            build_sparse_query(sparse_spec, text),
            size=RESULT_SIZE,
            pagination_depth=PAGINATION_DEPTH,
        )
        response = client.search(index, request, pipeline=PIPELINE_ID)
        records.extend(_response_records(response, query_id=query_id, tag=tag))
    verify_live_search_pipeline(
        client,
        pipeline_id=PIPELINE_ID,
        expected_definition=pipeline,
    )
    return records


def execute_guarded_known_item_sparse_hybrid(
    client: OpenSearchClient,
    *,
    index: str,
    queries: dict[str, dict[str, str]],
    sparse_spec: NeuralSparseSpec,
    sku_spec: SkuSliceSpec,
    tag: str,
) -> tuple[list[RunRecord], dict[str, int]]:
    pipeline = build_normalization_pipeline(
        lexical_weight=sku_spec.lexical_weight
    )
    verify_live_search_pipeline(
        client,
        pipeline_id=PIPELINE_ID,
        expected_definition=pipeline,
    )
    records: list[RunRecord] = []
    routing = {
        "lexical_only_sku": 0,
        "lexical_only_exact_title": 0,
        "hybrid": 0,
    }
    for query_id in sorted(queries, key=identifier_sort_key):
        text = queries[query_id]["query"]
        lexical_response = client.search(
            index,
            {
                "size": RESULT_SIZE,
                "_source": ["title"],
                "track_scores": True,
                "sort": [
                    {"_score": {"order": "desc"}},
                    {"product_id": {"order": "asc"}},
                ],
                "query": build_known_item_lexical_query(text),
            },
        )
        lexical_hits = cast(
            list[dict[str, Any]],
            cast(dict[str, Any], lexical_response["hits"])["hits"],
        )
        top_source = (
            cast(dict[str, Any], lexical_hits[0].get("_source", {}))
            if lexical_hits
            else {}
        )
        top_title = top_source.get("title")
        route = known_item_route(
            text,
            sku_prefix=sku_spec.sku_prefix,
            lexical_top_title=str(top_title) if top_title is not None else None,
        )
        routing[route] += 1
        if route != "hybrid":
            response = lexical_response
        else:
            request = build_two_clause_hybrid_request(
                build_known_item_lexical_query(text),
                build_sparse_query(sparse_spec, text),
                size=RESULT_SIZE,
                pagination_depth=PAGINATION_DEPTH,
            )
            response = client.search(index, request, pipeline=PIPELINE_ID)
        records.extend(_response_records(response, query_id=query_id, tag=tag))
    verify_live_search_pipeline(
        client,
        pipeline_id=PIPELINE_ID,
        expected_definition=pipeline,
    )
    return records, routing


def reciprocal_rank_scores(
    records: list[RunRecord],
    queries: dict[str, dict[str, str]],
) -> dict[str, float]:
    scores = {query_id: 0.0 for query_id in queries}
    for record in records:
        if (
            record.document_id == queries[record.query_id]["expected_document_id"]
            and scores[record.query_id] == 0.0
        ):
            scores[record.query_id] = 1.0 / record.rank
    return scores


def write_sku_bundle(
    client: OpenSearchClient,
    *,
    root: Path,
    sku_spec: SkuSliceSpec,
    query_path: Path,
    qrels_path: Path,
    tag: str,
    method: str,
    method_config: dict[str, object],
    records: list[RunRecord],
    benchmark_provenance: dict[str, Any],
    completion_code_revision: dict[str, Any],
    provenance_valid: bool,
    upstream_evidence: dict[str, object],
    completion_upstream_evidence: dict[str, object],
    evidence_eligible: bool,
) -> dict[str, object]:
    run_path = root / f"runs/{tag}.trec"
    manifest_path = root / f"runs/{tag}.manifest.json"
    metrics_path = root / f"results/wands/sku-slice/{tag}.metrics.json"
    write_run(run_path, records)
    evaluation = evaluate_run(qrels_path, run_path)
    crosscheck = cross_check_run(qrels_path, run_path, root=root)
    write_json(
        metrics_path,
        {
            "schema_version": 1,
            "dataset": sku_spec.dataset,
            "tag": tag,
            "metrics": evaluation.metrics,
            "per_query": evaluation.per_query,
            "evaluator_crosscheck": crosscheck.to_dict(),
        },
    )
    index_manifest_path = root / "results/wands/sku-slice/index-manifest.json"
    preparation_path = root / "results/wands/sku-slice/preparation.json"
    root_response = cast(dict[str, Any], client.request("GET", "/"))
    opensearch_version = str(
        cast(dict[str, Any], root_response["version"])["number"]
    )
    manifest: dict[str, object] = {
        "schema_version": 2,
        "dataset": sku_spec.dataset,
        "tag": tag,
        "method": method,
        "decision_scope": "synthetic_known_item_regression",
        "eligible_as_authentic_merchant_sku_evidence": False,
        "eligible_as_synthetic_known_item_regression_evidence": (
            evidence_eligible
        ),
        "quality_evidence_eligible_for_decision": evidence_eligible,
        "latency_evidence_eligible_for_decision": False,
        "benchmark_provenance": benchmark_provenance,
        "benchmark_provenance_valid": provenance_valid,
        "completion_code_revision": completion_code_revision,
        "upstream_evidence": upstream_evidence,
        "completion_upstream_evidence": completion_upstream_evidence,
        "upstream_evidence_unchanged": (
            upstream_evidence == completion_upstream_evidence
        ),
        "declared_variable": "retrieval_arm",
        "opensearch_version": opensearch_version,
        "index": asdict(collect_index_facts(client, sku_spec.index_name)),
        "index_manifest": _artifact(root, index_manifest_path),
        "preparation": _artifact(root, preparation_path),
        "method_config": method_config,
        "method_config_sha256": canonical_sha256(method_config),
        "query_file": _artifact(root, query_path),
        "qrels_file": _artifact(root, qrels_path),
        "run_file": {
            **_artifact(root, run_path),
            "records": len(records),
        },
        "metrics_file": _artifact(root, metrics_path),
    }
    write_json(manifest_path, manifest)
    return {
        "run": str(run_path.relative_to(root)),
        "manifest": str(manifest_path.relative_to(root)),
        "metrics_file": str(metrics_path.relative_to(root)),
        "metrics": evaluation.metrics,
        "eligible_as_synthetic_known_item_regression_evidence": (
            evidence_eligible
        ),
    }


def verify_sku_slice_summary(
    client: OpenSearchClient,
    *,
    root: Path,
    sparse_spec: NeuralSparseSpec,
    sku_spec: SkuSliceSpec,
) -> dict[str, Any]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    verify_registered_sku_benchmark_inputs(
        root=root,
        sparse_spec=sparse_spec,
        sku_spec=sku_spec,
    )
    summary = cast(
        dict[str, Any], read_json(root / "results/wands/sku-slice/summary.json")
    )
    if (
        summary.get("schema_version") != 2
        or summary.get("dataset") != sku_spec.dataset
        or summary.get("provenance")
        != "deterministic synthetic SKU and exact-title proxy"
        or summary.get("eligible_as_authentic_merchant_sku_evidence")
        is not False
        or summary.get("latency_evidence_eligible_for_decision") is not False
    ):
        raise DatasetIntegrityError("synthetic SKU summary metadata differs")
    provenance = summary.get("benchmark_provenance")
    completion_revision = summary.get("completion_code_revision")
    provenance_valid = (
        wands_recorded_provenance_valid(
            root,
            provenance=provenance,
            completion_revision=completion_revision,
        )
        and isinstance(provenance, Mapping)
        and verify_decision_provenance(
            provenance,
            root=root,
            profile_path=root / "config/benchmark.toml",
            environment_path=(
                root / "results/environment/benchmark-profile.json"
            ),
            require_current_code_revision=True,
        )
    )
    start_upstream = summary.get("upstream_evidence")
    completion_upstream = summary.get("completion_upstream_evidence")
    if not isinstance(start_upstream, dict) or not isinstance(
        completion_upstream, dict
    ):
        raise DatasetIntegrityError("synthetic SKU upstream evidence is missing")
    current_upstream = _sku_upstream_evidence(
        client,
        root=root,
        sparse_spec=sparse_spec,
        sku_spec=sku_spec,
    )
    if completion_upstream != current_upstream:
        raise DatasetIntegrityError("synthetic SKU current upstream evidence differs")
    upstream_unchanged = start_upstream == completion_upstream
    evidence_eligible, evidence_reasons = derive_sku_benchmark_eligibility(
        provenance_valid=provenance_valid,
        start_upstream_eligible=start_upstream.get("eligible_for_decision"),
        completion_upstream_eligible=completion_upstream.get(
            "eligible_for_decision"
        ),
        upstream_unchanged=upstream_unchanged,
    )
    if (
        summary.get("benchmark_provenance_valid") is not provenance_valid
        or summary.get("upstream_evidence_unchanged") is not upstream_unchanged
        or summary.get("quality_ineligibility_reasons") != evidence_reasons
    ):
        raise DatasetIntegrityError("synthetic SKU evidence derivation differs")
    artifacts_value = summary.get("artifacts")
    if not isinstance(artifacts_value, dict) or set(artifacts_value) != {
        "baseline",
        "unprotected_hybrid",
        "guarded_hybrid",
    }:
        raise DatasetIntegrityError("synthetic SKU artifact set differs")
    artifacts = cast(dict[str, dict[str, Any]], artifacts_value)
    queries = load_sku_queries(
        root / "data/prepared/wands/queries.sku-slice.jsonl"
    )
    pipeline = build_normalization_pipeline(
        lexical_weight=sku_spec.lexical_weight
    )
    try:
        verify_live_search_pipeline(
            client,
            pipeline_id=PIPELINE_ID,
            expected_definition=pipeline,
        )
    except DatasetIntegrityError as error:
        raise DatasetIntegrityError(
            f"synthetic SKU live search pipeline differs: {error}"
        ) from error
    guarded_records, routing = execute_guarded_known_item_sparse_hybrid(
        client,
        index=sku_spec.index_name,
        queries=queries,
        sparse_spec=sparse_spec,
        sku_spec=sku_spec,
        tag=_entry_tag(root, artifacts["guarded_hybrid"]),
    )
    expected_runs = {
        "baseline": execute_known_item_lexical(
            client,
            index=sku_spec.index_name,
            queries=queries,
            tag=_entry_tag(root, artifacts["baseline"]),
        ),
        "unprotected_hybrid": execute_known_item_sparse_hybrid(
            client,
            index=sku_spec.index_name,
            queries=queries,
            sparse_spec=sparse_spec,
            sku_spec=sku_spec,
            tag=_entry_tag(root, artifacts["unprotected_hybrid"]),
        ),
        "guarded_hybrid": guarded_records,
    }
    method_configs = _sku_method_configs(
        sparse_spec=sparse_spec,
        sku_spec=sku_spec,
        pipeline=pipeline,
        routing=routing,
    )
    methods = {
        "baseline": "known_item_lexical",
        "unprotected_hybrid": (
            "known_item_bm25_neural_sparse_unprotected_diagnostic"
        ),
        "guarded_hybrid": "known_item_guarded_bm25_neural_sparse",
    }
    manifests: dict[str, dict[str, Any]] = {}
    for name, entry in artifacts.items():
        manifests[name] = _verify_sku_artifact(
            client,
            root=root,
            sku_spec=sku_spec,
            entry=entry,
            expected_method=methods[name],
            expected_method_config=method_configs[name],
            expected_provenance=provenance,
            expected_completion_revision=completion_revision,
            expected_start_upstream=start_upstream,
            expected_completion_upstream=completion_upstream,
            expected_provenance_valid=provenance_valid,
            expected_eligible=evidence_eligible,
        )
    with tempfile.TemporaryDirectory(prefix="opensearch-hybrid-sku-verify-") as temp:
        for name, records in expected_runs.items():
            repeated = Path(temp) / f"{name}.trec"
            write_run(repeated, records)
            if repeated.read_bytes() != (root / str(artifacts[name]["run"])).read_bytes():
                raise DatasetIntegrityError(
                    f"synthetic SKU {name} rerun is not byte-identical"
                )
    baseline_scores = reciprocal_rank_scores(expected_runs["baseline"], queries)
    raw_hybrid_scores = reciprocal_rank_scores(
        expected_runs["unprotected_hybrid"], queries
    )
    guarded_scores = reciprocal_rank_scores(
        expected_runs["guarded_hybrid"], queries
    )
    assessments = {
        "overall": assess_known_item_regression(
            baseline=baseline_scores,
            candidate=guarded_scores,
            maximum_mrr_regression=sku_spec.maximum_mrr_regression,
            alpha=sku_spec.alpha,
        )
    }
    for kind in ("synthetic_sku", "exact_title"):
        query_ids = {
            query_id
            for query_id, query in queries.items()
            if query["kind"] == kind
        }
        assessments[kind] = assess_known_item_regression(
            baseline={
                query_id: baseline_scores[query_id] for query_id in query_ids
            },
            candidate={
                query_id: guarded_scores[query_id] for query_id in query_ids
            },
            maximum_mrr_regression=sku_spec.maximum_mrr_regression,
            alpha=sku_spec.alpha,
        )
    unprotected_assessment = assess_known_item_regression(
        baseline=baseline_scores,
        candidate=raw_hybrid_scores,
        maximum_mrr_regression=sku_spec.maximum_mrr_regression,
        alpha=sku_spec.alpha,
    )
    quality_eligible = evidence_eligible and all(
        manifest.get(
            "eligible_as_synthetic_known_item_regression_evidence"
        )
        is True
        for manifest in manifests.values()
    )
    gate_passes = quality_eligible and all(
        assessment.get("passes_no_significant_regression_gate") is True
        for assessment in assessments.values()
    )
    expected_fields: dict[str, object] = {
        "eligible_as_synthetic_known_item_regression_evidence": (
            quality_eligible
        ),
        "quality_evidence_eligible_for_decision": quality_eligible,
        "query_counts": {
            "synthetic_sku": sku_spec.sku_queries,
            "exact_title": sku_spec.exact_title_queries,
            "total": sku_spec.query_count,
        },
        "baseline_metrics": artifacts["baseline"]["metrics"],
        "unprotected_hybrid_metrics": artifacts["unprotected_hybrid"][
            "metrics"
        ],
        "guarded_hybrid_metrics": artifacts["guarded_hybrid"]["metrics"],
        "unprotected_hybrid_assessment": unprotected_assessment,
        "guarded_routing": routing,
        "no_regression_assessment": assessments,
        "passes_registered_synthetic_gate": gate_passes,
    }
    if any(summary.get(key) != value for key, value in expected_fields.items()):
        raise DatasetIntegrityError(
            "synthetic SKU summary routing, assessment, or eligibility differs"
        )
    completion_upstream = _sku_upstream_evidence(
        client,
        root=root,
        sparse_spec=sparse_spec,
        sku_spec=sku_spec,
    )
    if completion_upstream != current_upstream:
        raise DatasetIntegrityError(
            "synthetic SKU index changed during decision replay"
        )
    return summary


def _verify_sku_artifact(
    client: OpenSearchClient,
    *,
    root: Path,
    sku_spec: SkuSliceSpec,
    entry: dict[str, Any],
    expected_method: str,
    expected_method_config: dict[str, object],
    expected_provenance: object,
    expected_completion_revision: object,
    expected_start_upstream: dict[str, Any],
    expected_completion_upstream: dict[str, Any],
    expected_provenance_valid: bool,
    expected_eligible: bool,
) -> dict[str, Any]:
    if set(entry) != {
        "run",
        "manifest",
        "metrics_file",
        "metrics",
        "eligible_as_synthetic_known_item_regression_evidence",
    }:
        raise DatasetIntegrityError("synthetic SKU artifact entry fields differ")
    manifest = cast(dict[str, Any], read_json(root / str(entry["manifest"])))
    metrics = cast(dict[str, Any], read_json(root / str(entry["metrics_file"])))
    run_path = root / str(entry["run"])
    tag = manifest.get("tag")
    expected_manifest_path = root / f"runs/{tag}.manifest.json"
    expected_metrics_path = root / f"results/wands/sku-slice/{tag}.metrics.json"
    query_path = root / "data/prepared/wands/queries.sku-slice.jsonl"
    qrels_path = root / "data/prepared/wands/qrels.sku-slice.trec"
    index_manifest_path = root / "results/wands/sku-slice/index-manifest.json"
    preparation_path = root / "results/wands/sku-slice/preparation.json"
    root_response = cast(dict[str, Any], client.request("GET", "/"))
    live_version = str(
        cast(dict[str, Any], root_response["version"])["number"]
    )
    if (
        manifest.get("schema_version") != 2
        or manifest.get("dataset") != sku_spec.dataset
        or not isinstance(tag, str)
        or not tag
        or manifest.get("method") != expected_method
        or manifest.get("decision_scope")
        != "synthetic_known_item_regression"
        or manifest.get("declared_variable") != "retrieval_arm"
        or manifest.get("opensearch_version") != live_version
        or root / str(entry["manifest"]) != expected_manifest_path
        or root / str(entry["metrics_file"]) != expected_metrics_path
        or run_path != root / f"runs/{tag}.trec"
    ):
        raise DatasetIntegrityError("synthetic SKU run metadata differs")
    if (
        manifest.get("benchmark_provenance") != expected_provenance
        or manifest.get("completion_code_revision")
        != expected_completion_revision
        or manifest.get("upstream_evidence") != expected_start_upstream
        or manifest.get("completion_upstream_evidence")
        != expected_completion_upstream
        or manifest.get("upstream_evidence_unchanged")
        is not (expected_start_upstream == expected_completion_upstream)
        or manifest.get("benchmark_provenance_valid")
        is not expected_provenance_valid
        or manifest.get(
            "eligible_as_synthetic_known_item_regression_evidence"
        )
        is not expected_eligible
        or manifest.get("quality_evidence_eligible_for_decision")
        is not expected_eligible
        or manifest.get("eligible_as_authentic_merchant_sku_evidence")
        is not False
        or manifest.get("latency_evidence_eligible_for_decision") is not False
        or entry.get(
            "eligible_as_synthetic_known_item_regression_evidence"
        )
        is not expected_eligible
    ):
        raise DatasetIntegrityError("synthetic SKU run evidence eligibility differs")
    if manifest.get("index") != asdict(
        collect_index_facts(client, sku_spec.index_name)
    ):
        raise DatasetIntegrityError("synthetic SKU run index differs")
    if (
        manifest.get("index_manifest") != _artifact(root, index_manifest_path)
        or manifest.get("preparation") != _artifact(root, preparation_path)
        or manifest.get("method_config") != expected_method_config
        or manifest.get("method_config_sha256")
        != canonical_sha256(expected_method_config)
        or manifest.get("query_file") != _artifact(root, query_path)
        or manifest.get("qrels_file") != _artifact(root, qrels_path)
        or manifest.get("metrics_file") != _artifact(root, expected_metrics_path)
    ):
        raise DatasetIntegrityError("synthetic SKU run artifact binding differs")
    run_file = cast(dict[str, Any], manifest["run_file"])
    run_records = read_run(run_path)
    if run_file != {**_artifact(root, run_path), "records": len(run_records)}:
        raise DatasetIntegrityError("synthetic SKU run hash differs")
    evaluation = evaluate_run(qrels_path, run_path)
    if (
        metrics.get("schema_version") != 1
        or metrics.get("dataset") != sku_spec.dataset
        or metrics.get("tag") != tag
        or metrics.get("metrics") != evaluation.metrics
        or metrics.get("per_query") != evaluation.per_query
        or entry.get("metrics") != evaluation.metrics
    ):
        raise DatasetIntegrityError("synthetic SKU metrics do not reproduce")
    if metrics.get("evaluator_crosscheck") != cross_check_run(
        qrels_path, run_path, root=root
    ).to_dict():
        raise DatasetIntegrityError("synthetic SKU evaluator cross-check differs")
    return manifest


def _response_records(
    response: dict[str, Any],
    *,
    query_id: str,
    tag: str,
) -> list[RunRecord]:
    hits = cast(list[dict[str, Any]], cast(dict[str, Any], response["hits"])["hits"])
    if not hits:
        raise DatasetIntegrityError(f"known-item query {query_id} returned no hits")
    records: list[RunRecord] = []
    for rank, hit in enumerate(hits, start=1):
        if hit.get("_score") is None:
            raise DatasetIntegrityError(f"known-item query {query_id} has no score")
        records.append(
            RunRecord(
                query_id=query_id,
                document_id=str(hit["_id"]),
                rank=rank,
                score=hybrid_trec_score_for_rank(rank),
                tag=tag,
            )
        )
    return records


def _entry_tag(root: Path, entry: dict[str, Any]) -> str:
    return str(cast(dict[str, Any], read_json(root / str(entry["manifest"])))["tag"])


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }
