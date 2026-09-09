from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from poc.bm25 import load_prepared_queries
from poc.datasets import (
    DatasetIntegrityError,
    file_facts,
    load_trec_product_search_config,
)
from poc.experiments import BM25Experiments, BM25Profile, load_bm25_experiments
from poc.index_evidence import (
    content_addressed_pipeline_id,
    verify_live_search_pipeline,
)
from poc.manifest import canonical_sha256, read_json, write_json
from poc.neural_sparse import load_neural_sparse_spec
from poc.os_client import OpenSearchClient
from poc.provenance import collect_manifest_provenance, require_registered_opensearch_client
from poc.search import build_normalization_pipeline, build_two_clause_hybrid_request
from poc.significance import apply_family_eligibility, compare_run_family
from poc.sparse_benchmark import verify_wands_sparse_summary
from poc.trec import RunRecord, identifier_sort_key, write_run
from poc.trec_benchmark import (
    _artifact,
    _trec_evidence_ineligibility_reasons,
    _verified_upstream_component,
    derive_trec_evidence_eligibility,
    trec_completion_provenance_evidence,
    trec_completion_revision_is_current,
    trec_recorded_provenance_valid,
    verify_trec_artifact,
    verify_trec_bm25_summary,
    verify_trec_method_attribution,
    write_trec_bundle,
)
from poc.trec_indexing import load_trec_index_config
from poc.trec_sparse import (
    TrecSparseSpec,
    build_trec_sparse_query,
    extract_sparse_prediction,
    load_trec_sparse_config,
    select_registered_model_identity,
)
from poc.trec_sparse_indexing import verify_trec_sparse_index

RESULT_SIZE = 100
FROZEN_LEXICAL_WEIGHT = 0.3
PIPELINE_DEFINITION = build_normalization_pipeline(
    lexical_weight=FROZEN_LEXICAL_WEIGHT
)
PIPELINE_ID = content_addressed_pipeline_id(
    "opensearch-hybrid-trec-sparse-minmax-lw030-v1",
    PIPELINE_DEFINITION,
)


def trec_sparse_method_configs(
    *,
    root: Path,
    spec: TrecSparseSpec,
    model_id: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], BM25Profile]:
    query_model_path = (
        root / "results/trec-product-search/neural-sparse/query-model.json"
    )
    bm25_selection = cast(
        dict[str, Any], read_json(root / "results/wands/bm25-selection.json")
    )
    profile = BM25Profile.from_mapping(
        cast(dict[str, Any], bm25_selection["selected_profile"])
    )
    model_config: dict[str, object] = {
        "mode": "document_only_custom_query_tokenizer",
        "document_model": {
            "name": spec.document_model_name,
            "version": spec.document_model_version,
            "package_sha256": spec.document_model_sha256,
        },
        "query_tokenizer": {
            "name": spec.query_tokenizer_name,
            "version": spec.query_tokenizer_version,
            "model_id": model_id,
            "artifact_path": str(query_model_path.relative_to(root)),
            "artifact_sha256": file_facts(query_model_path).sha256,
        },
    }
    pipeline = build_normalization_pipeline(lexical_weight=FROZEN_LEXICAL_WEIGHT)
    hybrid_config: dict[str, object] = {
        **model_config,
        "lexical_profile": profile.to_dict(),
        "fusion": {
            "normalization": "min_max",
            "combination": "arithmetic_mean",
            "lexical_weight": FROZEN_LEXICAL_WEIGHT,
            "sparse_weight": 1.0 - FROZEN_LEXICAL_WEIGHT,
            "pipeline_id": PIPELINE_ID,
            "pipeline_sha256": canonical_sha256(pipeline),
        },
    }
    return model_config, hybrid_config, pipeline, profile


def run_trec_sparse_benchmark(
    client: OpenSearchClient,
    wands_client: OpenSearchClient,
    *,
    root: Path,
    spec: TrecSparseSpec,
    experiments: BM25Experiments,
) -> dict[str, object]:
    require_registered_opensearch_client(
        client,
        root / "config/benchmark-2.19.toml",
    )
    require_registered_opensearch_client(
        wands_client,
        root / "config/benchmark.toml",
    )
    verify_registered_trec_sparse_inputs(
        root=root,
        spec=spec,
        experiments=experiments,
    )
    benchmark_provenance = collect_manifest_provenance(
        root,
        profile_path=root / "config/benchmark-2.19.toml",
        environment_path=(
            root / "results/environment/benchmark-profile-opensearch-2.19.json"
        ),
    )
    query_path = (
        root / "data/prepared/trec-product-search-2024/queries.test.jsonl"
    )
    qrels_path = root / "data/prepared/trec-product-search-2024/qrels.test.trec"
    index_manifest_path = (
        root / "results/trec-product-search/neural-sparse/index-manifest.json"
    )
    selection_path = root / "results/wands/neural-sparse/summary.json"
    queries = load_prepared_queries(query_path, expected_split="test")
    wands_sparse = cast(dict[str, Any], read_json(selection_path))
    if wands_sparse.get("selected_lexical_weight") != FROZEN_LEXICAL_WEIGHT:
        raise DatasetIntegrityError("registered WANDS sparse weight differs")
    upstream_evidence, upstream_sources = _collect_trec_sparse_upstream_evidence(
        client,
        wands_client,
        root=root,
        spec=spec,
        experiments=experiments,
    )
    query_model = upstream_sources["query_model"]
    model_id = str(query_model["model_id"])
    bm25 = upstream_sources["bm25_summary"]
    model_config, hybrid_config, pipeline, profile = trec_sparse_method_configs(
        root=root,
        spec=spec,
        model_id=model_id,
    )
    sparse_tag = "trec-product-search-2024-neural-sparse-doc-only"
    sparse_records = execute_trec_sparse(
        client,
        index=spec.index_name,
        queries=queries,
        spec=spec,
        model_id=model_id,
        tag=sparse_tag,
    )
    client.put_search_pipeline(PIPELINE_ID, pipeline)
    verify_live_search_pipeline(
        client,
        pipeline_id=PIPELINE_ID,
        expected_definition=pipeline,
    )
    hybrid_tag = "trec-product-search-2024-bm25-sparse-minmax-lw030"
    hybrid_records = execute_trec_sparse_hybrid(
        client,
        index=spec.index_name,
        queries=queries,
        spec=spec,
        model_id=model_id,
        profile=profile,
        tag=hybrid_tag,
    )
    completion_upstream_evidence, _ = _collect_trec_sparse_upstream_evidence(
        client,
        wands_client,
        root=root,
        spec=spec,
        experiments=experiments,
    )
    provenance_valid, completion_revision = trec_completion_provenance_evidence(
        root,
        benchmark_provenance,
    )
    sparse_entry = write_trec_bundle(
        client,
        root=root,
        index=spec.index_name,
        query_path=query_path,
        qrels_path=qrels_path,
        tag=sparse_tag,
        method="neural-sparse",
        records=sparse_records,
        declared_variable="registered_retrieval_arm",
        method_config=model_config,
        selection_source=selection_path,
        index_manifest_path=index_manifest_path,
        benchmark_provenance=benchmark_provenance,
        provenance_valid=provenance_valid,
        completion_revision=completion_revision,
        upstream_evidence=upstream_evidence,
        completion_upstream_evidence=completion_upstream_evidence,
    )
    hybrid_entry = write_trec_bundle(
        client,
        root=root,
        index=spec.index_name,
        query_path=query_path,
        qrels_path=qrels_path,
        tag=hybrid_tag,
        method="neural-sparse",
        records=hybrid_records,
        declared_variable="registered_retrieval_arm",
        method_config=hybrid_config,
        selection_source=selection_path,
        index_manifest_path=index_manifest_path,
        benchmark_provenance=benchmark_provenance,
        provenance_valid=provenance_valid,
        completion_revision=completion_revision,
        upstream_evidence=upstream_evidence,
        completion_upstream_evidence=completion_upstream_evidence,
    )
    baseline_metrics = cast(dict[str, dict[str, float]], bm25["held_out_metrics"])
    sparse_ndcg = float(cast(dict[str, float], sparse_entry["metrics"])["ndcg@10"])
    hybrid_ndcg = float(cast(dict[str, float], hybrid_entry["metrics"])["ndcg@10"])
    source_eligibilities = {
        "sparse": sparse_entry.get("eligible_for_decision"),
        "hybrid": hybrid_entry.get("eligible_for_decision"),
    }
    quality_eligible = derive_trec_evidence_eligibility(
        provenance_valid=provenance_valid,
        start_upstream=upstream_evidence,
        completion_upstream=completion_upstream_evidence,
        source_eligibilities=tuple(source_eligibilities.values()),
    )
    summary: dict[str, object] = {
        "schema_version": 2,
        "dataset": "TREC Product Search 2024",
        "split": "held_out_test",
        "tuning_on_trec": False,
        "quality_evidence_eligible_for_decision": quality_eligible,
        "quality_ineligibility_reasons": _trec_evidence_ineligibility_reasons(
            provenance_valid=provenance_valid,
            start_upstream=upstream_evidence,
            completion_upstream=completion_upstream_evidence,
            source_eligibilities=source_eligibilities,
        ),
        "latency_evidence_eligible_for_decision": False,
        "latency_ineligibility_reason": "commodity x86 concurrency-4 run not recorded",
        "registered_architecture_source": str(selection_path.relative_to(root)),
        "frozen_lexical_weight": FROZEN_LEXICAL_WEIGHT,
        "document_model": spec.document_model_name,
        "query_tokenizer": spec.query_tokenizer_name,
        "sparse_only_metrics": sparse_entry["metrics"],
        "hybrid_metrics": hybrid_entry["metrics"],
        "tuned_bm25_ndcg@10": baseline_metrics["tuned"]["ndcg@10"],
        "sparse_minus_tuned_bm25_ndcg@10": (
            sparse_ndcg - baseline_metrics["tuned"]["ndcg@10"]
        ),
        "hybrid_minus_tuned_bm25_ndcg@10": (
            hybrid_ndcg - baseline_metrics["tuned"]["ndcg@10"]
        ),
        "competent_integrator_bm25_ndcg@10": (
            baseline_metrics["competent_integrator"]["ndcg@10"]
        ),
        "hybrid_minus_competent_integrator_bm25_ndcg@10": (
            hybrid_ndcg - baseline_metrics["competent_integrator"]["ndcg@10"]
        ),
        "artifacts": {
            "sparse": sparse_entry,
            "hybrid": hybrid_entry,
        },
        "benchmark_provenance": benchmark_provenance,
        "benchmark_provenance_valid": provenance_valid,
        "completion_code_revision": completion_revision,
        "upstream_evidence": upstream_evidence,
        "completion_upstream_evidence": completion_upstream_evidence,
        "upstream_evidence_unchanged": (
            completion_upstream_evidence == upstream_evidence
        ),
    }
    write_json(
        root / "results/trec-product-search/neural-sparse/summary.json",
        summary,
    )
    return summary


def execute_trec_sparse(
    client: OpenSearchClient,
    *,
    index: str,
    queries: dict[str, str],
    spec: TrecSparseSpec,
    model_id: str,
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
                "query": build_trec_sparse_query(
                    spec, queries[query_id], model_id=model_id
                ),
            },
        )
        records.extend(_response_records(response, query_id=query_id, tag=tag))
    return records


def execute_trec_sparse_hybrid(
    client: OpenSearchClient,
    *,
    index: str,
    queries: dict[str, str],
    spec: TrecSparseSpec,
    model_id: str,
    profile: BM25Profile,
    tag: str,
) -> list[RunRecord]:
    verify_live_search_pipeline(
        client,
        pipeline_id=PIPELINE_ID,
        expected_definition=PIPELINE_DEFINITION,
    )
    records: list[RunRecord] = []
    for query_id in sorted(queries, key=identifier_sort_key):
        request = build_two_clause_hybrid_request(
            profile.query(queries[query_id]),
            build_trec_sparse_query(spec, queries[query_id], model_id=model_id),
            size=RESULT_SIZE,
            pagination_depth=RESULT_SIZE,
        )
        response = client.search(index, request, pipeline=PIPELINE_ID)
        records.extend(_response_records(response, query_id=query_id, tag=tag))
    verify_live_search_pipeline(
        client,
        pipeline_id=PIPELINE_ID,
        expected_definition=PIPELINE_DEFINITION,
    )
    return records


def verify_trec_sparse_summary(
    client: OpenSearchClient,
    wands_client: OpenSearchClient,
    *,
    root: Path,
    spec: TrecSparseSpec,
) -> dict[str, Any]:
    require_registered_opensearch_client(
        client,
        root / "config/benchmark-2.19.toml",
    )
    require_registered_opensearch_client(
        wands_client,
        root / "config/benchmark.toml",
    )
    experiments = load_bm25_experiments(root / "config/experiments.toml")
    verify_registered_trec_sparse_inputs(
        root=root,
        spec=spec,
        experiments=experiments,
    )
    summary = cast(
        dict[str, Any],
        read_json(root / "results/trec-product-search/neural-sparse/summary.json"),
    )
    if (
        summary.get("schema_version") != 2
        or summary.get("dataset") != "TREC Product Search 2024"
        or summary.get("split") != "held_out_test"
        or summary.get("registered_architecture_source")
        != "results/wands/neural-sparse/summary.json"
    ):
        raise DatasetIntegrityError("TREC sparse summary metadata differs")
    if summary.get("tuning_on_trec") is not False:
        raise DatasetIntegrityError("TREC sparse transfer was tuned on test queries")
    if summary.get("frozen_lexical_weight") != FROZEN_LEXICAL_WEIGHT:
        raise DatasetIntegrityError("TREC sparse transfer weight differs")
    if summary.get("document_model") != spec.document_model_name:
        raise DatasetIntegrityError("TREC sparse document model differs")
    if summary.get("query_tokenizer") != spec.query_tokenizer_name:
        raise DatasetIntegrityError("TREC sparse query tokenizer differs")
    current_upstream, upstream_sources = _collect_trec_sparse_upstream_evidence(
        client,
        wands_client,
        root=root,
        spec=spec,
        experiments=experiments,
    )
    query_model = upstream_sources["query_model"]
    model_id = str(query_model["model_id"])
    model_config, hybrid_config, pipeline, profile = trec_sparse_method_configs(
        root=root,
        spec=spec,
        model_id=model_id,
    )
    summary_provenance = summary.get("benchmark_provenance")
    completion_revision = summary.get("completion_code_revision")
    provenance_valid = trec_recorded_provenance_valid(
        root,
        provenance=summary_provenance,
        completion_revision=completion_revision,
    )
    completion_is_current = trec_completion_revision_is_current(
        root, completion_revision
    )
    start_upstream = summary.get("upstream_evidence")
    completion_upstream = summary.get("completion_upstream_evidence")
    upstream_unchanged = (
        isinstance(start_upstream, dict)
        and isinstance(completion_upstream, dict)
        and start_upstream == completion_upstream == current_upstream
    )
    if (
        not isinstance(summary_provenance, dict)
        or not isinstance(completion_revision, dict)
        or not isinstance(start_upstream, dict)
        or not isinstance(completion_upstream, dict)
        or not completion_is_current
        or summary.get("benchmark_provenance_valid") is not provenance_valid
        or summary.get("upstream_evidence_unchanged") is not upstream_unchanged
        or not upstream_unchanged
    ):
        raise DatasetIntegrityError(
            "TREC sparse summary provenance or upstream evidence differs"
        )
    bm25 = upstream_sources["bm25_summary"]
    artifacts_value = summary.get("artifacts")
    if not isinstance(artifacts_value, dict):
        raise DatasetIntegrityError("TREC sparse artifacts are missing")
    artifacts = cast(dict[str, dict[str, Any]], artifacts_value)
    if set(artifacts) != {"sparse", "hybrid"}:
        raise DatasetIntegrityError("TREC sparse artifact set differs")
    expected_configs = {"sparse": model_config, "hybrid": hybrid_config}
    for arm, entry in artifacts.items():
        verify_trec_artifact(
            client,
            root=root,
            index=spec.index_name,
            entry=entry,
            index_manifest_path=(
                root / "results/trec-product-search/neural-sparse/index-manifest.json"
            ),
            expected_benchmark_provenance=summary_provenance,
            expected_completion_revision=completion_revision,
            expected_upstream_evidence=current_upstream,
            expected_query_path=(
                root / "data/prepared/trec-product-search-2024/queries.test.jsonl"
            ),
            expected_qrels_path=(
                root / "data/prepared/trec-product-search-2024/qrels.test.trec"
            ),
            expected_selection_source_path=(
                root / "results/wands/neural-sparse/summary.json"
            ),
        )
        manifest = cast(dict[str, Any], read_json(root / str(entry["manifest"])))
        verify_trec_method_attribution(
            manifest,
            expected_method="neural-sparse",
            expected_declared_variable="registered_retrieval_arm",
            expected_config=expected_configs[arm],
        )
    try:
        verify_live_search_pipeline(
            client,
            pipeline_id=PIPELINE_ID,
            expected_definition=pipeline,
        )
    except DatasetIntegrityError as error:
        raise DatasetIntegrityError(
            f"TREC sparse live hybrid pipeline differs: {error}"
        ) from error
    baseline_metrics = cast(dict[str, dict[str, float]], bm25["held_out_metrics"])
    source_eligibilities = {
        arm: entry.get("eligible_for_decision") for arm, entry in artifacts.items()
    }
    quality_eligible = derive_trec_evidence_eligibility(
        provenance_valid=provenance_valid,
        start_upstream=start_upstream,
        completion_upstream=completion_upstream,
        source_eligibilities=tuple(source_eligibilities.values()),
    )
    verify_trec_sparse_summary_fields(
        summary,
        sparse_metrics=cast(dict[str, float], artifacts["sparse"]["metrics"]),
        hybrid_metrics=cast(dict[str, float], artifacts["hybrid"]["metrics"]),
        baseline_metrics=baseline_metrics,
        quality_eligible=quality_eligible,
    )
    if summary.get(
        "quality_ineligibility_reasons"
    ) != _trec_evidence_ineligibility_reasons(
        provenance_valid=provenance_valid,
        start_upstream=start_upstream,
        completion_upstream=completion_upstream,
        source_eligibilities=source_eligibilities,
    ):
        raise DatasetIntegrityError("TREC sparse summary eligibility reasons differ")
    _verify_sparse_rerun(
        client,
        root=root,
        spec=spec,
        entry=artifacts["sparse"],
        model_id=model_id,
        profile=profile,
        hybrid=False,
    )
    _verify_sparse_rerun(
        client,
        root=root,
        spec=spec,
        entry=artifacts["hybrid"],
        model_id=model_id,
        profile=profile,
        hybrid=True,
    )
    completion_sparse_index = _verify_trec_sparse_index_surface(
        client,
        root=root,
        spec=spec,
    )
    if completion_sparse_index != upstream_sources["sparse_index"]:
        raise DatasetIntegrityError(
            "TREC sparse index changed during decision replay"
        )
    return summary


def verify_trec_query_model(
    client: OpenSearchClient,
    *,
    root: Path,
    spec: TrecSparseSpec,
) -> dict[str, Any]:
    require_registered_opensearch_client(
        client,
        root / "config/benchmark-2.19.toml",
    )
    artifact = cast(
        dict[str, Any],
        read_json(
            root / "results/trec-product-search/neural-sparse/query-model.json"
        ),
    )
    root_response = cast(dict[str, Any], client.request("GET", "/"))
    live_version = str(cast(dict[str, Any], root_response["version"])["number"])
    search = cast(
        dict[str, Any],
        client.request(
            "POST",
            "/_plugins/_ml/models/_search",
            json_body={"size": 1000, "query": {"match_all": {}}},
        ),
    )
    hits = cast(
        list[dict[str, Any]], cast(dict[str, Any], search["hits"])["hits"]
    )
    selected = select_registered_model_identity(
        hits,
        name=spec.query_tokenizer_name,
        version=spec.query_tokenizer_version,
    )
    if selected is None:
        raise DatasetIntegrityError("TREC sparse registered query model is absent")
    selected_model_id = str(selected.get("_id", ""))
    live_model = cast(
        dict[str, Any],
        client.request("GET", f"/_plugins/_ml/models/{selected_model_id}"),
    )
    model_id = verify_trec_query_model_identity(
        artifact,
        live_model=live_model,
        selected_model_id=selected_model_id,
        spec=spec,
    )
    probe = artifact.get("probe")
    if not isinstance(probe, dict) or not isinstance(probe.get("text"), str):
        raise DatasetIntegrityError("TREC sparse query model probe is missing")
    prediction_response = cast(
        dict[str, Any],
        client.request(
            "POST",
            f"/_plugins/_ml/_predict/sparse_encoding/{model_id}",
            json_body={"text_docs": [probe["text"]]},
        ),
    )
    prediction = extract_sparse_prediction(prediction_response)
    provenance = artifact.get("benchmark_provenance")
    provenance_valid = trec_recorded_provenance_valid(
        root,
        provenance=provenance,
        completion_revision=artifact.get("completion_code_revision"),
    )
    completion_is_current = trec_completion_revision_is_current(
        root, artifact.get("completion_code_revision")
    )
    expected_quality = provenance_valid
    if (
        artifact.get("schema_version") != 2
        or artifact.get("dataset") != "TREC Product Search 2024"
        or artifact.get("status") != "passed"
        or artifact.get("opensearch_version") != live_version
        or live_version != spec.expected_opensearch_version
        or artifact.get("benchmark_provenance_valid") is not provenance_valid
        or not completion_is_current
        or artifact.get("quality_evidence_eligible_for_decision")
        is not expected_quality
        or not expected_quality
        or probe.get("token_count") != len(prediction)
        or probe.get("tokens") != prediction
    ):
        raise DatasetIntegrityError(
            "TREC sparse query model evidence does not reproduce"
        )
    return artifact


def verify_trec_query_model_identity(
    artifact: dict[str, Any],
    *,
    live_model: dict[str, Any],
    selected_model_id: str,
    spec: TrecSparseSpec,
) -> str:
    registered_spec = artifact.get("registered_spec")
    recorded_model = artifact.get("registered_model")
    identity_keys = (
        "algorithm",
        "model_content_hash_value",
        "model_content_size_in_bytes",
        "model_format",
        "model_state",
        "name",
    )
    live_identity = {key: live_model.get(key) for key in identity_keys}
    recorded_identity = (
        {key: recorded_model.get(key) for key in identity_keys}
        if isinstance(recorded_model, dict)
        else None
    )
    content_hash = live_model.get("model_content_hash_value")
    content_size = live_model.get("model_content_size_in_bytes")
    if (
        artifact.get("status") != "passed"
        or artifact.get("quality_evidence_eligible_for_decision") is not True
        or not selected_model_id
        or artifact.get("model_id") != selected_model_id
        or registered_spec
        != {
            "name": spec.query_tokenizer_name,
            "version": spec.query_tokenizer_version,
            "format": spec.query_tokenizer_format,
        }
        or live_model.get("name") != spec.query_tokenizer_name
        or live_model.get("model_format") != spec.query_tokenizer_format
        or live_model.get("model_state") != "DEPLOYED"
        or live_model.get("algorithm") != "SPARSE_TOKENIZE"
        or content_hash != spec.query_tokenizer_content_sha256
        or content_size != spec.query_tokenizer_content_bytes
        or not isinstance(content_hash, str)
        or len(content_hash) != 64
        or any(character not in "0123456789abcdef" for character in content_hash)
        or not isinstance(content_size, int)
        or isinstance(content_size, bool)
        or content_size <= 0
        or recorded_identity != live_identity
    ):
        raise DatasetIntegrityError("TREC sparse query model identity differs")
    return selected_model_id


def verify_trec_sparse_summary_fields(
    summary: dict[str, Any],
    *,
    sparse_metrics: dict[str, float],
    hybrid_metrics: dict[str, float],
    baseline_metrics: dict[str, dict[str, float]],
    quality_eligible: bool,
) -> None:
    sparse_ndcg = float(sparse_metrics["ndcg@10"])
    hybrid_ndcg = float(hybrid_metrics["ndcg@10"])
    tuned_ndcg = float(baseline_metrics["tuned"]["ndcg@10"])
    competent_ndcg = float(
        baseline_metrics["competent_integrator"]["ndcg@10"]
    )
    expected = {
        "quality_evidence_eligible_for_decision": quality_eligible,
        "latency_evidence_eligible_for_decision": False,
        "latency_ineligibility_reason": (
            "commodity x86 concurrency-4 run not recorded"
        ),
        "sparse_only_metrics": sparse_metrics,
        "hybrid_metrics": hybrid_metrics,
        "tuned_bm25_ndcg@10": tuned_ndcg,
        "sparse_minus_tuned_bm25_ndcg@10": sparse_ndcg - tuned_ndcg,
        "hybrid_minus_tuned_bm25_ndcg@10": hybrid_ndcg - tuned_ndcg,
        "competent_integrator_bm25_ndcg@10": competent_ndcg,
        "hybrid_minus_competent_integrator_bm25_ndcg@10": (
            hybrid_ndcg - competent_ndcg
        ),
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise DatasetIntegrityError("TREC sparse summary metrics or eligibility differ")


def verify_registered_trec_sparse_inputs(
    *,
    root: Path,
    spec: TrecSparseSpec,
    experiments: BM25Experiments,
) -> None:
    registered_spec = load_trec_sparse_config(
        root / "config/trec_neural_sparse.toml"
    )
    registered_experiments = load_bm25_experiments(
        root / "config/experiments.toml"
    )
    if spec != registered_spec or experiments != registered_experiments:
        raise DatasetIntegrityError("TREC sparse registered configuration differs")


def verify_registered_trec_significance_thresholds(
    *,
    root: Path,
    metric: str,
    alpha: float,
    minimum_delta: float,
) -> None:
    experiments = load_bm25_experiments(root / "config/experiments.toml")
    if (
        metric != experiments.primary_metric
        or alpha != experiments.alpha
        or minimum_delta != experiments.minimum_paired_delta
    ):
        raise DatasetIntegrityError("TREC sparse significance thresholds differ")


def _collect_trec_sparse_upstream_evidence(
    client: OpenSearchClient,
    wands_client: OpenSearchClient,
    *,
    root: Path,
    spec: TrecSparseSpec,
    experiments: BM25Experiments,
) -> tuple[dict[str, object], dict[str, dict[str, Any]]]:
    index_manifest = _verify_trec_sparse_index_surface(
        client,
        root=root,
        spec=spec,
    )
    wands_spec = load_neural_sparse_spec(root / "config/neural_sparse.toml")
    wands_summary = verify_wands_sparse_summary(
        wands_client,
        root=root,
        spec=wands_spec,
        experiments=experiments,
    )
    query_model = verify_trec_query_model(client, root=root, spec=spec)
    trec_lexical_index = load_trec_index_config(root / "config/indexes.toml")
    bm25_summary = verify_trec_bm25_summary(
        client,
        wands_client,
        root=root,
        index=trec_lexical_index.name,
        experiments=experiments,
    )
    sources = {
        "sparse_index": index_manifest,
        "wands_sparse_summary": wands_summary,
        "query_model": query_model,
        "bm25_summary": bm25_summary,
    }
    components = {
        "trec_sparse_index": _verified_upstream_component(
            root=root,
            path=(
                root
                / "results/trec-product-search/neural-sparse/index-manifest.json"
            ),
            verified=index_manifest,
            semantic_verifier="verify_trec_sparse_index",
        ),
        "wands_sparse_summary": _verified_upstream_component(
            root=root,
            path=root / "results/wands/neural-sparse/summary.json",
            verified=wands_summary,
            semantic_verifier="verify_wands_sparse_summary",
        ),
        "trec_query_model": _verified_upstream_component(
            root=root,
            path=(
                root
                / "results/trec-product-search/neural-sparse/query-model.json"
            ),
            verified=query_model,
            semantic_verifier="verify_trec_query_model",
        ),
        "trec_bm25_summary": _verified_upstream_component(
            root=root,
            path=root / "results/trec-product-search/bm25-summary.json",
            verified=bm25_summary,
            semantic_verifier="verify_trec_bm25_summary",
        ),
    }
    eligible = all(
        component.get("eligible_for_decision") is True
        for component in components.values()
    )
    return (
        {
            "components": components,
            "registered_config_artifacts": {
                name: _artifact(root, root / relative)
                for name, relative in {
                    "datasets": "config/datasets.toml",
                    "indexes": "config/indexes.toml",
                    "experiments": "config/experiments.toml",
                    "neural_sparse": "config/neural_sparse.toml",
                    "trec_neural_sparse": "config/trec_neural_sparse.toml",
                }.items()
            },
            "eligible_for_decision": eligible,
            "ineligibility_reasons": [
                name
                for name, component in components.items()
                if component.get("eligible_for_decision") is not True
            ],
        },
        sources,
    )


def _verify_trec_sparse_index_surface(
    client: OpenSearchClient,
    *,
    root: Path,
    spec: TrecSparseSpec,
) -> dict[str, Any]:
    dataset_spec = load_trec_product_search_config(root / "config/datasets.toml")
    model_directory = root / "data/cache/models/neural-sparse/doc-v2-distill"
    return verify_trec_sparse_index(
        client,
        root=root,
        dataset_spec=dataset_spec,
        sparse_spec=spec,
        corpus_path=(
            root / "data/raw/trec-product-search-2024/collection.trec.gz"
        ),
        embeddings_path=(
            root
            / "data/cache/neural-sparse/trec-product-search-doc-v2-distill-full.jsonl"
        ),
        precompute_manifest_path=(
            root
            / "results/trec-product-search/neural-sparse/precompute-full-manifest.json"
        ),
        validation_manifest_path=(
            root
            / "results/trec-product-search/neural-sparse/"
            "precompute-full-validation-manifest.json"
        ),
        package_path=(
            root / "data/cache/models/neural-sparse/doc-v2-distill.zip"
        ),
        model_path=(
            model_directory
            / "opensearch-neural-sparse-encoding-doc-v2-distill.pt"
        ),
        tokenizer_path=model_directory / "tokenizer.json",
        manifest_path=(
            root
            / "results/trec-product-search/neural-sparse/index-manifest.json"
        ),
    )


def derive_trec_significance_family_eligibility(
    *,
    provenance_valid: object,
    upstream_unchanged: object,
    source_eligibilities: tuple[object, ...],
) -> bool:
    return (
        provenance_valid is True
        and upstream_unchanged is True
        and all(value is True for value in source_eligibilities)
    )


def build_trec_sparse_significance(
    client: OpenSearchClient,
    wands_client: OpenSearchClient,
    *,
    root: Path,
    spec: TrecSparseSpec,
    metric: str,
    alpha: float,
    minimum_delta: float,
) -> dict[str, object]:
    require_registered_opensearch_client(
        client,
        root / "config/benchmark-2.19.toml",
    )
    require_registered_opensearch_client(
        wands_client,
        root / "config/benchmark.toml",
    )
    registered_inputs = _registered_trec_significance_inputs(
        root=root,
        metric=metric,
        alpha=alpha,
        minimum_delta=minimum_delta,
    )
    benchmark_provenance = collect_manifest_provenance(
        root,
        profile_path=root / "config/benchmark-2.19.toml",
        environment_path=(
            root / "results/environment/benchmark-profile-opensearch-2.19.json"
        ),
    )
    summaries = _collect_verified_trec_significance_summaries(
        client,
        wands_client,
        root=root,
        spec=spec,
    )
    source_evidence = _trec_significance_source_evidence(
        root=root,
        summaries=summaries,
    )
    qrels_artifact = _artifact(
        root,
        root / "data/prepared/trec-product-search-2024/qrels.test.trec",
    )
    comparisons = _compute_trec_sparse_comparisons(
        root=root,
        summaries=summaries,
        metric=metric,
        alpha=alpha,
        minimum_delta=minimum_delta,
    )
    completion_summaries = _collect_verified_trec_significance_summaries(
        client,
        wands_client,
        root=root,
        spec=spec,
    )
    completion_source_evidence = _trec_significance_source_evidence(
        root=root,
        summaries=completion_summaries,
    )
    completion_qrels_artifact = _artifact(
        root,
        root / "data/prepared/trec-product-search-2024/qrels.test.trec",
    )
    provenance_valid, completion_revision = trec_completion_provenance_evidence(
        root,
        benchmark_provenance,
    )
    result = _assemble_trec_sparse_significance(
        comparisons=comparisons,
        source_evidence=source_evidence,
        completion_source_evidence=completion_source_evidence,
        qrels_artifact=qrels_artifact,
        completion_qrels_artifact=completion_qrels_artifact,
        registered_inputs=registered_inputs,
        benchmark_provenance=benchmark_provenance,
        provenance_valid=provenance_valid,
        completion_revision=completion_revision,
    )
    write_json(
        root / "results/trec-product-search/neural-sparse/significance.json",
        result,
    )
    return result


def verify_trec_sparse_significance(
    client: OpenSearchClient,
    wands_client: OpenSearchClient,
    *,
    root: Path,
    spec: TrecSparseSpec,
    metric: str,
    alpha: float,
    minimum_delta: float,
) -> dict[str, Any]:
    require_registered_opensearch_client(
        client,
        root / "config/benchmark-2.19.toml",
    )
    require_registered_opensearch_client(
        wands_client,
        root / "config/benchmark.toml",
    )
    registered_inputs = _registered_trec_significance_inputs(
        root=root,
        metric=metric,
        alpha=alpha,
        minimum_delta=minimum_delta,
    )
    recorded = cast(
        dict[str, Any],
        read_json(
            root / "results/trec-product-search/neural-sparse/significance.json"
        ),
    )
    summaries = _collect_verified_trec_significance_summaries(
        client,
        wands_client,
        root=root,
        spec=spec,
    )
    current_source_evidence = _trec_significance_source_evidence(
        root=root,
        summaries=summaries,
    )
    current_qrels_artifact = _artifact(
        root,
        root / "data/prepared/trec-product-search-2024/qrels.test.trec",
    )
    provenance = recorded.get("benchmark_provenance")
    completion_revision = recorded.get("completion_code_revision")
    provenance_valid = trec_recorded_provenance_valid(
        root,
        provenance=provenance,
        completion_revision=completion_revision,
    )
    completion_is_current = trec_completion_revision_is_current(
        root, completion_revision
    )
    start_evidence = recorded.get("source_evidence")
    completion_evidence = recorded.get("completion_source_evidence")
    start_qrels = recorded.get("qrels_artifact")
    completion_qrels = recorded.get("completion_qrels_artifact")
    upstream_unchanged = (
        isinstance(start_evidence, Mapping)
        and isinstance(completion_evidence, Mapping)
        and dict(start_evidence)
        == dict(completion_evidence)
        == current_source_evidence
        and isinstance(start_qrels, Mapping)
        and isinstance(completion_qrels, Mapping)
        and dict(start_qrels)
        == dict(completion_qrels)
        == current_qrels_artifact
    )
    if (
        recorded.get("schema_version") != 2
        or recorded.get("dataset") != "TREC Product Search 2024"
        or recorded.get("split") != "held_out_test"
        or recorded.get("classification")
        != "registered_held_out_paired_significance"
        or recorded.get("registered_inputs") != registered_inputs
        or not completion_is_current
        or recorded.get("benchmark_provenance_valid") is not provenance_valid
        or recorded.get("source_evidence_unchanged_at_completion")
        is not upstream_unchanged
        or not upstream_unchanged
    ):
        raise DatasetIntegrityError(
            "TREC sparse significance evidence or provenance differs"
        )
    comparisons = _compute_trec_sparse_comparisons(
        root=root,
        summaries=summaries,
        metric=metric,
        alpha=alpha,
        minimum_delta=minimum_delta,
    )
    if not isinstance(provenance, Mapping) or not isinstance(
        completion_revision, Mapping
    ):
        raise DatasetIntegrityError("TREC sparse significance provenance is missing")
    expected = _assemble_trec_sparse_significance(
        comparisons=comparisons,
        source_evidence=current_source_evidence,
        completion_source_evidence=current_source_evidence,
        qrels_artifact=current_qrels_artifact,
        completion_qrels_artifact=current_qrels_artifact,
        registered_inputs=registered_inputs,
        benchmark_provenance=provenance,
        provenance_valid=provenance_valid,
        completion_revision=completion_revision,
    )
    if recorded != expected:
        raise DatasetIntegrityError("TREC sparse significance does not reproduce")
    return recorded


def _registered_trec_significance_inputs(
    *,
    root: Path,
    metric: str,
    alpha: float,
    minimum_delta: float,
) -> dict[str, object]:
    verify_registered_trec_significance_thresholds(
        root=root,
        metric=metric,
        alpha=alpha,
        minimum_delta=minimum_delta,
    )
    return {
        "decision_thresholds": {
            "metric": metric,
            "alpha": alpha,
            "minimum_delta": minimum_delta,
        },
        "config_artifacts": {
            name: _artifact(root, root / relative)
            for name, relative in {
                "experiments": "config/experiments.toml",
                "indexes": "config/indexes.toml",
                "neural_sparse": "config/neural_sparse.toml",
                "trec_neural_sparse": "config/trec_neural_sparse.toml",
            }.items()
        },
    }


def _collect_verified_trec_significance_summaries(
    client: OpenSearchClient,
    wands_client: OpenSearchClient,
    *,
    root: Path,
    spec: TrecSparseSpec,
) -> dict[str, dict[str, Any]]:
    experiments = load_bm25_experiments(root / "config/experiments.toml")
    trec_index = load_trec_index_config(root / "config/indexes.toml")
    bm25 = verify_trec_bm25_summary(
        client,
        wands_client,
        root=root,
        index=trec_index.name,
        experiments=experiments,
    )
    sparse = verify_trec_sparse_summary(
        client,
        wands_client,
        root=root,
        spec=spec,
    )
    return {"bm25": bm25, "sparse": sparse}


def _trec_significance_source_evidence(
    *,
    root: Path,
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, object]]:
    bm25 = summaries["bm25"]
    sparse = summaries["sparse"]
    return {
        "bm25_tuned": _trec_significance_source(
            root=root,
            summary_path=root / "results/trec-product-search/bm25-summary.json",
            summary=bm25,
            artifact_key="tuned",
            semantic_verifier="verify_trec_bm25_summary",
        ),
        "bm25_competent_integrator": _trec_significance_source(
            root=root,
            summary_path=root / "results/trec-product-search/bm25-summary.json",
            summary=bm25,
            artifact_key="competent_integrator",
            semantic_verifier="verify_trec_bm25_summary",
        ),
        "sparse_only": _trec_significance_source(
            root=root,
            summary_path=(
                root / "results/trec-product-search/neural-sparse/summary.json"
            ),
            summary=sparse,
            artifact_key="sparse",
            semantic_verifier="verify_trec_sparse_summary",
        ),
        "sparse_hybrid": _trec_significance_source(
            root=root,
            summary_path=(
                root / "results/trec-product-search/neural-sparse/summary.json"
            ),
            summary=sparse,
            artifact_key="hybrid",
            semantic_verifier="verify_trec_sparse_summary",
        ),
    }


def _trec_significance_source(
    *,
    root: Path,
    summary_path: Path,
    summary: Mapping[str, Any],
    artifact_key: str,
    semantic_verifier: str,
) -> dict[str, object]:
    recorded_summary = read_json(summary_path)
    if not isinstance(recorded_summary, Mapping) or dict(recorded_summary) != dict(
        summary
    ):
        raise DatasetIntegrityError(
            f"TREC significance summary changed after verification: {semantic_verifier}"
        )
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise DatasetIntegrityError("TREC significance source artifacts are missing")
    entry = artifacts.get(artifact_key)
    if not isinstance(entry, Mapping):
        raise DatasetIntegrityError(
            f"TREC significance source is missing: {artifact_key}"
        )
    files: dict[str, dict[str, object]] = {}
    for key in ("run", "manifest", "metrics_file"):
        relative = entry.get(key)
        if not isinstance(relative, str) or not relative:
            raise DatasetIntegrityError(
                f"TREC significance source has no {key}: {artifact_key}"
            )
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise DatasetIntegrityError(
                f"TREC significance source path differs: {artifact_key}"
            )
        files[key] = _artifact(root, path)
    manifest = read_json(root / str(entry["manifest"]))
    if not isinstance(manifest, Mapping):
        raise DatasetIntegrityError("TREC significance manifest is not an object")
    summary_eligible = (
        summary.get("quality_evidence_eligible_for_decision") is True
    )
    manifest_eligible = manifest.get("eligible_for_decision") is True
    return {
        "semantic_verifier": semantic_verifier,
        "summary": _artifact(root, summary_path),
        "summary_artifact_key": artifact_key,
        "summary_eligible_for_decision": summary_eligible,
        "artifact_entry": dict(entry),
        "files": files,
        "manifest_eligible_for_decision": manifest_eligible,
        "eligible_for_decision": summary_eligible and manifest_eligible,
    }


def _compute_trec_sparse_comparisons(
    *,
    root: Path,
    summaries: Mapping[str, Mapping[str, Any]],
    metric: str,
    alpha: float,
    minimum_delta: float,
) -> dict[str, dict[str, object]]:
    sparse = summaries["sparse"]
    bm25 = summaries["bm25"]
    sparse_artifacts = cast(dict[str, dict[str, Any]], sparse["artifacts"])
    bm25_artifacts = cast(dict[str, dict[str, Any]], bm25["artifacts"])
    candidates = {
        "neural_sparse_doc_only_v2_distill": (
            root / str(sparse_artifacts["sparse"]["run"])
        ),
        "bm25_neural_sparse_minmax_lw030": (
            root / str(sparse_artifacts["hybrid"]["run"])
        ),
    }
    qrels_path = root / "data/prepared/trec-product-search-2024/qrels.test.trec"
    primary = compare_run_family(
        root=root,
        qrels_path=qrels_path,
        baseline_name="tuned_bm25",
        baseline_run_path=root / str(bm25_artifacts["tuned"]["run"]),
        candidate_run_paths=candidates,
        metric=metric,
        alpha=alpha,
        minimum_delta=minimum_delta,
        eligible_for_decision=False,
        dataset="TREC Product Search 2024",
        split="held_out_test",
        exact_gain=3,
    )
    challenge = compare_run_family(
        root=root,
        qrels_path=qrels_path,
        baseline_name="competent_integrator_bm25",
        baseline_run_path=(
            root / str(bm25_artifacts["competent_integrator"]["run"])
        ),
        candidate_run_paths=candidates,
        metric=metric,
        alpha=alpha,
        minimum_delta=minimum_delta,
        eligible_for_decision=False,
        dataset="TREC Product Search 2024",
        split="held_out_test",
        exact_gain=3,
    )
    return {"primary": primary, "post_review_challenge": challenge}


def _assemble_trec_sparse_significance(
    *,
    comparisons: Mapping[str, Mapping[str, object]],
    source_evidence: Mapping[str, Mapping[str, object]],
    completion_source_evidence: Mapping[str, Mapping[str, object]],
    qrels_artifact: Mapping[str, object],
    completion_qrels_artifact: Mapping[str, object],
    registered_inputs: Mapping[str, object],
    benchmark_provenance: Mapping[str, Any],
    provenance_valid: bool,
    completion_revision: Mapping[str, Any],
) -> dict[str, object]:
    upstream_unchanged = (
        dict(source_evidence) == dict(completion_source_evidence)
        and dict(qrels_artifact) == dict(completion_qrels_artifact)
    )
    families = {
        "primary": ("bm25_tuned", "sparse_only", "sparse_hybrid"),
        "post_review_challenge": (
            "bm25_competent_integrator",
            "sparse_only",
            "sparse_hybrid",
        ),
    }
    family_evidence: dict[str, dict[str, object]] = {}
    for name, required_sources in families.items():
        source_eligibilities = {
            source: source_evidence[source].get("eligible_for_decision")
            for source in required_sources
        }
        eligible = derive_trec_significance_family_eligibility(
            provenance_valid=provenance_valid,
            upstream_unchanged=upstream_unchanged,
            source_eligibilities=tuple(source_eligibilities.values()),
        )
        reasons: list[str] = []
        if provenance_valid is not True:
            reasons.append(
                "no valid clean committed significance start and completion provenance"
            )
        if not upstream_unchanged:
            reasons.append("significance source evidence changed during computation")
        reasons.extend(
            f"source is not decision eligible: {source}"
            for source, source_eligible in source_eligibilities.items()
            if source_eligible is not True
        )
        family_evidence[name] = {
            "required_sources": list(required_sources),
            "source_eligibilities": source_eligibilities,
            "benchmark_provenance_valid": provenance_valid,
            "upstream_evidence_unchanged": upstream_unchanged,
            "eligible_for_decision": eligible,
            "ineligibility_reasons": reasons,
        }
    primary = dict(comparisons["primary"])
    primary["schema_version"] = 2
    apply_family_eligibility(
        primary,
        family_evidence["primary"]["eligible_for_decision"] is True,
    )
    challenge = dict(comparisons["post_review_challenge"])
    challenge["schema_version"] = 2
    apply_family_eligibility(
        challenge,
        family_evidence["post_review_challenge"]["eligible_for_decision"]
        is True,
    )
    primary.update(
        {
            "classification": "registered_held_out_paired_significance",
            "benchmark_provenance": dict(benchmark_provenance),
            "benchmark_provenance_valid": provenance_valid,
            "completion_code_revision": dict(completion_revision),
            "registered_inputs": dict(registered_inputs),
            "qrels_artifact": dict(qrels_artifact),
            "completion_qrels_artifact": dict(completion_qrels_artifact),
            "source_evidence": dict(source_evidence),
            "completion_source_evidence": dict(completion_source_evidence),
            "source_evidence_unchanged_at_completion": upstream_unchanged,
            "family_eligibility": family_evidence,
            "post_review_challenge": challenge,
        }
    )
    return primary


def _response_records(
    response: dict[str, Any],
    *,
    query_id: str,
    tag: str,
) -> list[RunRecord]:
    hits = cast(list[dict[str, Any]], cast(dict[str, Any], response["hits"])["hits"])
    if not hits:
        raise DatasetIntegrityError(f"TREC sparse query {query_id} returned no hits")
    records: list[RunRecord] = []
    for rank, hit in enumerate(hits, start=1):
        score = hit.get("_score")
        if score is None:
            raise DatasetIntegrityError(f"TREC sparse query {query_id} has no score")
        records.append(
            RunRecord(
                query_id=query_id,
                document_id=str(hit["_id"]),
                rank=rank,
                score=float(score),
                tag=tag,
            )
        )
    return records


def _verify_sparse_rerun(
    client: OpenSearchClient,
    *,
    root: Path,
    spec: TrecSparseSpec,
    entry: dict[str, Any],
    model_id: str,
    profile: BM25Profile,
    hybrid: bool,
) -> None:
    queries = load_prepared_queries(
        root / "data/prepared/trec-product-search-2024/queries.test.jsonl",
        expected_split="test",
    )
    manifest = cast(dict[str, Any], read_json(root / str(entry["manifest"])))
    if hybrid:
        repeated = execute_trec_sparse_hybrid(
            client,
            index=spec.index_name,
            queries=queries,
            spec=spec,
            model_id=model_id,
            profile=profile,
            tag=str(manifest["tag"]),
        )
    else:
        repeated = execute_trec_sparse(
            client,
            index=spec.index_name,
            queries=queries,
            spec=spec,
            model_id=model_id,
            tag=str(manifest["tag"]),
        )
    with tempfile.TemporaryDirectory(prefix="opensearch-hybrid-trec-sparse-verify-") as temp:
        repeated_path = Path(temp) / "repeated.trec"
        write_run(repeated_path, repeated)
        if repeated_path.read_bytes() != (root / str(entry["run"])).read_bytes():
            label = "hybrid" if hybrid else "sparse"
            raise DatasetIntegrityError(
                f"TREC {label} rerun is not byte-identical"
            )
