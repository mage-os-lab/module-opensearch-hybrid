from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from poc.bm25 import load_prepared_queries, verify_wands_bm25_selection
from poc.config import WANDS_INT8_MODEL_NAMES, ModelSpec, load_model_registry
from poc.datasets import DatasetIntegrityError, file_facts, load_wands_config
from poc.dense import encode_queries, query_vectors_sha256
from poc.experiments import BM25Experiments, BM25Profile, load_bm25_experiments
from poc.hybrid import (
    HybridRunResult,
    derive_hybrid_run_eligibility,
    hybrid_trec_score_for_rank,
    hybrid_upstream_artifact_bindings_valid,
    verify_hybrid_pipeline_artifact,
    verify_wands_hybrid_summary,
    write_hybrid_pipeline_bundle,
)
from poc.index_evidence import (
    content_addressed_pipeline_id,
    verify_live_search_pipeline,
)
from poc.indexing import (
    load_wands_index_config,
    vector_field_name,
    wands_completion_provenance_evidence,
    wands_recorded_provenance_valid,
)
from poc.int8_dense import verify_wands_int8_dense_summary
from poc.manifest import canonical_sha256, read_json
from poc.neural_sparse import (
    NeuralSparseSpec,
    build_sparse_query,
    load_neural_sparse_spec,
)
from poc.os_client import OpenSearchClient
from poc.provenance import collect_manifest_provenance, require_registered_opensearch_client
from poc.query_runtime import (
    QueryRuntimeSpec,
    create_int8_query_backend,
    load_query_runtime_spec,
)
from poc.search import (
    build_exact_knn_request,
    build_weighted_normalization_pipeline,
)
from poc.sparse_benchmark import verify_wands_sparse_summary
from poc.three_way import ThreeWayCandidate, ThreeWaySpec, load_three_way_spec
from poc.three_way_indexing import verify_three_way_index
from poc.trec import RunRecord, identifier_sort_key, read_run

RESULT_SIZE = 100
THREE_WAY_METHOD = "native_hybrid_three_way_exact_dense"
_SUMMARY_PATH = Path("results/wands/three-way/summary.json")
_INDEX_MANIFEST_PATH = Path("results/wands/three-way/index-manifest.json")


def _json_dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    return cast(dict[str, Any], json.loads(json.dumps(value)))


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }


def _write_json_exact(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_registered_three_way_benchmark_inputs(
    *,
    root: Path,
    spec: ThreeWaySpec,
    sparse: NeuralSparseSpec,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
    experiments: BM25Experiments,
) -> dict[str, Any]:
    registered_spec = load_three_way_spec(root / "config/three_way.toml")
    registered_sparse = load_neural_sparse_spec(root / "config/neural_sparse.toml")
    registry = load_model_registry(root / "config/models.toml")
    registered_runtime = load_query_runtime_spec(root / "config/query_runtime.toml")
    registered_experiments = load_bm25_experiments(root / "config/experiments.toml")
    dataset = load_wands_config(root / "config/datasets.toml")
    vector_index = load_wands_index_config(root / "config/indexes.toml")
    if spec != registered_spec:
        raise DatasetIntegrityError("three-way configuration differs from registry")
    if sparse != registered_sparse:
        raise DatasetIntegrityError(
            "three-way sparse configuration differs from registry"
        )
    if model.name != spec.dense_model or registry.get(spec.dense_model) != model:
        raise DatasetIntegrityError(
            "three-way dense model differs from registered configuration"
        )
    if runtime != registered_runtime:
        raise DatasetIntegrityError(
            "three-way query runtime differs from registered configuration"
        )
    if experiments != registered_experiments:
        raise DatasetIntegrityError(
            "three-way experiment configuration differs from registry"
        )
    return _json_dict(
        {
            "dataset": asdict(dataset),
            "three_way": asdict(registered_spec),
            "neural_sparse": asdict(registered_sparse),
            "dense_model": asdict(model),
            "query_runtime": asdict(registered_runtime),
            "experiments": asdict(registered_experiments),
            "vector_index": asdict(vector_index),
            "int8_model_order": list(WANDS_INT8_MODEL_NAMES),
            "config_artifacts": {
                name: _artifact(root, root / f"config/{filename}")
                for name, filename in (
                    ("datasets", "datasets.toml"),
                    ("indexes", "indexes.toml"),
                    ("models", "models.toml"),
                    ("neural_sparse", "neural_sparse.toml"),
                    ("query_runtime", "query_runtime.toml"),
                    ("experiments", "experiments.toml"),
                    ("three_way", "three_way.toml"),
                )
            },
        }
    )


def expected_three_way_artifact_keys(spec: ThreeWaySpec) -> tuple[str, ...]:
    return tuple(
        [f"dev/{candidate.name}" for candidate in spec.candidates]
        + ["test/selected"]
    )


def derive_three_way_summary_eligibility(
    *,
    provenance_valid: object,
    upstream_eligible: object,
    upstream_unchanged: object,
    upstream_artifact_bindings_valid: object,
    source_bindings_valid: object,
    held_out_artifact_eligible: object,
) -> tuple[bool, list[str]]:
    checks = (
        (
            provenance_valid is True,
            "three-way summary has no valid clean committed start provenance",
        ),
        (
            upstream_eligible is True,
            "three-way upstream evidence is not decision eligible",
        ),
        (
            upstream_unchanged is True,
            "three-way upstream evidence changed during benchmark",
        ),
        (
            upstream_artifact_bindings_valid is True,
            "three-way upstream artifact bindings differ",
        ),
        (
            source_bindings_valid is True,
            "three-way source manifest bindings differ",
        ),
        (
            held_out_artifact_eligible is True,
            "selected held-out three-way artifact is not decision eligible",
        ),
    )
    reasons = [reason for passed, reason in checks if not passed]
    return not reasons, reasons


def three_way_pipeline_id(candidate: ThreeWayCandidate) -> str:
    definition = three_way_pipeline(candidate)
    return content_addressed_pipeline_id(
        f"opensearch-hybrid-wands-three-way-{candidate.name.replace('_', '-')}-v1",
        definition,
    )


def three_way_pipeline(candidate: ThreeWayCandidate) -> dict[str, Any]:
    return build_weighted_normalization_pipeline(
        candidate.weights,
        description="OpenSearch Hybrid BM25 plus exact dense plus neural sparse min_max fusion",
    )


def build_three_way_request(
    *,
    text: str,
    vector: NDArray[np.float32],
    profile: BM25Profile,
    model: ModelSpec,
    sparse: NeuralSparseSpec,
) -> dict[str, Any]:
    return {
        "size": RESULT_SIZE,
        "_source": False,
        "track_scores": True,
        "query": {
            "hybrid": {
                "pagination_depth": RESULT_SIZE,
                "queries": [
                    profile.query(text),
                    build_exact_knn_request(
                        vector.tolist(),
                        vector_field=vector_field_name(model),
                        k=RESULT_SIZE,
                    )["query"],
                    build_sparse_query(sparse, text),
                ],
            }
        },
    }


def execute_three_way_run(
    client: OpenSearchClient,
    *,
    index: str,
    queries: dict[str, str],
    query_vectors: dict[str, NDArray[np.float32]],
    profile: BM25Profile,
    model: ModelSpec,
    sparse: NeuralSparseSpec,
    candidate: ThreeWayCandidate,
    tag: str,
) -> list[RunRecord]:
    if set(queries) != set(query_vectors):
        raise ValueError("three-way queries and vectors differ")
    pipeline_id = three_way_pipeline_id(candidate)
    pipeline = three_way_pipeline(candidate)
    client.put_search_pipeline(pipeline_id, pipeline)
    verify_live_search_pipeline(
        client,
        pipeline_id=pipeline_id,
        expected_definition=pipeline,
    )
    records: list[RunRecord] = []
    for query_id in sorted(queries, key=identifier_sort_key):
        response = client.search(
            index,
            build_three_way_request(
                text=queries[query_id],
                vector=query_vectors[query_id],
                profile=profile,
                model=model,
                sparse=sparse,
            ),
            pipeline=pipeline_id,
        )
        hits = cast(dict[str, Any], response.get("hits", {})).get("hits")
        if not isinstance(hits, list) or len(hits) != RESULT_SIZE:
            raise DatasetIntegrityError(
                f"three-way query {query_id} did not return {RESULT_SIZE} hits"
            )
        records.extend(
            RunRecord(
                query_id=query_id,
                document_id=str(hit["_id"]),
                rank=rank,
                score=hybrid_trec_score_for_rank(rank),
                tag=tag,
            )
            for rank, hit in enumerate(hits, start=1)
        )
    verify_live_search_pipeline(
        client,
        pipeline_id=pipeline_id,
        expected_definition=pipeline,
    )
    return records


def run_wands_three_way_benchmark(
    client: OpenSearchClient,
    *,
    root: Path,
    spec: ThreeWaySpec,
    sparse: NeuralSparseSpec,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
    experiments: BM25Experiments,
) -> dict[str, object]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    benchmark_provenance = _json_dict(collect_manifest_provenance(root))
    registered_inputs = verify_registered_three_way_benchmark_inputs(
        root=root,
        spec=spec,
        sparse=sparse,
        model=model,
        runtime=runtime,
        experiments=experiments,
    )
    if len(spec.candidates) != len(experiments.tuning_candidates):
        raise DatasetIntegrityError("three-way and BM25 tuning budgets differ")
    upstream_evidence, sources = _collect_verified_three_way_upstream_evidence(
        client,
        root=root,
        spec=spec,
        sparse=sparse,
        model=model,
        runtime=runtime,
        experiments=experiments,
        registered_inputs=registered_inputs,
    )
    prepared = root / "data/prepared/wands"
    query_paths = {
        split: prepared / f"queries.{split}.jsonl" for split in ("dev", "test")
    }
    qrels_paths = {
        split: prepared / f"qrels.{split}.trec" for split in ("dev", "test")
    }
    queries = {
        split: load_prepared_queries(query_paths[split], expected_split=split)
        for split in ("dev", "test")
    }
    all_queries = {**queries["dev"], **queries["test"]}
    bm25 = sources["bm25"]
    profile = BM25Profile.from_mapping(cast(dict[str, Any], bm25["selected_profile"]))
    int8_models = cast(dict[str, dict[str, Any]], sources["int8"]["models"])
    selected_model_eligible = (
        int8_models[model.name].get("eligible_for_decision") is True
    )
    backend = create_int8_query_backend(root=root, model=model, runtime=runtime)
    started = time.perf_counter()
    vectors = encode_queries(
        backend,
        model=model,
        queries=all_queries,
        batch_size=runtime.batch_size,
    )
    encoding_seconds = time.perf_counter() - started
    candidates: dict[str, HybridRunResult] = {}
    artifacts: dict[str, dict[str, object]] = {}
    index_manifest_path = root / _INDEX_MANIFEST_PATH
    dev_vectors = {key: vectors[key] for key in queries["dev"]}
    test_vectors = {key: vectors[key] for key in queries["test"]}
    for candidate in spec.candidates:
        tag = _three_way_tag(candidate, "dev")
        result = write_hybrid_pipeline_bundle(
            client,
            root=root,
            index=spec.index_name,
            model=model,
            backend=backend,
            profile=profile,
            pipeline_id=three_way_pipeline_id(candidate),
            pipeline_definition=three_way_pipeline(candidate),
            fusion=_fusion(candidate),
            declared_variable="lexical_dense_sparse_weights",
            metrics_directory="three-way",
            query_path=query_paths["dev"],
            qrels_path=qrels_paths["dev"],
            split="dev",
            tag=tag,
            records=execute_three_way_run(
                client,
                index=spec.index_name,
                queries=queries["dev"],
                query_vectors=dev_vectors,
                profile=profile,
                model=model,
                sparse=sparse,
                candidate=candidate,
                tag=tag,
            ),
            query_vectors_hash=query_vectors_sha256(dev_vectors),
            encoding_seconds=encoding_seconds,
            eligible_for_decision=False,
            ineligibility_reason="development tuning result",
            decision_scope="tuning",
            index_manifest_path=index_manifest_path,
            benchmark_provenance=benchmark_provenance,
            upstream_evidence=upstream_evidence,
            selected_model_eligible=selected_model_eligible,
            method_name=THREE_WAY_METHOD,
        )
        candidates[candidate.name] = result
        artifacts[f"dev/{candidate.name}"] = result.selection_entry(root)
    selected = max(
        spec.candidates,
        key=lambda candidate: candidates[candidate.name].evaluation.metrics[
            experiments.primary_metric
        ],
    )
    tag = _three_way_tag(selected, "test")
    test = write_hybrid_pipeline_bundle(
        client,
        root=root,
        index=spec.index_name,
        model=model,
        backend=backend,
        profile=profile,
        pipeline_id=three_way_pipeline_id(selected),
        pipeline_definition=three_way_pipeline(selected),
        fusion=_fusion(selected),
        declared_variable="retrieval_arm",
        metrics_directory="three-way",
        query_path=query_paths["test"],
        qrels_path=qrels_paths["test"],
        split="test",
        tag=tag,
        records=execute_three_way_run(
            client,
            index=spec.index_name,
            queries=queries["test"],
            query_vectors=test_vectors,
            profile=profile,
            model=model,
            sparse=sparse,
            candidate=selected,
            tag=tag,
        ),
        query_vectors_hash=query_vectors_sha256(test_vectors),
        encoding_seconds=encoding_seconds,
        eligible_for_decision=False,
        ineligibility_reason=None,
        decision_scope="held_out",
        index_manifest_path=index_manifest_path,
        benchmark_provenance=benchmark_provenance,
        upstream_evidence=upstream_evidence,
        selected_model_eligible=selected_model_eligible,
        method_name=THREE_WAY_METHOD,
    )
    artifacts["test/selected"] = test.selection_entry(root)
    completion_upstream, completion_sources = (
        _collect_verified_three_way_upstream_evidence(
            client,
            root=root,
            spec=spec,
            sparse=sparse,
            model=model,
            runtime=runtime,
            experiments=experiments,
            registered_inputs=registered_inputs,
        )
    )
    upstream_unchanged = completion_upstream == upstream_evidence
    if completion_sources != sources:
        upstream_unchanged = False
    upstream_artifact_bindings_valid = hybrid_upstream_artifact_bindings_valid(
        root,
        completion_upstream,
    )
    provenance_valid, completion_revision = wands_completion_provenance_evidence(
        root,
        benchmark_provenance,
    )
    source_bindings, source_bindings_valid, binding_reasons = (
        _three_way_source_manifest_bindings(
            root=root,
            spec=spec,
            artifacts=artifacts,
            benchmark_provenance=benchmark_provenance,
            upstream_evidence=upstream_evidence,
            selected_model_eligible=selected_model_eligible,
        )
    )
    held_out_manifest = cast(
        dict[str, Any],
        read_json(root / str(artifacts["test/selected"]["manifest"])),
    )
    eligible, eligibility_reasons = derive_three_way_summary_eligibility(
        provenance_valid=provenance_valid,
        upstream_eligible=upstream_evidence.get("eligible_for_decision"),
        upstream_unchanged=upstream_unchanged,
        upstream_artifact_bindings_valid=upstream_artifact_bindings_valid,
        source_bindings_valid=source_bindings_valid,
        held_out_artifact_eligible=held_out_manifest.get(
            "eligible_for_decision"
        ),
    )
    dense = sources["dense"]
    sparse_summary = sources["sparse"]
    tuned = float(bm25["held_out_test_metrics"]["tuned"]["ndcg@10"])
    dense_score = float(dense["held_out_test_metrics"]["ndcg@10"])
    sparse_score = float(sparse_summary["hybrid_test_metrics"]["ndcg@10"])
    score = test.evaluation.metrics["ndcg@10"]
    method_definition = _three_way_method_definition(
        spec=spec,
        model=model,
        experiments=experiments,
        selected=selected,
    )
    summary: dict[str, object] = {
        "schema_version": 2,
        "dataset": "WANDS",
        "method": THREE_WAY_METHOD,
        "method_definition": method_definition,
        "method_definition_sha256": canonical_sha256(method_definition),
        "registered_inputs": registered_inputs,
        "benchmark_provenance": benchmark_provenance,
        "benchmark_provenance_valid": provenance_valid,
        "completion_code_revision": completion_revision,
        "upstream_evidence": upstream_evidence,
        "completion_upstream_evidence": completion_upstream,
        "upstream_evidence_unchanged": upstream_unchanged,
        "upstream_artifact_bindings_valid_at_completion": (
            upstream_artifact_bindings_valid
        ),
        "required_source_manifest_keys": list(
            expected_three_way_artifact_keys(spec)
        ),
        "source_manifest_bindings": source_bindings,
        "source_manifest_bindings_valid": source_bindings_valid,
        "model": model.name,
        "query_runtime": "dynamic_int8_onnx_cpu_batch_1",
        "query_runtime_spec": asdict(runtime),
        "eligible_for_decision": eligible,
        "quality_evidence_eligible_for_decision": eligible,
        "ineligibility_reason": "; ".join(eligibility_reasons) or None,
        "quality_ineligibility_reasons": eligibility_reasons + binding_reasons,
        "selection_metric": experiments.primary_metric,
        "selection_split": "dev",
        "tie_break": "first_candidate_in_registered_order",
        "tuning_budget": len(spec.candidates),
        "bm25_tuning_budget": len(experiments.tuning_candidates),
        "registered_candidates": [
            {"name": candidate.name, "weights": list(candidate.weights)}
            for candidate in spec.candidates
        ],
        "candidate_dev_metrics": [
            {
                "name": candidate.name,
                "weights": list(candidate.weights),
                "metrics": candidates[candidate.name].evaluation.metrics,
            }
            for candidate in spec.candidates
        ],
        "selected_candidate": selected.name,
        "selected_weights": list(selected.weights),
        "held_out_test_metrics": test.evaluation.metrics,
        "three_way_minus_tuned_bm25_ndcg@10": score - tuned,
        "three_way_minus_dense_minmax_ndcg@10": score - dense_score,
        "three_way_minus_sparse_minmax_ndcg@10": score - sparse_score,
        "bm25_profile_sha256": canonical_sha256(profile.to_dict()),
        "artifacts": artifacts,
    }
    persisted = _json_dict(summary)
    _write_json_exact(root / _SUMMARY_PATH, persisted)
    return cast(dict[str, object], persisted)


def verify_wands_three_way_benchmark(
    client: OpenSearchClient,
    *,
    root: Path,
    spec: ThreeWaySpec,
    sparse: NeuralSparseSpec,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
    experiments: BM25Experiments,
) -> dict[str, Any]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    registered_inputs = verify_registered_three_way_benchmark_inputs(
        root=root,
        spec=spec,
        sparse=sparse,
        model=model,
        runtime=runtime,
        experiments=experiments,
    )
    if len(spec.candidates) != len(experiments.tuning_candidates):
        raise DatasetIntegrityError("three-way and BM25 tuning budgets differ")
    current_upstream, sources = _collect_verified_three_way_upstream_evidence(
        client,
        root=root,
        spec=spec,
        sparse=sparse,
        model=model,
        runtime=runtime,
        experiments=experiments,
        registered_inputs=registered_inputs,
    )
    summary = cast(dict[str, Any], read_json(root / _SUMMARY_PATH))
    if (
        summary.get("schema_version") != 2
        or summary.get("dataset") != "WANDS"
        or summary.get("method") != THREE_WAY_METHOD
        or summary.get("registered_inputs") != registered_inputs
        or summary.get("model") != model.name
        or summary.get("query_runtime") != "dynamic_int8_onnx_cpu_batch_1"
        or summary.get("query_runtime_spec") != _json_dict(asdict(runtime))
        or summary.get("selection_metric") != experiments.primary_metric
        or summary.get("selection_split") != "dev"
        or summary.get("tie_break") != "first_candidate_in_registered_order"
        or summary.get("tuning_budget") != len(spec.candidates)
        or summary.get("bm25_tuning_budget")
        != len(experiments.tuning_candidates)
        or summary.get("registered_candidates")
        != [
            {"name": candidate.name, "weights": list(candidate.weights)}
            for candidate in spec.candidates
        ]
    ):
        raise DatasetIntegrityError("three-way summary identity or registry differs")
    provenance = summary.get("benchmark_provenance")
    provenance_valid = wands_recorded_provenance_valid(
        root,
        provenance=provenance,
        completion_revision=summary.get("completion_code_revision"),
    )
    if (
        not isinstance(provenance, Mapping)
        or summary.get("benchmark_provenance_valid") is not provenance_valid
    ):
        raise DatasetIntegrityError("three-way summary provenance validity differs")
    start_upstream = summary.get("upstream_evidence")
    completion_upstream = summary.get("completion_upstream_evidence")
    upstream_unchanged = (
        isinstance(start_upstream, Mapping)
        and start_upstream == completion_upstream
        and completion_upstream == current_upstream
    )
    if (
        not isinstance(start_upstream, Mapping)
        or start_upstream != current_upstream
        or completion_upstream != current_upstream
        or summary.get("upstream_evidence_unchanged") is not upstream_unchanged
    ):
        raise DatasetIntegrityError("three-way summary upstream evidence differs")
    upstream_artifact_bindings_valid = hybrid_upstream_artifact_bindings_valid(
        root,
        current_upstream,
    )
    if (
        summary.get("upstream_artifact_bindings_valid_at_completion")
        is not upstream_artifact_bindings_valid
    ):
        raise DatasetIntegrityError(
            "three-way summary upstream artifact bindings differ"
        )
    artifacts_value = summary.get("artifacts")
    if not isinstance(artifacts_value, dict):
        raise DatasetIntegrityError("three-way summary artifacts are missing")
    artifacts = cast(dict[str, dict[str, Any]], artifacts_value)
    expected_keys = expected_three_way_artifact_keys(spec)
    if set(artifacts) != set(expected_keys):
        raise DatasetIntegrityError("three-way summary artifact set differs")
    bm25 = sources["bm25"]
    profile = BM25Profile.from_mapping(cast(dict[str, Any], bm25["selected_profile"]))
    if summary.get("bm25_profile_sha256") != canonical_sha256(profile.to_dict()):
        raise DatasetIntegrityError("three-way BM25 profile binding differs")
    int8_models = cast(dict[str, dict[str, Any]], sources["int8"]["models"])
    selected_model_eligible = (
        int8_models[model.name].get("eligible_for_decision") is True
    )
    prepared = root / "data/prepared/wands"
    query_paths = {
        split: prepared / f"queries.{split}.jsonl" for split in ("dev", "test")
    }
    queries = {
        split: load_prepared_queries(query_paths[split], expected_split=split)
        for split in ("dev", "test")
    }
    all_queries = {**queries["dev"], **queries["test"]}
    backend = create_int8_query_backend(root=root, model=model, runtime=runtime)
    vectors = encode_queries(
        backend,
        model=model,
        queries=all_queries,
        batch_size=runtime.batch_size,
    )
    dev_vectors = {key: vectors[key] for key in queries["dev"]}
    test_vectors = {key: vectors[key] for key in queries["test"]}
    candidate_metrics: dict[str, dict[str, float]] = {}
    for candidate in spec.candidates:
        key = f"dev/{candidate.name}"
        entry = artifacts[key]
        _verify_three_way_artifact(
            client,
            root=root,
            spec=spec,
            sparse=sparse,
            model=model,
            runtime=runtime,
            profile=profile,
            backend_artifact_sha256=backend.model_artifact_sha256,
            candidate=candidate,
            split="dev",
            entry=entry,
            benchmark_provenance=cast(Mapping[str, Any], provenance),
            upstream_evidence=cast(Mapping[str, Any], start_upstream),
            selected_model_eligible=selected_model_eligible,
            queries=queries["dev"],
            query_vectors=dev_vectors,
        )
        candidate_metrics[candidate.name] = cast(
            dict[str, float], entry["metrics"]
        )
    expected_selected = max(
        spec.candidates,
        key=lambda candidate: candidate_metrics[candidate.name][
            experiments.primary_metric
        ],
    )
    expected_candidate_metrics = [
        {
            "name": candidate.name,
            "weights": list(candidate.weights),
            "metrics": candidate_metrics[candidate.name],
        }
        for candidate in spec.candidates
    ]
    if (
        summary.get("candidate_dev_metrics") != expected_candidate_metrics
        or summary.get("selected_candidate") != expected_selected.name
        or summary.get("selected_weights") != list(expected_selected.weights)
    ):
        raise DatasetIntegrityError("three-way development selection differs")
    test_entry = artifacts["test/selected"]
    test_manifest = _verify_three_way_artifact(
        client,
        root=root,
        spec=spec,
        sparse=sparse,
        model=model,
        runtime=runtime,
        profile=profile,
        backend_artifact_sha256=backend.model_artifact_sha256,
        candidate=expected_selected,
        split="test",
        entry=test_entry,
        benchmark_provenance=cast(Mapping[str, Any], provenance),
        upstream_evidence=cast(Mapping[str, Any], start_upstream),
        selected_model_eligible=selected_model_eligible,
        queries=queries["test"],
        query_vectors=test_vectors,
    )
    del backend
    held_out_metrics = cast(dict[str, float], test_entry["metrics"])
    if summary.get("held_out_test_metrics") != held_out_metrics:
        raise DatasetIntegrityError("three-way held-out metrics differ")
    tuned = float(bm25["held_out_test_metrics"]["tuned"]["ndcg@10"])
    dense_score = float(sources["dense"]["held_out_test_metrics"]["ndcg@10"])
    sparse_score = float(sources["sparse"]["hybrid_test_metrics"]["ndcg@10"])
    score = float(held_out_metrics["ndcg@10"])
    expected_metrics: dict[str, object] = {
        "three_way_minus_tuned_bm25_ndcg@10": score - tuned,
        "three_way_minus_dense_minmax_ndcg@10": score - dense_score,
        "three_way_minus_sparse_minmax_ndcg@10": score - sparse_score,
    }
    if any(summary.get(key) != value for key, value in expected_metrics.items()):
        raise DatasetIntegrityError("three-way held-out deltas differ")
    method_definition = _three_way_method_definition(
        spec=spec,
        model=model,
        experiments=experiments,
        selected=expected_selected,
    )
    if (
        summary.get("method_definition") != method_definition
        or summary.get("method_definition_sha256")
        != canonical_sha256(method_definition)
    ):
        raise DatasetIntegrityError("three-way method definition differs")
    source_bindings, bindings_valid, binding_reasons = (
        _three_way_source_manifest_bindings(
            root=root,
            spec=spec,
            artifacts=artifacts,
            benchmark_provenance=cast(Mapping[str, Any], provenance),
            upstream_evidence=cast(Mapping[str, Any], start_upstream),
            selected_model_eligible=selected_model_eligible,
        )
    )
    if (
        summary.get("required_source_manifest_keys") != list(expected_keys)
        or summary.get("source_manifest_bindings") != source_bindings
        or summary.get("source_manifest_bindings_valid") is not bindings_valid
    ):
        raise DatasetIntegrityError("three-way source manifest bindings differ")
    eligible, reasons = derive_three_way_summary_eligibility(
        provenance_valid=provenance_valid,
        upstream_eligible=current_upstream.get("eligible_for_decision"),
        upstream_unchanged=upstream_unchanged,
        upstream_artifact_bindings_valid=upstream_artifact_bindings_valid,
        source_bindings_valid=bindings_valid,
        held_out_artifact_eligible=test_manifest.get("eligible_for_decision"),
    )
    if (
        summary.get("eligible_for_decision") is not eligible
        or summary.get("quality_evidence_eligible_for_decision") is not eligible
        or summary.get("ineligibility_reason")
        != ("; ".join(reasons) or None)
        or summary.get("quality_ineligibility_reasons")
        != reasons + binding_reasons
    ):
        raise DatasetIntegrityError("three-way summary eligibility differs")
    completion_index = verify_three_way_index(
        client,
        root=root,
        dataset=load_wands_config(root / "config/datasets.toml"),
        spec=spec,
        sparse=sparse,
        model=model,
        products_path=root / "data/prepared/wands/products.jsonl",
        sparse_embeddings_path=(
            root / "data/cache/neural-sparse/wands-doc-v3-distill.jsonl"
        ),
        sparse_manifest_path=(
            root / "results/wands/neural-sparse/precompute-manifest.json"
        ),
        manifest_path=root / _INDEX_MANIFEST_PATH,
    )
    recorded_index_evidence = current_upstream.get("three_way_index")
    if (
        not isinstance(recorded_index_evidence, Mapping)
        or completion_index.get("index")
        != recorded_index_evidence.get("index")
        or completion_index.get("quality_evidence_eligible_for_decision")
        != recorded_index_evidence.get(
            "quality_evidence_eligible_for_decision"
        )
    ):
        raise DatasetIntegrityError(
            "three-way index changed during decision replay"
        )
    return summary


def _collect_verified_three_way_upstream_evidence(
    client: OpenSearchClient,
    *,
    root: Path,
    spec: ThreeWaySpec,
    sparse: NeuralSparseSpec,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
    experiments: BM25Experiments,
    registered_inputs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    dataset = load_wands_config(root / "config/datasets.toml")
    vector_index = load_wands_index_config(root / "config/indexes.toml")
    registry = load_model_registry(root / "config/models.toml")
    int8_models = tuple(registry[name] for name in WANDS_INT8_MODEL_NAMES)
    three_way_index = verify_three_way_index(
        client,
        root=root,
        dataset=dataset,
        spec=spec,
        sparse=sparse,
        model=model,
        products_path=root / "data/prepared/wands/products.jsonl",
        sparse_embeddings_path=(
            root / "data/cache/neural-sparse/wands-doc-v3-distill.jsonl"
        ),
        sparse_manifest_path=(
            root / "results/wands/neural-sparse/precompute-manifest.json"
        ),
        manifest_path=root / _INDEX_MANIFEST_PATH,
    )
    bm25 = verify_wands_bm25_selection(
        client,
        root=root,
        index=vector_index.name,
        experiments=experiments,
    )
    int8 = verify_wands_int8_dense_summary(
        client,
        root=root,
        index=vector_index.name,
        models=int8_models,
        runtime=runtime,
    )
    dense = verify_wands_hybrid_summary(
        client,
        root=root,
        index=vector_index.name,
        models=int8_models,
        runtime=runtime,
        experiments=experiments,
    )
    sparse_summary = verify_wands_sparse_summary(
        client,
        root=root,
        spec=sparse,
        experiments=experiments,
    )
    int8_model_values = int8.get("models")
    if not isinstance(int8_model_values, dict) or set(int8_model_values) != {
        item.name for item in int8_models
    }:
        raise DatasetIntegrityError("three-way int8 summary model set differs")
    selected_int8 = cast(dict[str, Any], int8_model_values[model.name])
    if dense.get("model") != model.name:
        raise DatasetIntegrityError(
            "three-way dense comparator uses a different selected model"
        )
    sources = {
        "bm25": bm25,
        "int8": int8,
        "dense": dense,
        "sparse": sparse_summary,
    }
    evidence = _json_dict(
        {
            "schema_version": 2,
            "registered_inputs": registered_inputs,
            "three_way_index": {
                "artifact": _artifact(root, root / _INDEX_MANIFEST_PATH),
                "index": three_way_index.get("index"),
                "quality_evidence_eligible_for_decision": three_way_index.get(
                    "quality_evidence_eligible_for_decision"
                ),
            },
            "bm25_selection": {
                "artifact": _artifact(
                    root, root / "results/wands/bm25-selection.json"
                ),
                "selected_profile_sha256": bm25.get("selected_profile_sha256"),
                "quality_evidence_eligible_for_decision": bm25.get(
                    "quality_evidence_eligible_for_decision"
                ),
            },
            "int8_dense": {
                "artifact": _artifact(
                    root, root / "results/wands/int8-dense-summary.json"
                ),
                "quality_evidence_eligible_for_decision": int8.get(
                    "quality_evidence_eligible_for_decision"
                ),
                "selected_model_eligible_for_decision": selected_int8.get(
                    "eligible_for_decision"
                ),
            },
            "dense_hybrid": {
                "artifact": _artifact(root, root / "results/wands/hybrid-summary.json"),
                "model": dense.get("model"),
                "eligible_for_decision": dense.get("eligible_for_decision"),
            },
            "neural_sparse": {
                "artifact": _artifact(
                    root, root / "results/wands/neural-sparse/summary.json"
                ),
                "quality_evidence_eligible_for_decision": sparse_summary.get(
                    "quality_evidence_eligible_for_decision"
                ),
            },
            "eligible_for_decision": (
                three_way_index.get("quality_evidence_eligible_for_decision")
                is True
                and bm25.get("quality_evidence_eligible_for_decision") is True
                and int8.get("quality_evidence_eligible_for_decision") is True
                and selected_int8.get("eligible_for_decision") is True
                and dense.get("eligible_for_decision") is True
                and sparse_summary.get("quality_evidence_eligible_for_decision")
                is True
            ),
        }
    )
    evidence["ineligibility_reasons"] = [
        name
        for name, eligible in (
            (
                "three_way_index",
                three_way_index.get("quality_evidence_eligible_for_decision"),
            ),
            ("bm25_selection", bm25.get("quality_evidence_eligible_for_decision")),
            (
                "int8_dense",
                int8.get("quality_evidence_eligible_for_decision"),
            ),
            ("selected_int8_model", selected_int8.get("eligible_for_decision")),
            ("dense_hybrid", dense.get("eligible_for_decision")),
            (
                "neural_sparse",
                sparse_summary.get("quality_evidence_eligible_for_decision"),
            ),
        )
        if eligible is not True
    ]
    return evidence, sources


def _verify_three_way_artifact(
    client: OpenSearchClient,
    *,
    root: Path,
    spec: ThreeWaySpec,
    sparse: NeuralSparseSpec,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
    profile: BM25Profile,
    backend_artifact_sha256: str,
    candidate: ThreeWayCandidate,
    split: str,
    entry: dict[str, Any],
    benchmark_provenance: Mapping[str, Any],
    upstream_evidence: Mapping[str, Any],
    selected_model_eligible: bool,
    queries: dict[str, str],
    query_vectors: dict[str, NDArray[np.float32]],
) -> dict[str, Any]:
    expected_tag = _three_way_tag(candidate, split)
    expected_paths = {
        "run": f"runs/{expected_tag}.trec",
        "manifest": f"runs/{expected_tag}.manifest.json",
        "metrics_file": f"results/wands/three-way/{expected_tag}.metrics.json",
    }
    if any(entry.get(key) != path for key, path in expected_paths.items()):
        raise DatasetIntegrityError(
            f"three-way artifact paths differ: {candidate.name}/{split}"
        )
    decision_scope = "tuning" if split == "dev" else "held_out"
    manifest = verify_hybrid_pipeline_artifact(
        client,
        root=root,
        index=spec.index_name,
        backend_artifact_sha256=backend_artifact_sha256,
        entry=entry,
        expected_pipeline_id=three_way_pipeline_id(candidate),
        expected_pipeline=three_way_pipeline(candidate),
        expected_fusion=_fusion(candidate),
        expected_index_manifest_path=root / _INDEX_MANIFEST_PATH,
        expected_model=model,
        expected_method_name=THREE_WAY_METHOD,
        expected_split=split,
        expected_decision_scope=decision_scope,
        expected_benchmark_provenance=benchmark_provenance,
        expected_upstream_evidence=upstream_evidence,
        expected_selected_model_eligible=selected_model_eligible,
        expected_runtime=runtime,
    )
    expected_variable = (
        "lexical_dense_sparse_weights" if split == "dev" else "retrieval_arm"
    )
    if (
        manifest.get("tag") != expected_tag
        or manifest.get("declared_variable") != expected_variable
        or manifest.get("query_vectors_sha256")
        != query_vectors_sha256(query_vectors)
        or manifest.get("bm25_profile") != _json_dict(profile.to_dict())
    ):
        raise DatasetIntegrityError(
            f"three-way run metadata differs: {candidate.name}/{split}"
        )
    rerun = execute_three_way_run(
        client,
        index=spec.index_name,
        queries=queries,
        query_vectors=query_vectors,
        profile=profile,
        model=model,
        sparse=sparse,
        candidate=candidate,
        tag=expected_tag,
    )
    recorded = read_run(root / expected_paths["run"])
    if sorted(rerun, key=lambda record: (record.query_id, record.rank)) != recorded:
        raise DatasetIntegrityError(
            f"three-way live rerun differs: {candidate.name}/{split}"
        )
    return manifest


def _three_way_source_manifest_bindings(
    *,
    root: Path,
    spec: ThreeWaySpec,
    artifacts: Mapping[str, Mapping[str, Any]],
    benchmark_provenance: Mapping[str, Any],
    upstream_evidence: Mapping[str, Any],
    selected_model_eligible: bool,
) -> tuple[dict[str, dict[str, object]], bool, list[str]]:
    bindings: dict[str, dict[str, object]] = {}
    reasons: list[str] = []
    expected_keys = expected_three_way_artifact_keys(spec)
    if set(artifacts) != set(expected_keys):
        return bindings, False, ["three-way artifact set differs"]
    upstream_bindings_valid = hybrid_upstream_artifact_bindings_valid(
        root,
        upstream_evidence,
    )
    for key in expected_keys:
        entry = artifacts[key]
        manifest_path = root / str(entry.get("manifest", ""))
        if not manifest_path.is_file():
            reasons.append(f"three-way source manifest is missing: {key}")
            continue
        bindings[key] = _artifact(root, manifest_path)
        manifest = cast(dict[str, Any], read_json(manifest_path))
        expected_split = "test" if key == "test/selected" else "dev"
        expected_scope = "held_out" if expected_split == "test" else "tuning"
        provenance_valid = wands_recorded_provenance_valid(
            root,
            provenance=manifest.get("benchmark_provenance"),
            completion_revision=manifest.get("completion_code_revision"),
        )
        expected_eligible, expected_reason = derive_hybrid_run_eligibility(
            decision_scope=expected_scope,
            benchmark_provenance_valid=provenance_valid,
            upstream_evidence_eligible=(
                upstream_evidence.get("eligible_for_decision") is True
                and upstream_bindings_valid
            ),
            selected_model_eligible=selected_model_eligible,
        )
        if not (
            manifest.get("schema_version") == 2
            and manifest.get("dataset") == "WANDS"
            and manifest.get("method") == THREE_WAY_METHOD
            and manifest.get("split") == expected_split
            and manifest.get("decision_scope") == expected_scope
            and manifest.get("benchmark_provenance") == benchmark_provenance
            and manifest.get("benchmark_provenance_valid") is provenance_valid
            and manifest.get("upstream_evidence") == upstream_evidence
            and manifest.get(
                "upstream_artifact_bindings_valid_at_completion"
            )
            is upstream_bindings_valid
            and manifest.get("selected_model_eligible")
            is selected_model_eligible
            and manifest.get("eligible_for_decision") is expected_eligible
            and manifest.get("ineligibility_reason") == expected_reason
            and entry.get("manifest_sha256")
            == file_facts(manifest_path).sha256
        ):
            reasons.append(f"three-way source manifest binding differs: {key}")
    return bindings, not reasons, reasons


def _three_way_method_definition(
    *,
    spec: ThreeWaySpec,
    model: ModelSpec,
    experiments: BM25Experiments,
    selected: ThreeWayCandidate,
) -> dict[str, object]:
    return {
        "method": THREE_WAY_METHOD,
        "quality_query": "native_hybrid_with_exact_knn_score_script",
        "query_clauses": ["tuned_bm25", "exact_dense", "neural_sparse"],
        "dense_model": model.name,
        "selection_metric": experiments.primary_metric,
        "selection_split": "dev",
        "tie_break": "first_candidate_in_registered_order",
        "registered_candidates": [
            {
                "name": candidate.name,
                "weights": list(candidate.weights),
                "pipeline_id": three_way_pipeline_id(candidate),
                "pipeline_sha256": canonical_sha256(
                    three_way_pipeline(candidate)
                ),
                "fusion_sha256": canonical_sha256(_fusion(candidate)),
            }
            for candidate in spec.candidates
        ],
        "selected_candidate": selected.name,
        "selected_pipeline_id": three_way_pipeline_id(selected),
        "selected_pipeline_sha256": canonical_sha256(
            three_way_pipeline(selected)
        ),
        "selected_fusion_sha256": canonical_sha256(_fusion(selected)),
    }


def _three_way_tag(candidate: ThreeWayCandidate, split: str) -> str:
    if split not in {"dev", "test"}:
        raise ValueError("three-way split must be dev or test")
    return f"wands-three-way-{candidate.name.replace('_', '-')}-{split}"


def _fusion(candidate: ThreeWayCandidate) -> dict[str, object]:
    return {
        "normalization": "min_max",
        "combination": "arithmetic_mean",
        "lexical_weight": candidate.lexical_weight,
        "dense_weight": candidate.dense_weight,
        "sparse_weight": candidate.sparse_weight,
    }
