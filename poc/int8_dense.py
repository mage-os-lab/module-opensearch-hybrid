from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from poc.bm25 import load_prepared_queries
from poc.config import WANDS_INT8_MODEL_NAMES, ModelSpec, load_model_registry
from poc.datasets import DatasetIntegrityError, file_facts
from poc.dense import (
    encode_queries,
    execute_exact_dense_run,
    query_vectors_sha256,
    verify_dense_artifact,
    verify_wands_dense_summary,
    write_dense_bundle,
)
from poc.evaluation import evaluate_run
from poc.indexing import (
    load_wands_index_config,
    wands_completion_provenance_evidence,
    wands_recorded_provenance_valid,
)
from poc.manifest import read_json, write_json
from poc.os_client import OpenSearchClient
from poc.protocol_amendments import (
    assess_query_runtime_quality,
    compare_ranking_stability,
    load_query_runtime_quality_guard,
)
from poc.provenance import (
    collect_code_revision,
    collect_manifest_provenance,
    require_registered_opensearch_client,
)
from poc.query_runtime import (
    QueryRuntimeSpec,
    create_int8_query_backend,
    load_query_runtime_spec,
    onnx_manifest_path,
    verify_onnx_int8_artifact,
    verify_query_parity_artifact,
    verify_self_retrieval_artifact,
)
from poc.trec import read_run, write_run

INT8_CANDIDATE_DEPTH = 200
INT8_SCORE_TIE_DECIMAL_PLACES = 6
INT8_RANKING_POLICY = "rounded_6_decimal_score_then_product_id_from_exact_top_200"
_SPLITS = ("dev", "test")


def _json_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(json.dumps(value)))


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }


def classify_int8_evidence(
    *,
    model_decision_eligible: bool,
    run_provenance_valid: bool,
    vector_index_eligible: bool,
    fp32_model_selection_eligible: bool,
    onnx_artifact_eligible: bool,
    parity_artifact_verified: bool,
    quality_guard_passed: bool,
    self_retrieval_passed: bool,
) -> tuple[bool, str | None]:
    if not model_decision_eligible:
        return False, "model registry excludes this model from decision evidence"
    if not run_provenance_valid:
        return False, "benchmark start/completion provenance is ineligible"
    if not vector_index_eligible:
        return False, "semantic WANDS vector index evidence is ineligible"
    if not fp32_model_selection_eligible:
        return False, "exploratory fp32 model-selection evidence is ineligible"
    if not onnx_artifact_eligible:
        return False, "registered ONNX artifact evidence is ineligible"
    if not parity_artifact_verified:
        return False, "query-runtime parity artifact is not semantically verified"
    if not quality_guard_passed:
        return False, "amended query-runtime quality guard failed"
    if not self_retrieval_passed:
        return False, "registered int8 self-retrieval gate failed"
    return True, None


def build_query_runtime_quality_artifact(
    *,
    root: Path,
    model: ModelSpec,
    fp32_entry: Mapping[str, Any],
    int8_entry: Mapping[str, Any],
    benchmark_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    _verify_quality_input_entry(
        root=root,
        model=model,
        entry=fp32_entry,
        required_eligibility_field="eligible_as_model_selection_input",
    )
    _verify_quality_input_entry(
        root=root,
        model=model,
        entry=int8_entry,
        required_eligibility_field="eligible_as_quality_guard_input",
    )
    guard = load_query_runtime_quality_guard(root / "config/protocol_amendments.toml")
    fp32_path = root / str(fp32_entry["run"])
    int8_path = root / str(int8_entry["run"])
    result = assess_query_runtime_quality(
        fp32_ndcg_at_10=float(cast(Mapping[str, Any], fp32_entry["metrics"])["ndcg@10"]),
        candidate_ndcg_at_10=float(
            cast(Mapping[str, Any], int8_entry["metrics"])["ndcg@10"]
        ),
        stability=compare_ranking_stability(
            read_run(fp32_path), read_run(int8_path), cutoff=10
        ),
        guard=guard,
    )
    fp32_manifest_path = root / str(fp32_entry["manifest"])
    int8_manifest_path = root / str(int8_entry["manifest"])
    fp32_manifest = cast(dict[str, Any], read_json(fp32_manifest_path))
    int8_manifest = cast(dict[str, Any], read_json(int8_manifest_path))
    completion_revision = asdict(collect_code_revision(root))
    provenance_valid = wands_recorded_provenance_valid(
        root,
        provenance=benchmark_provenance,
        completion_revision=completion_revision,
    )
    source_inputs_eligible = (
        fp32_manifest.get("eligible_as_model_selection_input") is True
        and int8_manifest.get("eligible_as_quality_guard_input") is True
    )
    result.update(
        {
            "schema_version": 2,
            "model": asdict(model),
            "selection_surface": "WANDS development split",
            "held_out_test_metrics_used": False,
            "reference_run": _artifact(root, fp32_path),
            "candidate_run": _artifact(root, int8_path),
            "protocol_registry": _artifact(
                root, root / "config/protocol_amendments.toml"
            ),
            "benchmark_provenance": _json_dict(benchmark_provenance),
            "completion_code_revision": completion_revision,
            "source_inputs_eligible": source_inputs_eligible,
            "quality_evidence_eligible_for_decision": (
                provenance_valid and source_inputs_eligible
            ),
        }
    )
    return result


def verify_query_runtime_quality_artifact(
    *,
    root: Path,
    model: ModelSpec,
    fp32_entry: Mapping[str, Any],
    int8_entry: Mapping[str, Any],
) -> dict[str, Any]:
    _verify_quality_input_entry(
        root=root,
        model=model,
        entry=fp32_entry,
        required_eligibility_field="eligible_as_model_selection_input",
    )
    _verify_quality_input_entry(
        root=root,
        model=model,
        entry=int8_entry,
        required_eligibility_field="eligible_as_quality_guard_input",
    )
    path = root / f"results/wands/query-runtime/{model.name}.quality-guard.json"
    artifact = cast(dict[str, Any], read_json(path))
    if artifact.get("schema_version") != 2 or artifact.get("model") != asdict(model):
        raise DatasetIntegrityError(f"query-runtime quality schema differs for {model.name}")
    guard = load_query_runtime_quality_guard(root / "config/protocol_amendments.toml")
    fp32_path = root / str(fp32_entry["run"])
    int8_path = root / str(int8_entry["run"])
    expected = assess_query_runtime_quality(
        fp32_ndcg_at_10=float(cast(Mapping[str, Any], fp32_entry["metrics"])["ndcg@10"]),
        candidate_ndcg_at_10=float(
            cast(Mapping[str, Any], int8_entry["metrics"])["ndcg@10"]
        ),
        stability=compare_ranking_stability(
            read_run(fp32_path), read_run(int8_path), cutoff=10
        ),
        guard=guard,
    )
    for key in (
        "amendment",
        "fp32_ndcg@10",
        "candidate_ndcg@10",
        "ndcg@10_loss",
        "ranking_stability",
        "checks",
        "status",
    ):
        if artifact.get(key) != expected[key]:
            raise DatasetIntegrityError(
                f"query-runtime quality result is not derived for {model.name}"
            )
    if (
        artifact.get("selection_surface") != "WANDS development split"
        or artifact.get("held_out_test_metrics_used") is not False
        or artifact.get("reference_run") != _artifact(root, fp32_path)
        or artifact.get("candidate_run") != _artifact(root, int8_path)
        or artifact.get("protocol_registry")
        != _artifact(root, root / "config/protocol_amendments.toml")
    ):
        raise DatasetIntegrityError(f"query-runtime quality sources differ for {model.name}")
    provenance_valid = wands_recorded_provenance_valid(
        root,
        provenance=artifact.get("benchmark_provenance"),
        completion_revision=artifact.get("completion_code_revision"),
    )
    fp32_manifest = cast(
        dict[str, Any], read_json(root / str(fp32_entry["manifest"]))
    )
    int8_manifest = cast(
        dict[str, Any], read_json(root / str(int8_entry["manifest"]))
    )
    source_inputs_eligible = (
        fp32_manifest.get("eligible_as_model_selection_input") is True
        and int8_manifest.get("eligible_as_quality_guard_input") is True
    )
    expected_eligible = provenance_valid and source_inputs_eligible
    if (
        artifact.get("source_inputs_eligible") is not source_inputs_eligible
        or artifact.get("quality_evidence_eligible_for_decision") is not expected_eligible
    ):
        raise DatasetIntegrityError(f"query-runtime quality eligibility differs for {model.name}")
    return artifact


def _verify_quality_input_entry(
    *,
    root: Path,
    model: ModelSpec,
    entry: Mapping[str, Any],
    required_eligibility_field: str,
) -> None:
    run_path = root / str(entry["run"])
    manifest_path = root / str(entry["manifest"])
    metrics_path = root / str(entry["metrics_file"])
    manifest = cast(dict[str, Any], read_json(manifest_path))
    if (
        manifest.get("schema_version") != 2
        or manifest.get("dataset") != "WANDS"
        or manifest.get("split") != "dev"
        or manifest.get("model") != asdict(model)
    ):
        raise DatasetIntegrityError(
            f"query-runtime quality input differs for {model.name}"
        )
    provenance_valid = wands_recorded_provenance_valid(
        root,
        provenance=manifest.get("benchmark_provenance"),
        completion_revision=manifest.get("completion_code_revision"),
    )
    upstream_eligible = manifest.get("upstream_evidence_eligible") is True
    expected_eligible = provenance_valid and upstream_eligible
    if (
        manifest.get("benchmark_provenance_valid") is not provenance_valid
        or manifest.get(required_eligibility_field) is not expected_eligible
        or not expected_eligible
    ):
        raise DatasetIntegrityError(
            f"query-runtime quality input eligibility differs for {model.name}"
        )
    run_file = manifest.get("run_file")
    query_file = manifest.get("query_file")
    qrels_file = manifest.get("qrels_file")
    if not isinstance(run_file, dict) or not isinstance(query_file, dict) or not isinstance(
        qrels_file, dict
    ):
        raise DatasetIntegrityError(
            f"query-runtime quality input bindings are missing for {model.name}"
        )
    query_path = root / str(query_file.get("path"))
    qrels_path = root / str(qrels_file.get("path"))
    if (
        query_path != root / "data/prepared/wands/queries.dev.jsonl"
        or qrels_path != root / "data/prepared/wands/qrels.dev.trec"
        or file_facts(query_path).sha256 != query_file.get("sha256")
        or file_facts(qrels_path).sha256 != qrels_file.get("sha256")
        or file_facts(run_path).sha256 != run_file.get("sha256")
        or manifest.get("metrics_file") != _artifact(root, metrics_path)
    ):
        raise DatasetIntegrityError(
            f"query-runtime quality input bytes differ for {model.name}"
        )
    evaluation = evaluate_run(qrels_path, run_path)
    metrics = cast(dict[str, Any], read_json(metrics_path))
    if (
        metrics.get("schema_version") != 2
        or metrics.get("metrics") != evaluation.metrics
        or metrics.get("per_query") != evaluation.per_query
        or entry.get("metrics") != evaluation.metrics
    ):
        raise DatasetIntegrityError(
            f"query-runtime quality input metrics differ for {model.name}"
        )


def _runtime_evidence(
    *,
    root: Path,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
    fp32_entry: Mapping[str, Any],
    int8_entry: Mapping[str, Any] | None,
) -> dict[str, object]:
    onnx = verify_onnx_int8_artifact(root=root, model=model, runtime=runtime)
    parity = verify_query_parity_artifact(root=root, model=model, runtime=runtime)
    self_retrieval = verify_self_retrieval_artifact(
        root=root, model=model, runtime=runtime
    )
    quality_path = root / f"results/wands/query-runtime/{model.name}.quality-guard.json"
    quality: dict[str, Any] | None = None
    quality_error: str | None = None
    if int8_entry is not None and quality_path.is_file():
        try:
            quality = verify_query_runtime_quality_artifact(
                root=root,
                model=model,
                fp32_entry=fp32_entry,
                int8_entry=int8_entry,
            )
        except (DatasetIntegrityError, KeyError, OSError, ValueError) as error:
            quality_error = str(error)
    return {
        "onnx_manifest": _artifact(root, onnx_manifest_path(root, model, runtime)),
        "onnx_artifact_eligible": onnx.get(
            "quality_evidence_eligible_for_decision"
        )
        is True,
        "parity_artifact": _artifact(
            root, root / f"results/wands/query-runtime/{model.name}.parity.json"
        ),
        "parity_status": parity["status"],
        "parity_artifact_verified": (
            parity.get("quality_evidence_eligible_for_decision") is True
        ),
        "self_retrieval_artifact": _artifact(
            root,
            root / f"results/wands/query-runtime/{model.name}.self-retrieval.json",
        ),
        "self_retrieval_status": self_retrieval["status"],
        "self_retrieval_artifact_verified": (
            self_retrieval.get("quality_evidence_eligible_for_decision") is True
        ),
        "self_retrieval_passed": (
            self_retrieval["status"] == "passed"
            and self_retrieval.get("quality_evidence_eligible_for_decision") is True
        ),
        "quality_guard_artifact": (
            _artifact(root, quality_path) if quality is not None else None
        ),
        "quality_guard_status": quality.get("status") if quality is not None else None,
        "quality_guard_artifact_verified": (
            quality is not None
            and quality.get("quality_evidence_eligible_for_decision") is True
        ),
        "quality_guard_passed": (
            quality is not None
            and quality.get("status") == "passed"
            and quality.get("quality_evidence_eligible_for_decision") is True
        ),
        "quality_guard_verification_error": quality_error,
    }


def _load_previous_int8_entries(root: Path) -> dict[str, Mapping[str, Any]]:
    path = root / "results/wands/int8-dense-summary.json"
    if not path.is_file():
        return {}
    value = read_json(path)
    if not isinstance(value, dict) or not isinstance(value.get("models"), dict):
        return {}
    entries: dict[str, Mapping[str, Any]] = {}
    for model_name, model_value in cast(dict[str, Any], value["models"]).items():
        if not isinstance(model_value, dict) or not isinstance(
            model_value.get("splits"), dict
        ):
            continue
        dev = cast(dict[str, Any], model_value["splits"]).get("dev")
        if isinstance(dev, dict):
            entries[model_name] = dev
    return entries


def _registered_int8_context(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    models: tuple[ModelSpec, ...],
    runtime: QueryRuntimeSpec,
) -> tuple[dict[str, Any], dict[str, dict[str, object]]]:
    registry = load_model_registry(root / "config/models.toml")
    expected_models = tuple(registry[name] for name in WANDS_INT8_MODEL_NAMES)
    if models != expected_models:
        raise DatasetIntegrityError("int8 dense models differ from registered order")
    if runtime != load_query_runtime_spec(root / "config/query_runtime.toml"):
        raise DatasetIntegrityError("int8 dense runtime differs from registered config")
    if index != load_wands_index_config(root / "config/indexes.toml").name:
        raise DatasetIntegrityError("int8 dense index differs from registered config")
    fp32_summary = verify_wands_dense_summary(
        client, root=root, index=index, models=expected_models
    )
    fp32_models = cast(
        dict[str, dict[str, Mapping[str, Any]]], fp32_summary["models"]
    )
    previous = _load_previous_int8_entries(root)
    runtime_evidence = {
        model.name: _runtime_evidence(
            root=root,
            model=model,
            runtime=runtime,
            fp32_entry=fp32_models[model.name]["dev"],
            int8_entry=previous.get(model.name),
        )
        for model in models
    }
    return fp32_summary, runtime_evidence


def _base_source_eligible(
    fp32_summary: Mapping[str, Any],
    runtime_evidence: Mapping[str, Any],
) -> bool:
    fp32_source = fp32_summary.get("source_evidence")
    return (
        isinstance(fp32_source, Mapping)
        and fp32_source.get("eligible_for_model_selection") is True
        and fp32_summary.get("model_selection_evidence_eligible") is True
        and runtime_evidence.get("onnx_artifact_eligible") is True
        and runtime_evidence.get("parity_artifact_verified") is True
        and runtime_evidence.get("self_retrieval_artifact_verified") is True
    )


def _int8_summary_quality_eligible(
    *,
    provenance_valid: bool,
    runtime_by_model: Mapping[str, Mapping[str, Any]],
) -> bool:
    required_verified_artifacts = (
        "onnx_artifact_eligible",
        "parity_artifact_verified",
        "self_retrieval_artifact_verified",
        "quality_guard_artifact_verified",
    )
    return provenance_valid and all(
        all(evidence.get(key) is True for key in required_verified_artifacts)
        for evidence in runtime_by_model.values()
    )


def run_wands_int8_dense_benchmark(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    models: tuple[ModelSpec, ...],
    runtime: QueryRuntimeSpec,
) -> dict[str, object]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    benchmark_provenance = _json_dict(collect_manifest_provenance(root))
    fp32_summary, runtime_by_model = _registered_int8_context(
        client, root=root, index=index, models=models, runtime=runtime
    )
    fp32_models = cast(
        dict[str, dict[str, dict[str, Any]]], fp32_summary["models"]
    )
    prepared = root / "data/prepared/wands"
    query_paths = {split: prepared / f"queries.{split}.jsonl" for split in _SPLITS}
    qrels_paths = {split: prepared / f"qrels.{split}.trec" for split in _SPLITS}
    queries = {
        split: load_prepared_queries(query_paths[split], expected_split=split)
        for split in _SPLITS
    }
    all_queries = {**queries["dev"], **queries["test"]}
    provenance_valid, _ = wands_completion_provenance_evidence(
        root, benchmark_provenance
    )
    fp32_source = cast(dict[str, Any], fp32_summary["source_evidence"])
    vector_index_eligible = fp32_source.get("eligible_for_model_selection") is True
    summary_models: dict[str, object] = {}
    for model in models:
        runtime_evidence = runtime_by_model[model.name]
        base_eligible = _base_source_eligible(fp32_summary, runtime_evidence)
        eligible, reason = classify_int8_evidence(
            model_decision_eligible=model.decision_eligible,
            run_provenance_valid=provenance_valid,
            vector_index_eligible=vector_index_eligible,
            fp32_model_selection_eligible=(
                fp32_summary.get("model_selection_evidence_eligible") is True
            ),
            onnx_artifact_eligible=(
                runtime_evidence.get("onnx_artifact_eligible") is True
            ),
            parity_artifact_verified=(
                runtime_evidence.get("parity_artifact_verified") is True
            ),
            quality_guard_passed=(
                runtime_evidence.get("quality_guard_passed") is True
            ),
            self_retrieval_passed=(
                runtime_evidence.get("self_retrieval_passed") is True
            ),
        )
        source_evidence = {
            "fp32_summary": _artifact(root, root / "results/wands/dense-summary.json"),
            "fp32_source_evidence": fp32_source,
            "runtime": runtime_evidence,
            "base_evidence_eligible": base_eligible,
        }
        backend = create_int8_query_backend(root=root, model=model, runtime=runtime)
        started = time.perf_counter()
        vectors = encode_queries(
            backend, model=model, queries=all_queries, batch_size=runtime.batch_size
        )
        encoding_seconds = time.perf_counter() - started
        split_results: dict[str, object] = {}
        for split in _SPLITS:
            tag = (
                f"wands-dense-{model.name.replace('_', '-')}-{split}-"
                f"int8-onnx-{runtime.quantization_config}"
            )
            split_vectors = {
                query_id: vectors[query_id] for query_id in queries[split]
            }
            records = execute_exact_dense_run(
                client,
                index=index,
                model=model,
                query_vectors=split_vectors,
                tag=tag,
                score_tie_decimal_places=INT8_SCORE_TIE_DECIMAL_PLACES,
                candidate_depth=INT8_CANDIDATE_DEPTH,
            )
            result = write_dense_bundle(
                client,
                root=root,
                index=index,
                index_manifest_path=root / "results/wands/index-manifest.json",
                model=model,
                model_manifest_path=(
                    root / f"results/wands/embeddings/{model.name}.manifest.json"
                ),
                backend=backend,
                query_path=query_paths[split],
                qrels_path=qrels_paths[split],
                tag=tag,
                split=split,
                records=records,
                encoding_seconds=encoding_seconds,
                decision_scope="int8_onnx_query_runtime_candidate",
                eligible_for_decision=eligible,
                ineligibility_reason=reason,
                query_encoder_artifact_sha256=backend.model_artifact_sha256,
                ranking_policy=INT8_RANKING_POLICY,
                candidate_depth=INT8_CANDIDATE_DEPTH,
                query_vectors_hash=query_vectors_sha256(split_vectors),
                benchmark_provenance=benchmark_provenance,
                source_evidence=source_evidence,
                upstream_evidence_eligible=base_eligible,
                requested_quality_guard_input=True,
            )
            fp32_ndcg = float(fp32_models[model.name][split]["metrics"]["ndcg@10"])
            entry = result.summary_entry(root)
            entry.update(
                {
                    "fp32_ndcg@10": fp32_ndcg,
                    "int8_minus_fp32_ndcg@10": (
                        result.evaluation.metrics["ndcg@10"] - fp32_ndcg
                    ),
                }
            )
            split_results[split] = entry
        summary_models[model.name] = {
            "eligible_for_decision": eligible,
            "ineligibility_reason": reason,
            "runtime_evidence": runtime_evidence,
            "splits": split_results,
        }
    provenance_valid, completion_revision = wands_completion_provenance_evidence(
        root, benchmark_provenance
    )
    artifacts = {
        f"{model.name}/{split}": cast(
            dict[str, dict[str, Any]],
            cast(dict[str, Any], summary_models[model.name])["splits"],
        )[split]
        for model in models
        for split in _SPLITS
    }
    quality_eligible = _int8_summary_quality_eligible(
        provenance_valid=provenance_valid,
        runtime_by_model=runtime_by_model,
    )
    summary: dict[str, object] = {
        "schema_version": 2,
        "dataset": "WANDS",
        "quality_query": "exact_knn_score_script",
        "query_runtime": "dynamic_int8_onnx_cpu",
        "runtime": asdict(runtime),
        "quantization_config": runtime.quantization_config,
        "benchmark_provenance": benchmark_provenance,
        "benchmark_provenance_valid": provenance_valid,
        "completion_code_revision": completion_revision,
        "fp32_summary": _artifact(root, root / "results/wands/dense-summary.json"),
        "required_artifact_keys": sorted(artifacts),
        "artifacts": artifacts,
        "models": summary_models,
        "quality_evidence_eligible_for_decision": quality_eligible,
    }
    write_json(root / "results/wands/int8-dense-summary.json", summary)
    return summary


def verify_wands_int8_dense_summary(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    models: tuple[ModelSpec, ...],
    runtime: QueryRuntimeSpec,
) -> dict[str, Any]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    fp32_summary, runtime_by_model = _registered_int8_context(
        client, root=root, index=index, models=models, runtime=runtime
    )
    summary = cast(
        dict[str, Any], read_json(root / "results/wands/int8-dense-summary.json")
    )
    if summary.get("schema_version") != 2 or summary.get("dataset") != "WANDS":
        raise DatasetIntegrityError("int8 dense summary schema or dataset differs")
    if (
        summary.get("quality_query") != "exact_knn_score_script"
        or summary.get("query_runtime") != "dynamic_int8_onnx_cpu"
        or summary.get("runtime") != asdict(runtime)
        or summary.get("quantization_config") != runtime.quantization_config
        or summary.get("fp32_summary")
        != _artifact(root, root / "results/wands/dense-summary.json")
    ):
        raise DatasetIntegrityError("int8 dense summary method metadata differs")
    provenance = summary.get("benchmark_provenance")
    provenance_valid = wands_recorded_provenance_valid(
        root,
        provenance=provenance,
        completion_revision=summary.get("completion_code_revision"),
    )
    if summary.get("benchmark_provenance_valid") is not provenance_valid:
        raise DatasetIntegrityError("int8 dense summary provenance validity differs")
    models_value = summary.get("models")
    artifacts_value = summary.get("artifacts")
    if not isinstance(models_value, dict) or set(models_value) != {
        model.name for model in models
    }:
        raise DatasetIntegrityError("int8 dense summary model set differs")
    if not isinstance(artifacts_value, dict):
        raise DatasetIntegrityError("int8 dense summary artifacts are missing")
    summary_models = cast(dict[str, dict[str, Any]], models_value)
    artifacts = cast(dict[str, dict[str, Any]], artifacts_value)
    expected_keys = {f"{model.name}/{split}" for model in models for split in _SPLITS}
    if (
        set(artifacts) != expected_keys
        or summary.get("required_artifact_keys") != sorted(expected_keys)
    ):
        raise DatasetIntegrityError("int8 dense artifact set differs")
    fp32_models = cast(
        dict[str, dict[str, dict[str, Any]]], fp32_summary["models"]
    )
    fp32_source = cast(dict[str, Any], fp32_summary["source_evidence"])
    for model in models:
        model_entry = summary_models[model.name]
        runtime_evidence = runtime_by_model[model.name]
        if model_entry.get("runtime_evidence") != runtime_evidence:
            raise DatasetIntegrityError(f"int8 runtime evidence differs for {model.name}")
        base_eligible = _base_source_eligible(fp32_summary, runtime_evidence)
        eligible, reason = classify_int8_evidence(
            model_decision_eligible=model.decision_eligible,
            run_provenance_valid=provenance_valid,
            vector_index_eligible=(
                fp32_source.get("eligible_for_model_selection") is True
            ),
            fp32_model_selection_eligible=(
                fp32_summary.get("model_selection_evidence_eligible") is True
            ),
            onnx_artifact_eligible=(
                runtime_evidence.get("onnx_artifact_eligible") is True
            ),
            parity_artifact_verified=(
                runtime_evidence.get("parity_artifact_verified") is True
            ),
            quality_guard_passed=(
                runtime_evidence.get("quality_guard_passed") is True
            ),
            self_retrieval_passed=(
                runtime_evidence.get("self_retrieval_passed") is True
            ),
        )
        if (
            model_entry.get("eligible_for_decision") is not eligible
            or model_entry.get("ineligibility_reason") != reason
        ):
            raise DatasetIntegrityError(f"int8 eligibility differs for {model.name}")
        splits_value = model_entry.get("splits")
        if not isinstance(splits_value, dict) or set(splits_value) != set(_SPLITS):
            raise DatasetIntegrityError(f"int8 split set differs for {model.name}")
        splits = cast(dict[str, dict[str, Any]], splits_value)
        source_evidence = {
            "fp32_summary": _artifact(root, root / "results/wands/dense-summary.json"),
            "fp32_source_evidence": fp32_source,
            "runtime": runtime_evidence,
            "base_evidence_eligible": base_eligible,
        }
        onnx = verify_onnx_int8_artifact(root=root, model=model, runtime=runtime)
        for split in _SPLITS:
            entry = splits[split]
            if artifacts[f"{model.name}/{split}"] != entry:
                raise DatasetIntegrityError(
                    f"int8 artifact binding differs for {model.name}/{split}"
                )
            verify_dense_artifact(
                client,
                root=root,
                index=index,
                entry=entry,
                expected_eligible_for_decision=eligible,
                expected_quality_guard_input=True,
                expected_model=model,
                expected_split=split,
                expected_benchmark_provenance=cast(Mapping[str, Any], provenance),
                expected_source_evidence=source_evidence,
                expected_upstream_eligible=base_eligible,
            )
            manifest = cast(dict[str, Any], read_json(root / str(entry["manifest"])))
            if (
                manifest.get("decision_scope")
                != "int8_onnx_query_runtime_candidate"
                or manifest.get("ranking_policy") != INT8_RANKING_POLICY
                or manifest.get("candidate_depth") != INT8_CANDIDATE_DEPTH
                or manifest.get("query_encoder_artifact_sha256")
                != onnx.get("artifact_sha256")
            ):
                raise DatasetIntegrityError(
                    f"int8 method binding differs for {model.name}/{split}"
                )
            int8_ndcg = float(cast(dict[str, Any], entry["metrics"])["ndcg@10"])
            fp32_ndcg = float(fp32_models[model.name][split]["metrics"]["ndcg@10"])
            if (
                entry.get("fp32_ndcg@10") != fp32_ndcg
                or entry.get("int8_minus_fp32_ndcg@10")
                != int8_ndcg - fp32_ndcg
            ):
                raise DatasetIntegrityError(
                    f"int8 quality delta differs for {model.name}/{split}"
                )
    expected_quality_eligible = _int8_summary_quality_eligible(
        provenance_valid=provenance_valid,
        runtime_by_model=runtime_by_model,
    )
    if (
        summary.get("quality_evidence_eligible_for_decision")
        is not expected_quality_eligible
    ):
        raise DatasetIntegrityError("int8 dense summary eligibility differs")
    _verify_int8_live_reruns(
        client,
        root=root,
        index=index,
        models=models,
        runtime=runtime,
        summary_models=summary_models,
    )
    return summary


def _verify_int8_live_reruns(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    models: tuple[ModelSpec, ...],
    runtime: QueryRuntimeSpec,
    summary_models: Mapping[str, Mapping[str, Any]],
) -> None:
    prepared = root / "data/prepared/wands"
    queries = {
        split: load_prepared_queries(
            prepared / f"queries.{split}.jsonl",
            expected_split=split,
        )
        for split in _SPLITS
    }
    all_queries = {**queries["dev"], **queries["test"]}
    with tempfile.TemporaryDirectory(
        prefix="opensearch-hybrid-int8-dense-verify-"
    ) as temporary:
        temporary_root = Path(temporary)
        for model in models:
            backend = create_int8_query_backend(
                root=root,
                model=model,
                runtime=runtime,
            )
            vectors = encode_queries(
                backend,
                model=model,
                queries=all_queries,
                batch_size=runtime.batch_size,
            )
            repeated_vectors = encode_queries(
                backend,
                model=model,
                queries=all_queries,
                batch_size=runtime.batch_size,
            )
            for query_id in vectors:
                if not np.array_equal(vectors[query_id], repeated_vectors[query_id]):
                    maximum_delta = float(
                        np.max(
                            np.abs(vectors[query_id] - repeated_vectors[query_id])
                        )
                    )
                    raise DatasetIntegrityError(
                        f"{model.name} int8 query encoding is nondeterministic for "
                        f"{query_id}: maximum delta {maximum_delta:.12f}"
                    )
            splits = cast(
                Mapping[str, Mapping[str, Any]],
                summary_models[model.name]["splits"],
            )
            for split in _SPLITS:
                entry = splits[split]
                manifest = cast(
                    dict[str, Any], read_json(root / str(entry["manifest"]))
                )
                if backend.model_artifact_sha256 != manifest.get(
                    "query_encoder_artifact_sha256"
                ):
                    raise DatasetIntegrityError(
                        f"{model.name}/{split} rerun loaded a different artifact"
                    )
                split_vectors = {
                    query_id: vectors[query_id] for query_id in queries[split]
                }
                vector_sha256 = query_vectors_sha256(split_vectors)
                if vector_sha256 != manifest.get("query_vectors_sha256"):
                    raise DatasetIntegrityError(
                        f"{model.name}/{split} query vectors differ: "
                        f"{vector_sha256} != "
                        f"{manifest.get('query_vectors_sha256')}"
                    )
                tag = str(manifest["tag"])
                records = execute_exact_dense_run(
                    client,
                    index=index,
                    model=model,
                    query_vectors=split_vectors,
                    tag=tag,
                    score_tie_decimal_places=INT8_SCORE_TIE_DECIMAL_PLACES,
                    candidate_depth=INT8_CANDIDATE_DEPTH,
                )
                repeated_records = execute_exact_dense_run(
                    client,
                    index=index,
                    model=model,
                    query_vectors=split_vectors,
                    tag=tag,
                    score_tie_decimal_places=INT8_SCORE_TIE_DECIMAL_PLACES,
                    candidate_depth=INT8_CANDIDATE_DEPTH,
                )
                if records != repeated_records:
                    raise DatasetIntegrityError(
                        f"identical int8 dense requests are not deterministic: "
                        f"{model.name}/{split}"
                    )
                repeated_path = temporary_root / f"{model.name}-{split}.trec"
                write_run(repeated_path, records)
                if repeated_path.read_bytes() != (
                    root / str(entry["run"])
                ).read_bytes():
                    raise DatasetIntegrityError(
                        f"{model.name}/{split} int8 dense live rerun is not "
                        "byte-identical"
                    )
