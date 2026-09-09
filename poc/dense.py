from __future__ import annotations

import hashlib
import importlib.metadata
import json
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from poc.bm25 import load_prepared_queries
from poc.config import (
    WANDS_FINALIST_MODEL_NAMES,
    ModelSpec,
    load_model_registry,
)
from poc.datasets import DatasetIntegrityError, file_facts, load_wands_config
from poc.embedding_cache import EmbeddingBackend, verify_embedding_cache
from poc.evaluation import Evaluation, evaluate_run
from poc.evaluation_crosscheck import cross_check_run
from poc.indexing import (
    load_wands_index_config,
    vector_field_name,
    verify_wands_lexical_index,
    wands_completion_provenance_evidence,
    wands_recorded_provenance_valid,
)
from poc.manifest import canonical_sha256, collect_index_facts, read_json, write_json
from poc.model_runtime import create_bulk_backend, release_device_memory
from poc.os_client import OpenSearchClient
from poc.provenance import collect_manifest_provenance, require_registered_opensearch_client
from poc.search import build_exact_knn_request
from poc.trec import RunRecord, identifier_sort_key, read_run, write_run

RESULT_SIZE = 100
_DENSE_SPLITS = ("dev", "test")


@dataclass(frozen=True, slots=True)
class DenseRunResult:
    tag: str
    run_path: Path
    manifest_path: Path
    metrics_path: Path
    evaluation: Evaluation
    records: list[RunRecord]

    def summary_entry(self, root: Path) -> dict[str, object]:
        return {
            "run": str(self.run_path.relative_to(root)),
            "manifest": str(self.manifest_path.relative_to(root)),
            "metrics_file": str(self.metrics_path.relative_to(root)),
            "metrics": self.evaluation.metrics,
        }


def _json_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(json.dumps(value)))


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }


def _dense_source_evidence(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    models: tuple[ModelSpec, ...],
) -> dict[str, object]:
    index_spec = load_wands_index_config(root / "config/indexes.toml")
    registry = load_model_registry(root / "config/models.toml")
    expected_models = tuple(registry[name] for name in WANDS_FINALIST_MODEL_NAMES)
    if index != index_spec.name:
        raise DatasetIntegrityError("dense index differs from registered WANDS index")
    if models != expected_models:
        raise DatasetIntegrityError("dense models differ from registered finalist order")
    dataset_spec = load_wands_config(root / "config/datasets.toml")
    index_summary = verify_wands_lexical_index(
        client,
        dataset_spec=dataset_spec,
        index_spec=index_spec,
        prepared_directory=root / "data/prepared/wands",
        manifest_path=root / "results/wands/index-manifest.json",
    )
    if index_summary.get("vector_models") != list(WANDS_FINALIST_MODEL_NAMES):
        raise DatasetIntegrityError("live WANDS index has the wrong vector model set")
    embedding_bindings: dict[str, dict[str, object]] = {}
    embedding_eligible = True
    products_path = root / "data/prepared/wands/products.jsonl"
    for model in models:
        verify_embedding_cache(root=root, products_path=products_path, model=model)
        manifest_path = root / f"results/wands/embeddings/{model.name}.manifest.json"
        manifest = cast(dict[str, Any], read_json(manifest_path))
        embedding_bindings[model.name] = _artifact(root, manifest_path)
        embedding_eligible = embedding_eligible and (
            manifest.get("quality_evidence_eligible_for_decision") is True
        )
    return {
        "index_manifest": _artifact(root, root / "results/wands/index-manifest.json"),
        "index_verification": index_summary,
        "embedding_manifests": embedding_bindings,
        "eligible_for_model_selection": (
            index_summary.get("quality_evidence_eligible_for_decision") is True
            and embedding_eligible
        ),
    }


def _dense_source_bindings(
    root: Path,
    models: tuple[ModelSpec, ...],
) -> dict[str, object]:
    prepared = root / "data/prepared/wands"
    fixed_paths = {
        "dataset_registry": root / "config/datasets.toml",
        "index_registry": root / "config/indexes.toml",
        "model_registry": root / "config/models.toml",
        "prepared_manifest": prepared / "manifest.json",
        "products": prepared / "products.jsonl",
        "queries_dev": prepared / "queries.dev.jsonl",
        "queries_test": prepared / "queries.test.jsonl",
        "qrels_dev": prepared / "qrels.dev.trec",
        "qrels_test": prepared / "qrels.test.trec",
        "index_manifest": root / "results/wands/index-manifest.json",
    }
    return {
        **{name: _artifact(root, path) for name, path in sorted(fixed_paths.items())},
        "embedding_manifests": {
            model.name: _artifact(
                root,
                root / f"results/wands/embeddings/{model.name}.manifest.json",
            )
            for model in models
        },
    }


def trec_score_for_rank(rank: int) -> float:
    """Persist engine order without serializing nondeterministic float noise."""
    if not 1 <= rank <= RESULT_SIZE:
        raise ValueError(f"dense rank must be between 1 and {RESULT_SIZE}")
    return float(RESULT_SIZE - rank + 1)


def query_vectors_sha256(vectors: dict[str, NDArray[np.float32]]) -> str:
    digest = hashlib.sha256()
    for query_id in sorted(vectors, key=identifier_sort_key):
        encoded_id = query_id.encode()
        vector = np.asarray(vectors[query_id], dtype=np.float32)
        digest.update(len(encoded_id).to_bytes(4, "big"))
        digest.update(encoded_id)
        digest.update(vector.shape[0].to_bytes(4, "big"))
        digest.update(vector.tobytes())
    return digest.hexdigest()


def encode_queries(
    backend: EmbeddingBackend,
    *,
    model: ModelSpec,
    queries: dict[str, str],
    batch_size: int = 32,
) -> dict[str, NDArray[np.float32]]:
    if batch_size <= 0:
        raise ValueError("query encoding batch size must be positive")
    query_ids = sorted(queries, key=identifier_sort_key)
    encoded: dict[str, NDArray[np.float32]] = {}
    for start in range(0, len(query_ids), batch_size):
        batch_ids = query_ids[start : start + batch_size]
        texts = [model.render_query(queries[query_id]) for query_id in batch_ids]
        vectors = np.asarray(backend.encode(texts), dtype=np.float32)
        if vectors.shape != (len(batch_ids), model.dims) or not np.isfinite(vectors).all():
            raise DatasetIntegrityError(f"invalid query embeddings from {model.name}")
        for query_id, vector in zip(batch_ids, vectors, strict=True):
            encoded[query_id] = vector
    return encoded


def execute_exact_dense_run(
    client: OpenSearchClient,
    *,
    index: str,
    model: ModelSpec,
    query_vectors: dict[str, NDArray[np.float32]],
    tag: str,
    score_tie_decimal_places: int | None = None,
    candidate_depth: int = RESULT_SIZE,
) -> list[RunRecord]:
    if candidate_depth < RESULT_SIZE:
        raise ValueError(f"dense candidate depth must be at least {RESULT_SIZE}")
    if score_tie_decimal_places is not None and score_tie_decimal_places < 0:
        raise ValueError("dense score tie decimal places must be non-negative")
    records: list[RunRecord] = []
    field = vector_field_name(model)
    for query_id in sorted(query_vectors, key=identifier_sort_key):
        response = client.search(
            index,
            build_exact_knn_request(
                query_vectors[query_id].tolist(),
                vector_field=field,
                k=candidate_depth,
            ),
        )
        hits = cast(list[dict[str, Any]], cast(dict[str, Any], response["hits"])["hits"])
        if score_tie_decimal_places is not None:
            for hit in hits:
                if hit.get("_score") is None:
                    raise DatasetIntegrityError(f"dense result for {query_id} has no score")
            hits = sorted(
                hits,
                key=lambda hit: (
                    -round(float(hit["_score"]), score_tie_decimal_places),
                    identifier_sort_key(str(hit["_id"])),
                ),
            )[:RESULT_SIZE]
        for rank, hit in enumerate(hits, start=1):
            score = hit.get("_score")
            if score is None:
                raise DatasetIntegrityError(f"dense result for {query_id} has no score")
            records.append(
                RunRecord(
                    query_id=query_id,
                    document_id=str(hit["_id"]),
                    rank=rank,
                    score=trec_score_for_rank(rank),
                    tag=tag,
                )
            )
    return records


def write_dense_bundle(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    index_manifest_path: Path,
    model: ModelSpec,
    model_manifest_path: Path,
    backend: EmbeddingBackend,
    query_path: Path,
    qrels_path: Path,
    tag: str,
    split: str,
    records: list[RunRecord],
    encoding_seconds: float,
    decision_scope: str = "exploratory_fp32_query_runtime",
    eligible_for_decision: bool = False,
    ineligibility_reason: str | None = (
        "query encoder is fp32 CPU, not registered int8 ONNX CPU"
    ),
    query_encoder_artifact_sha256: str | None = None,
    ranking_policy: str = "opensearch_score_then_product_id",
    candidate_depth: int = RESULT_SIZE,
    query_vectors_hash: str | None = None,
    benchmark_provenance: Mapping[str, Any] | None = None,
    source_evidence: Mapping[str, Any] | None = None,
    upstream_evidence_eligible: bool = False,
    requested_model_selection_input: bool = False,
    requested_quality_guard_input: bool = False,
) -> DenseRunResult:
    if benchmark_provenance is None or source_evidence is None:
        raise DatasetIntegrityError("dense bundle requires explicit start and source evidence")
    run_path = root / f"runs/{tag}.trec"
    manifest_path = root / f"runs/{tag}.manifest.json"
    metrics_path = root / f"results/wands/dense/{tag}.metrics.json"
    write_run(run_path, records)
    evaluation = evaluate_run(qrels_path, run_path)
    crosscheck = cross_check_run(qrels_path, run_path, root=root)
    write_json(
        metrics_path,
        {
            "schema_version": 2,
            "tag": tag,
            "split": split,
            "metrics": evaluation.metrics,
            "per_query": evaluation.per_query,
            "evaluator_crosscheck": crosscheck.to_dict(),
        },
    )

    root_response = cast(dict[str, Any], client.request("GET", "/"))
    version = str(cast(dict[str, Any], root_response["version"])["number"])
    index_manifest = cast(dict[str, Any], read_json(index_manifest_path))
    model_manifest = cast(dict[str, Any], read_json(model_manifest_path))
    query_facts = file_facts(query_path)
    qrels_facts = file_facts(qrels_path)
    run_facts = file_facts(run_path)
    provenance_valid, completion_revision = wands_completion_provenance_evidence(
        root,
        benchmark_provenance,
    )
    model_selection_input_eligible = (
        requested_model_selection_input
        and provenance_valid
        and upstream_evidence_eligible
    )
    quality_guard_input_eligible = (
        requested_quality_guard_input
        and provenance_valid
        and upstream_evidence_eligible
    )
    derived_eligible = (
        eligible_for_decision and provenance_valid and upstream_evidence_eligible
    )
    derived_reason = ineligibility_reason
    if eligible_for_decision and not derived_eligible:
        reasons: list[str] = []
        if not provenance_valid:
            reasons.append("run has no valid clean committed start provenance")
        if not upstream_evidence_eligible:
            reasons.append("dense upstream evidence is ineligible")
        derived_reason = "; ".join(reasons)
    manifest: dict[str, object] = {
        "schema_version": 2,
        "dataset": "WANDS",
        "split": split,
        "tag": tag,
        "decision_scope": decision_scope,
        "eligible_for_decision": derived_eligible,
        "ineligibility_reason": derived_reason,
        "eligible_as_model_selection_input": model_selection_input_eligible,
        "eligible_as_quality_guard_input": quality_guard_input_eligible,
        "benchmark_provenance": _json_dict(benchmark_provenance),
        "benchmark_provenance_valid": provenance_valid,
        "completion_code_revision": completion_revision,
        "source_evidence": _json_dict(source_evidence),
        "upstream_evidence_eligible": upstream_evidence_eligible,
        "declared_variable": "embedding_model",
        "opensearch_version": version,
        "index": asdict(collect_index_facts(client, index)),
        "index_manifest_sha256": file_facts(index_manifest_path).sha256,
        "index_definition_sha256": index_manifest["index_definition_sha256"],
        "vector_field": vector_field_name(model),
        "quality_query": "exact_knn_score_script",
        "ranking_policy": ranking_policy,
        "candidate_depth": candidate_depth,
        "trec_score_source": "opensearch_rank_derived_100_to_1",
        "engine_score_persistence": "omitted_due_sub_micro_runtime_noise",
        "pagination_depth": None,
        "pipeline_sha256": None,
        "model": asdict(model),
        "model_sha256": model_manifest["model_artifact_sha256"],
        "document_encoder_runtime": model_manifest["encoder_runtime"],
        "query_encoder_runtime": backend.runtime,
        "query_encoder_device": backend.device,
        "query_encoder_artifact_sha256": query_encoder_artifact_sha256,
        "query_encoding_seconds": encoding_seconds,
        "query_vectors_sha256": query_vectors_hash,
        "query_template_sha256": canonical_sha256(
            {"prefix": model.query_prefix, "template": model.query_template}
        ),
        "query_template_example": model.render_query("example product query"),
        "query_file": {
            "path": str(query_path.relative_to(root)),
            "sha256": query_facts.sha256,
        },
        "qrels_file": {
            "path": str(qrels_path.relative_to(root)),
            "sha256": qrels_facts.sha256,
        },
        "run_file": {
            "path": str(run_path.relative_to(root)),
            "sha256": run_facts.sha256,
            "records": len(records),
        },
        "evaluation_packages": {
            "ranx": importlib.metadata.version("ranx"),
            "pytrec_eval": importlib.metadata.version("pytrec-eval-terrier"),
        },
        "metrics_file": _artifact(root, metrics_path),
    }
    write_json(manifest_path, manifest)
    return DenseRunResult(
        tag=tag,
        run_path=run_path,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        evaluation=evaluation,
        records=records,
    )


def run_wands_dense_benchmark(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    models: tuple[ModelSpec, ...],
) -> dict[str, object]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    benchmark_provenance = _json_dict(collect_manifest_provenance(root))
    source_evidence = _dense_source_evidence(
        client,
        root=root,
        index=index,
        models=models,
    )
    upstream_eligible = source_evidence.get("eligible_for_model_selection") is True
    prepared = root / "data/prepared/wands"
    query_paths = {split: prepared / f"queries.{split}.jsonl" for split in ("dev", "test")}
    qrels_paths = {split: prepared / f"qrels.{split}.trec" for split in ("dev", "test")}
    queries = {
        split: load_prepared_queries(query_paths[split], expected_split=split)
        for split in ("dev", "test")
    }
    all_queries = {**queries["dev"], **queries["test"]}
    summary_models: dict[str, object] = {}
    for model in models:
        backend = create_bulk_backend(root=root, model=model, device="cpu")
        started = time.perf_counter()
        vectors = encode_queries(backend, model=model, queries=all_queries)
        encoding_seconds = time.perf_counter() - started
        split_results: dict[str, object] = {}
        for split in ("dev", "test"):
            tag = f"wands-dense-{model.name.replace('_', '-')}-{split}-fp32-cpu"
            records = execute_exact_dense_run(
                client,
                index=index,
                model=model,
                query_vectors={query_id: vectors[query_id] for query_id in queries[split]},
                tag=tag,
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
                benchmark_provenance=benchmark_provenance,
                source_evidence=source_evidence,
                upstream_evidence_eligible=upstream_eligible,
                requested_model_selection_input=True,
            )
            split_results[split] = result.summary_entry(root)
        summary_models[model.name] = split_results
        del backend
        release_device_memory()

    provenance_valid, completion_revision = wands_completion_provenance_evidence(
        root,
        benchmark_provenance,
    )
    required_artifacts = {
        f"{model.name}/{split}": cast(
            dict[str, dict[str, Any]], summary_models[model.name]
        )[split]
        for model in models
        for split in _DENSE_SPLITS
    }
    model_selection_eligible = (
        provenance_valid
        and upstream_eligible
        and all(
            cast(dict[str, Any], read_json(root / str(entry["manifest"]))).get(
                "eligible_as_model_selection_input"
            )
            is True
            for entry in required_artifacts.values()
        )
    )
    selected_model = max(
        models,
        key=lambda model: float(
            cast(
                dict[str, Any],
                cast(dict[str, dict[str, Any]], summary_models[model.name])["dev"][
                    "metrics"
                ],
            )["ndcg@10"]
        ),
    )
    summary: dict[str, object] = {
        "schema_version": 2,
        "dataset": "WANDS",
        "quality_query": "exact_knn_score_script",
        "query_runtime": "fp32_cpu_exploratory",
        "eligible_for_decision": False,
        "ineligibility_reason": "query encoder is fp32 CPU, not registered int8 ONNX CPU",
        "model_selection_evidence_eligible": model_selection_eligible,
        "quality_evidence_eligible_for_decision": False,
        "benchmark_provenance": benchmark_provenance,
        "benchmark_provenance_valid": provenance_valid,
        "completion_code_revision": completion_revision,
        "source_evidence": source_evidence,
        "source_bindings": _dense_source_bindings(root, models),
        "selection_metric": "ndcg@10",
        "selection_split": "dev",
        "tie_break": "first_model_in_registered_order",
        "selected_model": selected_model.name,
        "required_artifact_keys": sorted(required_artifacts),
        "artifacts": required_artifacts,
        "models": summary_models,
    }
    write_json(root / "results/wands/dense-summary.json", summary)
    return summary


def verify_wands_dense_summary(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    models: tuple[ModelSpec, ...],
) -> dict[str, Any]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    source_evidence = _dense_source_evidence(
        client,
        root=root,
        index=index,
        models=models,
    )
    summary_path = root / "results/wands/dense-summary.json"
    summary = cast(dict[str, Any], read_json(summary_path))
    if summary.get("schema_version") != 2 or summary.get("dataset") != "WANDS":
        raise DatasetIntegrityError("dense summary schema or dataset differs")
    if (
        summary.get("quality_query") != "exact_knn_score_script"
        or summary.get("query_runtime") != "fp32_cpu_exploratory"
        or summary.get("eligible_for_decision") is not False
        or summary.get("quality_evidence_eligible_for_decision") is not False
    ):
        raise DatasetIntegrityError("dense summary method metadata differs")
    if summary.get("source_evidence") != source_evidence:
        raise DatasetIntegrityError("dense summary source evidence differs")
    if summary.get("source_bindings") != _dense_source_bindings(root, models):
        raise DatasetIntegrityError("dense summary source bindings differ")
    provenance = summary.get("benchmark_provenance")
    provenance_valid = wands_recorded_provenance_valid(
        root,
        provenance=provenance,
        completion_revision=summary.get("completion_code_revision"),
    )
    if summary.get("benchmark_provenance_valid") is not provenance_valid:
        raise DatasetIntegrityError("dense summary provenance validity differs")
    models_value = summary.get("models")
    artifacts_value = summary.get("artifacts")
    if not isinstance(models_value, dict) or set(models_value) != {
        model.name for model in models
    }:
        raise DatasetIntegrityError("dense summary model set differs")
    if not isinstance(artifacts_value, dict):
        raise DatasetIntegrityError("dense summary artifacts are missing")
    summary_models = cast(dict[str, dict[str, dict[str, Any]]], models_value)
    artifacts = cast(dict[str, dict[str, Any]], artifacts_value)
    expected_keys = {
        f"{model.name}/{split}" for model in models for split in _DENSE_SPLITS
    }
    if (
        set(artifacts) != expected_keys
        or summary.get("required_artifact_keys") != sorted(expected_keys)
    ):
        raise DatasetIntegrityError("dense summary artifact set differs")
    all_inputs_eligible = True
    for model in models:
        if set(summary_models[model.name]) != set(_DENSE_SPLITS):
            raise DatasetIntegrityError(f"dense split set differs for {model.name}")
        for split in _DENSE_SPLITS:
            entry = summary_models[model.name][split]
            if artifacts[f"{model.name}/{split}"] != entry:
                raise DatasetIntegrityError(
                    f"dense artifact binding differs for {model.name}/{split}"
                )
            verify_dense_artifact(
                client,
                root=root,
                index=index,
                entry=entry,
                expected_eligible_for_decision=False,
                expected_model_selection_input=True,
                expected_model=model,
                expected_split=split,
                expected_benchmark_provenance=cast(
                    Mapping[str, Any], provenance
                ),
                expected_source_evidence=source_evidence,
                expected_upstream_eligible=(
                    source_evidence.get("eligible_for_model_selection") is True
                ),
            )
            manifest = cast(dict[str, Any], read_json(root / str(entry["manifest"])))
            all_inputs_eligible = all_inputs_eligible and (
                manifest.get("eligible_as_model_selection_input") is True
            )
    expected_selection_eligible = (
        provenance_valid
        and source_evidence.get("eligible_for_model_selection") is True
        and all_inputs_eligible
    )
    if (
        summary.get("model_selection_evidence_eligible")
        is not expected_selection_eligible
    ):
        raise DatasetIntegrityError("dense model-selection eligibility differs")
    selected_model = max(
        models,
        key=lambda model: float(
            cast(dict[str, Any], summary_models[model.name]["dev"]["metrics"])[
                "ndcg@10"
            ]
        ),
    )
    if (
        summary.get("selection_metric") != "ndcg@10"
        or summary.get("selection_split") != "dev"
        or summary.get("tie_break") != "first_model_in_registered_order"
        or summary.get("selected_model") != selected_model.name
    ):
        raise DatasetIntegrityError("dense model selection differs")
    _verify_dense_live_reruns(
        client,
        root=root,
        index=index,
        models=models,
        summary_models=summary_models,
    )
    return summary


def _verify_dense_live_reruns(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    models: tuple[ModelSpec, ...],
    summary_models: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    prepared = root / "data/prepared/wands"
    queries = {
        split: load_prepared_queries(
            prepared / f"queries.{split}.jsonl",
            expected_split=split,
        )
        for split in _DENSE_SPLITS
    }
    all_queries = {**queries["dev"], **queries["test"]}
    with tempfile.TemporaryDirectory(
        prefix="opensearch-hybrid-dense-verify-"
    ) as temporary:
        temporary_root = Path(temporary)
        for model in models:
            backend = create_bulk_backend(root=root, model=model, device="cpu")
            try:
                vectors = encode_queries(
                    backend,
                    model=model,
                    queries=all_queries,
                )
                for split in _DENSE_SPLITS:
                    entry = summary_models[model.name][split]
                    manifest = cast(
                        dict[str, Any],
                        read_json(root / str(entry["manifest"])),
                    )
                    if backend.model_artifact_sha256 != manifest.get(
                        "model_sha256"
                    ):
                        raise DatasetIntegrityError(
                            f"{model.name} rerun loaded a different model artifact"
                        )
                    records = execute_exact_dense_run(
                        client,
                        index=index,
                        model=model,
                        query_vectors={
                            query_id: vectors[query_id]
                            for query_id in queries[split]
                        },
                        tag=str(manifest["tag"]),
                    )
                    repeated_path = (
                        temporary_root / f"{model.name}-{split}.trec"
                    )
                    write_run(repeated_path, records)
                    if repeated_path.read_bytes() != (
                        root / str(entry["run"])
                    ).read_bytes():
                        raise DatasetIntegrityError(
                            f"{model.name} {split} dense live rerun is not byte-identical"
                        )
            finally:
                del backend
                release_device_memory()


def verify_dense_artifact(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    entry: dict[str, Any],
    expected_eligible_for_decision: bool = False,
    expected_model_selection_input: bool = False,
    expected_quality_guard_input: bool = False,
    expected_model: ModelSpec | None = None,
    expected_split: str | None = None,
    expected_benchmark_provenance: Mapping[str, Any] | None = None,
    expected_source_evidence: Mapping[str, Any] | None = None,
    expected_upstream_eligible: bool | None = None,
) -> None:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    run_path = root / str(entry["run"])
    manifest_path = root / str(entry["manifest"])
    metrics_path = root / str(entry["metrics_file"])
    manifest = cast(dict[str, Any], read_json(manifest_path))
    if manifest.get("schema_version") != 2 or manifest.get("dataset") != "WANDS":
        raise DatasetIntegrityError(f"dense run schema or dataset differs: {run_path}")
    index_spec = load_wands_index_config(root / "config/indexes.toml")
    if index != index_spec.name:
        raise DatasetIntegrityError("dense verifier index differs from registered index")
    model_value = manifest.get("model")
    if not isinstance(model_value, dict) or not isinstance(model_value.get("name"), str):
        raise DatasetIntegrityError(f"dense run model is invalid: {run_path}")
    recorded_model = ModelSpec.from_mapping(
        str(model_value["name"]),
        cast(dict[str, Any], model_value),
    )
    registry = load_model_registry(root / "config/models.toml")
    if registry.get(recorded_model.name) != recorded_model:
        raise DatasetIntegrityError(f"dense run model is not registered: {run_path}")
    if expected_model is not None and recorded_model != expected_model:
        raise DatasetIntegrityError(f"dense run model differs: {run_path}")
    split = manifest.get("split")
    if split not in _DENSE_SPLITS or (
        expected_split is not None and split != expected_split
    ):
        raise DatasetIntegrityError(f"dense run split differs: {run_path}")
    tag = manifest.get("tag")
    if (
        not isinstance(tag, str)
        or not tag
        or run_path != root / f"runs/{tag}.trec"
        or manifest_path != root / f"runs/{tag}.manifest.json"
        or metrics_path != root / f"results/wands/dense/{tag}.metrics.json"
    ):
        raise DatasetIntegrityError(f"dense run paths or tag differ: {run_path}")
    if manifest.get("quality_query") != "exact_knn_score_script":
        raise DatasetIntegrityError(f"dense run is not exact: {run_path}")
    if manifest.get("trec_score_source") != "opensearch_rank_derived_100_to_1":
        raise DatasetIntegrityError(f"dense run has unstable TREC scores: {run_path}")
    provenance = manifest.get("benchmark_provenance")
    provenance_valid = wands_recorded_provenance_valid(
        root,
        provenance=provenance,
        completion_revision=manifest.get("completion_code_revision"),
    )
    if manifest.get("benchmark_provenance_valid") is not provenance_valid:
        raise DatasetIntegrityError(f"dense run provenance validity differs: {run_path}")
    if expected_benchmark_provenance is not None and canonical_sha256(
        provenance
    ) != canonical_sha256(expected_benchmark_provenance):
        raise DatasetIntegrityError(f"dense run start provenance differs: {run_path}")
    source_evidence = manifest.get("source_evidence")
    if not isinstance(source_evidence, dict):
        raise DatasetIntegrityError(f"dense source evidence is missing: {run_path}")
    if expected_source_evidence is None:
        expected_source_evidence = _dense_source_evidence(
            client,
            root=root,
            index=index,
            models=tuple(registry[name] for name in WANDS_FINALIST_MODEL_NAMES),
        )
    if expected_source_evidence is not None and source_evidence != _json_dict(
        expected_source_evidence
    ):
        raise DatasetIntegrityError(f"dense source evidence differs: {run_path}")
    upstream_eligible = manifest.get("upstream_evidence_eligible") is True
    if (
        expected_upstream_eligible is not None
        and upstream_eligible is not expected_upstream_eligible
    ):
        raise DatasetIntegrityError(f"dense upstream eligibility differs: {run_path}")
    derived_decision_eligible = (
        expected_eligible_for_decision and provenance_valid and upstream_eligible
    )
    derived_selection_input = (
        expected_model_selection_input and provenance_valid and upstream_eligible
    )
    derived_quality_guard_input = (
        expected_quality_guard_input and provenance_valid and upstream_eligible
    )
    if (
        manifest.get("eligible_for_decision") is not derived_decision_eligible
        or manifest.get("eligible_as_model_selection_input")
        is not derived_selection_input
        or manifest.get("eligible_as_quality_guard_input")
        is not derived_quality_guard_input
    ):
        raise DatasetIntegrityError(f"dense run has incorrect decision eligibility: {run_path}")
    root_response = cast(dict[str, Any], client.request("GET", "/"))
    live_version = str(cast(dict[str, Any], root_response["version"])["number"])
    if manifest.get("opensearch_version") != live_version:
        raise DatasetIntegrityError(f"dense OpenSearch version differs: {run_path}")
    if manifest.get("vector_field") != vector_field_name(recorded_model):
        raise DatasetIntegrityError(f"dense vector field differs: {run_path}")
    model_manifest_path = (
        root / f"results/wands/embeddings/{recorded_model.name}.manifest.json"
    )
    verify_embedding_cache(
        root=root,
        products_path=root / "data/prepared/wands/products.jsonl",
        model=recorded_model,
    )
    model_manifest = cast(dict[str, Any], read_json(model_manifest_path))
    if (
        manifest.get("model_sha256") != model_manifest.get("model_artifact_sha256")
        or manifest.get("document_encoder_runtime")
        != model_manifest.get("encoder_runtime")
    ):
        raise DatasetIntegrityError(f"dense document model binding differs: {run_path}")
    run_file = cast(dict[str, Any], manifest["run_file"])
    run_records = read_run(run_path)
    if (
        run_file.get("path") != str(run_path.relative_to(root))
        or file_facts(run_path).sha256 != run_file.get("sha256")
        or run_file.get("records") != len(run_records)
        or any(record.tag != tag for record in run_records)
    ):
        raise DatasetIntegrityError(f"dense run differs from manifest: {run_path}")
    if asdict(collect_index_facts(client, index)) != manifest.get("index"):
        raise DatasetIntegrityError(f"dense run references a different index: {run_path}")
    query_file = cast(dict[str, Any], manifest["query_file"])
    qrels_file = cast(dict[str, Any], manifest["qrels_file"])
    query_path = root / str(query_file["path"])
    qrels_path = root / str(qrels_file["path"])
    expected_query_path = root / f"data/prepared/wands/queries.{split}.jsonl"
    expected_qrels_path = root / f"data/prepared/wands/qrels.{split}.trec"
    if (
        query_path != expected_query_path
        or query_file.get("path") != str(query_path.relative_to(root))
        or file_facts(query_path).sha256 != query_file.get("sha256")
    ):
        raise DatasetIntegrityError(f"dense query file differs: {run_path}")
    if (
        qrels_path != expected_qrels_path
        or qrels_file.get("path") != str(qrels_path.relative_to(root))
        or file_facts(qrels_path).sha256 != qrels_file.get("sha256")
    ):
        raise DatasetIntegrityError(f"dense qrels file differs: {run_path}")
    evaluation = evaluate_run(qrels_path, run_path)
    metrics = cast(dict[str, Any], read_json(metrics_path))
    if (
        metrics.get("schema_version") != 2
        or metrics.get("tag") != tag
        or metrics.get("split") != split
        or metrics.get("metrics") != evaluation.metrics
        or metrics.get("per_query") != evaluation.per_query
        or manifest.get("metrics_file") != _artifact(root, metrics_path)
    ):
        raise DatasetIntegrityError(f"dense metrics do not reproduce: {run_path}")
    if entry.get("metrics") != evaluation.metrics:
        raise DatasetIntegrityError(f"dense summary metrics do not reproduce: {run_path}")
    crosscheck = cross_check_run(qrels_path, run_path, root=root)
    if metrics.get("evaluator_crosscheck") != crosscheck.to_dict():
        raise DatasetIntegrityError(f"dense evaluator cross-check differs: {run_path}")
