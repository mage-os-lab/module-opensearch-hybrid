from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from typing import Any, cast

from poc.bm25 import load_prepared_queries
from poc.config import WANDS_INT8_MODEL_NAMES, ModelSpec, load_model_registry
from poc.datasets import (
    DatasetIntegrityError,
    file_facts,
    load_wands_config,
)
from poc.dense import encode_queries, query_vectors_sha256
from poc.experiments import load_bm25_experiments
from poc.indexing import (
    load_wands_index_config,
    vector_field_name,
    verify_wands_lexical_index,
    wands_completion_provenance_evidence,
    wands_recorded_provenance_valid,
)
from poc.int8_dense import verify_wands_int8_dense_summary
from poc.latency import summarize_latency_ms
from poc.manifest import collect_index_facts, read_json, write_json
from poc.os_client import OpenSearchClient
from poc.provenance import collect_manifest_provenance, require_registered_opensearch_client
from poc.query_runtime import (
    QueryRuntimeSpec,
    create_int8_query_backend,
    load_query_runtime_spec,
)
from poc.search import build_standalone_knn_request
from poc.three_way import load_three_way_spec
from poc.trec import RunRecord, identifier_sort_key, read_run, write_run

ANN_RECALL_DEPTH = 100


def verify_registered_ann_configuration(
    *,
    root: Path,
    index: str,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
    minimum_recall: float,
    ef_search_grid: tuple[int, ...],
    depth: int,
) -> dict[str, Any]:
    dataset = load_wands_config(root / "config/datasets.toml")
    registered_index = load_wands_index_config(root / "config/indexes.toml")
    registered_three_way = load_three_way_spec(root / "config/three_way.toml")
    registered_models = load_model_registry(root / "config/models.toml")
    registered_runtime = load_query_runtime_spec(root / "config/query_runtime.toml")
    registered_experiments = load_bm25_experiments(
        root / "config/experiments.toml"
    )
    registered_model = registered_models[registered_three_way.dense_model]
    if (
        index != registered_index.name
        or model != registered_model
        or runtime != registered_runtime
        or minimum_recall
        != registered_experiments.minimum_ann_recall_at_100
        or ef_search_grid != registered_experiments.ann_ef_search_grid
        or depth != ANN_RECALL_DEPTH
    ):
        raise DatasetIntegrityError("ANN registered configuration differs")
    return {
        "dataset": asdict(dataset),
        "index": asdict(registered_index),
        "model": asdict(registered_model),
        "int8_model_order": list(WANDS_INT8_MODEL_NAMES),
        "query_runtime": asdict(registered_runtime),
        "thresholds": {
            "minimum_recall@100": registered_experiments.minimum_ann_recall_at_100,
            "ef_search_grid": list(registered_experiments.ann_ef_search_grid),
            "depth": ANN_RECALL_DEPTH,
        },
        "config_artifacts": {
            name: _artifact(root, root / f"config/{filename}")
            for name, filename in (
                ("datasets", "datasets.toml"),
                ("experiments", "experiments.toml"),
                ("indexes", "indexes.toml"),
                ("models", "models.toml"),
                ("query_runtime", "query_runtime.toml"),
                ("three_way", "three_way.toml"),
            )
        },
    }


def derive_ann_quality_eligibility(
    *,
    development_passed: object,
    held_out_passed: object,
    provenance_valid: object,
    upstream_unchanged: object,
    upstream_eligible: object,
    selected_model_eligible: object,
) -> bool:
    return all(
        value is True
        for value in (
            development_passed,
            held_out_passed,
            provenance_valid,
            upstream_unchanged,
            upstream_eligible,
            selected_model_eligible,
        )
    )


def assess_ann_recall(
    *,
    exact: dict[str, list[str]],
    approximate: dict[str, list[str]],
    depth: int,
    minimum_recall: float,
) -> dict[str, object]:
    if set(exact) != set(approximate) or not exact:
        raise ValueError("exact and approximate ANN queries differ")
    if depth <= 0 or not 0.0 <= minimum_recall <= 1.0:
        raise ValueError("ANN recall depth or floor is invalid")
    per_query: dict[str, float] = {}
    for query_id in sorted(exact, key=identifier_sort_key):
        exact_documents = exact[query_id][:depth]
        approximate_documents = approximate[query_id][:depth]
        if len(exact_documents) != depth or len(approximate_documents) != depth:
            raise ValueError(f"ANN query {query_id} has fewer than {depth} results")
        if len(set(exact_documents)) != depth or len(set(approximate_documents)) != depth:
            raise ValueError(f"ANN query {query_id} contains duplicate results")
        per_query[query_id] = len(
            set(exact_documents) & set(approximate_documents)
        ) / depth
    mean_recall = fmean(per_query.values())
    minimum_query_recall = min(per_query.values())
    return {
        "depth": depth,
        f"mean_recall@{depth}": mean_recall,
        f"minimum_query_recall@{depth}": minimum_query_recall,
        "minimum_required_mean_recall": minimum_recall,
        "meets_mean_recall_floor": mean_recall >= minimum_recall,
        "per_query": per_query,
    }


def select_ef_search(
    candidates: tuple[int, ...],
    development_mean_recall: dict[int, float],
    *,
    minimum_recall: float,
) -> tuple[int, bool]:
    if (
        not candidates
        or tuple(sorted(set(candidates))) != candidates
        or set(development_mean_recall) != set(candidates)
        or not 0.0 <= minimum_recall <= 1.0
    ):
        raise ValueError("ANN ef_search selection inputs are invalid")
    for candidate in candidates:
        if development_mean_recall[candidate] >= minimum_recall:
            return candidate, True
    return candidates[-1], False


def run_wands_ann_recall(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
    minimum_recall: float,
    ef_search_grid: tuple[int, ...],
    depth: int = 100,
) -> dict[str, object]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    benchmark_provenance = collect_manifest_provenance(
        root,
        profile_path=root / "config/benchmark.toml",
        environment_path=root / "results/environment/benchmark-profile.json",
    )
    registered_configuration = verify_registered_ann_configuration(
        root=root,
        index=index,
        model=model,
        runtime=runtime,
        minimum_recall=minimum_recall,
        ef_search_grid=ef_search_grid,
        depth=depth,
    )
    client.wait_until_ready(expected_version="3.8.0")
    upstream_evidence = _ann_upstream_evidence(
        client,
        root=root,
        index=index,
        runtime=runtime,
        model=model,
    )
    queries = {
        split: load_prepared_queries(
            root / f"data/prepared/wands/queries.{split}.jsonl",
            expected_split=split,
        )
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
    exact = {
        split: _run_documents(read_run(path))
        for split, path in _exact_reference_paths(root, model).items()
    }
    candidates: list[dict[str, object]] = []
    candidate_means: dict[int, float] = {}
    for ef_search in ef_search_grid:
        tag = (
            f"wands-ann-{model.name.replace('_', '-')}-"
            f"ef{ef_search:04d}-dev-int8"
        )
        records, latency = _execute_ann_split(
            client,
            index=index,
            queries=queries["dev"],
            vectors=vectors,
            model=model,
            depth=depth,
            ef_search=ef_search,
            tag=tag,
        )
        run_path = root / f"runs/{tag}.trec"
        write_run(run_path, records)
        recall = assess_ann_recall(
            exact=exact["dev"],
            approximate=_run_documents(records),
            depth=depth,
            minimum_recall=minimum_recall,
        )
        mean_recall = cast(float, recall[f"mean_recall@{depth}"])
        candidate_means[ef_search] = mean_recall
        candidates.append(
            {
                "ef_search": ef_search,
                "split": "dev",
                "recall": recall,
                "search_latency_diagnostic": summarize_latency_ms(latency),
                "run_file": _artifact(root, run_path),
            }
        )
    selected, development_passed = select_ef_search(
        ef_search_grid,
        candidate_means,
        minimum_recall=minimum_recall,
    )
    test_tag = (
        f"wands-ann-{model.name.replace('_', '-')}-"
        f"ef{selected:04d}-test-int8"
    )
    test_records, test_latency = _execute_ann_split(
        client,
        index=index,
        queries=queries["test"],
        vectors=vectors,
        model=model,
        depth=depth,
        ef_search=selected,
        tag=test_tag,
    )
    test_run_path = root / f"runs/{test_tag}.trec"
    write_run(test_run_path, test_records)
    test_recall = assess_ann_recall(
        exact=exact["test"],
        approximate=_run_documents(test_records),
        depth=depth,
        minimum_recall=minimum_recall,
    )
    test_passed = test_recall["meets_mean_recall_floor"] is True
    index_manifest_path = root / "results/wands/index-manifest.json"
    default_failure_path = root / "results/wands/ann-recall.json"
    completion_upstream_evidence = _ann_upstream_evidence(
        client,
        root=root,
        index=index,
        runtime=runtime,
        model=model,
    )
    upstream_unchanged = upstream_evidence == completion_upstream_evidence
    provenance_valid, completion_revision = wands_completion_provenance_evidence(
        root,
        benchmark_provenance,
    )
    quality_eligible = derive_ann_quality_eligibility(
        development_passed=development_passed,
        held_out_passed=test_passed,
        provenance_valid=provenance_valid,
        upstream_unchanged=upstream_unchanged,
        upstream_eligible=upstream_evidence.get("eligible_for_decision"),
        selected_model_eligible=upstream_evidence.get(
            "selected_model_eligible_for_decision"
        ),
    )
    result: dict[str, object] = {
        "schema_version": 2,
        "dataset": "WANDS",
        "method": "lucene_hnsw_ef_search_development_selection",
        "decision_scope": "development_selection_then_held_out_once",
        "model": asdict(model),
        "query_runtime": backend.runtime,
        "query_runtime_spec": asdict(runtime),
        "registered_configuration": registered_configuration,
        "query_encoder_artifact_sha256": backend.model_artifact_sha256,
        "query_vectors_sha256": query_vectors_sha256(vectors),
        "index": asdict(collect_index_facts(client, index)),
        "index_manifest": _artifact(root, index_manifest_path),
        "query_files": {
            split: _artifact(
                root, root / f"data/prepared/wands/queries.{split}.jsonl"
            )
            for split in ("dev", "test")
        },
        "ann": {
            "engine": "lucene",
            "algorithm": "hnsw",
            "m": 16,
            "ef_construction": 128,
            "query_k": depth,
            "ef_search_grid": list(ef_search_grid),
            "selected_ef_search": selected,
        },
        "development_candidates": candidates,
        "development_selection_passed": development_passed,
        "selected_ef_search": selected,
        "held_out_test": {
            "split": "test",
            "recall": test_recall,
            "search_latency_diagnostic": summarize_latency_ms(test_latency),
            "run_file": _artifact(root, test_run_path),
        },
        "request_cache": False,
        "latency_evidence_eligible_for_decision": False,
        "quality_evidence_eligible_for_decision": quality_eligible,
        "quality_ineligibility_reason": (
            None
            if quality_eligible
            else "ANN evidence, upstream bindings, or start provenance is ineligible"
        ),
        "exact_reference_runs": {
            split: _artifact(root, path)
            for split, path in _exact_reference_paths(root, model).items()
        },
        "engine_managed_default_failure": (
            _artifact(root, default_failure_path)
            if default_failure_path.exists()
            else None
        ),
        "upstream_evidence": upstream_evidence,
        "completion_upstream_evidence": completion_upstream_evidence,
        "upstream_evidence_unchanged": upstream_unchanged,
        "benchmark_provenance": benchmark_provenance,
        "completion_code_revision": completion_revision,
        "benchmark_provenance_valid": provenance_valid,
    }
    path = root / "results/wands/ann-recall-tuned.json"
    write_json(path, result)
    return result


def verify_wands_ann_recall(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    model: ModelSpec,
    runtime: QueryRuntimeSpec,
    minimum_recall: float,
    ef_search_grid: tuple[int, ...],
    depth: int = 100,
) -> dict[str, Any]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    registered_configuration = verify_registered_ann_configuration(
        root=root,
        index=index,
        model=model,
        runtime=runtime,
        minimum_recall=minimum_recall,
        ef_search_grid=ef_search_grid,
        depth=depth,
    )
    path = root / "results/wands/ann-recall-tuned.json"
    result = cast(dict[str, Any], read_json(path))
    if (
        result.get("schema_version") != 2
        or result.get("dataset") != "WANDS"
        or result.get("method")
        != "lucene_hnsw_ef_search_development_selection"
        or result.get("decision_scope")
        != "development_selection_then_held_out_once"
        or result.get("model") != asdict(model)
        or result.get("query_runtime_spec") != asdict(runtime)
        or result.get("registered_configuration") != registered_configuration
        or result.get("request_cache") is not False
        or result.get("latency_evidence_eligible_for_decision") is not False
    ):
        raise DatasetIntegrityError("ANN recall metadata differs")
    upstream_evidence = _ann_upstream_evidence(
        client,
        root=root,
        index=index,
        runtime=runtime,
        model=model,
    )
    if (
        result.get("upstream_evidence") != upstream_evidence
        or result.get("completion_upstream_evidence") != upstream_evidence
        or result.get("upstream_evidence_unchanged") is not True
    ):
        raise DatasetIntegrityError("ANN upstream evidence differs")
    if not isinstance(result.get("benchmark_provenance"), Mapping) or not isinstance(
        result.get("completion_code_revision"), Mapping
    ):
        raise DatasetIntegrityError("ANN provenance evidence is missing")
    provenance_valid = wands_recorded_provenance_valid(
        root,
        provenance=result.get("benchmark_provenance"),
        completion_revision=result.get("completion_code_revision"),
    )
    if result.get("benchmark_provenance_valid") is not provenance_valid:
        raise DatasetIntegrityError("ANN provenance validity differs")
    if result.get("index") != asdict(collect_index_facts(client, index)):
        raise DatasetIntegrityError("ANN recall index facts differ")
    if result.get("index_manifest") != _artifact(
        root, root / "results/wands/index-manifest.json"
    ):
        raise DatasetIntegrityError("ANN recall index manifest differs")
    exact = {
        split: _run_documents(read_run(source))
        for split, source in _exact_reference_paths(root, model).items()
    }
    query_files = {
        split: _artifact(
            root, root / f"data/prepared/wands/queries.{split}.jsonl"
        )
        for split in ("dev", "test")
    }
    if result.get("query_files") != query_files:
        raise DatasetIntegrityError("ANN query files differ")
    queries = {
        split: load_prepared_queries(
            root / f"data/prepared/wands/queries.{split}.jsonl",
            expected_split=split,
        )
        for split in ("dev", "test")
    }
    backend = create_int8_query_backend(root=root, model=model, runtime=runtime)
    vectors = encode_queries(
        backend,
        model=model,
        queries={**queries["dev"], **queries["test"]},
        batch_size=runtime.batch_size,
    )
    if (
        result.get("query_runtime") != backend.runtime
        or result.get("query_encoder_artifact_sha256")
        != backend.model_artifact_sha256
        or result.get("query_vectors_sha256") != query_vectors_sha256(vectors)
    ):
        raise DatasetIntegrityError("ANN query runtime evidence differs")
    recorded_candidates = cast(
        list[dict[str, Any]], result["development_candidates"]
    )
    if [candidate["ef_search"] for candidate in recorded_candidates] != list(
        ef_search_grid
    ):
        raise DatasetIntegrityError("ANN ef_search candidate grid differs")
    means: dict[int, float] = {}
    for candidate in recorded_candidates:
        if candidate.get("split") != "dev":
            raise DatasetIntegrityError("ANN development split differs")
        ef_search = int(candidate["ef_search"])
        run_artifact = cast(dict[str, Any], candidate["run_file"])
        run_path = root / str(run_artifact["path"])
        if run_artifact != _artifact(root, run_path):
            raise DatasetIntegrityError("ANN development run differs")
        recall = assess_ann_recall(
            exact=exact["dev"],
            approximate=_run_documents(read_run(run_path)),
            depth=depth,
            minimum_recall=minimum_recall,
        )
        if candidate.get("recall") != recall:
            raise DatasetIntegrityError("ANN development recall does not reproduce")
        _verify_ann_live_rerun(
            client,
            root=root,
            index=index,
            queries=queries["dev"],
            vectors=vectors,
            model=model,
            depth=depth,
            ef_search=ef_search,
            tag=f"wands-ann-{model.name.replace('_', '-')}-ef{ef_search:04d}-dev-int8",
            expected_path=run_path,
        )
        means[ef_search] = cast(float, recall[f"mean_recall@{depth}"])
    selected, development_passed = select_ef_search(
        ef_search_grid,
        means,
        minimum_recall=minimum_recall,
    )
    if (
        result.get("selected_ef_search") != selected
        or result.get("development_selection_passed") is not development_passed
    ):
        raise DatasetIntegrityError("ANN development selection differs")
    if result.get("ann") != {
        "engine": "lucene",
        "algorithm": "hnsw",
        "m": 16,
        "ef_construction": 128,
        "query_k": depth,
        "ef_search_grid": list(ef_search_grid),
        "selected_ef_search": selected,
    }:
        raise DatasetIntegrityError("ANN method configuration differs")
    test = cast(dict[str, Any], result["held_out_test"])
    if test.get("split") != "test":
        raise DatasetIntegrityError("ANN held-out split differs")
    test_artifact = cast(dict[str, Any], test["run_file"])
    test_path = root / str(test_artifact["path"])
    if test_artifact != _artifact(root, test_path):
        raise DatasetIntegrityError("ANN held-out run differs")
    test_recall = assess_ann_recall(
        exact=exact["test"],
        approximate=_run_documents(read_run(test_path)),
        depth=depth,
        minimum_recall=minimum_recall,
    )
    if test.get("recall") != test_recall:
        raise DatasetIntegrityError("ANN held-out recall does not reproduce")
    _verify_ann_live_rerun(
        client,
        root=root,
        index=index,
        queries=queries["test"],
        vectors=vectors,
        model=model,
        depth=depth,
        ef_search=selected,
        tag=f"wands-ann-{model.name.replace('_', '-')}-ef{selected:04d}-test-int8",
        expected_path=test_path,
    )
    eligible = derive_ann_quality_eligibility(
        development_passed=development_passed,
        held_out_passed=test_recall["meets_mean_recall_floor"],
        provenance_valid=provenance_valid,
        upstream_unchanged=result.get("upstream_evidence_unchanged"),
        upstream_eligible=upstream_evidence.get("eligible_for_decision"),
        selected_model_eligible=upstream_evidence.get(
            "selected_model_eligible_for_decision"
        ),
    )
    if result.get("quality_evidence_eligible_for_decision") is not eligible:
        raise DatasetIntegrityError("ANN quality eligibility differs")
    expected_reason = (
        None
        if eligible
        else "ANN evidence, upstream bindings, or start provenance is ineligible"
    )
    if result.get("quality_ineligibility_reason") != expected_reason:
        raise DatasetIntegrityError("ANN quality ineligibility reason differs")
    expected_references = {
        split: _artifact(root, source)
        for split, source in _exact_reference_paths(root, model).items()
    }
    if result.get("exact_reference_runs") != expected_references:
        raise DatasetIntegrityError("ANN exact references differ")
    default = result.get("engine_managed_default_failure")
    if default is not None:
        recorded_default = cast(dict[str, Any], default)
        default_path = root / str(recorded_default["path"])
        if recorded_default != _artifact(root, default_path):
            raise DatasetIntegrityError("ANN default failure artifact differs")
    return result


def _ann_upstream_evidence(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    runtime: QueryRuntimeSpec,
    model: ModelSpec,
) -> dict[str, object]:
    dataset = load_wands_config(root / "config/datasets.toml")
    index_spec = load_wands_index_config(root / "config/indexes.toml")
    models_by_name = load_model_registry(root / "config/models.toml")
    models = tuple(models_by_name[name] for name in WANDS_INT8_MODEL_NAMES)
    index_verification = verify_wands_lexical_index(
        client,
        dataset_spec=dataset,
        index_spec=index_spec,
        prepared_directory=root / "data/prepared/wands",
        manifest_path=root / "results/wands/index-manifest.json",
    )
    dense_summary = verify_wands_int8_dense_summary(
        client,
        root=root,
        index=index,
        models=models,
        runtime=runtime,
    )
    dense_models = cast(dict[str, Any], dense_summary["models"])
    selected_dense_model = cast(dict[str, Any], dense_models[model.name])
    selected_model_present = model.name in cast(
        list[str], index_verification["vector_models"]
    )
    selected_model_eligible = (
        model.decision_eligible
        and selected_model_present
        and selected_dense_model.get("eligible_for_decision") is True
    )
    return {
        "index_manifest": _artifact(
            root, root / "results/wands/index-manifest.json"
        ),
        "int8_dense_summary": _artifact(
            root, root / "results/wands/int8-dense-summary.json"
        ),
        "index_quality_eligible_for_decision": index_verification.get(
            "quality_evidence_eligible_for_decision"
        )
        is True,
        "dense_quality_eligible_for_decision": dense_summary.get(
            "quality_evidence_eligible_for_decision"
        )
        is True,
        "selected_model": model.name,
        "selected_model_present_in_live_vector_index": selected_model_present,
        "selected_model_eligible_for_decision": selected_model_eligible,
        "eligible_for_decision": (
            index_verification.get("quality_evidence_eligible_for_decision")
            is True
            and dense_summary.get("quality_evidence_eligible_for_decision")
            is True
            and selected_model_eligible
        ),
    }


def _verify_ann_live_rerun(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    queries: dict[str, str],
    vectors: dict[str, Any],
    model: ModelSpec,
    depth: int,
    ef_search: int,
    tag: str,
    expected_path: Path,
) -> None:
    records, _ = _execute_ann_split(
        client,
        index=index,
        queries=queries,
        vectors=vectors,
        model=model,
        depth=depth,
        ef_search=ef_search,
        tag=tag,
    )
    with TemporaryDirectory(prefix="opensearch-hybrid-ann-verify-") as temporary:
        rerun_path = Path(temporary) / "rerun.trec"
        write_run(rerun_path, records)
        if rerun_path.read_bytes() != expected_path.read_bytes():
            raise DatasetIntegrityError("ANN live rerun differs")


def _execute_ann_split(
    client: OpenSearchClient,
    *,
    index: str,
    queries: dict[str, str],
    vectors: dict[str, Any],
    model: ModelSpec,
    depth: int,
    ef_search: int,
    tag: str,
) -> tuple[list[RunRecord], list[float]]:
    records: list[RunRecord] = []
    latency_samples: list[float] = []
    for query_id in sorted(queries, key=identifier_sort_key):
        request = build_standalone_knn_request(
            vectors[query_id].tolist(),
            vector_field=vector_field_name(model),
            k=depth,
            ef_search=ef_search,
        )
        started = time.perf_counter()
        response = client.search(index, request, request_cache=False)
        latency_samples.append((time.perf_counter() - started) * 1000.0)
        hits = cast(dict[str, Any], response.get("hits", {})).get("hits")
        if not isinstance(hits, list) or len(hits) != depth:
            raise DatasetIntegrityError(
                f"ANN query {query_id} did not return {depth} hits"
            )
        records.extend(
            RunRecord(
                query_id=query_id,
                document_id=str(hit["_id"]),
                rank=rank,
                score=float(depth - rank + 1),
                tag=tag,
            )
            for rank, hit in enumerate(hits, start=1)
        )
    return records, latency_samples


def _exact_reference_paths(root: Path, model: ModelSpec) -> dict[str, Path]:
    summary = cast(
        dict[str, Any], read_json(root / "results/wands/int8-dense-summary.json")
    )
    entry = cast(dict[str, Any], cast(dict[str, Any], summary["models"])[model.name])
    splits = cast(dict[str, Any], entry["splits"])
    return {
        split: root / str(cast(dict[str, Any], splits[split])["run"])
        for split in ("dev", "test")
    }


def _run_documents(records: list[RunRecord]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for record in sorted(records, key=lambda item: (item.query_id, item.rank)):
        output.setdefault(record.query_id, []).append(record.document_id)
    return output


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }
