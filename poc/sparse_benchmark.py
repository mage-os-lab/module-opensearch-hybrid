from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from poc.bm25 import load_prepared_queries, verify_wands_bm25_selection
from poc.config import WANDS_INT8_MODEL_NAMES, load_model_registry
from poc.datasets import DatasetIntegrityError, file_facts, load_wands_config
from poc.evaluation import Evaluation, evaluate_run
from poc.evaluation_crosscheck import cross_check_run
from poc.experiments import BM25Experiments, BM25Profile
from poc.hybrid import (
    hybrid_trec_score_for_rank,
    select_hybrid_weight,
    verify_wands_hybrid_summary,
)
from poc.index_content import verify_live_index_content
from poc.index_evidence import (
    INDEX_WRITE_BLOCK_EVIDENCE,
    content_addressed_pipeline_id,
    verify_live_index_settings,
    verify_live_search_pipeline,
)
from poc.indexing import (
    verify_wands_preparation_evidence,
    wands_completion_provenance_evidence,
    wands_recorded_provenance_valid,
)
from poc.manifest import canonical_sha256, collect_index_facts, read_json, write_json
from poc.neural_sparse import (
    NeuralSparseSpec,
    build_sparse_index_definition,
    build_sparse_ingest_pipeline,
    build_sparse_query,
    iter_precomputed_sparse_documents,
    verify_wands_sparse_precompute,
    verify_wands_sparse_tokenizer_probe,
)
from poc.os_client import OpenSearchClient
from poc.provenance import collect_manifest_provenance, require_registered_opensearch_client
from poc.query_runtime import load_query_runtime_spec
from poc.rrf import verify_wands_rrf_summary
from poc.search import build_normalization_pipeline, build_two_clause_hybrid_request
from poc.trec import RunRecord, identifier_sort_key, read_run, write_run

RESULT_SIZE = 100
_INDEX_MANIFEST = Path("results/wands/neural-sparse/index-manifest.json")
_TOKENIZER_INSPECTION = Path(
    "results/wands/neural-sparse/tokenizer-inspection.json"
)
_BM25_SELECTION = Path("results/wands/bm25-selection.json")


def _json_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(json.dumps(value)))


def _write_json_exact(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }


def _hash_matches(path: Path, expected: object) -> bool:
    return isinstance(expected, str) and path.is_file() and file_facts(path).sha256 == expected


def _standard_wands_index_name(root: Path, fallback: str) -> str:
    path = root / "results/wands/index-manifest.json"
    if not path.is_file():
        return fallback
    value = read_json(path)
    if not isinstance(value, dict):
        return fallback
    index = value.get("index")
    if not isinstance(index, dict):
        return fallback
    name = index.get("name")
    return name if isinstance(name, str) and name else fallback


def _sparse_index_evidence(
    client: OpenSearchClient,
    *,
    root: Path,
    spec: NeuralSparseSpec,
) -> dict[str, object]:
    manifest_path = root / _INDEX_MANIFEST
    manifest_value = read_json(manifest_path)
    if not isinstance(manifest_value, dict):
        raise DatasetIntegrityError("neural-sparse index manifest must be an object")
    manifest = cast(dict[str, Any], manifest_value)
    model_id = manifest.get("model_id")
    definition = build_sparse_index_definition(spec, use_default_pipeline=False)
    pipeline = (
        build_sparse_ingest_pipeline(spec, model_id=model_id)
        if isinstance(model_id, str) and model_id
        else None
    )
    live_index = asdict(collect_index_facts(client, spec.index_name))
    live_mapping_bound = False
    live_settings_bound = False
    live_ingest_pipeline_bound = False
    try:
        mapping = cast(
            dict[str, Any],
            client.request("GET", f"/{spec.index_name}/_mapping"),
        )
        live_mapping_bound = (
            cast(dict[str, Any], mapping[spec.index_name]).get("mappings")
            == definition["mappings"]
        )
        verify_live_index_settings(
            client,
            index=spec.index_name,
            definition=definition,
        )
        live_settings_bound = True
        if pipeline is not None:
            live_pipeline_response = cast(
                dict[str, Any],
                client.request(
                    "GET",
                    f"/_ingest/pipeline/{spec.ingest_pipeline}",
                ),
            )
            live_pipeline_value = live_pipeline_response.get(
                spec.ingest_pipeline
            )
            if isinstance(live_pipeline_value, dict):
                live_pipeline = dict(live_pipeline_value)
                live_pipeline.pop("version", None)
                live_ingest_pipeline_bound = live_pipeline == pipeline
    except (DatasetIntegrityError, KeyError, TypeError):
        pass
    preparation_bound = False
    precompute_evidence: dict[str, object] = {}
    precompute_bound = False
    live_content_bound = False
    try:
        dataset_spec = load_wands_config(root / "config/datasets.toml")
        preparation = _json_dict(
            verify_wands_preparation_evidence(
                root=root,
                dataset_spec=dataset_spec,
                prepared_directory=root / "data/prepared/wands",
            )
        )
        preparation_bound = manifest.get("preparation_evidence") == preparation
        precompute_evidence = verify_wands_sparse_precompute(
            root=root,
            dataset=dataset_spec,
            spec=spec,
            products_path=root / "data/prepared/wands/products.jsonl",
            embeddings_path=(
                root / "data/cache/neural-sparse/wands-doc-v3-distill.jsonl"
            ),
            manifest_path=(
                root / "results/wands/neural-sparse/precompute-manifest.json"
            ),
        )
        precompute_bound = (
            manifest.get("precompute_evidence") == precompute_evidence
            and precompute_evidence.get("eligible_for_decision") is True
        )
        live_content = verify_live_index_content(
            client,
            index=spec.index_name,
            expected_documents=iter_precomputed_sparse_documents(
                root / "data/prepared/wands/products.jsonl",
                root / "data/cache/neural-sparse/wands-doc-v3-distill.jsonl",
                spec,
            ),
            expected_count=dataset_spec.expected_products,
            float32_fields=(spec.embedding_field,),
        )
        live_content_bound = manifest.get("index_content") == live_content
    except (DatasetIntegrityError, KeyError, OSError, TypeError, ValueError):
        pass
    checks = {
        "schema_valid": (
            manifest.get("schema_version") == 2
            and manifest.get("dataset") == "WANDS"
        ),
        "live_index_bound": manifest.get("index") == live_index,
        "live_mapping_bound": live_mapping_bound,
        "live_settings_bound": live_settings_bound,
        "live_ingest_pipeline_bound": live_ingest_pipeline_bound,
        "live_content_bound": live_content_bound,
        "index_definition_bound": (
            manifest.get("index_definition_sha256")
            == canonical_sha256(definition)
        ),
        "index_write_block_bound": (
            manifest.get("index_write_block") == INDEX_WRITE_BLOCK_EVIDENCE
        ),
        "ingest_pipeline_bound": (
            pipeline is not None
            and manifest.get("ingest_pipeline_sha256")
            == canonical_sha256(pipeline)
        ),
        "model_artifact_bound": _hash_matches(
            root / "results/wands/neural-sparse/model-deployment.json",
            manifest.get("model_artifact_sha256"),
        ),
        "prepared_products_bound": _hash_matches(
            root / "data/prepared/wands/products.jsonl",
            manifest.get("prepared_products_sha256"),
        ),
        "precomputed_embeddings_bound": _hash_matches(
            root / "data/cache/neural-sparse/wands-doc-v3-distill.jsonl",
            manifest.get("precomputed_embeddings_sha256"),
        ),
        "precompute_manifest_bound": _hash_matches(
            root / "results/wands/neural-sparse/precompute-manifest.json",
            manifest.get("precompute_manifest_sha256"),
        ),
        "precompute_evidence_bound": precompute_bound,
        "preparation_bound": preparation_bound,
        "query_contract_bound": (
            manifest.get("query_mode") == "doc_only_builtin_analyzer"
            and manifest.get("query_analyzer") == spec.query_analyzer
        ),
        "provenance_valid": wands_recorded_provenance_valid(
            root,
            provenance=manifest.get("benchmark_provenance"),
            completion_revision=manifest.get("completion_code_revision"),
        ),
        "latency_ineligible": (
            manifest.get("latency_evidence_eligible_for_decision") is False
        ),
    }
    derived_quality = all(checks.values())
    declared_quality = manifest.get("quality_evidence_eligible_for_decision")
    declaration_consistent = declared_quality is derived_quality
    reasons = [name for name, passed in checks.items() if not passed]
    if not declaration_consistent:
        reasons.append("quality_declaration_consistent")
    return {
        "artifact": _artifact(root, manifest_path),
        **checks,
        "quality_declared": declared_quality is True,
        "quality_declaration_consistent": declaration_consistent,
        "eligible_for_decision": derived_quality and declaration_consistent,
        "ineligibility_reasons": reasons,
    }


def _tokenizer_evidence(
    client: OpenSearchClient,
    *,
    root: Path,
    spec: NeuralSparseSpec,
    index_evidence: Mapping[str, Any],
) -> dict[str, object]:
    try:
        return verify_wands_sparse_tokenizer_probe(
            client,
            root=root,
            spec=spec,
            index_evidence=index_evidence,
        )
    except (DatasetIntegrityError, KeyError, OSError, TypeError, ValueError):
        path = root / "results/wands/neural-sparse/tokenizer-inspection.json"
        return {
            "artifact": _artifact(root, path),
            "schema_valid": False,
            "status_passed": False,
            "index_manifest_bound": False,
            "live_probe_verified": False,
            "quality_declaration_consistent": False,
            "quality_declared": False,
            "eligible_for_decision": False,
            "ineligibility_reasons": ["semantic_verification_failed"],
        }


def _bm25_evidence(
    client: OpenSearchClient,
    *,
    root: Path,
    spec: NeuralSparseSpec,
    experiments: BM25Experiments,
) -> dict[str, object]:
    path = root / _BM25_SELECTION
    value = read_json(path)
    if not isinstance(value, dict):
        raise DatasetIntegrityError("WANDS BM25 selection must be an object")
    selection = cast(dict[str, Any], value)
    verified = True
    try:
        verify_wands_bm25_selection(
            client,
            root=root,
            index=_standard_wands_index_name(root, spec.index_name),
            experiments=experiments,
        )
    except (DatasetIntegrityError, KeyError, TypeError, ValueError):
        verified = False
    quality_declared = selection.get("quality_evidence_eligible_for_decision") is True
    return {
        "artifact": _artifact(root, path),
        "schema_valid": (
            selection.get("schema_version") == 2
            and selection.get("dataset") == "WANDS"
        ),
        "verified": verified,
        "quality_declared": quality_declared,
        "eligible_for_decision": (
            verified
            and quality_declared
            and selection.get("schema_version") == 2
            and selection.get("dataset") == "WANDS"
        ),
    }


def _sparse_upstream_evidence(
    client: OpenSearchClient,
    *,
    root: Path,
    spec: NeuralSparseSpec,
    experiments: BM25Experiments,
) -> dict[str, object]:
    index = _sparse_index_evidence(client, root=root, spec=spec)
    tokenizer = _tokenizer_evidence(
        client,
        root=root,
        spec=spec,
        index_evidence=index,
    )
    bm25 = _bm25_evidence(
        client,
        root=root,
        spec=spec,
        experiments=experiments,
    )
    eligible = all(
        evidence.get("eligible_for_decision") is True
        for evidence in (index, tokenizer, bm25)
    )
    reasons = [
        name
        for name, evidence in (
            ("neural_sparse_index", index),
            ("tokenizer_inspection", tokenizer),
            ("bm25_selection", bm25),
        )
        if evidence.get("eligible_for_decision") is not True
    ]
    return {
        "neural_sparse_index": index,
        "tokenizer_inspection": tokenizer,
        "bm25_selection": bm25,
        "eligible_for_decision": eligible,
        "ineligibility_reasons": reasons,
    }


def _verified_dense_comparisons(
    client: OpenSearchClient,
    *,
    root: Path,
    spec: NeuralSparseSpec,
    experiments: BM25Experiments,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, object]]:
    registry = load_model_registry(root / "config/models.toml")
    models = tuple(registry[name] for name in WANDS_INT8_MODEL_NAMES)
    runtime = load_query_runtime_spec(root / "config/query_runtime.toml")
    index = _standard_wands_index_name(root, spec.index_name)
    minmax = verify_wands_hybrid_summary(
        client,
        root=root,
        index=index,
        models=models,
        runtime=runtime,
        experiments=experiments,
    )
    rrf = verify_wands_rrf_summary(
        client,
        root=root,
        index=index,
        models=models,
        runtime=runtime,
        experiments=experiments,
    )
    minmax_eligible = minmax.get("eligible_for_decision") is True
    rrf_eligible = rrf.get("eligible_for_decision") is True
    evidence: dict[str, object] = {
        "dense_minmax_summary": _artifact(
            root, root / "results/wands/hybrid-summary.json"
        ),
        "dense_rrf_summary": _artifact(
            root, root / "results/wands/rrf-summary.json"
        ),
        "dense_minmax_semantically_verified": True,
        "dense_rrf_semantically_verified": True,
        "dense_minmax_eligible_for_decision": minmax_eligible,
        "dense_rrf_eligible_for_decision": rrf_eligible,
        "eligible_for_decision": minmax_eligible and rrf_eligible,
    }
    return minmax, rrf, evidence


def _run_eligibility(
    *,
    decision_scope: str,
    provenance_valid: bool,
    upstream_eligible: bool,
    upstream_unchanged: bool,
) -> tuple[bool, str | None]:
    if decision_scope != "held_out":
        return False, "development or context-only result"
    reasons: list[str] = []
    if not provenance_valid:
        reasons.append("run has no valid clean committed start provenance")
    if not upstream_eligible:
        reasons.append("run upstream evidence is not decision eligible")
    if not upstream_unchanged:
        reasons.append("run upstream evidence changed after benchmark start")
    return not reasons, "; ".join(reasons) or None


def _expected_artifact_keys(experiments: BM25Experiments) -> tuple[str, ...]:
    return tuple(
        ["sparse/dev", "sparse/test"]
        + [
            f"hybrid/dev/lw{round(weight * 100):03d}"
            for weight in experiments.lexical_weight_grid
        ]
        + ["hybrid/test/selected"]
    )


def _source_manifest_bindings(
    *,
    root: Path,
    artifacts: Mapping[str, Mapping[str, Any]],
    experiments: BM25Experiments,
    benchmark_provenance: Mapping[str, Any],
    upstream_evidence: Mapping[str, Any],
) -> tuple[dict[str, dict[str, object]], bool, list[str]]:
    bindings: dict[str, dict[str, object]] = {}
    reasons: list[str] = []
    for key in _expected_artifact_keys(experiments):
        entry = artifacts.get(key)
        if entry is None:
            reasons.append(f"source manifest is missing: {key}")
            continue
        manifest_path = root / str(entry.get("manifest", ""))
        if not manifest_path.is_file():
            reasons.append(f"source manifest file is missing: {key}")
            continue
        bindings[key] = _artifact(root, manifest_path)
        value = read_json(manifest_path)
        if not isinstance(value, dict):
            reasons.append(f"source manifest is not an object: {key}")
            continue
        manifest = cast(dict[str, Any], value)
        expected_split = "dev" if "/dev" in key else "test"
        if key == "sparse/dev":
            expected_scope = "context_only"
        elif expected_split == "dev":
            expected_scope = "tuning"
        else:
            expected_scope = "held_out"
        provenance = manifest.get("benchmark_provenance")
        provenance_valid = wands_recorded_provenance_valid(
            root,
            provenance=provenance,
            completion_revision=manifest.get("completion_code_revision"),
        )
        upstream_unchanged = (
            manifest.get("upstream_evidence") == upstream_evidence
            and manifest.get("completion_upstream_evidence") == upstream_evidence
            and manifest.get("upstream_evidence_unchanged") is True
        )
        eligible, reason = _run_eligibility(
            decision_scope=expected_scope,
            provenance_valid=provenance_valid,
            upstream_eligible=(
                upstream_evidence.get("eligible_for_decision") is True
            ),
            upstream_unchanged=upstream_unchanged,
        )
        valid = (
            manifest.get("schema_version") == 2
            and manifest.get("dataset") == "WANDS"
            and manifest.get("split") == expected_split
            and manifest.get("decision_scope") == expected_scope
            and provenance == benchmark_provenance
            and manifest.get("benchmark_provenance_valid") is provenance_valid
            and upstream_unchanged
            and manifest.get("eligible_for_decision") is eligible
            and manifest.get("ineligibility_reason") == reason
        )
        if not valid:
            reasons.append(f"source manifest binding differs: {key}")
    return bindings, not reasons, reasons


def _summary_eligibility(
    *,
    provenance_valid: bool,
    upstream_eligible: bool,
    upstream_unchanged: bool,
    source_bindings_valid: bool,
    held_out_sources_eligible: bool,
    comparison_sources_eligible: bool,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not provenance_valid:
        reasons.append("summary has no valid clean committed start provenance")
    if not upstream_eligible:
        reasons.append("summary upstream evidence is not decision eligible")
    if not upstream_unchanged:
        reasons.append("summary upstream evidence changed after benchmark start")
    if not source_bindings_valid:
        reasons.append("summary source manifest bindings differ")
    if not held_out_sources_eligible:
        reasons.append("summary held-out source manifests are not decision eligible")
    if not comparison_sources_eligible:
        reasons.append("summary dense comparison sources are not decision eligible")
    return not reasons, reasons


@dataclass(frozen=True, slots=True)
class SparseRunResult:
    run_path: Path
    manifest_path: Path
    metrics_path: Path
    evaluation: Evaluation
    records: list[RunRecord]

    def selection_entry(self, root: Path) -> dict[str, object]:
        return {
            "run": str(self.run_path.relative_to(root)),
            "manifest": str(self.manifest_path.relative_to(root)),
            "metrics_file": str(self.metrics_path.relative_to(root)),
            "metrics": self.evaluation.metrics,
        }


def sparse_pipeline_id(lexical_weight: float) -> str:
    definition = build_normalization_pipeline(lexical_weight=lexical_weight)
    return content_addressed_pipeline_id(
        (
            "opensearch-hybrid-wands-sparse-minmax-"
            f"lw{round(lexical_weight * 100):03d}-v1"
        ),
        definition,
    )


def execute_sparse_run(
    client: OpenSearchClient,
    *,
    index: str,
    queries: dict[str, str],
    spec: NeuralSparseSpec,
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
                "query": build_sparse_query(spec, queries[query_id]),
            },
        )
        records.extend(_response_records(response, query_id=query_id, tag=tag))
    return records


def execute_sparse_hybrid_run(
    client: OpenSearchClient,
    *,
    index: str,
    queries: dict[str, str],
    spec: NeuralSparseSpec,
    profile: BM25Profile,
    lexical_weight: float,
    tag: str,
) -> list[RunRecord]:
    pipeline_id = sparse_pipeline_id(lexical_weight)
    pipeline = build_normalization_pipeline(lexical_weight=lexical_weight)
    client.put_search_pipeline(
        pipeline_id,
        pipeline,
    )
    verify_live_search_pipeline(
        client,
        pipeline_id=pipeline_id,
        expected_definition=pipeline,
    )
    records: list[RunRecord] = []
    for query_id in sorted(queries, key=identifier_sort_key):
        request = build_two_clause_hybrid_request(
            profile.query(queries[query_id]),
            build_sparse_query(spec, queries[query_id]),
            size=RESULT_SIZE,
            pagination_depth=RESULT_SIZE,
        )
        response = client.search(index, request, pipeline=pipeline_id)
        records.extend(_response_records(response, query_id=query_id, tag=tag))
    verify_live_search_pipeline(
        client,
        pipeline_id=pipeline_id,
        expected_definition=pipeline,
    )
    return records


def write_sparse_bundle(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    spec: NeuralSparseSpec,
    query_path: Path,
    qrels_path: Path,
    split: str,
    tag: str,
    decision_scope: str,
    records: list[RunRecord],
    experiments: BM25Experiments,
    benchmark_provenance: dict[str, Any],
    upstream_evidence: dict[str, Any],
    lexical_profile: BM25Profile | None = None,
    lexical_weight: float | None = None,
) -> SparseRunResult:
    run_path = root / f"runs/{tag}.trec"
    manifest_path = root / f"runs/{tag}.manifest.json"
    metrics_path = root / f"results/wands/neural-sparse/{tag}.metrics.json"
    write_run(run_path, records)
    evaluation = evaluate_run(qrels_path, run_path)
    crosscheck = cross_check_run(qrels_path, run_path, root=root)
    write_json(
        metrics_path,
        {
            "schema_version": 1,
            "tag": tag,
            "split": split,
            "metrics": evaluation.metrics,
            "per_query": evaluation.per_query,
            "evaluator_crosscheck": crosscheck.to_dict(),
        },
    )
    index_manifest_path = root / "results/wands/neural-sparse/index-manifest.json"
    index_manifest = cast(dict[str, Any], read_json(index_manifest_path))
    current_upstream = _sparse_upstream_evidence(
        client,
        root=root,
        spec=spec,
        experiments=experiments,
    )
    upstream_unchanged = current_upstream == upstream_evidence
    provenance_valid, completion_revision = wands_completion_provenance_evidence(
        root,
        benchmark_provenance,
    )
    eligible, ineligibility_reason = _run_eligibility(
        decision_scope=decision_scope,
        provenance_valid=provenance_valid,
        upstream_eligible=(
            upstream_evidence.get("eligible_for_decision") is True
            and current_upstream.get("eligible_for_decision") is True
        ),
        upstream_unchanged=upstream_unchanged,
    )
    pipeline = (
        build_normalization_pipeline(lexical_weight=lexical_weight)
        if lexical_weight is not None
        else None
    )
    manifest = {
        "schema_version": 2,
        "dataset": "WANDS",
        "split": split,
        "tag": tag,
        "decision_scope": decision_scope,
        "eligible_for_decision": eligible,
        "ineligibility_reason": ineligibility_reason,
        "benchmark_provenance": benchmark_provenance,
        "benchmark_provenance_valid": provenance_valid,
        "completion_code_revision": completion_revision,
        "upstream_evidence": upstream_evidence,
        "completion_upstream_evidence": current_upstream,
        "upstream_evidence_unchanged": upstream_unchanged,
        "declared_variable": "lexical_weight" if lexical_weight is not None else "retrieval_arm",
        "index": asdict(collect_index_facts(client, index)),
        "index_manifest_sha256": file_facts(index_manifest_path).sha256,
        "index_manifest": _artifact(root, index_manifest_path),
        "index_definition_sha256": index_manifest["index_definition_sha256"],
        "query_mode": "doc_only_builtin_analyzer",
        "query_model_id": None,
        "ingestion_model_id": index_manifest["model_id"],
        "query_analyzer": spec.query_analyzer,
        "lexical_profile": lexical_profile.to_dict() if lexical_profile else None,
        "fusion": (
            {
                "normalization": "min_max",
                "combination": "arithmetic_mean",
                "lexical_weight": lexical_weight,
                "sparse_weight": 1.0 - lexical_weight,
            }
            if lexical_weight is not None
            else None
        ),
        "pipeline_id": sparse_pipeline_id(lexical_weight) if lexical_weight is not None else None,
        "pipeline_sha256": canonical_sha256(pipeline) if pipeline else None,
        "query_file": {
            "path": str(query_path.relative_to(root)),
            "sha256": file_facts(query_path).sha256,
        },
        "qrels_file": {
            "path": str(qrels_path.relative_to(root)),
            "sha256": file_facts(qrels_path).sha256,
        },
        "run_file": {
            "path": str(run_path.relative_to(root)),
            "sha256": file_facts(run_path).sha256,
            "records": len(records),
        },
    }
    _write_json_exact(manifest_path, manifest)
    return SparseRunResult(run_path, manifest_path, metrics_path, evaluation, records)


def run_wands_sparse_benchmark(
    client: OpenSearchClient,
    *,
    root: Path,
    spec: NeuralSparseSpec,
    experiments: BM25Experiments,
) -> dict[str, object]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    benchmark_provenance = _json_dict(collect_manifest_provenance(root))
    upstream_evidence = _json_dict(
        _sparse_upstream_evidence(
            client,
            root=root,
            spec=spec,
            experiments=experiments,
        )
    )
    prepared = root / "data/prepared/wands"
    query_paths = {split: prepared / f"queries.{split}.jsonl" for split in ("dev", "test")}
    qrels_paths = {split: prepared / f"qrels.{split}.trec" for split in ("dev", "test")}
    queries = {
        split: load_prepared_queries(query_paths[split], expected_split=split)
        for split in ("dev", "test")
    }
    bm25 = cast(dict[str, Any], read_json(root / "results/wands/bm25-selection.json"))
    tokenizer_inspection = cast(
        dict[str, Any],
        read_json(root / "results/wands/neural-sparse/tokenizer-inspection.json"),
    )
    if tokenizer_inspection.get("status") != "passed":
        raise ValueError("neural-sparse tokenizer compatibility is not verified")
    profile = BM25Profile.from_mapping(cast(dict[str, Any], bm25["selected_profile"]))
    artifacts: dict[str, dict[str, object]] = {}

    for split in ("dev", "test"):
        tag = f"wands-neural-sparse-doc-only-{split}"
        result = write_sparse_bundle(
            client,
            root=root,
            index=spec.index_name,
            spec=spec,
            query_path=query_paths[split],
            qrels_path=qrels_paths[split],
            split=split,
            tag=tag,
            decision_scope="held_out" if split == "test" else "context_only",
            records=execute_sparse_run(
                client,
                index=spec.index_name,
                queries=queries[split],
                spec=spec,
                tag=tag,
            ),
            experiments=experiments,
            benchmark_provenance=benchmark_provenance,
            upstream_evidence=upstream_evidence,
        )
        artifacts[f"sparse/{split}"] = result.selection_entry(root)

    candidates: dict[float, SparseRunResult] = {}
    for weight in experiments.lexical_weight_grid:
        tag = f"wands-bm25-sparse-minmax-lw{round(weight * 100):03d}-dev"
        candidates[weight] = write_sparse_bundle(
            client,
            root=root,
            index=spec.index_name,
            spec=spec,
            query_path=query_paths["dev"],
            qrels_path=qrels_paths["dev"],
            split="dev",
            tag=tag,
            decision_scope="tuning",
            records=execute_sparse_hybrid_run(
                client,
                index=spec.index_name,
                queries=queries["dev"],
                spec=spec,
                profile=profile,
                lexical_weight=weight,
                tag=tag,
            ),
            lexical_profile=profile,
            lexical_weight=weight,
            experiments=experiments,
            benchmark_provenance=benchmark_provenance,
            upstream_evidence=upstream_evidence,
        )
        artifacts[f"hybrid/dev/lw{round(weight * 100):03d}"] = candidates[
            weight
        ].selection_entry(root)
    selected_weight = select_hybrid_weight(
        experiments.lexical_weight_grid,
        {weight: result.evaluation.metrics for weight, result in candidates.items()},
        metric=experiments.primary_metric,
    )
    test_tag = f"wands-bm25-sparse-minmax-lw{round(selected_weight * 100):03d}-test"
    selected = write_sparse_bundle(
        client,
        root=root,
        index=spec.index_name,
        spec=spec,
        query_path=query_paths["test"],
        qrels_path=qrels_paths["test"],
        split="test",
        tag=test_tag,
        decision_scope="held_out",
        records=execute_sparse_hybrid_run(
            client,
            index=spec.index_name,
            queries=queries["test"],
            spec=spec,
            profile=profile,
            lexical_weight=selected_weight,
            tag=test_tag,
        ),
        lexical_profile=profile,
        lexical_weight=selected_weight,
        experiments=experiments,
        benchmark_provenance=benchmark_provenance,
        upstream_evidence=upstream_evidence,
    )
    artifacts["hybrid/test/selected"] = selected.selection_entry(root)
    tuned_bm25 = float(bm25["held_out_test_metrics"]["tuned"]["ndcg@10"])
    competent_bm25 = float(
        bm25["held_out_test_metrics"]["competent_integrator"]["ndcg@10"]
    )
    dense_minmax, dense_rrf, dense_comparison_evidence = (
        _verified_dense_comparisons(
            client,
            root=root,
            spec=spec,
            experiments=experiments,
        )
    )
    sparse_hybrid_ndcg = selected.evaluation.metrics["ndcg@10"]
    dense_minmax_ndcg = float(dense_minmax["held_out_test_metrics"]["ndcg@10"])
    dense_rrf_ndcg = float(dense_rrf["held_out_test_metrics"]["ndcg@10"])
    completion_upstream = _json_dict(
        _sparse_upstream_evidence(
            client,
            root=root,
            spec=spec,
            experiments=experiments,
        )
    )
    upstream_unchanged = completion_upstream == upstream_evidence
    provenance_valid, completion_revision = wands_completion_provenance_evidence(
        root,
        benchmark_provenance,
    )
    source_bindings, source_bindings_valid, source_binding_reasons = (
        _source_manifest_bindings(
            root=root,
            artifacts=artifacts,
            experiments=experiments,
            benchmark_provenance=benchmark_provenance,
            upstream_evidence=upstream_evidence,
        )
    )
    held_out_sources_eligible = all(
        cast(dict[str, Any], read_json(root / str(artifacts[key]["manifest"]))).get(
            "eligible_for_decision"
        )
        is True
        for key in ("sparse/test", "hybrid/test/selected")
    )
    quality_eligible, quality_reasons = _summary_eligibility(
        provenance_valid=provenance_valid,
        upstream_eligible=(
            upstream_evidence.get("eligible_for_decision") is True
            and completion_upstream.get("eligible_for_decision") is True
        ),
        upstream_unchanged=upstream_unchanged,
        source_bindings_valid=source_bindings_valid,
        held_out_sources_eligible=held_out_sources_eligible,
        comparison_sources_eligible=(
            dense_comparison_evidence.get("eligible_for_decision") is True
        ),
    )
    summary: dict[str, object] = {
        "schema_version": 2,
        "dataset": "WANDS",
        "mode": "document_only_neural_sparse_builtin_query_analyzer",
        "query_model_deployed": False,
        "benchmark_provenance": benchmark_provenance,
        "benchmark_provenance_valid": provenance_valid,
        "completion_code_revision": completion_revision,
        "upstream_evidence": upstream_evidence,
        "completion_upstream_evidence": completion_upstream,
        "upstream_evidence_unchanged": upstream_unchanged,
        "required_source_manifest_keys": list(
            _expected_artifact_keys(experiments)
        ),
        "source_manifest_bindings": source_bindings,
        "dense_comparison_evidence": dense_comparison_evidence,
        "quality_evidence_eligible_for_decision": quality_eligible,
        "quality_ineligibility_reasons": quality_reasons
        + source_binding_reasons,
        "latency_evidence_eligible_for_decision": False,
        "latency_ineligibility_reason": "commodity x86 concurrency-4 run not recorded",
        "query_analyzer": spec.query_analyzer,
        "ingestion_model": spec.model_name,
        "selected_lexical_weight": selected_weight,
        "candidate_dev_metrics": [
            {"lexical_weight": weight, "metrics": candidates[weight].evaluation.metrics}
            for weight in experiments.lexical_weight_grid
        ],
        "sparse_only_test_metrics": artifacts["sparse/test"]["metrics"],
        "hybrid_test_metrics": selected.evaluation.metrics,
        "tuned_bm25_test_ndcg@10": tuned_bm25,
        "sparse_hybrid_minus_tuned_bm25_ndcg@10": (
            sparse_hybrid_ndcg - tuned_bm25
        ),
        "competent_integrator_bm25_test_ndcg@10": competent_bm25,
        "sparse_hybrid_minus_competent_integrator_bm25_ndcg@10": (
            sparse_hybrid_ndcg - competent_bm25
        ),
        "dense_minmax_test_ndcg@10": dense_minmax_ndcg,
        "dense_minmax_minus_sparse_hybrid_ndcg@10": (
            dense_minmax_ndcg - sparse_hybrid_ndcg
        ),
        "dense_rrf_test_ndcg@10": dense_rrf_ndcg,
        "dense_rrf_minus_sparse_hybrid_ndcg@10": (
            dense_rrf_ndcg - sparse_hybrid_ndcg
        ),
        "within_registered_sparse_alternative_band": (
            dense_minmax_ndcg - sparse_hybrid_ndcg <= 0.03
        ),
        "tokenizer_inspection": tokenizer_inspection,
        "artifacts": artifacts,
    }
    persisted_summary = _json_dict(summary)
    _write_json_exact(
        root / "results/wands/neural-sparse/summary.json",
        persisted_summary,
    )
    return cast(dict[str, object], persisted_summary)


def verify_wands_sparse_summary(
    client: OpenSearchClient,
    *,
    root: Path,
    spec: NeuralSparseSpec,
    experiments: BM25Experiments,
) -> dict[str, Any]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    summary = cast(
        dict[str, Any],
        read_json(root / "results/wands/neural-sparse/summary.json"),
    )
    if (
        summary.get("schema_version") != 2
        or summary.get("dataset") != "WANDS"
        or summary.get("mode")
        != "document_only_neural_sparse_builtin_query_analyzer"
        or summary.get("query_model_deployed") is not False
        or summary.get("query_analyzer") != spec.query_analyzer
        or summary.get("ingestion_model") != spec.model_name
    ):
        raise DatasetIntegrityError("neural-sparse summary metadata differs")
    provenance = summary.get("benchmark_provenance")
    if not isinstance(provenance, Mapping):
        raise DatasetIntegrityError("neural-sparse summary provenance is missing")
    provenance_valid = wands_recorded_provenance_valid(
        root,
        provenance=provenance,
        completion_revision=summary.get("completion_code_revision"),
    )
    if summary.get("benchmark_provenance_valid") is not provenance_valid:
        raise DatasetIntegrityError("neural-sparse summary provenance validity differs")
    start_upstream = summary.get("upstream_evidence")
    completion_upstream = summary.get("completion_upstream_evidence")
    if not isinstance(start_upstream, Mapping) or not isinstance(
        completion_upstream, Mapping
    ):
        raise DatasetIntegrityError("neural-sparse summary upstream evidence is missing")
    current_upstream = _sparse_upstream_evidence(
        client,
        root=root,
        spec=spec,
        experiments=experiments,
    )
    expected_upstream_unchanged = start_upstream == completion_upstream
    if (
        completion_upstream != current_upstream
        or summary.get("upstream_evidence_unchanged")
        is not expected_upstream_unchanged
    ):
        raise DatasetIntegrityError("neural-sparse summary upstream binding differs")

    artifacts_value = summary.get("artifacts")
    if not isinstance(artifacts_value, dict):
        raise DatasetIntegrityError("neural-sparse summary artifacts are missing")
    artifacts = cast(dict[str, dict[str, Any]], artifacts_value)
    expected_keys = set(_expected_artifact_keys(experiments))
    if set(artifacts) != expected_keys:
        raise DatasetIntegrityError("neural-sparse summary artifact set differs")
    manifests: dict[str, dict[str, Any]] = {}
    for artifact_name, entry in artifacts.items():
        if set(entry) != {"run", "manifest", "metrics_file", "metrics"}:
            raise DatasetIntegrityError(
                f"neural-sparse artifact entry fields differ: {artifact_name}"
            )
        expected_split = "dev" if "/dev" in artifact_name else "test"
        if artifact_name == "sparse/dev":
            expected_scope = "context_only"
        elif expected_split == "dev":
            expected_scope = "tuning"
        else:
            expected_scope = "held_out"
        manifests[artifact_name] = verify_sparse_artifact(
            client,
            root=root,
            index=spec.index_name,
            spec=spec,
            entry=entry,
            expected_split=expected_split,
            expected_scope=expected_scope,
            expected_provenance=provenance,
            expected_upstream=start_upstream,
            current_upstream=current_upstream,
        )

    candidate_metrics = {
        weight: cast(
            dict[str, float],
            artifacts[f"hybrid/dev/lw{round(weight * 100):03d}"]["metrics"],
        )
        for weight in experiments.lexical_weight_grid
    }
    expected_weight = select_hybrid_weight(
        experiments.lexical_weight_grid,
        candidate_metrics,
        metric=experiments.primary_metric,
    )
    bm25 = cast(dict[str, Any], read_json(root / _BM25_SELECTION))
    profile = BM25Profile.from_mapping(cast(dict[str, Any], bm25["selected_profile"]))
    for key, manifest in manifests.items():
        is_hybrid = key.startswith("hybrid/")
        if (manifest.get("lexical_profile") is not None) is not is_hybrid:
            raise DatasetIntegrityError(
                f"neural-sparse lexical profile attribution differs: {key}"
            )
        if is_hybrid and manifest.get("lexical_profile") != _json_dict(
            profile.to_dict()
        ):
            raise DatasetIntegrityError(
                f"neural-sparse lexical profile differs: {key}"
            )
    expected_candidate_metrics = [
        {"lexical_weight": weight, "metrics": candidate_metrics[weight]}
        for weight in experiments.lexical_weight_grid
    ]
    if (
        summary.get("selected_lexical_weight") != expected_weight
        or summary.get("candidate_dev_metrics") != expected_candidate_metrics
    ):
        raise DatasetIntegrityError("neural-sparse selected weight or dev metrics differ")
    selected_manifest = manifests["hybrid/test/selected"]
    selected_fusion = cast(dict[str, Any], selected_manifest.get("fusion"))
    if selected_fusion.get("lexical_weight") != expected_weight:
        raise DatasetIntegrityError("neural-sparse selected run weight differs")
    tuned = float(bm25["held_out_test_metrics"]["tuned"]["ndcg@10"])
    competent = float(
        bm25["held_out_test_metrics"]["competent_integrator"]["ndcg@10"]
    )
    sparse_metrics = cast(dict[str, float], artifacts["sparse/test"]["metrics"])
    hybrid_metrics = cast(
        dict[str, float], artifacts["hybrid/test/selected"]["metrics"]
    )
    sparse_hybrid = float(hybrid_metrics["ndcg@10"])
    dense_minmax, dense_rrf, dense_comparison_evidence = (
        _verified_dense_comparisons(
            client,
            root=root,
            spec=spec,
            experiments=experiments,
        )
    )
    dense_minmax_ndcg = float(dense_minmax["held_out_test_metrics"]["ndcg@10"])
    dense_rrf_ndcg = float(dense_rrf["held_out_test_metrics"]["ndcg@10"])
    expected_fields: dict[str, object] = {
        "sparse_only_test_metrics": sparse_metrics,
        "hybrid_test_metrics": hybrid_metrics,
        "tuned_bm25_test_ndcg@10": tuned,
        "sparse_hybrid_minus_tuned_bm25_ndcg@10": sparse_hybrid - tuned,
        "competent_integrator_bm25_test_ndcg@10": competent,
        "sparse_hybrid_minus_competent_integrator_bm25_ndcg@10": (
            sparse_hybrid - competent
        ),
        "dense_minmax_test_ndcg@10": dense_minmax_ndcg,
        "dense_minmax_minus_sparse_hybrid_ndcg@10": (
            dense_minmax_ndcg - sparse_hybrid
        ),
        "dense_rrf_test_ndcg@10": dense_rrf_ndcg,
        "dense_rrf_minus_sparse_hybrid_ndcg@10": (
            dense_rrf_ndcg - sparse_hybrid
        ),
        "within_registered_sparse_alternative_band": (
            dense_minmax_ndcg - sparse_hybrid <= 0.03
        ),
        "dense_comparison_evidence": dense_comparison_evidence,
        "tokenizer_inspection": read_json(root / _TOKENIZER_INSPECTION),
        "latency_evidence_eligible_for_decision": False,
        "latency_ineligibility_reason": (
            "commodity x86 concurrency-4 run not recorded"
        ),
    }
    if any(summary.get(key) != value for key, value in expected_fields.items()):
        raise DatasetIntegrityError("neural-sparse summary metrics or metadata differ")

    bindings, bindings_valid, binding_reasons = _source_manifest_bindings(
        root=root,
        artifacts=artifacts,
        experiments=experiments,
        benchmark_provenance=provenance,
        upstream_evidence=start_upstream,
    )
    required_keys = list(_expected_artifact_keys(experiments))
    if (
        summary.get("required_source_manifest_keys") != required_keys
        or summary.get("source_manifest_bindings") != bindings
    ):
        raise DatasetIntegrityError("neural-sparse source manifest bindings differ")
    held_out_eligible = all(
        manifests[key].get("eligible_for_decision") is True
        for key in ("sparse/test", "hybrid/test/selected")
    )
    eligible, reasons = _summary_eligibility(
        provenance_valid=provenance_valid,
        upstream_eligible=(
            start_upstream.get("eligible_for_decision") is True
            and current_upstream.get("eligible_for_decision") is True
        ),
        upstream_unchanged=expected_upstream_unchanged,
        source_bindings_valid=bindings_valid,
        held_out_sources_eligible=held_out_eligible,
        comparison_sources_eligible=(
            dense_comparison_evidence.get("eligible_for_decision") is True
        ),
    )
    if (
        summary.get("quality_evidence_eligible_for_decision") is not eligible
        or summary.get("quality_ineligibility_reasons")
        != reasons + binding_reasons
    ):
        raise DatasetIntegrityError("neural-sparse summary quality eligibility differs")
    _verify_sparse_live_reruns(
        client,
        root=root,
        spec=spec,
        profile=profile,
        artifacts=artifacts,
        manifests=manifests,
    )
    completion_index_evidence = _sparse_index_evidence(
        client,
        root=root,
        spec=spec,
    )
    if completion_index_evidence != start_upstream.get(
        "neural_sparse_index"
    ):
        raise DatasetIntegrityError(
            "neural-sparse index changed during decision replay"
        )
    return summary


def _verify_sparse_live_reruns(
    client: OpenSearchClient,
    *,
    root: Path,
    spec: NeuralSparseSpec,
    profile: BM25Profile,
    artifacts: Mapping[str, Mapping[str, Any]],
    manifests: Mapping[str, Mapping[str, Any]],
) -> None:
    prepared = root / "data/prepared/wands"
    queries = {
        split: load_prepared_queries(
            prepared / f"queries.{split}.jsonl",
            expected_split=split,
        )
        for split in ("dev", "test")
    }
    with tempfile.TemporaryDirectory(
        prefix="opensearch-hybrid-sparse-verify-"
    ) as temporary:
        temporary_root = Path(temporary)
        for key, entry in artifacts.items():
            manifest = manifests[key]
            split = "dev" if "/dev" in key else "test"
            tag = str(manifest["tag"])
            if key.startswith("sparse/"):
                records = execute_sparse_run(
                    client,
                    index=spec.index_name,
                    queries=queries[split],
                    spec=spec,
                    tag=tag,
                )
            else:
                fusion = manifest.get("fusion")
                if not isinstance(fusion, Mapping):
                    raise DatasetIntegrityError(
                        f"neural-sparse fusion metadata is missing: {key}"
                    )
                lexical_weight = fusion.get("lexical_weight")
                if not isinstance(lexical_weight, int | float) or isinstance(
                    lexical_weight, bool
                ):
                    raise DatasetIntegrityError(
                        f"neural-sparse lexical weight is invalid: {key}"
                    )
                records = execute_sparse_hybrid_run(
                    client,
                    index=spec.index_name,
                    queries=queries[split],
                    spec=spec,
                    profile=profile,
                    lexical_weight=float(lexical_weight),
                    tag=tag,
                )
            repeated_path = temporary_root / f"{key.replace('/', '-')}.trec"
            write_run(repeated_path, records)
            if repeated_path.read_bytes() != (
                root / str(entry["run"])
            ).read_bytes():
                raise DatasetIntegrityError(
                    f"neural-sparse {key} live rerun is not byte-identical"
                )


def verify_sparse_artifact(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    spec: NeuralSparseSpec,
    entry: dict[str, Any],
    expected_split: str,
    expected_scope: str,
    expected_provenance: Mapping[str, Any],
    expected_upstream: Mapping[str, Any],
    current_upstream: Mapping[str, Any],
) -> dict[str, Any]:
    run_path = root / str(entry["run"])
    manifest_path = root / str(entry["manifest"])
    metrics_path = root / str(entry["metrics_file"])
    manifest = cast(dict[str, Any], read_json(manifest_path))
    tag = manifest.get("tag")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("dataset") != "WANDS"
        or not isinstance(tag, str)
        or not tag
        or run_path != root / f"runs/{tag}.trec"
        or manifest_path != root / f"runs/{tag}.manifest.json"
        or metrics_path
        != root / f"results/wands/neural-sparse/{tag}.metrics.json"
        or manifest.get("split") != expected_split
        or manifest.get("decision_scope") != expected_scope
    ):
        raise DatasetIntegrityError(
            f"neural-sparse run metadata or paths differ: {run_path}"
        )
    provenance = manifest.get("benchmark_provenance")
    provenance_valid = wands_recorded_provenance_valid(
        root,
        provenance=provenance,
        completion_revision=manifest.get("completion_code_revision"),
    )
    completion_upstream = manifest.get("completion_upstream_evidence")
    expected_unchanged = (
        manifest.get("upstream_evidence") == expected_upstream
        and completion_upstream == current_upstream
        and expected_upstream == current_upstream
    )
    eligible, ineligibility_reason = _run_eligibility(
        decision_scope=expected_scope,
        provenance_valid=provenance_valid,
        upstream_eligible=(
            expected_upstream.get("eligible_for_decision") is True
            and current_upstream.get("eligible_for_decision") is True
        ),
        upstream_unchanged=expected_unchanged,
    )
    if (
        provenance != expected_provenance
        or manifest.get("benchmark_provenance_valid") is not provenance_valid
        or manifest.get("upstream_evidence") != expected_upstream
        or completion_upstream != current_upstream
        or manifest.get("upstream_evidence_unchanged") is not expected_unchanged
        or manifest.get("eligible_for_decision") is not eligible
        or manifest.get("ineligibility_reason") != ineligibility_reason
    ):
        raise DatasetIntegrityError(
            f"neural-sparse run provenance or eligibility differs: {run_path}"
        )
    index_manifest_path = root / "results/wands/neural-sparse/index-manifest.json"
    index_manifest = cast(dict[str, Any], read_json(index_manifest_path))
    if asdict(collect_index_facts(client, index)) != manifest.get("index"):
        raise DatasetIntegrityError(f"neural-sparse run references another index: {run_path}")
    if (
        manifest.get("index_manifest_sha256")
        != file_facts(index_manifest_path).sha256
        or manifest.get("index_manifest") != _artifact(root, index_manifest_path)
    ):
        raise DatasetIntegrityError(f"neural-sparse index manifest differs: {run_path}")
    if manifest.get("index_definition_sha256") != index_manifest.get(
        "index_definition_sha256"
    ):
        raise DatasetIntegrityError(f"neural-sparse index definition differs: {run_path}")
    if manifest.get("query_mode") != "doc_only_builtin_analyzer":
        raise DatasetIntegrityError(f"neural-sparse query mode differs: {run_path}")
    if manifest.get("query_model_id") is not None:
        raise DatasetIntegrityError(f"neural-sparse query model must be absent: {run_path}")
    if manifest.get("ingestion_model_id") != index_manifest.get("model_id"):
        raise DatasetIntegrityError(f"neural-sparse ingestion model differs: {run_path}")
    if manifest.get("query_analyzer") != spec.query_analyzer:
        raise DatasetIntegrityError(f"neural-sparse query analyzer differs: {run_path}")
    run_file = cast(dict[str, Any], manifest["run_file"])
    run_records = read_run(run_path)
    if (
        run_file.get("path") != str(run_path.relative_to(root))
        or file_facts(run_path).sha256 != run_file.get("sha256")
        or run_file.get("records") != len(run_records)
        or any(record.tag != tag for record in run_records)
    ):
        raise DatasetIntegrityError(f"neural-sparse run differs from manifest: {run_path}")
    for key in ("query_file", "qrels_file"):
        facts = cast(dict[str, Any], manifest[key])
        if file_facts(root / str(facts["path"])).sha256 != facts.get("sha256"):
            raise DatasetIntegrityError(f"neural-sparse {key} differs: {run_path}")
    fusion = cast(dict[str, Any] | None, manifest.get("fusion"))
    if fusion is None:
        if (
            manifest.get("pipeline_id") is not None
            or manifest.get("pipeline_sha256") is not None
            or manifest.get("declared_variable") != "retrieval_arm"
            or manifest.get("lexical_profile") is not None
        ):
            raise DatasetIntegrityError(f"sparse-only pipeline must be absent: {run_path}")
    else:
        lexical_weight = float(fusion["lexical_weight"])
        expected_pipeline = build_normalization_pipeline(
            lexical_weight=lexical_weight
        )
        expected_pipeline_id = sparse_pipeline_id(lexical_weight)
        if manifest.get("pipeline_id") != expected_pipeline_id:
            raise DatasetIntegrityError(f"neural-sparse pipeline ID differs: {run_path}")
        expected_fusion = {
            "normalization": "min_max",
            "combination": "arithmetic_mean",
            "lexical_weight": lexical_weight,
            "sparse_weight": 1.0 - lexical_weight,
        }
        if (
            fusion != expected_fusion
            or manifest.get("declared_variable") != "lexical_weight"
            or manifest.get("pipeline_sha256")
            != canonical_sha256(expected_pipeline)
        ):
            raise DatasetIntegrityError(f"neural-sparse pipeline hash differs: {run_path}")
        try:
            verify_live_search_pipeline(
                client,
                pipeline_id=expected_pipeline_id,
                expected_definition=expected_pipeline,
            )
        except DatasetIntegrityError as error:
            raise DatasetIntegrityError(
                f"live neural-sparse pipeline differs: {run_path}: {error}"
            ) from error
    qrels_path = root / str(cast(dict[str, Any], manifest["qrels_file"])["path"])
    evaluation = evaluate_run(qrels_path, run_path)
    metrics = cast(dict[str, Any], read_json(metrics_path))
    if (
        metrics.get("schema_version") != 1
        or metrics.get("tag") != tag
        or metrics.get("split") != expected_split
        or metrics.get("metrics") != evaluation.metrics
        or metrics.get("per_query") != evaluation.per_query
    ):
        raise DatasetIntegrityError(f"neural-sparse metrics do not reproduce: {run_path}")
    if entry.get("metrics") != evaluation.metrics:
        raise DatasetIntegrityError(f"neural-sparse summary metrics differ: {run_path}")
    if (
        metrics.get("evaluator_crosscheck")
        != cross_check_run(qrels_path, run_path, root=root).to_dict()
    ):
        raise DatasetIntegrityError(f"neural-sparse evaluator cross-check differs: {run_path}")
    return manifest


def _response_records(
    response: dict[str, Any],
    *,
    query_id: str,
    tag: str,
) -> list[RunRecord]:
    hits = cast(list[dict[str, Any]], cast(dict[str, Any], response["hits"])["hits"])
    if not hits:
        raise ValueError(f"neural-sparse query {query_id} returned no hits")
    return [
        RunRecord(
            query_id=query_id,
            document_id=str(hit["_id"]),
            rank=rank,
            score=hybrid_trec_score_for_rank(rank),
            tag=tag,
        )
        for rank, hit in enumerate(hits, start=1)
    ]
