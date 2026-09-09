from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from poc.datasets import DatasetIntegrityError, file_facts, load_wands_config
from poc.evaluation import Evaluation, evaluate_run
from poc.experiments import BM25Experiments, BM25Profile
from poc.manifest import (
    IndexFacts,
    canonical_sha256,
    collect_index_facts,
    read_json,
    write_json,
)
from poc.os_client import OpenSearchClient
from poc.provenance import (
    collect_code_revision,
    collect_manifest_provenance,
    require_registered_opensearch_client,
    verify_decision_provenance,
)
from poc.trec import RunRecord, identifier_sort_key, read_run, write_run

RESULT_SIZE = 100
_PROFILE_RELATIVE_PATH = Path("config/benchmark.toml")
_ENVIRONMENT_RELATIVE_PATH = Path(
    "results/environment/benchmark-profile.json"
)


@dataclass(frozen=True, slots=True)
class BM25RunResult:
    tag: str
    run_path: Path
    manifest_path: Path
    metrics_path: Path
    profile: BM25Profile
    evaluation: Evaluation
    records: list[RunRecord]

    def selection_entry(self, root: Path) -> dict[str, object]:
        return {
            "run": str(self.run_path.relative_to(root)),
            "manifest": str(self.manifest_path.relative_to(root)),
            "metrics_file": str(self.metrics_path.relative_to(root)),
            "metrics": self.evaluation.metrics,
            "profile": self.profile.to_dict(),
        }


def load_prepared_queries(path: Path, *, expected_split: str) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = cast(dict[str, Any], json.loads(line))
            query_id = str(record.get("query_id", ""))
            query = str(record.get("query", ""))
            split = str(record.get("split", ""))
            if not query_id or not query or split != expected_split or query_id in queries:
                raise DatasetIntegrityError(f"invalid query in {path}:{line_number}")
            queries[query_id] = query
    if not queries:
        raise DatasetIntegrityError(f"no {expected_split} queries in {path}")
    return queries


def _valid_decision_provenance(
    root: Path,
    provenance: Mapping[str, Any],
    *,
    require_current_code_revision: bool = False,
) -> bool:
    return verify_decision_provenance(
        provenance,
        root=root,
        profile_path=root / _PROFILE_RELATIVE_PATH,
        environment_path=root / _ENVIRONMENT_RELATIVE_PATH,
        require_current_code_revision=require_current_code_revision,
    )


def _json_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(json.dumps(value)))


def _write_json_exact(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _completion_provenance_evidence(
    root: Path,
    provenance: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    completion_revision = _json_dict(asdict(collect_code_revision(root)))
    start_revision = provenance.get("code_revision")
    valid = (
        isinstance(start_revision, Mapping)
        and completion_revision == dict(start_revision)
        and _valid_decision_provenance(
            root,
            provenance,
            require_current_code_revision=True,
        )
    )
    return valid, completion_revision


def _recorded_completion_provenance_valid(
    root: Path,
    provenance: object,
    completion_revision: object,
) -> bool:
    if not isinstance(provenance, Mapping) or not isinstance(
        completion_revision, Mapping
    ):
        return False
    return (
        _valid_decision_provenance(root, provenance)
        and completion_revision == provenance.get("code_revision")
    )


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }


def _bm25_index_evidence(
    *,
    root: Path,
    index_manifest_path: Path,
    live_index: IndexFacts,
) -> dict[str, object]:
    manifest_value = read_json(index_manifest_path)
    if not isinstance(manifest_value, dict):
        raise DatasetIntegrityError("WANDS index manifest must be a JSON object")
    manifest = cast(dict[str, Any], manifest_value)
    definition_hash = manifest.get("index_definition_sha256")
    manifest_valid = (
        manifest.get("schema_version") == 2
        and manifest.get("dataset") == "WANDS"
        and isinstance(definition_hash, str)
        and len(definition_hash) == 64
        and all(character in "0123456789abcdef" for character in definition_hash)
    )
    provenance = manifest.get("benchmark_provenance")
    provenance_valid = isinstance(provenance, Mapping) and _valid_decision_provenance(
        root, provenance
    )
    live_index_bound = manifest.get("index") == asdict(live_index)
    quality_declared = (
        manifest.get("quality_evidence_eligible_for_decision") is True
    )
    eligible = (
        manifest_valid
        and provenance_valid
        and live_index_bound
        and quality_declared
    )
    reasons: list[str] = []
    if not manifest_valid:
        reasons.append("index manifest schema or identity is invalid")
    if not provenance_valid:
        reasons.append("index manifest has no valid clean committed provenance")
    if not live_index_bound:
        reasons.append("index manifest does not bind the live index")
    if not quality_declared:
        reasons.append("index manifest is not quality eligible")
    return {
        "artifact": _artifact(root, index_manifest_path),
        "manifest_valid": manifest_valid,
        "quality_declared": quality_declared,
        "provenance_valid": provenance_valid,
        "live_index_bound": live_index_bound,
        "eligible_for_decision": eligible,
        "ineligibility_reasons": reasons,
    }


def _verify_registered_wands_lexical_index(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
) -> dict[str, object]:
    from poc.indexing import load_wands_index_config, verify_wands_lexical_index

    dataset_spec = load_wands_config(root / "config/datasets.toml")
    index_spec = load_wands_index_config(root / "config/indexes.toml")
    if index != index_spec.name:
        raise DatasetIntegrityError("BM25 verifier index differs from registered index")
    return verify_wands_lexical_index(
        client,
        dataset_spec=dataset_spec,
        index_spec=index_spec,
        prepared_directory=root / "data/prepared/wands",
        manifest_path=root / "results/wands/index-manifest.json",
    )


def _bm25_run_eligibility(
    *,
    decision_scope: str,
    benchmark_provenance_valid: bool,
    index_evidence_eligible: bool,
) -> tuple[bool, str | None]:
    if decision_scope != "held_out":
        return False, "development or context-only result"
    reasons: list[str] = []
    if not benchmark_provenance_valid:
        reasons.append("run has no valid clean committed start provenance")
    if not index_evidence_eligible:
        reasons.append("run index evidence is not decision eligible")
    return not reasons, "; ".join(reasons) or None


def execute_bm25_profile(
    client: OpenSearchClient,
    *,
    index: str,
    queries: dict[str, str],
    profile: BM25Profile,
    tag: str,
) -> list[RunRecord]:
    records: list[RunRecord] = []
    for query_id in sorted(queries, key=identifier_sort_key):
        response = client.search(
            index,
            {
                "size": RESULT_SIZE,
                "_source": False,
                "track_total_hits": False,
                "query": profile.query(queries[query_id]),
            },
        )
        hits = cast(list[dict[str, Any]], cast(dict[str, Any], response["hits"])["hits"])
        for rank, hit in enumerate(hits, start=1):
            score = hit.get("_score")
            if score is None:
                raise DatasetIntegrityError(f"BM25 result for {query_id} has no score")
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


def write_bm25_bundle(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    index_manifest_path: Path,
    query_path: Path,
    qrels_path: Path,
    profile: BM25Profile,
    tag: str,
    split: str,
    decision_scope: str,
    records: list[RunRecord],
    benchmark_provenance: dict[str, Any],
) -> BM25RunResult:
    run_path = root / f"runs/{tag}.trec"
    manifest_path = root / f"runs/{tag}.manifest.json"
    metrics_path = root / f"results/wands/bm25/{tag}.metrics.json"
    write_run(run_path, records)
    evaluation = evaluate_run(qrels_path, run_path)
    write_json(
        metrics_path,
        {
            "schema_version": 1,
            "tag": tag,
            "split": split,
            "metrics": evaluation.metrics,
            "per_query": evaluation.per_query,
        },
    )

    root_response = cast(dict[str, Any], client.request("GET", "/"))
    version = str(cast(dict[str, Any], root_response["version"])["number"])
    index_facts = collect_index_facts(client, index)
    index_manifest = cast(dict[str, Any], read_json(index_manifest_path))
    index_evidence = _bm25_index_evidence(
        root=root,
        index_manifest_path=index_manifest_path,
        live_index=index_facts,
    )
    benchmark_provenance_valid, completion_revision = (
        _completion_provenance_evidence(root, benchmark_provenance)
    )
    eligible, ineligibility_reason = _bm25_run_eligibility(
        decision_scope=decision_scope,
        benchmark_provenance_valid=benchmark_provenance_valid,
        index_evidence_eligible=index_evidence["eligible_for_decision"] is True,
    )
    query_facts = file_facts(query_path)
    qrels_facts = file_facts(qrels_path)
    run_facts = file_facts(run_path)
    manifest: dict[str, object] = {
        "schema_version": 2,
        "dataset": "WANDS",
        "split": split,
        "tag": tag,
        "decision_scope": decision_scope,
        "eligible_for_decision": eligible,
        "ineligibility_reason": ineligibility_reason,
        "benchmark_provenance": benchmark_provenance,
        "benchmark_provenance_valid": benchmark_provenance_valid,
        "completion_code_revision": completion_revision,
        "declared_variable": "bm25_profile",
        "opensearch_version": version,
        "index": asdict(index_facts),
        "index_manifest_sha256": file_facts(index_manifest_path).sha256,
        "index_manifest": index_evidence["artifact"],
        "index_evidence": index_evidence,
        "index_definition_sha256": index_manifest["index_definition_sha256"],
        "hnsw": None,
        "pagination_depth": None,
        "pipeline_sha256": None,
        "model_sha256": None,
        "encoder_runtime": "not_applicable_lexical",
        "profile": profile.to_dict(),
        "profile_sha256": canonical_sha256(profile.to_dict()),
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
    }
    _write_json_exact(manifest_path, manifest)
    return BM25RunResult(
        tag=tag,
        run_path=run_path,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        profile=profile,
        evaluation=evaluation,
        records=records,
    )


def run_bm25_bundle(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    index_manifest_path: Path,
    query_path: Path,
    qrels_path: Path,
    queries: dict[str, str],
    profile: BM25Profile,
    tag: str,
    split: str,
    decision_scope: str,
    benchmark_provenance: dict[str, Any],
) -> BM25RunResult:
    records = execute_bm25_profile(
        client,
        index=index,
        queries=queries,
        profile=profile,
        tag=tag,
    )
    return write_bm25_bundle(
        client,
        root=root,
        index=index,
        index_manifest_path=index_manifest_path,
        query_path=query_path,
        qrels_path=qrels_path,
        profile=profile,
        tag=tag,
        split=split,
        decision_scope=decision_scope,
        records=records,
        benchmark_provenance=benchmark_provenance,
    )


def retag(records: list[RunRecord], tag: str) -> list[RunRecord]:
    return [
        RunRecord(
            query_id=record.query_id,
            document_id=record.document_id,
            rank=record.rank,
            score=record.score,
            tag=tag,
        )
        for record in records
    ]


def _required_selection_source_keys(
    experiments: BM25Experiments,
) -> tuple[str, ...]:
    return tuple(
        [
            f"tuning/{profile.name}/dev"
            for profile in experiments.tuning_candidates
        ]
        + [
            "magento/dev",
            "tuned/dev",
            "naive/test",
            "magento/test",
            "tuned/test",
            "competent_integrator/test",
        ]
    )


def _bm25_selection_source_bindings(
    *,
    root: Path,
    artifacts: Mapping[str, Mapping[str, Any]],
    experiments: BM25Experiments,
    benchmark_provenance: Mapping[str, Any],
    index_evidence: Mapping[str, Any],
) -> tuple[dict[str, dict[str, object]], bool, list[str]]:
    required = _required_selection_source_keys(experiments)
    bindings: dict[str, dict[str, object]] = {}
    reasons: list[str] = []
    for key in required:
        entry = artifacts.get(key)
        if entry is None:
            raise DatasetIntegrityError(f"BM25 selection source is missing: {key}")
        manifest_path = root / str(entry["manifest"])
        manifest = cast(dict[str, Any], read_json(manifest_path))
        bindings[key] = _artifact(root, manifest_path)
        expected_split = key.rsplit("/", 1)[-1]
        expected_scope = "tuning" if expected_split == "dev" else "held_out"
        provenance = manifest.get("benchmark_provenance")
        provenance_valid = _recorded_completion_provenance_valid(
            root,
            provenance,
            manifest.get("completion_code_revision"),
        )
        eligible, ineligibility_reason = _bm25_run_eligibility(
            decision_scope=expected_scope,
            benchmark_provenance_valid=provenance_valid,
            index_evidence_eligible=index_evidence.get("eligible_for_decision") is True,
        )
        valid = (
            manifest.get("schema_version") == 2
            and manifest.get("decision_scope") == expected_scope
            and manifest.get("split") == expected_split
            and provenance_valid
            and canonical_sha256(provenance)
            == canonical_sha256(benchmark_provenance)
            and manifest.get("benchmark_provenance_valid") is provenance_valid
            and manifest.get("index_evidence") == index_evidence
            and manifest.get("index_manifest") == index_evidence.get("artifact")
            and manifest.get("eligible_for_decision") is eligible
            and manifest.get("ineligibility_reason") == ineligibility_reason
        )
        if not valid:
            reasons.append(f"selection source binding differs: {key}")
    return bindings, not reasons, reasons


def _bm25_selection_eligibility(
    *,
    benchmark_provenance_valid: bool,
    index_evidence_eligible: bool,
    source_bindings_valid: bool,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not benchmark_provenance_valid:
        reasons.append("selection has no valid clean committed start provenance")
    if not index_evidence_eligible:
        reasons.append("selection index evidence is not decision eligible")
    if not source_bindings_valid:
        reasons.append("selection source manifest bindings differ")
    return not reasons, reasons


def run_wands_bm25_benchmark(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    experiments: BM25Experiments,
) -> dict[str, object]:
    require_registered_opensearch_client(client, root / _PROFILE_RELATIVE_PATH)
    benchmark_provenance = _json_dict(collect_manifest_provenance(root))
    if len(experiments.tuning_candidates) != experiments.hybrid_weight_count:
        raise DatasetIntegrityError("BM25 and hybrid tuning budgets differ")
    prepared = root / "data/prepared/wands"
    index_manifest = root / "results/wands/index-manifest.json"
    query_paths = {split: prepared / f"queries.{split}.jsonl" for split in ("dev", "test")}
    qrels_paths = {split: prepared / f"qrels.{split}.trec" for split in ("dev", "test")}
    queries = {
        split: load_prepared_queries(query_paths[split], expected_split=split)
        for split in ("dev", "test")
    }

    artifacts: dict[str, dict[str, object]] = {}
    candidate_results: dict[str, BM25RunResult] = {}
    for profile in experiments.tuning_candidates:
        tag = f"wands-bm25-tune-{profile.name.replace('_', '-')}-dev"
        result = run_bm25_bundle(
            client,
            root=root,
            index=index,
            index_manifest_path=index_manifest,
            query_path=query_paths["dev"],
            qrels_path=qrels_paths["dev"],
            queries=queries["dev"],
            profile=profile,
            tag=tag,
            split="dev",
            decision_scope="tuning",
            benchmark_provenance=benchmark_provenance,
        )
        candidate_results[profile.name] = result
        artifacts[f"tuning/{profile.name}/dev"] = result.selection_entry(root)

    selected = max(
        experiments.tuning_candidates,
        key=lambda profile: candidate_results[profile.name].evaluation.metrics["ndcg@10"],
    )

    naive_dev = run_bm25_bundle(
        client,
        root=root,
        index=index,
        index_manifest_path=index_manifest,
        query_path=query_paths["dev"],
        qrels_path=qrels_paths["dev"],
        queries=queries["dev"],
        profile=experiments.naive,
        tag="wands-bm25-naive-dev",
        split="dev",
        decision_scope="context_only",
        benchmark_provenance=benchmark_provenance,
    )
    artifacts["naive/dev"] = naive_dev.selection_entry(root)

    competent_dev = run_bm25_bundle(
        client,
        root=root,
        index=index,
        index_manifest_path=index_manifest,
        query_path=query_paths["dev"],
        qrels_path=qrels_paths["dev"],
        queries=queries["dev"],
        profile=experiments.competent_integrator,
        tag="wands-bm25-competent-integrator-dev",
        split="dev",
        decision_scope="post_review_challenge_arm",
        benchmark_provenance=benchmark_provenance,
    )
    artifacts["competent_integrator/dev"] = competent_dev.selection_entry(root)

    for arm, profile in (
        ("magento", experiments.magento_emulation),
        ("tuned", selected),
    ):
        source = candidate_results[profile.name]
        tag = f"wands-bm25-{arm}-dev"
        result = write_bm25_bundle(
            client,
            root=root,
            index=index,
            index_manifest_path=index_manifest,
            query_path=query_paths["dev"],
            qrels_path=qrels_paths["dev"],
            profile=profile,
            tag=tag,
            split="dev",
            decision_scope="tuning",
            records=retag(source.records, tag),
            benchmark_provenance=benchmark_provenance,
        )
        artifacts[f"{arm}/dev"] = result.selection_entry(root)

    test_results: dict[str, BM25RunResult] = {}
    for arm, profile in (
        ("naive", experiments.naive),
        ("magento", experiments.magento_emulation),
        ("competent_integrator", experiments.competent_integrator),
    ):
        tag = f"wands-bm25-{arm}-test"
        test_results[arm] = run_bm25_bundle(
            client,
            root=root,
            index=index,
            index_manifest_path=index_manifest,
            query_path=query_paths["test"],
            qrels_path=qrels_paths["test"],
            queries=queries["test"],
            profile=profile,
            tag=tag,
            split="test",
            decision_scope="held_out",
            benchmark_provenance=benchmark_provenance,
        )
        artifacts[f"{arm}/test"] = test_results[arm].selection_entry(root)

    tuned_tag = "wands-bm25-tuned-test"
    if selected == experiments.magento_emulation:
        tuned_test = write_bm25_bundle(
            client,
            root=root,
            index=index,
            index_manifest_path=index_manifest,
            query_path=query_paths["test"],
            qrels_path=qrels_paths["test"],
            profile=selected,
            tag=tuned_tag,
            split="test",
            decision_scope="held_out",
            records=retag(test_results["magento"].records, tuned_tag),
            benchmark_provenance=benchmark_provenance,
        )
    else:
        tuned_test = run_bm25_bundle(
            client,
            root=root,
            index=index,
            index_manifest_path=index_manifest,
            query_path=query_paths["test"],
            qrels_path=qrels_paths["test"],
            queries=queries["test"],
            profile=selected,
            tag=tuned_tag,
            split="test",
            decision_scope="held_out",
            benchmark_provenance=benchmark_provenance,
        )
    artifacts["tuned/test"] = tuned_test.selection_entry(root)

    index_facts = collect_index_facts(client, index)
    index_evidence = _bm25_index_evidence(
        root=root,
        index_manifest_path=index_manifest,
        live_index=index_facts,
    )
    benchmark_provenance_valid, completion_revision = (
        _completion_provenance_evidence(root, benchmark_provenance)
    )
    source_bindings, source_bindings_valid, source_binding_reasons = (
        _bm25_selection_source_bindings(
            root=root,
            artifacts=artifacts,
            experiments=experiments,
            benchmark_provenance=benchmark_provenance,
            index_evidence=index_evidence,
        )
    )
    quality_eligible, quality_reasons = _bm25_selection_eligibility(
        benchmark_provenance_valid=benchmark_provenance_valid,
        index_evidence_eligible=index_evidence["eligible_for_decision"] is True,
        source_bindings_valid=source_bindings_valid,
    )
    selection: dict[str, object] = {
        "schema_version": 2,
        "dataset": "WANDS",
        "benchmark_provenance": benchmark_provenance,
        "benchmark_provenance_valid": benchmark_provenance_valid,
        "completion_code_revision": completion_revision,
        "quality_evidence_eligible_for_decision": quality_eligible,
        "quality_ineligibility_reasons": quality_reasons + source_binding_reasons,
        "index_manifest": index_evidence["artifact"],
        "index_evidence": index_evidence,
        "required_source_manifest_keys": list(
            _required_selection_source_keys(experiments)
        ),
        "source_manifest_bindings": source_bindings,
        "selection_metric": "ndcg@10",
        "selection_split": "dev",
        "tie_break": "first_candidate_in_registered_order",
        "bm25_tuning_budget": len(experiments.tuning_candidates),
        "hybrid_weight_tuning_budget": experiments.hybrid_weight_count,
        "selected_profile": selected.to_dict(),
        "selected_profile_sha256": canonical_sha256(selected.to_dict()),
        "index_uuid": index_facts.uuid,
        "candidate_dev_metrics": {
            profile.name: candidate_results[profile.name].evaluation.metrics
            for profile in experiments.tuning_candidates
        },
        "held_out_test_metrics": {
            arm: artifacts[f"{arm}/test"]["metrics"]
            for arm in ("naive", "magento", "tuned", "competent_integrator")
        },
        "post_review_challenge_arm": {
            "profile": experiments.competent_integrator.to_dict(),
            "selection_surface": "fixed before rerunning the held-out test split",
            "development_metrics": competent_dev.evaluation.metrics,
            "limitations": [
                "WANDS contains no merchant SKU field",
                "no merchant synonym set exists for this public dataset",
            ],
        },
        "artifacts": artifacts,
    }
    persisted_selection = _json_dict(selection)
    write_json(root / "results/wands/bm25-selection.json", persisted_selection)
    return cast(dict[str, object], persisted_selection)


def verify_bm25_artifact(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    entry: dict[str, Any],
) -> dict[str, Any]:
    require_registered_opensearch_client(client, root / _PROFILE_RELATIVE_PATH)
    run_path = root / str(entry["run"])
    manifest_path = root / str(entry["manifest"])
    metrics_path = root / str(entry["metrics_file"])
    manifest = cast(dict[str, Any], read_json(manifest_path))
    if manifest.get("schema_version") != 2:
        raise DatasetIntegrityError(f"BM25 run manifest schema differs: {run_path}")
    if manifest.get("dataset") != "WANDS":
        raise DatasetIntegrityError(f"BM25 run dataset differs: {run_path}")
    tag = manifest.get("tag")
    if (
        not isinstance(tag, str)
        or not tag
        or run_path != root / f"runs/{tag}.trec"
        or manifest_path != root / f"runs/{tag}.manifest.json"
        or metrics_path != root / f"results/wands/bm25/{tag}.metrics.json"
    ):
        raise DatasetIntegrityError(f"BM25 artifact paths or tag differ: {run_path}")
    expected_method_metadata = {
        "declared_variable": "bm25_profile",
        "hnsw": None,
        "pagination_depth": None,
        "pipeline_sha256": None,
        "model_sha256": None,
        "encoder_runtime": "not_applicable_lexical",
    }
    if any(
        manifest.get(key) != value
        for key, value in expected_method_metadata.items()
    ):
        raise DatasetIntegrityError(f"BM25 method metadata differs: {run_path}")
    root_response = cast(dict[str, Any], client.request("GET", "/"))
    live_version = str(cast(dict[str, Any], root_response["version"])["number"])
    if manifest.get("opensearch_version") != live_version:
        raise DatasetIntegrityError(f"BM25 OpenSearch version differs: {run_path}")
    if manifest.get("profile") != entry.get("profile") or canonical_sha256(
        manifest.get("profile")
    ) != manifest.get("profile_sha256"):
        raise DatasetIntegrityError(f"BM25 profile differs from manifest: {run_path}")
    run_file = cast(dict[str, Any], manifest["run_file"])
    run_records = read_run(run_path)
    if (
        run_file.get("path") != str(run_path.relative_to(root))
        or file_facts(run_path).sha256 != run_file.get("sha256")
        or run_file.get("records") != len(run_records)
        or any(record.tag != tag for record in run_records)
    ):
        raise DatasetIntegrityError(f"run file differs from manifest: {run_path}")
    live_index = collect_index_facts(client, index)
    if asdict(live_index) != manifest.get("index"):
        raise DatasetIntegrityError(f"live index differs from run manifest: {run_path}")
    index_manifest_path = root / "results/wands/index-manifest.json"
    index_evidence = _bm25_index_evidence(
        root=root,
        index_manifest_path=index_manifest_path,
        live_index=live_index,
    )
    if (
        manifest.get("index_manifest") != index_evidence["artifact"]
        or manifest.get("index_manifest_sha256")
        != cast(dict[str, Any], index_evidence["artifact"])["sha256"]
        or manifest.get("index_evidence") != index_evidence
    ):
        raise DatasetIntegrityError(f"BM25 index evidence differs: {run_path}")
    index_manifest = cast(dict[str, Any], read_json(index_manifest_path))
    if manifest.get("index_definition_sha256") != index_manifest.get(
        "index_definition_sha256"
    ):
        raise DatasetIntegrityError(f"BM25 index definition differs: {run_path}")
    provenance = manifest.get("benchmark_provenance")
    provenance_valid = _recorded_completion_provenance_valid(
        root,
        provenance,
        manifest.get("completion_code_revision"),
    )
    if manifest.get("benchmark_provenance_valid") is not provenance_valid:
        raise DatasetIntegrityError(f"BM25 run provenance validity differs: {run_path}")
    decision_scope = str(manifest.get("decision_scope", ""))
    eligible, ineligibility_reason = _bm25_run_eligibility(
        decision_scope=decision_scope,
        benchmark_provenance_valid=provenance_valid,
        index_evidence_eligible=index_evidence["eligible_for_decision"] is True,
    )
    if (
        manifest.get("eligible_for_decision") is not eligible
        or manifest.get("ineligibility_reason") != ineligibility_reason
    ):
        raise DatasetIntegrityError(f"BM25 run eligibility differs: {run_path}")
    query_file = cast(dict[str, Any], manifest["query_file"])
    qrels_file = cast(dict[str, Any], manifest["qrels_file"])
    query_path = root / str(query_file["path"])
    qrels_path = root / str(qrels_file["path"])
    if (
        query_file.get("path") != str(query_path.relative_to(root))
        or file_facts(query_path).sha256 != query_file.get("sha256")
    ):
        raise DatasetIntegrityError(f"query file differs from run manifest: {run_path}")
    if (
        qrels_file.get("path") != str(qrels_path.relative_to(root))
        or file_facts(qrels_path).sha256 != qrels_file.get("sha256")
    ):
        raise DatasetIntegrityError(f"qrels file differs from run manifest: {run_path}")
    metrics = cast(dict[str, Any], read_json(metrics_path))
    if (
        metrics.get("schema_version") != 1
        or metrics.get("tag") != tag
        or metrics.get("split") != manifest.get("split")
    ):
        raise DatasetIntegrityError(f"BM25 metrics metadata differs: {run_path}")
    evaluation = evaluate_run(qrels_path, run_path)
    if (
        metrics.get("metrics") != evaluation.metrics
        or metrics.get("per_query") != evaluation.per_query
    ):
        raise DatasetIntegrityError(f"stored metrics do not reproduce for {run_path}")
    if entry.get("metrics") != evaluation.metrics:
        raise DatasetIntegrityError(f"selection metrics do not reproduce for {run_path}")
    return manifest


def verify_wands_bm25_selection(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    experiments: BM25Experiments,
) -> dict[str, Any]:
    require_registered_opensearch_client(client, root / _PROFILE_RELATIVE_PATH)
    selection_path = root / "results/wands/bm25-selection.json"
    if len(experiments.tuning_candidates) != experiments.hybrid_weight_count:
        raise DatasetIntegrityError("BM25 and hybrid tuning budgets differ")
    selection = cast(dict[str, Any], read_json(selection_path))
    if selection.get("schema_version") != 2 or selection.get("dataset") != "WANDS":
        raise DatasetIntegrityError("BM25 selection schema or dataset differs")
    _verify_registered_wands_lexical_index(
        client,
        root=root,
        index=index,
    )
    artifacts_value = selection.get("artifacts")
    if not isinstance(artifacts_value, dict):
        raise DatasetIntegrityError("BM25 selection artifacts are missing")
    artifacts = cast(dict[str, dict[str, Any]], artifacts_value)
    expected_artifact_keys = {
        *(
            f"tuning/{profile.name}/dev"
            for profile in experiments.tuning_candidates
        ),
        "naive/dev",
        "magento/dev",
        "tuned/dev",
        "competent_integrator/dev",
        "naive/test",
        "magento/test",
        "tuned/test",
        "competent_integrator/test",
    }
    if set(artifacts) != expected_artifact_keys:
        raise DatasetIntegrityError("BM25 selection artifact set differs")

    provenance = selection.get("benchmark_provenance")
    if not isinstance(provenance, Mapping):
        raise DatasetIntegrityError("BM25 selection provenance is missing")
    provenance_valid = _recorded_completion_provenance_valid(
        root,
        provenance,
        selection.get("completion_code_revision"),
    )
    if selection.get("benchmark_provenance_valid") is not provenance_valid:
        raise DatasetIntegrityError("BM25 selection provenance validity differs")
    live_index = collect_index_facts(client, index)
    index_manifest_path = root / "results/wands/index-manifest.json"
    index_evidence = _bm25_index_evidence(
        root=root,
        index_manifest_path=index_manifest_path,
        live_index=live_index,
    )
    if (
        selection.get("index_manifest") != index_evidence["artifact"]
        or selection.get("index_evidence") != index_evidence
        or selection.get("index_uuid") != live_index.uuid
    ):
        raise DatasetIntegrityError("BM25 selection index evidence differs")

    manifests: dict[str, dict[str, Any]] = {}
    for key, entry in artifacts.items():
        if set(entry) != {"run", "manifest", "metrics_file", "metrics", "profile"}:
            raise DatasetIntegrityError(f"BM25 artifact entry fields differ: {key}")
        manifest = verify_bm25_artifact(
            client,
            root=root,
            index=index,
            entry=entry,
        )
        manifests[key] = manifest
        if manifest.get("benchmark_provenance") != provenance:
            raise DatasetIntegrityError(
                f"BM25 artifact start provenance differs: {key}"
            )
        expected_split = key.rsplit("/", 1)[-1]
        if manifest.get("split") != expected_split:
            raise DatasetIntegrityError(f"BM25 artifact split differs: {key}")
        expected_query_path = f"data/prepared/wands/queries.{expected_split}.jsonl"
        expected_qrels_path = f"data/prepared/wands/qrels.{expected_split}.trec"
        if (
            cast(dict[str, Any], manifest.get("query_file")).get("path")
            != expected_query_path
            or cast(dict[str, Any], manifest.get("qrels_file")).get("path")
            != expected_qrels_path
        ):
            raise DatasetIntegrityError(f"BM25 artifact split files differ: {key}")

    candidate_metrics = {
        profile.name: artifacts[f"tuning/{profile.name}/dev"]["metrics"]
        for profile in experiments.tuning_candidates
    }
    selected = max(
        experiments.tuning_candidates,
        key=lambda profile: cast(
            dict[str, float], candidate_metrics[profile.name]
        )["ndcg@10"],
    )
    expected_profiles = {
        **{
            f"tuning/{profile.name}/dev": profile
            for profile in experiments.tuning_candidates
        },
        "naive/dev": experiments.naive,
        "naive/test": experiments.naive,
        "magento/dev": experiments.magento_emulation,
        "magento/test": experiments.magento_emulation,
        "tuned/dev": selected,
        "tuned/test": selected,
        "competent_integrator/dev": experiments.competent_integrator,
        "competent_integrator/test": experiments.competent_integrator,
    }
    expected_scopes = {
        **{
            f"tuning/{profile.name}/dev": "tuning"
            for profile in experiments.tuning_candidates
        },
        "naive/dev": "context_only",
        "magento/dev": "tuning",
        "tuned/dev": "tuning",
        "competent_integrator/dev": "post_review_challenge_arm",
        "naive/test": "held_out",
        "magento/test": "held_out",
        "tuned/test": "held_out",
        "competent_integrator/test": "held_out",
    }
    for key, profile in expected_profiles.items():
        if (
            artifacts[key].get("profile") != _json_dict(profile.to_dict())
            or manifests[key].get("decision_scope") != expected_scopes[key]
        ):
            raise DatasetIntegrityError(f"BM25 artifact profile or scope differs: {key}")

    _verify_bm25_live_reruns(
        client,
        root=root,
        index=index,
        profiles=expected_profiles,
        artifacts=artifacts,
        manifests=manifests,
    )

    held_out_metrics = {
        arm: artifacts[f"{arm}/test"]["metrics"]
        for arm in ("naive", "magento", "tuned", "competent_integrator")
    }
    challenge = {
        "profile": experiments.competent_integrator.to_dict(),
        "selection_surface": "fixed before rerunning the held-out test split",
        "development_metrics": artifacts["competent_integrator/dev"]["metrics"],
        "limitations": [
            "WANDS contains no merchant SKU field",
            "no merchant synonym set exists for this public dataset",
        ],
    }
    expected_selection_fields = _json_dict(
        {
            "selection_metric": "ndcg@10",
            "selection_split": "dev",
            "tie_break": "first_candidate_in_registered_order",
            "bm25_tuning_budget": len(experiments.tuning_candidates),
            "hybrid_weight_tuning_budget": experiments.hybrid_weight_count,
            "selected_profile": selected.to_dict(),
            "selected_profile_sha256": canonical_sha256(selected.to_dict()),
            "candidate_dev_metrics": candidate_metrics,
            "held_out_test_metrics": held_out_metrics,
            "post_review_challenge_arm": challenge,
        }
    )
    if any(
        selection.get(key) != value
        for key, value in expected_selection_fields.items()
    ):
        raise DatasetIntegrityError("BM25 selection fields do not reproduce")

    bindings, bindings_valid, binding_reasons = _bm25_selection_source_bindings(
        root=root,
        artifacts=artifacts,
        experiments=experiments,
        benchmark_provenance=provenance,
        index_evidence=index_evidence,
    )
    required_keys = list(_required_selection_source_keys(experiments))
    if (
        selection.get("required_source_manifest_keys") != required_keys
        or selection.get("source_manifest_bindings") != bindings
    ):
        raise DatasetIntegrityError("BM25 selection source manifest bindings differ")
    eligible, reasons = _bm25_selection_eligibility(
        benchmark_provenance_valid=provenance_valid,
        index_evidence_eligible=index_evidence["eligible_for_decision"] is True,
        source_bindings_valid=bindings_valid,
    )
    if (
        selection.get("quality_evidence_eligible_for_decision") is not eligible
        or selection.get("quality_ineligibility_reasons")
        != reasons + binding_reasons
    ):
        raise DatasetIntegrityError("BM25 selection quality eligibility differs")
    return selection


def _verify_bm25_live_reruns(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    profiles: Mapping[str, BM25Profile],
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
        prefix="opensearch-hybrid-bm25-verify-"
    ) as temporary:
        temporary_root = Path(temporary)
        for key, profile in profiles.items():
            split = key.rsplit("/", 1)[-1]
            entry = artifacts[key]
            manifest = manifests[key]
            tag = str(manifest["tag"])
            records = execute_bm25_profile(
                client,
                index=index,
                queries=queries[split],
                profile=profile,
                tag=tag,
            )
            repeated_path = temporary_root / f"{key.replace('/', '-')}.trec"
            write_run(repeated_path, records)
            if repeated_path.read_bytes() != (
                root / str(entry["run"])
            ).read_bytes():
                raise DatasetIntegrityError(
                    f"BM25 live rerun is not byte-identical: {key}"
                )
