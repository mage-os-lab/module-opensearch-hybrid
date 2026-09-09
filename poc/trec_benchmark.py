from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from poc.bm25 import (
    execute_bm25_profile,
    load_prepared_queries,
    verify_wands_bm25_selection,
)
from poc.datasets import (
    DatasetIntegrityError,
    file_facts,
    load_trec_product_search_config,
    load_wands_config,
)
from poc.evaluation import evaluate_run
from poc.evaluation_crosscheck import cross_check_run
from poc.experiments import BM25Experiments, BM25Profile, load_bm25_experiments
from poc.indexing import load_wands_index_config, verify_wands_lexical_index
from poc.manifest import canonical_sha256, collect_index_facts, read_json, write_json
from poc.os_client import OpenSearchClient
from poc.provenance import (
    collect_code_revision,
    collect_manifest_provenance,
    require_registered_opensearch_client,
    verify_decision_provenance,
)
from poc.trec import RunRecord, read_run, write_run
from poc.trec_indexing import load_trec_index_config, verify_trec_lexical_index

_PROFILE_PATH = "config/benchmark-2.19.toml"
_ENVIRONMENT_PATH = (
    "results/environment/benchmark-profile-opensearch-2.19.json"
)


def run_trec_bm25_benchmark(
    client: OpenSearchClient,
    wands_client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    experiments: BM25Experiments,
) -> dict[str, object]:
    require_registered_opensearch_client(client, root / _PROFILE_PATH)
    require_registered_opensearch_client(
        wands_client,
        root / "config/benchmark.toml",
    )
    verify_registered_trec_bm25_inputs(
        root=root,
        index=index,
        experiments=experiments,
    )
    benchmark_provenance = collect_manifest_provenance(
        root,
        profile_path=root / _PROFILE_PATH,
        environment_path=root / _ENVIRONMENT_PATH,
    )
    upstream_evidence = _trec_bm25_upstream_evidence(
        client,
        wands_client,
        root=root,
        experiments=experiments,
    )
    prepared = root / "data/prepared/trec-product-search-2024"
    query_path = prepared / "queries.test.jsonl"
    qrels_path = prepared / "qrels.test.trec"
    queries = load_prepared_queries(query_path, expected_split="test")
    wands_selection_path = root / "results/wands/bm25-selection.json"
    wands = cast(dict[str, Any], read_json(wands_selection_path))
    tuned = BM25Profile.from_mapping(
        cast(dict[str, Any], wands["selected_profile"])
    )
    profiles = {
        "tuned": tuned,
        "magento": experiments.magento_emulation,
        "competent_integrator": experiments.competent_integrator,
    }
    records_by_arm: dict[str, list[RunRecord]] = {}
    for arm, profile in profiles.items():
        tag = f"trec-product-search-2024-bm25-{arm.replace('_', '-')}"
        records_by_arm[arm] = execute_bm25_profile(
            client,
            index=index,
            queries=queries,
            profile=profile,
            tag=tag,
        )
    completion_upstream_evidence = _trec_bm25_upstream_evidence(
        client,
        wands_client,
        root=root,
        experiments=experiments,
    )
    provenance_valid, completion_revision = trec_completion_provenance_evidence(
        root,
        benchmark_provenance,
    )
    artifacts: dict[str, dict[str, object]] = {}
    for arm, profile in profiles.items():
        tag = f"trec-product-search-2024-bm25-{arm.replace('_', '-')}"
        artifacts[arm] = write_trec_bundle(
            client,
            root=root,
            index=index,
            query_path=query_path,
            qrels_path=qrels_path,
            tag=tag,
            method="bm25",
            records=records_by_arm[arm],
            declared_variable="bm25_profile",
            method_config={"profile": profile.to_dict()},
            selection_source=wands_selection_path,
            index_manifest_path=(
                root / "results/trec-product-search/index-manifest.json"
            ),
            benchmark_provenance=benchmark_provenance,
            provenance_valid=provenance_valid,
            completion_revision=completion_revision,
            upstream_evidence=upstream_evidence,
            completion_upstream_evidence=completion_upstream_evidence,
        )
    quality_eligible = derive_trec_evidence_eligibility(
        provenance_valid=provenance_valid,
        start_upstream=upstream_evidence,
        completion_upstream=completion_upstream_evidence,
        source_eligibilities=tuple(
            entry.get("eligible_for_decision") for entry in artifacts.values()
        ),
    )
    summary: dict[str, object] = {
        "schema_version": 2,
        "dataset": "TREC Product Search 2024",
        "split": "held_out_test",
        "tuning_on_trec": False,
        "selection_source": "results/wands/bm25-selection.json",
        "frozen_tuned_profile": tuned.to_dict(),
        "frozen_tuned_profile_sha256": canonical_sha256(tuned.to_dict()),
        "held_out_metrics": {
            arm: entry["metrics"] for arm, entry in artifacts.items()
        },
        "quality_evidence_eligible_for_decision": quality_eligible,
        "quality_ineligibility_reasons": _trec_evidence_ineligibility_reasons(
            provenance_valid=provenance_valid,
            start_upstream=upstream_evidence,
            completion_upstream=completion_upstream_evidence,
            source_eligibilities={
                arm: entry.get("eligible_for_decision")
                for arm, entry in artifacts.items()
            },
        ),
        "artifacts": artifacts,
        "benchmark_provenance": benchmark_provenance,
        "benchmark_provenance_valid": provenance_valid,
        "completion_code_revision": completion_revision,
        "upstream_evidence": upstream_evidence,
        "completion_upstream_evidence": completion_upstream_evidence,
        "upstream_evidence_unchanged": (
            completion_upstream_evidence == upstream_evidence
        ),
    }
    write_json(root / "results/trec-product-search/bm25-summary.json", summary)
    return summary


def write_trec_bundle(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    query_path: Path,
    qrels_path: Path,
    tag: str,
    method: str,
    records: list[RunRecord],
    declared_variable: str,
    method_config: dict[str, object],
    selection_source: Path,
    index_manifest_path: Path,
    benchmark_provenance: dict[str, Any],
    provenance_valid: bool,
    completion_revision: dict[str, Any],
    upstream_evidence: dict[str, object],
    completion_upstream_evidence: dict[str, object],
) -> dict[str, object]:
    run_path = root / f"runs/{tag}.trec"
    manifest_path = root / f"runs/{tag}.manifest.json"
    metrics_path = root / f"results/trec-product-search/{method}/{tag}.metrics.json"
    write_run(run_path, records)
    evaluation = evaluate_run(qrels_path, run_path, exact_gain=3)
    crosscheck = cross_check_run(qrels_path, run_path, root=root)
    write_json(
        metrics_path,
        {
            "schema_version": 1,
            "dataset": "TREC Product Search 2024",
            "tag": tag,
            "split": "test",
            "exact_gain": 3,
            "metrics": evaluation.metrics,
            "per_query": evaluation.per_query,
            "evaluator_crosscheck": crosscheck.to_dict(),
        },
    )
    root_response = cast(dict[str, Any], client.request("GET", "/"))
    version = str(cast(dict[str, Any], root_response["version"])["number"])
    preparation_path = root / "results/trec-product-search/preparation.json"
    upstream_unchanged = completion_upstream_evidence == upstream_evidence
    eligible = derive_trec_evidence_eligibility(
        provenance_valid=provenance_valid,
        start_upstream=upstream_evidence,
        completion_upstream=completion_upstream_evidence,
        source_eligibilities=(),
    )
    manifest: dict[str, object] = {
        "schema_version": 2,
        "dataset": "TREC Product Search 2024",
        "split": "test",
        "tag": tag,
        "method": method,
        "decision_scope": "held_out_transfer",
        "eligible_for_decision": eligible,
        "ineligibility_reasons": _trec_evidence_ineligibility_reasons(
            provenance_valid=provenance_valid,
            start_upstream=upstream_evidence,
            completion_upstream=completion_upstream_evidence,
            source_eligibilities={},
        ),
        "upstream_quality_evidence_eligible_for_decision": (
            upstream_evidence.get("eligible_for_decision") is True
            and completion_upstream_evidence.get("eligible_for_decision") is True
        ),
        "upstream_evidence": upstream_evidence,
        "completion_upstream_evidence": completion_upstream_evidence,
        "upstream_evidence_unchanged": upstream_unchanged,
        "declared_variable": declared_variable,
        "opensearch_version": version,
        "index": asdict(collect_index_facts(client, index)),
        "index_manifest": _artifact(root, index_manifest_path),
        "index_manifest_sha256": file_facts(index_manifest_path).sha256,
        "preparation": _artifact(root, preparation_path),
        "preparation_sha256": file_facts(preparation_path).sha256,
        "selection_source": _artifact(root, selection_source),
        "method_config": method_config,
        "method_config_sha256": canonical_sha256(method_config),
        "query_file": _artifact(root, query_path),
        "qrels_file": _artifact(root, qrels_path),
        "run_file": {**_artifact(root, run_path), "records": len(records)},
        "metrics_file": _artifact(root, metrics_path),
        "benchmark_provenance": benchmark_provenance,
        "benchmark_provenance_valid": provenance_valid,
        "completion_code_revision": completion_revision,
    }
    write_json(manifest_path, manifest)
    return {
        "run": str(run_path.relative_to(root)),
        "manifest": str(manifest_path.relative_to(root)),
        "metrics_file": str(metrics_path.relative_to(root)),
        "metrics": evaluation.metrics,
        "eligible_for_decision": eligible,
        "artifact_files": {
            "run": _artifact(root, run_path),
            "manifest": _artifact(root, manifest_path),
            "metrics_file": _artifact(root, metrics_path),
        },
    }


def verify_trec_bm25_summary(
    client: OpenSearchClient,
    wands_client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    experiments: BM25Experiments,
) -> dict[str, Any]:
    require_registered_opensearch_client(client, root / _PROFILE_PATH)
    require_registered_opensearch_client(
        wands_client,
        root / "config/benchmark.toml",
    )
    verify_registered_trec_bm25_inputs(root=root, index=index, experiments=experiments)
    summary = cast(
        dict[str, Any],
        read_json(root / "results/trec-product-search/bm25-summary.json"),
    )
    if (
        summary.get("schema_version") != 2
        or summary.get("dataset") != "TREC Product Search 2024"
        or summary.get("split") != "held_out_test"
        or summary.get("tuning_on_trec") is not False
        or summary.get("selection_source")
        != "results/wands/bm25-selection.json"
    ):
        raise DatasetIntegrityError("TREC BM25 summary metadata differs")
    wands = cast(dict[str, Any], read_json(root / "results/wands/bm25-selection.json"))
    tuned = BM25Profile.from_mapping(cast(dict[str, Any], wands["selected_profile"]))
    expected_profiles = {
        "tuned": tuned,
        "magento": experiments.magento_emulation,
        "competent_integrator": experiments.competent_integrator,
    }
    if (
        summary.get("frozen_tuned_profile_sha256")
        != canonical_sha256(tuned.to_dict())
        or summary.get("frozen_tuned_profile") != tuned.to_dict()
    ):
        raise DatasetIntegrityError("TREC frozen BM25 profile differs")
    artifacts_value = summary.get("artifacts")
    if not isinstance(artifacts_value, dict):
        raise DatasetIntegrityError("TREC BM25 artifacts are missing")
    artifacts = cast(dict[str, dict[str, Any]], artifacts_value)
    if set(artifacts) != set(expected_profiles):
        raise DatasetIntegrityError("TREC BM25 artifact set differs")
    current_upstream = _trec_bm25_upstream_evidence(
        client, wands_client, root=root, experiments=experiments
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
        raise DatasetIntegrityError("TREC BM25 provenance or upstream evidence differs")
    index_manifest_path = root / "results/trec-product-search/index-manifest.json"
    for arm, profile in expected_profiles.items():
        entry = artifacts[arm]
        manifest = verify_trec_artifact(
            client,
            root=root,
            index=index,
            entry=entry,
            index_manifest_path=index_manifest_path,
            expected_benchmark_provenance=summary_provenance,
            expected_completion_revision=completion_revision,
            expected_upstream_evidence=current_upstream,
            expected_query_path=(
                root / "data/prepared/trec-product-search-2024/queries.test.jsonl"
            ),
            expected_qrels_path=(
                root / "data/prepared/trec-product-search-2024/qrels.test.trec"
            ),
            expected_selection_source_path=root / "results/wands/bm25-selection.json",
        )
        verify_trec_method_attribution(
            manifest,
            expected_method="bm25",
            expected_declared_variable="bm25_profile",
            expected_config={"profile": profile.to_dict()},
        )
    source_eligibilities = {
        arm: entry.get("eligible_for_decision") for arm, entry in artifacts.items()
    }
    quality_eligible = derive_trec_evidence_eligibility(
        provenance_valid=provenance_valid,
        start_upstream=start_upstream,
        completion_upstream=completion_upstream,
        source_eligibilities=tuple(source_eligibilities.values()),
    )
    expected_metrics = {arm: entry["metrics"] for arm, entry in artifacts.items()}
    if (
        summary.get("held_out_metrics") != expected_metrics
        or summary.get("quality_evidence_eligible_for_decision") is not quality_eligible
        or summary.get("quality_ineligibility_reasons")
        != _trec_evidence_ineligibility_reasons(
            provenance_valid=provenance_valid,
            start_upstream=start_upstream,
            completion_upstream=completion_upstream,
            source_eligibilities=source_eligibilities,
        )
    ):
        raise DatasetIntegrityError("TREC BM25 summary eligibility differs")
    for arm in ("tuned", "magento", "competent_integrator"):
        _verify_trec_bm25_rerun(
            client,
            root=root,
            index=index,
            profile=expected_profiles[arm],
            entry=artifacts[arm],
        )
    return summary


def verify_registered_trec_bm25_inputs(
    *, root: Path, index: str, experiments: BM25Experiments
) -> None:
    registered_index = load_trec_index_config(root / "config/indexes.toml")
    registered_experiments = load_bm25_experiments(root / "config/experiments.toml")
    if index != registered_index.name or experiments != registered_experiments:
        raise DatasetIntegrityError("TREC BM25 registered configuration differs")


def verify_trec_artifact(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    entry: dict[str, Any],
    index_manifest_path: Path,
    expected_benchmark_provenance: Mapping[str, Any],
    expected_completion_revision: Mapping[str, Any],
    expected_upstream_evidence: Mapping[str, Any],
    expected_query_path: Path,
    expected_qrels_path: Path,
    expected_selection_source_path: Path,
) -> dict[str, Any]:
    require_registered_opensearch_client(client, root / _PROFILE_PATH)
    expected_entry_keys = {
        "run", "manifest", "metrics_file", "metrics",
        "eligible_for_decision", "artifact_files",
    }
    if set(entry) != expected_entry_keys:
        raise DatasetIntegrityError("TREC artifact entry fields differ")
    run_path = root / str(entry["run"])
    manifest_path = root / str(entry["manifest"])
    metrics_path = root / str(entry["metrics_file"])
    manifest = cast(dict[str, Any], read_json(manifest_path))
    metrics = cast(dict[str, Any], read_json(metrics_path))
    expected_artifact_files = {
        "run": _artifact(root, run_path),
        "manifest": _artifact(root, manifest_path),
        "metrics_file": _artifact(root, metrics_path),
    }
    if entry.get("artifact_files") != expected_artifact_files:
        raise DatasetIntegrityError("TREC artifact file bindings differ")
    root_response = cast(dict[str, Any], client.request("GET", "/"))
    live_version = str(cast(dict[str, Any], root_response["version"])["number"])
    run_records = read_run(run_path)
    verify_trec_manifest_metadata(
        manifest,
        expected_version=live_version,
        expected_run_path=str(run_path.relative_to(root)),
        record_tags=[record.tag for record in run_records],
    )
    if manifest.get("index") != asdict(collect_index_facts(client, index)):
        raise DatasetIntegrityError(f"TREC run index differs: {run_path}")
    verify_trec_artifact_bindings(
        manifest,
        index_manifest_path=index_manifest_path,
        preparation_path=root / "results/trec-product-search/preparation.json",
    )
    if (
        manifest.get("index_manifest") != _artifact(root, index_manifest_path)
        or manifest.get("preparation")
        != _artifact(
            root, root / "results/trec-product-search/preparation.json"
        )
    ):
        raise DatasetIntegrityError("TREC run upstream artifact bindings differ")
    provenance = manifest.get("benchmark_provenance")
    completion_revision = manifest.get("completion_code_revision")
    provenance_valid = trec_recorded_provenance_valid(
        root, provenance=provenance, completion_revision=completion_revision
    )
    completion_is_current = trec_completion_revision_is_current(
        root, completion_revision
    )
    start_upstream = manifest.get("upstream_evidence")
    completion_upstream = manifest.get("completion_upstream_evidence")
    upstream_unchanged = (
        isinstance(start_upstream, dict)
        and isinstance(completion_upstream, dict)
        and start_upstream == completion_upstream == expected_upstream_evidence
    )
    expected_eligible = derive_trec_evidence_eligibility(
        provenance_valid=provenance_valid,
        start_upstream=start_upstream,
        completion_upstream=completion_upstream,
        source_eligibilities=(),
    )
    if (
        provenance != expected_benchmark_provenance
        or completion_revision != expected_completion_revision
        or not completion_is_current
        or manifest.get("benchmark_provenance_valid") is not provenance_valid
        or not upstream_unchanged
        or manifest.get("upstream_evidence_unchanged") is not upstream_unchanged
        or manifest.get("upstream_quality_evidence_eligible_for_decision")
        is not (expected_upstream_evidence.get("eligible_for_decision") is True)
        or manifest.get("eligible_for_decision") is not expected_eligible
        or entry.get("eligible_for_decision") is not expected_eligible
        or manifest.get("ineligibility_reasons")
        != _trec_evidence_ineligibility_reasons(
            provenance_valid=provenance_valid,
            start_upstream=start_upstream,
            completion_upstream=completion_upstream,
            source_eligibilities={},
        )
    ):
        raise DatasetIntegrityError("TREC run evidence or eligibility differs")
    run_file = manifest.get("run_file")
    if (
        not isinstance(run_file, dict)
        or run_file != {**_artifact(root, run_path), "records": len(run_records)}
    ):
        raise DatasetIntegrityError(f"TREC run binding differs: {run_path}")
    for key, expected_path in (
        ("query_file", expected_query_path),
        ("qrels_file", expected_qrels_path),
        ("selection_source", expected_selection_source_path),
        ("metrics_file", metrics_path),
    ):
        verify_trec_registered_artifact(
            manifest, key=key, root=root, expected_path=expected_path
        )
    evaluation = evaluate_run(expected_qrels_path, run_path, exact_gain=3)
    if (
        metrics.get("schema_version") != 1
        or metrics.get("dataset") != "TREC Product Search 2024"
        or metrics.get("tag") != manifest.get("tag")
        or metrics.get("split") != "test"
        or metrics.get("exact_gain") != 3
        or metrics.get("metrics") != evaluation.metrics
        or metrics.get("per_query") != evaluation.per_query
        or entry.get("metrics") != evaluation.metrics
    ):
        raise DatasetIntegrityError(f"TREC metrics do not reproduce: {run_path}")
    if metrics.get("evaluator_crosscheck") != cross_check_run(
        expected_qrels_path, run_path, root=root
    ).to_dict():
        raise DatasetIntegrityError(f"TREC evaluator cross-check differs: {run_path}")
    return manifest


def verify_trec_manifest_metadata(
    manifest: dict[str, Any],
    *,
    expected_version: str,
    expected_run_path: str,
    record_tags: list[str],
) -> None:
    run_file = manifest.get("run_file")
    expected_tag = manifest.get("tag")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("dataset") != "TREC Product Search 2024"
        or manifest.get("split") != "test"
        or manifest.get("decision_scope") != "held_out_transfer"
        or manifest.get("opensearch_version") != expected_version
        or not isinstance(expected_tag, str)
        or not expected_tag
        or not isinstance(run_file, dict)
        or run_file.get("path") != expected_run_path
        or run_file.get("records") != len(record_tags)
        or not record_tags
        or any(tag != expected_tag for tag in record_tags)
    ):
        raise DatasetIntegrityError("TREC run metadata differs")


def verify_trec_artifact_bindings(
    manifest: dict[str, Any], *, index_manifest_path: Path, preparation_path: Path
) -> None:
    index_facts = file_facts(index_manifest_path)
    preparation_facts = file_facts(preparation_path)
    if manifest.get("index_manifest_sha256") != index_facts.sha256:
        raise DatasetIntegrityError("TREC index manifest differs")
    if manifest.get("preparation_sha256") != preparation_facts.sha256:
        raise DatasetIntegrityError("TREC preparation artifact differs")
    for key, path, facts in (
        ("index_manifest", index_manifest_path, index_facts),
        ("preparation", preparation_path, preparation_facts),
    ):
        artifact = manifest.get(key)
        if artifact is None:
            continue
        if (
            not isinstance(artifact, dict)
            or artifact.get("sha256") != facts.sha256
            or artifact.get("bytes") != facts.bytes
            or not isinstance(artifact.get("path"), str)
            or not str(path).endswith(cast(str, artifact["path"]))
        ):
            raise DatasetIntegrityError(f"TREC {key.replace('_', ' ')} binding differs")


def verify_trec_method_attribution(
    manifest: dict[str, Any],
    *,
    expected_method: str,
    expected_declared_variable: str,
    expected_config: dict[str, object],
) -> None:
    expected_hash = canonical_sha256(expected_config)
    if (
        manifest.get("method") != expected_method
        or manifest.get("declared_variable") != expected_declared_variable
        or canonical_sha256(manifest.get("method_config")) != expected_hash
        or manifest.get("method_config_sha256") != expected_hash
    ):
        raise DatasetIntegrityError("TREC method attribution differs")


def verify_trec_registered_artifact(
    manifest: dict[str, Any], *, key: str, root: Path, expected_path: Path
) -> None:
    artifact = manifest.get(key)
    try:
        expected_relative = expected_path.relative_to(root).as_posix()
    except ValueError as error:
        raise DatasetIntegrityError(
            f"TREC registered {key.replace('_', ' ')} is outside the repository"
        ) from error
    expected_facts = file_facts(expected_path)
    if (
        not isinstance(artifact, dict)
        or artifact.get("path") != expected_relative
        or artifact.get("sha256") != expected_facts.sha256
        or artifact.get("bytes") != expected_facts.bytes
    ):
        raise DatasetIntegrityError(f"TREC registered {key.replace('_', ' ')} differs")


def trec_revision_chain_matches(
    *, start_revision: object, completion_revision: object, current_revision: object
) -> bool:
    revisions = (start_revision, completion_revision, current_revision)
    if not all(isinstance(revision, Mapping) for revision in revisions):
        return False
    normalized = [dict(cast(Mapping[str, Any], revision)) for revision in revisions]
    expected_keys = {"git_commit", "source_tree_sha256", "source_dirty"}
    for revision in normalized:
        commit = revision.get("git_commit")
        source_hash = revision.get("source_tree_sha256")
        if (
            set(revision) != expected_keys
            or not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
            or not isinstance(source_hash, str)
            or len(source_hash) != 64
            or any(character not in "0123456789abcdef" for character in source_hash)
            or revision.get("source_dirty") is not False
        ):
            return False
    return normalized[0] == normalized[1] == normalized[2]


def trec_completion_provenance_evidence(
    root: Path, provenance: Mapping[str, Any]
) -> tuple[bool, dict[str, Any]]:
    completion_revision = asdict(collect_code_revision(root))
    valid = trec_revision_chain_matches(
        start_revision=provenance.get("code_revision"),
        completion_revision=completion_revision,
        current_revision=completion_revision,
    ) and verify_decision_provenance(
        provenance,
        root=root,
        profile_path=root / _PROFILE_PATH,
        environment_path=root / _ENVIRONMENT_PATH,
        require_current_code_revision=True,
    )
    return valid, completion_revision


def trec_recorded_provenance_valid(
    root: Path, *, provenance: object, completion_revision: object
) -> bool:
    if not isinstance(provenance, Mapping):
        return False
    current_revision = asdict(collect_code_revision(root))
    return trec_revision_chain_matches(
        start_revision=provenance.get("code_revision"),
        completion_revision=completion_revision,
        current_revision=current_revision,
    ) and verify_decision_provenance(
        provenance,
        root=root,
        profile_path=root / _PROFILE_PATH,
        environment_path=root / _ENVIRONMENT_PATH,
        require_current_code_revision=True,
    )


def trec_completion_revision_is_current(
    root: Path, completion_revision: object
) -> bool:
    return isinstance(completion_revision, Mapping) and dict(
        completion_revision
    ) == asdict(collect_code_revision(root))


def derive_trec_evidence_eligibility(
    *,
    provenance_valid: object,
    start_upstream: object,
    completion_upstream: object,
    source_eligibilities: tuple[object, ...],
) -> bool:
    return (
        provenance_valid is True
        and isinstance(start_upstream, Mapping)
        and isinstance(completion_upstream, Mapping)
        and dict(start_upstream) == dict(completion_upstream)
        and start_upstream.get("eligible_for_decision") is True
        and completion_upstream.get("eligible_for_decision") is True
        and all(value is True for value in source_eligibilities)
    )


def _trec_evidence_ineligibility_reasons(
    *,
    provenance_valid: object,
    start_upstream: object,
    completion_upstream: object,
    source_eligibilities: Mapping[str, object],
) -> list[str]:
    reasons: list[str] = []
    if provenance_valid is not True:
        reasons.append("no valid clean committed start and completion provenance")
    if (
        not isinstance(start_upstream, Mapping)
        or start_upstream.get("eligible_for_decision") is not True
        or not isinstance(completion_upstream, Mapping)
        or completion_upstream.get("eligible_for_decision") is not True
    ):
        reasons.append("semantic upstream evidence is not decision eligible")
    if (
        not isinstance(start_upstream, Mapping)
        or not isinstance(completion_upstream, Mapping)
        or dict(start_upstream) != dict(completion_upstream)
    ):
        reasons.append("semantic upstream evidence changed during computation")
    reasons.extend(
        f"source is not decision eligible: {name}"
        for name, eligible in source_eligibilities.items()
        if eligible is not True
    )
    return reasons


def _trec_bm25_upstream_evidence(
    client: OpenSearchClient,
    wands_client: OpenSearchClient,
    *,
    root: Path,
    experiments: BM25Experiments,
) -> dict[str, object]:
    dataset_spec = load_wands_config(root / "config/datasets.toml")
    index_spec = load_wands_index_config(root / "config/indexes.toml")
    wands_index = verify_wands_lexical_index(
        wands_client,
        dataset_spec=dataset_spec,
        index_spec=index_spec,
        prepared_directory=root / "data/prepared/wands",
        manifest_path=root / "results/wands/index-manifest.json",
    )
    selection = verify_wands_bm25_selection(
        wands_client,
        root=root,
        index=index_spec.name,
        experiments=experiments,
    )
    trec_dataset_spec = load_trec_product_search_config(root / "config/datasets.toml")
    trec_index_spec = load_trec_index_config(root / "config/indexes.toml")
    trec_index = verify_trec_lexical_index(
        client,
        dataset_spec=trec_dataset_spec,
        index_spec=trec_index_spec,
        corpus_path=root / "data/raw/trec-product-search-2024/collection.trec.gz",
        preparation_path=root / "results/trec-product-search/preparation.json",
        manifest_path=root / "results/trec-product-search/index-manifest.json",
        expected_version="2.19.6",
        maximum_orphan_rate=experiments.maximum_trec_orphan_rate,
    )
    components = {
        "wands_lexical_index": _verified_upstream_component(
            root=root,
            path=root / "results/wands/index-manifest.json",
            verified=wands_index,
            semantic_verifier="verify_wands_lexical_index",
        ),
        "wands_bm25_selection": _verified_upstream_component(
            root=root,
            path=root / "results/wands/bm25-selection.json",
            verified=selection,
            semantic_verifier="verify_wands_bm25_selection",
        ),
        "trec_lexical_index": _verified_upstream_component(
            root=root,
            path=root / "results/trec-product-search/index-manifest.json",
            verified=trec_index,
            semantic_verifier="verify_trec_lexical_index",
        ),
    }
    eligible = all(
        component.get("eligible_for_decision") is True
        for component in components.values()
    )
    return {
        "components": components,
        "registered_config_artifacts": {
            name: _artifact(root, root / relative)
            for name, relative in {
                "datasets": "config/datasets.toml",
                "indexes": "config/indexes.toml",
                "experiments": "config/experiments.toml",
            }.items()
        },
        "eligible_for_decision": eligible,
        "ineligibility_reasons": [
            name
            for name, component in components.items()
            if component.get("eligible_for_decision") is not True
        ],
    }


def _verified_upstream_component(
    *,
    root: Path,
    path: Path,
    verified: Mapping[str, Any],
    semantic_verifier: str,
    eligibility_key: str = "quality_evidence_eligible_for_decision",
) -> dict[str, object]:
    declared = verified.get(eligibility_key) is True
    return {
        "semantic_verifier": semantic_verifier,
        "artifact": _artifact(root, path),
        "verified_payload_sha256": canonical_sha256(verified),
        "eligibility_key": eligibility_key,
        "quality_declared": declared,
        "eligible_for_decision": declared,
    }


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }


def _verify_trec_bm25_rerun(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    profile: BM25Profile,
    entry: dict[str, Any],
) -> None:
    manifest = cast(dict[str, Any], read_json(root / str(entry["manifest"])))
    queries = load_prepared_queries(
        root / "data/prepared/trec-product-search-2024/queries.test.jsonl",
        expected_split="test",
    )
    records = execute_bm25_profile(
        client,
        index=index,
        queries=queries,
        profile=profile,
        tag=str(manifest["tag"]),
    )
    with tempfile.TemporaryDirectory(
        prefix="opensearch-hybrid-trec-bm25-verify-"
    ) as temporary:
        repeated_path = Path(temporary) / "repeated.trec"
        write_run(repeated_path, records)
        if repeated_path.read_bytes() != (root / str(entry["run"])).read_bytes():
            raise DatasetIntegrityError("TREC BM25 rerun is not byte-identical")
