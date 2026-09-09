from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from poc.datasets import DatasetIntegrityError, TrecProductSearchDatasetSpec, file_facts
from poc.index_content import verify_live_index_content
from poc.index_evidence import (
    INDEX_WRITE_BLOCK_EVIDENCE,
    lock_live_index_for_decisions,
    verify_live_index_settings,
)
from poc.manifest import canonical_sha256, collect_index_facts, read_json, write_json
from poc.os_client import OpenSearchClient
from poc.provenance import (
    collect_manifest_provenance,
    require_registered_opensearch_client,
    verify_decision_provenance,
)
from poc.trec_sparse import (
    TrecSparseSpec,
    build_trec_sparse_index_definition,
    iter_trec_precomputed_sparse_documents,
)
from poc.trec_sparse_validation import verify_trec_sparse_validation_manifest


def build_trec_sparse_index(
    client: OpenSearchClient,
    *,
    root: Path,
    dataset_spec: TrecProductSearchDatasetSpec,
    sparse_spec: TrecSparseSpec,
    corpus_path: Path,
    embeddings_path: Path,
    precompute_manifest_path: Path,
    validation_manifest_path: Path,
    package_path: Path,
    model_path: Path,
    tokenizer_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    require_registered_opensearch_client(
        client,
        root / "config/benchmark-2.19.toml",
    )
    indexing_provenance = collect_manifest_provenance(
        root,
        profile_path=root / "config/benchmark-2.19.toml",
        environment_path=(
            root / "results/environment/benchmark-profile-opensearch-2.19.json"
        ),
    )
    version = client.wait_until_ready(
        expected_version=sparse_spec.expected_opensearch_version
    )
    precompute, validation = _verify_precompute(
        root=root,
        dataset_spec=dataset_spec,
        sparse_spec=sparse_spec,
        corpus_path=corpus_path,
        embeddings_path=embeddings_path,
        manifest_path=precompute_manifest_path,
        validation_manifest_path=validation_manifest_path,
        package_path=package_path,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )
    definition = build_trec_sparse_index_definition(sparse_spec)
    client.delete_index(sparse_spec.index_name)
    client.create_index(sparse_spec.index_name, definition)
    started = time.perf_counter()
    client.bulk_index(
        sparse_spec.index_name,
        iter_trec_precomputed_sparse_documents(
            corpus_path, embeddings_path, sparse_spec
        ),
        batch_size=sparse_spec.bulk_request_size,
    )
    bulk_seconds = time.perf_counter() - started
    client.refresh(sparse_spec.index_name)
    merge_started = time.perf_counter()
    client.force_merge(sparse_spec.index_name)
    merge_seconds = time.perf_counter() - merge_started
    client.update_index_settings(
        sparse_spec.index_name,
        {"index": {"refresh_interval": "1s"}},
    )
    index_write_block = lock_live_index_for_decisions(
        client,
        index=sparse_spec.index_name,
    )
    facts = collect_index_facts(client, sparse_spec.index_name)
    if facts.document_count != dataset_spec.expected_products:
        raise DatasetIntegrityError("TREC sparse index has the wrong product count")
    if facts.segment_count != 1 or facts.deleted_document_count != 0:
        raise DatasetIntegrityError("TREC sparse index must have one segment and no deletions")
    index_content = verify_live_index_content(
        client,
        index=sparse_spec.index_name,
        expected_documents=iter_trec_precomputed_sparse_documents(
            corpus_path,
            embeddings_path,
            sparse_spec,
        ),
        expected_count=dataset_spec.expected_products,
        float32_fields=(sparse_spec.embedding_field,),
    )
    store_bytes = _primary_store_bytes(client, sparse_spec.index_name)
    indexing_provenance_eligible = verify_decision_provenance(
        indexing_provenance,
        root=root,
        profile_path=root / "config/benchmark-2.19.toml",
        environment_path=(
            root / "results/environment/benchmark-profile-opensearch-2.19.json"
        ),
        require_current_code_revision=True,
    )
    quality_eligible = (
        validation["quality_evidence_eligible_for_decision"] is True
        and indexing_provenance_eligible
    )
    quality_ineligibility_reason: str | None = None
    if validation["quality_evidence_eligible_for_decision"] is not True:
        quality_ineligibility_reason = cast(
            str | None, validation.get("quality_ineligibility_reason")
        )
    elif not indexing_provenance_eligible:
        quality_ineligibility_reason = (
            "index build did not retain current decision-eligible start provenance"
        )
    manifest: dict[str, object] = {
        "schema_version": 2,
        "dataset": "TREC Product Search 2024",
        "source_revision": dataset_spec.revision,
        "opensearch_version": version,
        "index": asdict(facts),
        "index_definition_sha256": canonical_sha256(definition),
        "index_content": index_content,
        "index_write_block": index_write_block,
        "source_corpus": {
            **_artifact(root, corpus_path),
            "records": dataset_spec.expected_products,
        },
        "precomputed_embeddings": _artifact(root, embeddings_path),
        "precompute_manifest": {
            **_artifact(root, precompute_manifest_path),
            "records": precompute["records"],
        },
        "validation_manifest": {
            **_artifact(root, validation_manifest_path),
            "schema_version": validation["schema_version"],
            "quality_evidence_eligible_for_decision": validation[
                "quality_evidence_eligible_for_decision"
            ],
            "latency_evidence_eligible_for_decision": validation[
                "latency_evidence_eligible_for_decision"
            ],
        },
        "document_model": {
            "name": sparse_spec.document_model_name,
            "version": sparse_spec.document_model_version,
            "format": sparse_spec.document_model_format,
            "package_sha256": sparse_spec.document_model_sha256,
            "package_bytes": sparse_spec.document_model_bytes,
            "maximum_token_length": sparse_spec.maximum_token_length,
            "maximum_value_ratio": sparse_spec.maximum_value_ratio,
        },
        "query_tokenizer": {
            "name": sparse_spec.query_tokenizer_name,
            "version": sparse_spec.query_tokenizer_version,
            "format": sparse_spec.query_tokenizer_format,
            "content_sha256": sparse_spec.query_tokenizer_content_sha256,
            "content_bytes": sparse_spec.query_tokenizer_content_bytes,
        },
        "bulk_request_size": sparse_spec.bulk_request_size,
        "bulk_index_seconds": bulk_seconds,
        "bulk_documents_per_second": dataset_spec.expected_products / bulk_seconds,
        "force_merge_seconds": merge_seconds,
        "primary_store_bytes": store_bytes,
        "refresh_interval": "1s",
        "vector_state": "document_only_neural_sparse_rank_features",
        "indexing_provenance_eligible_for_decision": indexing_provenance_eligible,
        "quality_evidence_eligible_for_decision": quality_eligible,
        "quality_ineligibility_reason": quality_ineligibility_reason,
        "latency_evidence_eligible_for_decision": False,
        "benchmark_provenance": indexing_provenance,
    }
    write_json(manifest_path, manifest)
    return manifest


def verify_trec_sparse_index(
    client: OpenSearchClient,
    *,
    root: Path,
    dataset_spec: TrecProductSearchDatasetSpec,
    sparse_spec: TrecSparseSpec,
    corpus_path: Path,
    embeddings_path: Path,
    precompute_manifest_path: Path,
    validation_manifest_path: Path,
    package_path: Path,
    model_path: Path,
    tokenizer_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    require_registered_opensearch_client(
        client,
        root / "config/benchmark-2.19.toml",
    )
    client.wait_until_ready(expected_version=sparse_spec.expected_opensearch_version)
    _, validation = _verify_precompute(
        root=root,
        dataset_spec=dataset_spec,
        sparse_spec=sparse_spec,
        corpus_path=corpus_path,
        embeddings_path=embeddings_path,
        manifest_path=precompute_manifest_path,
        validation_manifest_path=validation_manifest_path,
        package_path=package_path,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )
    manifest = cast(dict[str, Any], read_json(manifest_path))
    verify_trec_sparse_manifest_metadata(
        manifest,
        dataset_spec=dataset_spec,
        sparse_spec=sparse_spec,
    )
    facts = collect_index_facts(client, sparse_spec.index_name)
    if manifest.get("index") != asdict(facts):
        raise DatasetIntegrityError("live TREC sparse index facts differ")
    if facts.document_count != dataset_spec.expected_products:
        raise DatasetIntegrityError("live TREC sparse index count differs")
    if facts.segment_count != 1 or facts.deleted_document_count != 0:
        raise DatasetIntegrityError("live TREC sparse index segment state differs")
    definition = build_trec_sparse_index_definition(sparse_spec)
    if manifest.get("index_definition_sha256") != canonical_sha256(definition):
        raise DatasetIntegrityError("TREC sparse index definition hash differs")
    if manifest.get("index_write_block") != INDEX_WRITE_BLOCK_EVIDENCE:
        raise DatasetIntegrityError("TREC sparse index write block evidence differs")
    mapping_response = cast(
        dict[str, Any],
        client.request("GET", f"/{sparse_spec.index_name}/_mapping"),
    )
    live_mapping = cast(
        dict[str, Any], mapping_response[sparse_spec.index_name]
    )["mappings"]
    if live_mapping != definition["mappings"]:
        raise DatasetIntegrityError("live TREC sparse mapping differs")
    verify_live_index_settings(
        client,
        index=sparse_spec.index_name,
        definition=definition,
    )
    index_content = verify_live_index_content(
        client,
        index=sparse_spec.index_name,
        expected_documents=iter_trec_precomputed_sparse_documents(
            corpus_path,
            embeddings_path,
            sparse_spec,
        ),
        expected_count=dataset_spec.expected_products,
        float32_fields=(sparse_spec.embedding_field,),
    )
    if manifest.get("index_content") != index_content:
        raise DatasetIntegrityError("live TREC sparse index content differs")
    for key, path in (
        ("source_corpus", corpus_path),
        ("precomputed_embeddings", embeddings_path),
        ("precompute_manifest", precompute_manifest_path),
        ("validation_manifest", validation_manifest_path),
    ):
        recorded = cast(dict[str, Any], manifest[key])
        expected_artifact = _artifact(root, path)
        if any(
            recorded.get(field) != expected_artifact[field]
            for field in ("path", "sha256", "bytes")
        ):
            raise DatasetIntegrityError(f"TREC sparse {key} differs")
    validation_quality = validation["quality_evidence_eligible_for_decision"] is True
    expected_validation = {
        **_artifact(root, validation_manifest_path),
        "schema_version": validation["schema_version"],
        "quality_evidence_eligible_for_decision": validation_quality,
        "latency_evidence_eligible_for_decision": False,
    }
    if manifest.get("validation_manifest") != expected_validation:
        raise DatasetIntegrityError("TREC sparse validation manifest binding differs")
    expected_precompute = {
        **_artifact(root, precompute_manifest_path),
        "records": dataset_spec.expected_products,
    }
    if manifest.get("precompute_manifest") != expected_precompute:
        raise DatasetIntegrityError("TREC sparse precompute manifest binding differs")
    indexing_provenance = manifest.get("benchmark_provenance")
    indexing_provenance_eligible = isinstance(
        indexing_provenance, Mapping
    ) and verify_decision_provenance(
        indexing_provenance,
        root=root,
        profile_path=root / "config/benchmark-2.19.toml",
        environment_path=(
            root / "results/environment/benchmark-profile-opensearch-2.19.json"
        ),
        require_current_code_revision=False,
    )
    expected_quality = validation_quality and indexing_provenance_eligible
    expected_reason: str | None = None
    if not validation_quality:
        expected_reason = cast(
            str | None, validation.get("quality_ineligibility_reason")
        )
    elif not indexing_provenance_eligible:
        expected_reason = (
            "index build did not retain current decision-eligible start provenance"
        )
    if (
        manifest.get("indexing_provenance_eligible_for_decision")
        is not indexing_provenance_eligible
        or manifest.get("quality_evidence_eligible_for_decision")
        is not expected_quality
        or manifest.get("quality_ineligibility_reason") != expected_reason
        or manifest.get("latency_evidence_eligible_for_decision") is not False
    ):
        raise DatasetIntegrityError("TREC sparse index evidence eligibility differs")
    if manifest.get("primary_store_bytes") != _primary_store_bytes(
        client, sparse_spec.index_name
    ):
        raise DatasetIntegrityError("TREC sparse store size differs")
    return manifest


def verify_trec_sparse_manifest_metadata(
    manifest: dict[str, Any],
    *,
    dataset_spec: TrecProductSearchDatasetSpec,
    sparse_spec: TrecSparseSpec,
) -> None:
    source_corpus = manifest.get("source_corpus")
    expected_document_model = {
        "name": sparse_spec.document_model_name,
        "version": sparse_spec.document_model_version,
        "format": sparse_spec.document_model_format,
        "package_sha256": sparse_spec.document_model_sha256,
        "package_bytes": sparse_spec.document_model_bytes,
        "maximum_token_length": sparse_spec.maximum_token_length,
        "maximum_value_ratio": sparse_spec.maximum_value_ratio,
    }
    expected_query_tokenizer = {
        "name": sparse_spec.query_tokenizer_name,
        "version": sparse_spec.query_tokenizer_version,
        "format": sparse_spec.query_tokenizer_format,
        "content_sha256": sparse_spec.query_tokenizer_content_sha256,
        "content_bytes": sparse_spec.query_tokenizer_content_bytes,
    }
    expected = {
        "schema_version": 2,
        "dataset": "TREC Product Search 2024",
        "source_revision": dataset_spec.revision,
        "opensearch_version": sparse_spec.expected_opensearch_version,
        "document_model": expected_document_model,
        "query_tokenizer": expected_query_tokenizer,
        "bulk_request_size": sparse_spec.bulk_request_size,
        "refresh_interval": "1s",
        "vector_state": "document_only_neural_sparse_rank_features",
        "index_write_block": INDEX_WRITE_BLOCK_EVIDENCE,
    }
    if any(manifest.get(key) != value for key, value in expected.items()) or (
        not isinstance(source_corpus, dict)
        or source_corpus.get("records") != dataset_spec.expected_products
    ):
        raise DatasetIntegrityError("TREC sparse index metadata differs")


def _verify_precompute(
    *,
    root: Path,
    dataset_spec: TrecProductSearchDatasetSpec,
    sparse_spec: TrecSparseSpec,
    corpus_path: Path,
    embeddings_path: Path,
    manifest_path: Path,
    validation_manifest_path: Path,
    package_path: Path,
    model_path: Path,
    tokenizer_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = cast(dict[str, Any], read_json(manifest_path))
    validation = verify_trec_sparse_validation_manifest(
        root=root,
        dataset_spec=dataset_spec,
        sparse_spec=sparse_spec,
        corpus_path=corpus_path,
        embeddings_path=embeddings_path,
        completion_manifest_path=manifest_path,
        package_path=package_path,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        validation_manifest_path=validation_manifest_path,
    )
    return manifest, validation


def _primary_store_bytes(client: OpenSearchClient, index: str) -> int:
    response = cast(
        dict[str, Any], client.request("GET", f"/{index}/_stats/store")
    )
    primaries = cast(dict[str, Any], cast(dict[str, Any], response["_all"])["primaries"])
    return int(cast(dict[str, Any], primaries["store"])["size_in_bytes"])


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }
