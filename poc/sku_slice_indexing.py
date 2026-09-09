from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from poc.datasets import DatasetIntegrityError, WandsDatasetSpec, file_facts
from poc.index_content import verify_live_index_content
from poc.index_evidence import (
    INDEX_WRITE_BLOCK_EVIDENCE,
    lock_live_index_for_decisions,
    verify_live_index_settings,
)
from poc.manifest import canonical_sha256, collect_index_facts, read_json, write_json
from poc.neural_sparse import (
    NeuralSparseSpec,
    verify_wands_sparse_precompute,
)
from poc.os_client import OpenSearchClient
from poc.provenance import (
    collect_manifest_provenance,
    require_registered_opensearch_client,
    verify_decision_provenance,
)
from poc.sku_slice import (
    SkuSliceSpec,
    build_sku_index_definition,
    build_sku_slice_records,
    iter_sku_sparse_documents,
)
from poc.trec import identifier_sort_key


def build_sku_slice_index(
    client: OpenSearchClient,
    *,
    root: Path,
    dataset_spec: WandsDatasetSpec,
    sparse_spec: NeuralSparseSpec,
    sku_spec: SkuSliceSpec,
    products_path: Path,
    embeddings_path: Path,
    preparation_path: Path,
    precompute_manifest_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    indexing_provenance = collect_manifest_provenance(
        root,
        profile_path=root / "config/benchmark.toml",
        environment_path=root / "results/environment/benchmark-profile.json",
    )
    version = client.wait_until_ready(expected_version="3.8.0")
    precompute, upstream_eligible, upstream_reason = _verify_sku_upstreams(
        root=root,
        dataset_spec=dataset_spec,
        sparse_spec=sparse_spec,
        sku_spec=sku_spec,
        products_path=products_path,
        embeddings_path=embeddings_path,
        preparation_path=preparation_path,
        precompute_manifest_path=precompute_manifest_path,
    )
    definition = build_sku_index_definition(sparse_spec, sku_spec)
    client.delete_index(sku_spec.index_name)
    client.create_index(sku_spec.index_name, definition)
    started = time.perf_counter()
    client.bulk_index(
        sku_spec.index_name,
        iter_sku_sparse_documents(
            products_path, embeddings_path, sparse_spec, sku_spec
        ),
        batch_size=sparse_spec.bulk_request_size,
    )
    bulk_seconds = time.perf_counter() - started
    client.refresh(sku_spec.index_name)
    client.force_merge(sku_spec.index_name)
    client.update_index_settings(
        sku_spec.index_name,
        {"index": {"refresh_interval": "1s"}},
    )
    index_write_block = lock_live_index_for_decisions(
        client,
        index=sku_spec.index_name,
    )
    facts = collect_index_facts(client, sku_spec.index_name)
    if facts.document_count != dataset_spec.expected_products:
        raise DatasetIntegrityError("synthetic SKU index count differs")
    if facts.segment_count != 1 or facts.deleted_document_count != 0:
        raise DatasetIntegrityError("synthetic SKU index must have one clean segment")
    index_content = verify_live_index_content(
        client,
        index=sku_spec.index_name,
        expected_documents=iter_sku_sparse_documents(
            products_path,
            embeddings_path,
            sparse_spec,
            sku_spec,
        ),
        expected_count=dataset_spec.expected_products,
        float32_fields=(sparse_spec.embedding_field,),
    )
    indexing_eligible = verify_decision_provenance(
        indexing_provenance,
        root=root,
        profile_path=root / "config/benchmark.toml",
        environment_path=root / "results/environment/benchmark-profile.json",
        require_current_code_revision=True,
    )
    latency_eligible = upstream_eligible and indexing_eligible
    ineligibility_reason = upstream_reason
    if upstream_eligible and not indexing_eligible:
        ineligibility_reason = (
            "SKU index build did not retain current decision-eligible start provenance"
        )
    manifest: dict[str, object] = {
        "schema_version": 2,
        "dataset": sku_spec.dataset,
        "source_revision": dataset_spec.revision,
        "opensearch_version": version,
        "index": asdict(facts),
        "index_definition_sha256": canonical_sha256(definition),
        "index_content": index_content,
        "index_write_block": index_write_block,
        "source_products": _artifact(root, products_path),
        "precomputed_sparse_embeddings": _artifact(root, embeddings_path),
        "precompute_manifest": {
            **_artifact(root, precompute_manifest_path),
            "schema_version": precompute["schema_version"],
            "generation_provenance_eligible_for_decision": upstream_eligible,
        },
        "slice_preparation": {
            **_artifact(root, preparation_path),
            "schema_version": 1,
            "eligible_as_synthetic_known_item_regression_evidence": True,
        },
        "neural_sparse_spec": asdict(sparse_spec),
        "sku_slice_spec": asdict(sku_spec),
        "document_model": sparse_spec.model_name,
        "query_analyzer": sparse_spec.query_analyzer,
        "bulk_index_seconds": bulk_seconds,
        "bulk_documents_per_second": dataset_spec.expected_products / bulk_seconds,
        "refresh_interval": "1s",
        "upstream_evidence_eligible_for_decision": upstream_eligible,
        "indexing_provenance_eligible_for_decision": indexing_eligible,
        "quality_evidence_eligible_for_decision": latency_eligible,
        "latency_evidence_eligible_for_decision": latency_eligible,
        "latency_ineligibility_reason": ineligibility_reason,
        "benchmark_provenance": indexing_provenance,
    }
    write_json(manifest_path, manifest)
    return manifest


def verify_sku_slice_index(
    client: OpenSearchClient,
    *,
    root: Path,
    dataset_spec: WandsDatasetSpec,
    sparse_spec: NeuralSparseSpec,
    sku_spec: SkuSliceSpec,
    products_path: Path,
    embeddings_path: Path,
    preparation_path: Path,
    precompute_manifest_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    version = client.wait_until_ready(expected_version="3.8.0")
    manifest = cast(dict[str, Any], read_json(manifest_path))
    if manifest.get("schema_version") != 2:
        raise DatasetIntegrityError("unsupported synthetic SKU index manifest schema")
    if (
        manifest.get("dataset") != sku_spec.dataset
        or manifest.get("source_revision") != dataset_spec.revision
        or manifest.get("opensearch_version") != version
        or manifest.get("document_model") != sparse_spec.model_name
        or manifest.get("query_analyzer") != sparse_spec.query_analyzer
        or manifest.get("neural_sparse_spec") != asdict(sparse_spec)
        or manifest.get("sku_slice_spec") != asdict(sku_spec)
        or manifest.get("refresh_interval") != "1s"
        or manifest.get("index_write_block") != INDEX_WRITE_BLOCK_EVIDENCE
    ):
        raise DatasetIntegrityError("synthetic SKU index identity differs")
    precompute, upstream_eligible, upstream_reason = _verify_sku_upstreams(
        root=root,
        dataset_spec=dataset_spec,
        sparse_spec=sparse_spec,
        sku_spec=sku_spec,
        products_path=products_path,
        embeddings_path=embeddings_path,
        preparation_path=preparation_path,
        precompute_manifest_path=precompute_manifest_path,
    )
    facts = collect_index_facts(client, sku_spec.index_name)
    if manifest.get("index") != asdict(facts):
        raise DatasetIntegrityError("live synthetic SKU index facts differ")
    if facts.document_count != dataset_spec.expected_products:
        raise DatasetIntegrityError("live synthetic SKU index count differs")
    if facts.segment_count != 1 or facts.deleted_document_count != 0:
        raise DatasetIntegrityError("live synthetic SKU index segment state differs")
    definition = build_sku_index_definition(sparse_spec, sku_spec)
    if manifest.get("index_definition_sha256") != canonical_sha256(definition):
        raise DatasetIntegrityError("synthetic SKU index definition differs")
    mapping_response = cast(
        dict[str, Any], client.request("GET", f"/{sku_spec.index_name}/_mapping")
    )
    live_mapping = cast(dict[str, Any], mapping_response[sku_spec.index_name])[
        "mappings"
    ]
    if live_mapping != definition["mappings"]:
        raise DatasetIntegrityError("live synthetic SKU mapping differs")
    _verify_live_sku_settings(client, sku_spec.index_name, definition)
    index_content = verify_live_index_content(
        client,
        index=sku_spec.index_name,
        expected_documents=iter_sku_sparse_documents(
            products_path,
            embeddings_path,
            sparse_spec,
            sku_spec,
        ),
        expected_count=dataset_spec.expected_products,
        float32_fields=(sparse_spec.embedding_field,),
    )
    if manifest.get("index_content") != index_content:
        raise DatasetIntegrityError("live synthetic SKU index content differs")
    for key, path in (
        ("source_products", products_path),
        ("precomputed_sparse_embeddings", embeddings_path),
    ):
        artifact = manifest.get(key)
        if artifact != _artifact(root, path):
            raise DatasetIntegrityError(f"synthetic SKU {key} differs")
    expected_precompute = {
        **_artifact(root, precompute_manifest_path),
        "schema_version": precompute["schema_version"],
        "generation_provenance_eligible_for_decision": upstream_eligible,
    }
    if manifest.get("precompute_manifest") != expected_precompute:
        raise DatasetIntegrityError("synthetic SKU precompute manifest differs")
    expected_preparation = {
        **_artifact(root, preparation_path),
        "schema_version": 1,
        "eligible_as_synthetic_known_item_regression_evidence": True,
    }
    if manifest.get("slice_preparation") != expected_preparation:
        raise DatasetIntegrityError("synthetic SKU slice preparation differs")
    provenance = manifest.get("benchmark_provenance")
    indexing_eligible = isinstance(
        provenance, Mapping
    ) and verify_decision_provenance(
        provenance,
        root=root,
        profile_path=root / "config/benchmark.toml",
        environment_path=root / "results/environment/benchmark-profile.json",
        require_current_code_revision=False,
    )
    latency_eligible = upstream_eligible and indexing_eligible
    ineligibility_reason = upstream_reason
    if upstream_eligible and not indexing_eligible:
        ineligibility_reason = (
            "SKU index build did not retain current decision-eligible start provenance"
        )
    if (
        manifest.get("upstream_evidence_eligible_for_decision")
        is not upstream_eligible
        or manifest.get("indexing_provenance_eligible_for_decision")
        is not indexing_eligible
        or manifest.get("quality_evidence_eligible_for_decision")
        is not latency_eligible
        or manifest.get("latency_evidence_eligible_for_decision")
        is not latency_eligible
        or manifest.get("latency_ineligibility_reason") != ineligibility_reason
    ):
        raise DatasetIntegrityError("synthetic SKU index evidence eligibility differs")
    return manifest


def _verify_sku_upstreams(
    *,
    root: Path,
    dataset_spec: WandsDatasetSpec,
    sparse_spec: NeuralSparseSpec,
    sku_spec: SkuSliceSpec,
    products_path: Path,
    embeddings_path: Path,
    preparation_path: Path,
    precompute_manifest_path: Path,
) -> tuple[dict[str, Any], bool, str | None]:
    verify_sku_slice_preparation(
        root=root,
        sku_spec=sku_spec,
        products_path=products_path,
        preparation_path=preparation_path,
    )

    precompute = verify_wands_sparse_precompute(
        root=root,
        dataset=dataset_spec,
        spec=sparse_spec,
        products_path=products_path,
        embeddings_path=embeddings_path,
        manifest_path=precompute_manifest_path,
    )
    provenance_eligible = precompute.get("eligible_for_decision") is True
    return (
        precompute,
        provenance_eligible,
        None
        if provenance_eligible
        else "WANDS sparse precompute start provenance is not decision-eligible",
    )


def verify_sku_slice_preparation(
    *,
    root: Path,
    sku_spec: SkuSliceSpec,
    products_path: Path,
    preparation_path: Path,
) -> dict[str, Any]:
    preparation = cast(dict[str, Any], read_json(preparation_path))
    query_path = root / "data/prepared/wands/queries.sku-slice.jsonl"
    qrels_path = root / "data/prepared/wands/qrels.sku-slice.trec"
    if (
        preparation.get("schema_version") != 1
        or preparation.get("dataset") != sku_spec.dataset
        or preparation.get("seed") != sku_spec.seed
        or preparation.get("eligible_as_authentic_merchant_sku_evidence") is not False
        or preparation.get("eligible_as_synthetic_known_item_regression_evidence")
        is not True
        or preparation.get("source_products") != _artifact(root, products_path)
        or preparation.get("counts")
        != {
            "synthetic_sku": sku_spec.sku_queries,
            "exact_title": sku_spec.exact_title_queries,
            "total": sku_spec.query_count,
        }
        or preparation.get("queries") != _artifact(root, query_path)
        or preparation.get("qrels") != _artifact(root, qrels_path)
    ):
        raise DatasetIntegrityError("synthetic SKU slice preparation is not verified")
    _verify_deterministic_sku_queries(
        products_path=products_path,
        query_path=query_path,
        qrels_path=qrels_path,
        sku_spec=sku_spec,
    )
    return preparation


def _verify_deterministic_sku_queries(
    *,
    products_path: Path,
    query_path: Path,
    qrels_path: Path,
    sku_spec: SkuSliceSpec,
) -> None:
    records, qrels = build_sku_slice_records(products_path, sku_spec)
    expected_queries = "".join(
        json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        for record in records
    )
    if query_path.read_text() != expected_queries:
        raise DatasetIntegrityError(
            "synthetic SKU deterministic query content differs"
        )
    expected_qrels = "".join(
        record.line()
        for record in sorted(
            qrels,
            key=lambda record: (
                identifier_sort_key(record.query_id),
                identifier_sort_key(record.document_id),
            ),
        )
    )
    if qrels_path.read_text() != expected_qrels:
        raise DatasetIntegrityError("synthetic SKU deterministic qrels content differs")


def _verify_live_sku_settings(
    client: OpenSearchClient,
    index: str,
    definition: dict[str, Any],
) -> None:
    try:
        verify_live_index_settings(
            client,
            index=index,
            definition=definition,
            expected_refresh_interval="1s",
        )
    except DatasetIntegrityError as error:
        raise DatasetIntegrityError(
            f"live synthetic SKU settings differ: {error}"
        ) from error


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }
