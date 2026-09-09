from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from poc.config import ModelSpec, load_model_registry
from poc.datasets import (
    DatasetIntegrityError,
    WandsDatasetSpec,
    file_facts,
    load_wands_config,
)
from poc.embedding_cache import load_cached_vectors, verify_embedding_cache
from poc.index_content import verify_live_index_content
from poc.index_evidence import (
    INDEX_WRITE_BLOCK_EVIDENCE,
    lock_live_index_for_decisions,
    verify_live_index_settings,
)
from poc.indexing import (
    WandsIndexSpec,
    vector_field_name,
    verify_wands_preparation_evidence,
    wands_completion_provenance_evidence,
    wands_index_definition,
    wands_recorded_provenance_valid,
)
from poc.manifest import canonical_sha256, collect_index_facts, read_json
from poc.neural_sparse import (
    NeuralSparseSpec,
    iter_precomputed_sparse_documents,
    load_neural_sparse_spec,
    verify_wands_sparse_precompute,
)
from poc.os_client import OpenSearchClient
from poc.provenance import collect_manifest_provenance, require_registered_opensearch_client
from poc.three_way import ThreeWaySpec, load_three_way_spec

EXPECTED_OS_VERSION = "3.8.0"
_PROFILE_PATH = Path("config/benchmark.toml")
_ENVIRONMENT_PATH = Path("results/environment/benchmark-profile.json")
_PRODUCTS_PATH = Path("data/prepared/wands/products.jsonl")
_SPARSE_EMBEDDINGS_PATH = Path(
    "data/cache/neural-sparse/wands-doc-v3-distill.jsonl"
)
_SPARSE_MANIFEST_PATH = Path(
    "results/wands/neural-sparse/precompute-manifest.json"
)
_INDEX_MANIFEST_PATH = Path("results/wands/three-way/index-manifest.json")
def _json_dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    return cast(dict[str, Any], json.loads(json.dumps(value)))


def _write_json_exact(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_registered_three_way_index_inputs(
    *,
    root: Path,
    dataset: WandsDatasetSpec,
    spec: ThreeWaySpec,
    sparse: NeuralSparseSpec,
    model: ModelSpec,
) -> dict[str, Any]:
    registered_dataset = load_wands_config(root / "config/datasets.toml")
    registered_spec = load_three_way_spec(root / "config/three_way.toml")
    registered_sparse = load_neural_sparse_spec(root / "config/neural_sparse.toml")
    model_registry = load_model_registry(root / "config/models.toml")
    if dataset != registered_dataset:
        raise DatasetIntegrityError(
            "three-way dataset differs from registered WANDS configuration"
        )
    if spec != registered_spec:
        raise DatasetIntegrityError("three-way configuration differs from registry")
    if sparse != registered_sparse:
        raise DatasetIntegrityError(
            "three-way sparse configuration differs from registry"
        )
    if model.name != spec.dense_model or model_registry.get(spec.dense_model) != model:
        raise DatasetIntegrityError(
            "three-way dense model differs from registered configuration"
        )
    return _json_dict(
        {
            "dataset": asdict(registered_dataset),
            "three_way": asdict(registered_spec),
            "neural_sparse": asdict(registered_sparse),
            "dense_model": asdict(model),
            "config_artifacts": {
                "datasets": _artifact(root, root / "config/datasets.toml"),
                "models": _artifact(root, root / "config/models.toml"),
                "neural_sparse": _artifact(
                    root, root / "config/neural_sparse.toml"
                ),
                "three_way": _artifact(root, root / "config/three_way.toml"),
            },
        }
    )


def verify_three_way_sparse_precompute(
    *,
    root: Path,
    dataset: WandsDatasetSpec,
    sparse: NeuralSparseSpec,
    products_path: Path,
    embeddings_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    return verify_wands_sparse_precompute(
        root=root,
        dataset=dataset,
        spec=sparse,
        products_path=products_path,
        embeddings_path=embeddings_path,
        manifest_path=manifest_path,
    )


def derive_three_way_index_eligibility(
    *,
    provenance_valid: object,
    preparation_eligible: object,
    dense_cache_eligible: object,
    sparse_precompute_eligible: object,
    upstream_unchanged: object,
    live_index_clean: object,
) -> tuple[bool, list[str]]:
    checks = (
        (
            provenance_valid is True,
            "three-way index has no valid clean committed start provenance",
        ),
        (
            preparation_eligible is True,
            "WANDS preparation evidence is not decision eligible",
        ),
        (
            dense_cache_eligible is True,
            "three-way dense cache evidence is not decision eligible",
        ),
        (
            sparse_precompute_eligible is True,
            "three-way sparse precompute evidence is not decision eligible",
        ),
        (
            upstream_unchanged is True,
            "three-way index upstream evidence changed during build",
        ),
        (
            live_index_clean is True,
            "three-way live index is not a single clean exact index",
        ),
    )
    reasons = [reason for passed, reason in checks if not passed]
    return not reasons, reasons


def build_three_way_index_definition(
    spec: ThreeWaySpec,
    sparse: NeuralSparseSpec,
    model: ModelSpec,
) -> dict[str, Any]:
    definition = wands_index_definition(
        WandsIndexSpec(name=spec.index_name, shards=1, replicas=0),
        vector_models=(model,),
    )
    properties = definition["mappings"]["properties"]
    properties[sparse.text_field] = {"type": "text", "index": False}
    properties[sparse.embedding_field] = {"type": "rank_features"}
    return definition


def iter_three_way_documents(
    products_path: Path,
    sparse_embeddings_path: Path,
    sparse: NeuralSparseSpec,
    model: ModelSpec,
    dense_document_ids: list[str],
    dense_vectors: NDArray[np.float32],
) -> Iterator[tuple[str, dict[str, Any]]]:
    count = 0
    for row, (document_id, document) in enumerate(
        iter_precomputed_sparse_documents(
            products_path,
            sparse_embeddings_path,
            sparse,
        )
    ):
        if row >= len(dense_document_ids) or dense_document_ids[row] != document_id:
            raise DatasetIntegrityError(
                f"three-way dense embedding order differs at {document_id}"
            )
        output = dict(document)
        output[vector_field_name(model)] = dense_vectors[row].tolist()
        count = row + 1
        yield document_id, output
    if len(dense_document_ids) != count:
        raise DatasetIntegrityError("three-way dense embedding count differs")


def build_three_way_index(
    client: OpenSearchClient,
    *,
    root: Path,
    dataset: WandsDatasetSpec,
    spec: ThreeWaySpec,
    sparse: NeuralSparseSpec,
    model: ModelSpec,
    products_path: Path,
    sparse_embeddings_path: Path,
    sparse_manifest_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    require_registered_opensearch_client(client, root / _PROFILE_PATH)
    benchmark_provenance = _json_dict(
        collect_manifest_provenance(
            root,
            profile_path=root / _PROFILE_PATH,
            environment_path=root / _ENVIRONMENT_PATH,
        )
    )
    _verify_canonical_paths(
        root=root,
        products_path=products_path,
        sparse_embeddings_path=sparse_embeddings_path,
        sparse_manifest_path=sparse_manifest_path,
        manifest_path=manifest_path,
    )
    registered_inputs = verify_registered_three_way_index_inputs(
        root=root,
        dataset=dataset,
        spec=spec,
        sparse=sparse,
        model=model,
    )
    upstream_evidence = _three_way_index_upstream_evidence(
        root=root,
        dataset=dataset,
        sparse=sparse,
        model=model,
        products_path=products_path,
        sparse_embeddings_path=sparse_embeddings_path,
        sparse_manifest_path=sparse_manifest_path,
        registered_inputs=registered_inputs,
    )
    version = client.wait_until_ready(expected_version=EXPECTED_OS_VERSION)
    dense_ids, dense_vectors = load_cached_vectors(
        root=root,
        products_path=products_path,
        model=model,
    )
    definition = build_three_way_index_definition(spec, sparse, model)
    client.delete_index(spec.index_name)
    client.create_index(spec.index_name, definition)
    client.bulk_index(
        spec.index_name,
        iter_three_way_documents(
            products_path,
            sparse_embeddings_path,
            sparse,
            model,
            dense_ids,
            dense_vectors,
        ),
        batch_size=sparse.bulk_request_size,
    )
    client.refresh(spec.index_name)
    client.force_merge(spec.index_name)
    client.update_index_settings(
        spec.index_name,
        {"index": {"refresh_interval": "1s"}},
    )
    index_write_block = lock_live_index_for_decisions(
        client,
        index=spec.index_name,
    )
    facts = _verify_live_three_way_index(
        client,
        dataset=dataset,
        spec=spec,
        model=model,
        definition=definition,
    )
    index_content = verify_live_index_content(
        client,
        index=spec.index_name,
        expected_documents=iter_three_way_documents(
            products_path,
            sparse_embeddings_path,
            sparse,
            model,
            dense_ids,
            dense_vectors,
        ),
        expected_count=dataset.expected_products,
        float32_fields=(
            vector_field_name(model),
            sparse.embedding_field,
        ),
    )
    completion_upstream_evidence = _three_way_index_upstream_evidence(
        root=root,
        dataset=dataset,
        sparse=sparse,
        model=model,
        products_path=products_path,
        sparse_embeddings_path=sparse_embeddings_path,
        sparse_manifest_path=sparse_manifest_path,
        registered_inputs=registered_inputs,
    )
    upstream_unchanged = completion_upstream_evidence == upstream_evidence
    provenance_valid, completion_revision = wands_completion_provenance_evidence(
        root,
        benchmark_provenance,
    )
    eligible, reasons = derive_three_way_index_eligibility(
        provenance_valid=provenance_valid,
        preparation_eligible=upstream_evidence["preparation"][
            "eligible_for_decision"
        ],
        dense_cache_eligible=upstream_evidence["dense_embedding_cache"][
            "eligible_for_decision"
        ],
        sparse_precompute_eligible=upstream_evidence["sparse_precompute"][
            "eligible_for_decision"
        ],
        upstream_unchanged=upstream_unchanged,
        live_index_clean=True,
    )
    manifest: dict[str, object] = {
        "schema_version": 2,
        "dataset": "WANDS",
        "source_revision": dataset.revision,
        "opensearch_version": version,
        "index": asdict(facts),
        "index_definition_sha256": canonical_sha256(definition),
        "index_content": index_content,
        "index_write_block": index_write_block,
        "registered_inputs": registered_inputs,
        "upstream_evidence": upstream_evidence,
        "completion_upstream_evidence": completion_upstream_evidence,
        "upstream_evidence_unchanged": upstream_unchanged,
        "dense_model": asdict(model),
        "sparse_model": sparse.model_name,
        "query_analyzer": sparse.query_analyzer,
        "vector_state": "exact_dense_and_document_neural_sparse",
        "refresh_interval": "1s",
        "benchmark_provenance": benchmark_provenance,
        "benchmark_provenance_valid": provenance_valid,
        "completion_code_revision": completion_revision,
        "quality_evidence_eligible_for_decision": eligible,
        "quality_ineligibility_reasons": reasons,
        "latency_evidence_eligible_for_decision": False,
        "latency_ineligibility_reason": (
            "index construction is not end-to-end query latency evidence"
        ),
    }
    persisted = _json_dict(manifest)
    _write_json_exact(manifest_path, persisted)
    return cast(dict[str, object], persisted)


def verify_three_way_index(
    client: OpenSearchClient,
    *,
    root: Path,
    dataset: WandsDatasetSpec,
    spec: ThreeWaySpec,
    sparse: NeuralSparseSpec,
    model: ModelSpec,
    products_path: Path,
    sparse_embeddings_path: Path,
    sparse_manifest_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    require_registered_opensearch_client(client, root / _PROFILE_PATH)
    _verify_canonical_paths(
        root=root,
        products_path=products_path,
        sparse_embeddings_path=sparse_embeddings_path,
        sparse_manifest_path=sparse_manifest_path,
        manifest_path=manifest_path,
    )
    registered_inputs = verify_registered_three_way_index_inputs(
        root=root,
        dataset=dataset,
        spec=spec,
        sparse=sparse,
        model=model,
    )
    version = client.wait_until_ready(expected_version=EXPECTED_OS_VERSION)
    manifest = cast(dict[str, Any], read_json(manifest_path))
    if (
        manifest.get("schema_version") != 2
        or manifest.get("dataset") != "WANDS"
        or manifest.get("source_revision") != dataset.revision
        or manifest.get("opensearch_version") != version
        or manifest.get("registered_inputs") != registered_inputs
        or manifest.get("dense_model") != asdict(model)
        or manifest.get("sparse_model") != sparse.model_name
        or manifest.get("query_analyzer") != sparse.query_analyzer
        or manifest.get("vector_state")
        != "exact_dense_and_document_neural_sparse"
        or manifest.get("refresh_interval") != "1s"
        or manifest.get("index_write_block") != INDEX_WRITE_BLOCK_EVIDENCE
    ):
        raise DatasetIntegrityError("three-way index identity or metadata differs")
    current_upstream = _three_way_index_upstream_evidence(
        root=root,
        dataset=dataset,
        sparse=sparse,
        model=model,
        products_path=products_path,
        sparse_embeddings_path=sparse_embeddings_path,
        sparse_manifest_path=sparse_manifest_path,
        registered_inputs=registered_inputs,
    )
    start_upstream = manifest.get("upstream_evidence")
    completion_upstream = manifest.get("completion_upstream_evidence")
    expected_unchanged = (
        isinstance(start_upstream, dict)
        and start_upstream == completion_upstream
        and completion_upstream == current_upstream
    )
    if (
        start_upstream != current_upstream
        or completion_upstream != current_upstream
        or manifest.get("upstream_evidence_unchanged") is not expected_unchanged
    ):
        raise DatasetIntegrityError("three-way index upstream evidence differs")
    definition = build_three_way_index_definition(spec, sparse, model)
    if manifest.get("index_definition_sha256") != canonical_sha256(definition):
        raise DatasetIntegrityError("three-way index definition differs")
    facts = _verify_live_three_way_index(
        client,
        dataset=dataset,
        spec=spec,
        model=model,
        definition=definition,
    )
    if manifest.get("index") != asdict(facts):
        raise DatasetIntegrityError("live three-way index facts differ")
    dense_ids, dense_vectors = load_cached_vectors(
        root=root,
        products_path=products_path,
        model=model,
    )
    index_content = verify_live_index_content(
        client,
        index=spec.index_name,
        expected_documents=iter_three_way_documents(
            products_path,
            sparse_embeddings_path,
            sparse,
            model,
            dense_ids,
            dense_vectors,
        ),
        expected_count=dataset.expected_products,
        float32_fields=(
            vector_field_name(model),
            sparse.embedding_field,
        ),
    )
    if manifest.get("index_content") != index_content:
        raise DatasetIntegrityError("live three-way index content differs")
    provenance_valid = wands_recorded_provenance_valid(
        root,
        provenance=manifest.get("benchmark_provenance"),
        completion_revision=manifest.get("completion_code_revision"),
    )
    if manifest.get("benchmark_provenance_valid") is not provenance_valid:
        raise DatasetIntegrityError("three-way index provenance validity differs")
    eligible, reasons = derive_three_way_index_eligibility(
        provenance_valid=provenance_valid,
        preparation_eligible=current_upstream["preparation"][
            "eligible_for_decision"
        ],
        dense_cache_eligible=current_upstream["dense_embedding_cache"][
            "eligible_for_decision"
        ],
        sparse_precompute_eligible=current_upstream["sparse_precompute"][
            "eligible_for_decision"
        ],
        upstream_unchanged=expected_unchanged,
        live_index_clean=True,
    )
    if (
        manifest.get("quality_evidence_eligible_for_decision") is not eligible
        or manifest.get("quality_ineligibility_reasons") != reasons
        or manifest.get("latency_evidence_eligible_for_decision") is not False
        or manifest.get("latency_ineligibility_reason")
        != "index construction is not end-to-end query latency evidence"
    ):
        raise DatasetIntegrityError("three-way index quality eligibility differs")
    return manifest


def _verify_canonical_paths(
    *,
    root: Path,
    products_path: Path,
    sparse_embeddings_path: Path,
    sparse_manifest_path: Path,
    manifest_path: Path,
) -> None:
    expected = (
        (products_path, root / _PRODUCTS_PATH, "prepared products"),
        (
            sparse_embeddings_path,
            root / _SPARSE_EMBEDDINGS_PATH,
            "sparse embeddings",
        ),
        (
            sparse_manifest_path,
            root / _SPARSE_MANIFEST_PATH,
            "sparse precompute manifest",
        ),
        (manifest_path, root / _INDEX_MANIFEST_PATH, "index manifest"),
    )
    for actual, registered, label in expected:
        if actual.resolve() != registered.resolve():
            raise DatasetIntegrityError(f"three-way {label} path is not canonical")


def _three_way_index_upstream_evidence(
    *,
    root: Path,
    dataset: WandsDatasetSpec,
    sparse: NeuralSparseSpec,
    model: ModelSpec,
    products_path: Path,
    sparse_embeddings_path: Path,
    sparse_manifest_path: Path,
    registered_inputs: dict[str, Any],
) -> dict[str, Any]:
    preparation = _json_dict(
        verify_wands_preparation_evidence(
            root=root,
            dataset_spec=dataset,
            prepared_directory=products_path.parent,
        )
    )
    verify_embedding_cache(
        root=root,
        products_path=products_path,
        model=model,
    )
    dense_manifest_path = (
        root / f"results/wands/embeddings/{model.name}.manifest.json"
    )
    dense_manifest = cast(dict[str, Any], read_json(dense_manifest_path))
    dense_evidence = {
        "artifact": _artifact(root, dense_manifest_path),
        "model": asdict(model),
        "model_artifact_sha256": dense_manifest.get("model_artifact_sha256"),
        "encoder_runtime": dense_manifest.get("encoder_runtime"),
        "eligible_for_decision": (
            dense_manifest.get("schema_version") == 2
            and dense_manifest.get("dataset") == "WANDS"
            and dense_manifest.get("quality_evidence_eligible_for_decision") is True
        ),
    }
    sparse_evidence = _json_dict(
        verify_three_way_sparse_precompute(
            root=root,
            dataset=dataset,
            sparse=sparse,
            products_path=products_path,
            embeddings_path=sparse_embeddings_path,
            manifest_path=sparse_manifest_path,
        )
    )
    eligible = all(
        evidence.get("eligible_for_decision") is True
        for evidence in (preparation, dense_evidence, sparse_evidence)
    )
    return _json_dict(
        {
            "registered_inputs": registered_inputs,
            "preparation": preparation,
            "dense_embedding_cache": dense_evidence,
            "sparse_precompute": sparse_evidence,
            "eligible_for_decision": eligible,
            "ineligibility_reasons": [
                name
                for name, evidence in (
                    ("preparation", preparation),
                    ("dense_embedding_cache", dense_evidence),
                    ("sparse_precompute", sparse_evidence),
                )
                if evidence.get("eligible_for_decision") is not True
            ],
        }
    )


def _verify_live_three_way_index(
    client: OpenSearchClient,
    *,
    dataset: WandsDatasetSpec,
    spec: ThreeWaySpec,
    model: ModelSpec,
    definition: dict[str, Any],
) -> Any:
    facts = collect_index_facts(client, spec.index_name)
    if facts.document_count != dataset.expected_products:
        raise DatasetIntegrityError("live three-way index count differs")
    if facts.segment_count != 1 or facts.deleted_document_count != 0:
        raise DatasetIntegrityError("live three-way index must contain one clean segment")
    mapping_response = cast(
        dict[str, Any], client.request("GET", f"/{spec.index_name}/_mapping")
    )
    live_mapping = cast(dict[str, Any], mapping_response[spec.index_name]).get(
        "mappings"
    )
    if live_mapping != definition["mappings"]:
        raise DatasetIntegrityError("live three-way mapping differs")
    verify_live_index_settings(
        client,
        index=spec.index_name,
        definition=definition,
    )
    dense_field = vector_field_name(model)
    count = cast(
        dict[str, Any],
        client.request(
            "GET",
            f"/{spec.index_name}/_count",
            json_body={"query": {"exists": {"field": dense_field}}},
        ),
    )
    if int(count.get("count", -1)) != dataset.expected_products:
        raise DatasetIntegrityError(
            f"live three-way index is missing values for {dense_field}"
        )
    return facts


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }
