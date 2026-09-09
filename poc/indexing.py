from __future__ import annotations

import json
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from poc.config import ConfigError, ModelSpec
from poc.datasets import DatasetIntegrityError, WandsDatasetSpec, file_facts
from poc.embedding_cache import load_cached_vectors, verify_embedding_cache
from poc.index_content import verify_live_index_content
from poc.index_evidence import (
    INDEX_WRITE_BLOCK_EVIDENCE,
    lock_live_index_for_decisions,
    verify_live_index_settings,
)
from poc.manifest import canonical_sha256, collect_index_facts, read_json
from poc.os_client import OpenSearchClient
from poc.provenance import (
    collect_code_revision,
    collect_manifest_provenance,
    require_registered_opensearch_client,
    verify_decision_provenance,
)
from poc.wands import verify_prepared_wands

EXPECTED_OS_VERSION = "3.8.0"
_PROFILE_RELATIVE_PATH = Path("config/benchmark.toml")
_ENVIRONMENT_RELATIVE_PATH = Path(
    "results/environment/benchmark-profile.json"
)
_BOUND_PREPARED_FILES = (
    "products.jsonl",
    "queries.all.jsonl",
    "queries.dev.jsonl",
    "queries.test.jsonl",
    "qrels.all.trec",
    "qrels.dev.trec",
    "qrels.test.trec",
)


@dataclass(frozen=True, slots=True)
class WandsIndexSpec:
    name: str
    shards: int
    replicas: int


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


def verify_wands_preparation_evidence(
    *,
    root: Path,
    dataset_spec: WandsDatasetSpec,
    prepared_directory: Path,
) -> dict[str, object]:
    raw_directory = root / "data/raw/wands"
    verification = verify_prepared_wands(
        dataset_spec,
        raw_directory,
        prepared_directory,
    )
    return {
        "dataset_registry": _artifact(root, root / "config/datasets.toml"),
        "raw_files": {
            name: _artifact(root, raw_directory / file_spec.filename)
            for name, file_spec in sorted(dataset_spec.files.items())
        },
        "prepared_manifest": _artifact(
            root,
            prepared_directory / "manifest.json",
        ),
        "prepared_files": {
            name: _artifact(root, prepared_directory / name)
            for name in _BOUND_PREPARED_FILES
        },
        "semantic_verification": verification,
        "eligible_for_decision": True,
    }


def _wands_decision_provenance_valid(
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


def wands_completion_provenance_evidence(
    root: Path,
    provenance: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    completion_revision = _json_dict(asdict(collect_code_revision(root)))
    start_revision = provenance.get("code_revision")
    valid = (
        isinstance(start_revision, Mapping)
        and completion_revision == dict(start_revision)
        and _wands_decision_provenance_valid(
            root,
            provenance,
            require_current_code_revision=True,
        )
    )
    return valid, completion_revision


def wands_recorded_provenance_valid(
    root: Path,
    *,
    provenance: object,
    completion_revision: object,
) -> bool:
    if not isinstance(provenance, Mapping) or not isinstance(
        completion_revision, Mapping
    ):
        return False
    return (
        _wands_decision_provenance_valid(root, provenance)
        and completion_revision == provenance.get("code_revision")
    )


def load_wands_index_config(path: Path | str = Path("config/indexes.toml")) -> WandsIndexSpec:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    if raw.get("schema_version") != 1:
        raise ConfigError(f"unsupported index registry schema in {config_path}")
    indexes = raw.get("indexes")
    if not isinstance(indexes, dict) or not isinstance(indexes.get("wands"), dict):
        raise ConfigError(f"index registry {config_path} contains no WANDS entry")
    values = indexes["wands"]
    name = values.get("name")
    shards = values.get("shards")
    replicas = values.get("replicas")
    if not isinstance(name, str) or not name:
        raise ConfigError("WANDS index name must be a non-empty string")
    if not isinstance(shards, int) or isinstance(shards, bool) or shards <= 0:
        raise ConfigError("WANDS index shards must be a positive integer")
    if not isinstance(replicas, int) or isinstance(replicas, bool) or replicas < 0:
        raise ConfigError("WANDS index replicas must be a non-negative integer")
    return WandsIndexSpec(name=name, shards=shards, replicas=replicas)


def wands_index_definition(
    spec: WandsIndexSpec,
    *,
    vector_models: tuple[ModelSpec, ...] = (),
) -> dict[str, Any]:
    text_field: dict[str, object] = {
        "type": "text",
        "analyzer": "product_text",
        "fields": {
            "magento": {"type": "text", "analyzer": "magento_default"},
            "prefix": {"type": "text", "analyzer": "prefix_search"},
        },
    }
    properties: dict[str, Any] = {
        "average_rating": {"type": "float"},
        "brand": text_field,
        "category": text_field,
        "description": text_field,
        "features": text_field,
        "product_class": text_field,
        "product_id": {"type": "keyword"},
        "rating_count": {"type": "float"},
        "review_count": {"type": "float"},
        "title": text_field,
    }
    for model in vector_models:
        properties[vector_field_name(model)] = {
            "type": "knn_vector",
            "dimension": model.dims,
            "space_type": "cosinesimil",
            "method": {
                "name": "hnsw",
                "engine": "lucene",
                "space_type": "cosinesimil",
                "parameters": {"ef_construction": 128, "m": 16},
            },
        }
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": spec.shards,
                "number_of_replicas": spec.replicas,
                "refresh_interval": "-1",
            },
            "analysis": {
                "filter": {
                    "default_stemmer": {"type": "stemmer", "language": "english"},
                    "unique_stem": {"type": "unique", "only_on_same_position": True},
                },
                "analyzer": {
                    "product_text": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding"],
                    },
                    "magento_default": {
                        "type": "custom",
                        "char_filter": ["html_strip"],
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "keyword_repeat",
                            "asciifolding",
                            "default_stemmer",
                            "unique_stem",
                        ],
                    },
                    "prefix_search": {
                        "type": "custom",
                        "char_filter": ["html_strip"],
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding"],
                    },
                }
            },
        },
        "mappings": {
            "dynamic": "strict",
            "properties": properties,
        },
    }


def vector_field_name(model: ModelSpec) -> str:
    return f"embedding_{model.name}"


def iter_prepared_products(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = cast(dict[str, Any], json.loads(line))
            product_id = str(record.get("product_id", ""))
            if not product_id:
                raise DatasetIntegrityError(f"missing product_id in {path}:{line_number}")
            yield product_id, record


def iter_products_with_vectors(
    path: Path,
    vectors: dict[str, tuple[list[str], NDArray[np.float32]]],
) -> Iterator[tuple[str, dict[str, Any]]]:
    for row_index, (product_id, record) in enumerate(iter_prepared_products(path)):
        document = dict(record)
        for model_name, (document_ids, model_vectors) in vectors.items():
            if row_index >= len(document_ids) or document_ids[row_index] != product_id:
                raise DatasetIntegrityError(
                    f"embedding order differs for {model_name} at product {product_id}"
                )
            document[f"embedding_{model_name}"] = model_vectors[row_index].tolist()
        yield product_id, document


def build_wands_lexical_index(
    client: OpenSearchClient,
    *,
    dataset_spec: WandsDatasetSpec,
    index_spec: WandsIndexSpec,
    prepared_directory: Path,
    manifest_path: Path,
) -> dict[str, object]:
    root = manifest_path.parents[2]
    require_registered_opensearch_client(client, root / _PROFILE_RELATIVE_PATH)
    benchmark_provenance = _json_dict(collect_manifest_provenance(root))
    preparation_evidence = _json_dict(
        verify_wands_preparation_evidence(
            root=root,
            dataset_spec=dataset_spec,
            prepared_directory=prepared_directory,
        )
    )
    version = client.wait_until_ready(expected_version=EXPECTED_OS_VERSION)
    prepared_manifest = cast(dict[str, Any], read_json(prepared_directory / "manifest.json"))
    prepared_outputs = cast(dict[str, Any], prepared_manifest["outputs"])
    products_facts = cast(dict[str, Any], prepared_outputs["products.jsonl"])
    definition = wands_index_definition(index_spec)

    client.delete_index(index_spec.name)
    client.create_index(index_spec.name, definition)
    client.bulk_index(
        index_spec.name,
        iter_prepared_products(prepared_directory / "products.jsonl"),
    )
    client.refresh(index_spec.name)
    client.force_merge(index_spec.name)
    client.update_index_settings(
        index_spec.name,
        {"index": {"refresh_interval": "1s"}},
    )
    index_write_block = lock_live_index_for_decisions(
        client,
        index=index_spec.name,
    )
    facts = collect_index_facts(client, index_spec.name)
    if facts.document_count != dataset_spec.expected_products:
        raise DatasetIntegrityError(
            f"WANDS index contains {facts.document_count} products; "
            f"expected {dataset_spec.expected_products}"
        )
    if facts.segment_count != 1 or facts.deleted_document_count != 0:
        raise DatasetIntegrityError("WANDS index must have one segment and no deleted documents")
    index_content = verify_live_index_content(
        client,
        index=index_spec.name,
        expected_documents=iter_prepared_products(
            prepared_directory / "products.jsonl"
        ),
        expected_count=dataset_spec.expected_products,
    )

    provenance_eligible, completion_revision = wands_completion_provenance_evidence(
        root,
        benchmark_provenance,
    )
    quality_eligible = (
        provenance_eligible
        and preparation_evidence.get("eligible_for_decision") is True
    )
    manifest: dict[str, object] = {
        "schema_version": 2,
        "dataset": "WANDS",
        "source_revision": dataset_spec.revision,
        "opensearch_version": version,
        "index": asdict(facts),
        "index_definition_sha256": canonical_sha256(definition),
        "index_content": index_content,
        "index_write_block": index_write_block,
        "prepared_products": {
            "sha256": str(products_facts["sha256"]),
            "records": int(products_facts["records"]),
        },
        "preparation_evidence": preparation_evidence,
        "vector_state": "fields_not_added",
        "vector_cache_bindings_verified": True,
        "refresh_interval": "1s",
        "benchmark_provenance": benchmark_provenance,
        "completion_code_revision": completion_revision,
        "quality_evidence_eligible_for_decision": quality_eligible,
        "latency_evidence_eligible_for_decision": False,
    }
    persisted_manifest = _json_dict(manifest)
    _write_json_exact(manifest_path, persisted_manifest)
    return cast(dict[str, object], persisted_manifest)


def build_wands_vector_index(
    client: OpenSearchClient,
    *,
    dataset_spec: WandsDatasetSpec,
    index_spec: WandsIndexSpec,
    models: tuple[ModelSpec, ...],
    root: Path,
    prepared_directory: Path,
    manifest_path: Path,
) -> dict[str, object]:
    require_registered_opensearch_client(client, root / _PROFILE_RELATIVE_PATH)
    benchmark_provenance = _json_dict(collect_manifest_provenance(root))
    preparation_evidence = _json_dict(
        verify_wands_preparation_evidence(
            root=root,
            dataset_spec=dataset_spec,
            prepared_directory=prepared_directory,
        )
    )
    if not models:
        raise ValueError("at least one vector model is required")
    products_path = prepared_directory / "products.jsonl"
    vectors: dict[str, tuple[list[str], NDArray[np.float32]]] = {}
    vector_manifests: list[dict[str, object]] = []
    for model in models:
        verify_embedding_cache(root=root, products_path=products_path, model=model)
        vectors[model.name] = load_cached_vectors(
            root=root,
            products_path=products_path,
            model=model,
        )
        embedding_manifest_path = (
            root / f"results/wands/embeddings/{model.name}.manifest.json"
        )
        embedding_manifest = cast(dict[str, Any], read_json(embedding_manifest_path))
        vector_manifests.append(
            {
                "field": vector_field_name(model),
                "model": asdict(model),
                "model_artifact_sha256": embedding_manifest["model_artifact_sha256"],
                "encoder_runtime": embedding_manifest["encoder_runtime"],
                "embedding_manifest_sha256": file_facts(embedding_manifest_path).sha256,
                "embedding_manifest": _artifact(root, embedding_manifest_path),
            }
        )

    version = client.wait_until_ready(expected_version=EXPECTED_OS_VERSION)
    prepared_manifest = cast(dict[str, Any], read_json(prepared_directory / "manifest.json"))
    prepared_outputs = cast(dict[str, Any], prepared_manifest["outputs"])
    products_facts = cast(dict[str, Any], prepared_outputs["products.jsonl"])
    definition = wands_index_definition(index_spec, vector_models=models)
    client.delete_index(index_spec.name)
    client.create_index(index_spec.name, definition)
    client.bulk_index(
        index_spec.name,
        iter_products_with_vectors(products_path, vectors),
    )
    client.refresh(index_spec.name)
    client.force_merge(index_spec.name)
    client.update_index_settings(index_spec.name, {"index": {"refresh_interval": "1s"}})
    index_write_block = lock_live_index_for_decisions(
        client,
        index=index_spec.name,
    )
    facts = collect_index_facts(client, index_spec.name)
    if facts.document_count != dataset_spec.expected_products:
        raise DatasetIntegrityError("WANDS vector index has the wrong product count")
    if facts.segment_count != 1 or facts.deleted_document_count != 0:
        raise DatasetIntegrityError("WANDS vector index must have one segment and no deletions")
    index_content = verify_live_index_content(
        client,
        index=index_spec.name,
        expected_documents=iter_products_with_vectors(products_path, vectors),
        expected_count=dataset_spec.expected_products,
        float32_fields=tuple(vector_field_name(model) for model in models),
    )

    provenance_eligible, completion_revision = wands_completion_provenance_evidence(
        root,
        benchmark_provenance,
    )
    quality_eligible = (
        provenance_eligible
        and preparation_evidence.get("eligible_for_decision") is True
    )
    manifest: dict[str, object] = {
        "schema_version": 2,
        "dataset": "WANDS",
        "source_revision": dataset_spec.revision,
        "opensearch_version": version,
        "index": asdict(facts),
        "index_definition_sha256": canonical_sha256(definition),
        "index_content": index_content,
        "index_write_block": index_write_block,
        "prepared_products": {
            "sha256": str(products_facts["sha256"]),
            "records": int(products_facts["records"]),
        },
        "preparation_evidence": preparation_evidence,
        "vector_state": "loaded",
        "vector_models": vector_manifests,
        "vector_cache_bindings_verified": True,
        "refresh_interval": "1s",
        "benchmark_provenance": benchmark_provenance,
        "completion_code_revision": completion_revision,
        "quality_evidence_eligible_for_decision": quality_eligible,
        "latency_evidence_eligible_for_decision": False,
    }
    persisted_manifest = _json_dict(manifest)
    _write_json_exact(manifest_path, persisted_manifest)
    return cast(dict[str, object], persisted_manifest)


def verify_wands_lexical_index(
    client: OpenSearchClient,
    *,
    dataset_spec: WandsDatasetSpec,
    index_spec: WandsIndexSpec,
    prepared_directory: Path,
    manifest_path: Path,
) -> dict[str, object]:
    root = manifest_path.parents[2]
    require_registered_opensearch_client(client, root / _PROFILE_RELATIVE_PATH)
    client.wait_until_ready(expected_version=EXPECTED_OS_VERSION)
    manifest = cast(dict[str, Any], read_json(manifest_path))
    if manifest.get("schema_version") != 2 or manifest.get("dataset") != "WANDS":
        raise DatasetIntegrityError("WANDS index manifest schema or dataset differs")
    if manifest.get("source_revision") != dataset_spec.revision:
        raise DatasetIntegrityError("WANDS index manifest source revision is stale")
    preparation_evidence = _json_dict(
        verify_wands_preparation_evidence(
            root=root,
            dataset_spec=dataset_spec,
            prepared_directory=prepared_directory,
        )
    )
    if manifest.get("preparation_evidence") != preparation_evidence:
        raise DatasetIntegrityError("WANDS index preparation evidence differs")
    vector_entries = cast(list[dict[str, Any]], manifest.get("vector_models", []))
    vector_models = tuple(
        ModelSpec.from_mapping(str(entry["model"]["name"]), cast(dict[str, Any], entry["model"]))
        for entry in vector_entries
    )
    expected_vector_entries: list[dict[str, object]] = []
    expected_vectors: dict[
        str, tuple[list[str], NDArray[np.float32]]
    ] = {}
    products_path = prepared_directory / "products.jsonl"
    for model in vector_models:
        verify_embedding_cache(
            root=root,
            products_path=products_path,
            model=model,
        )
        expected_vectors[model.name] = load_cached_vectors(
            root=root,
            products_path=products_path,
            model=model,
        )
        embedding_manifest_path = (
            root / f"results/wands/embeddings/{model.name}.manifest.json"
        )
        embedding_manifest = cast(
            dict[str, Any], read_json(embedding_manifest_path)
        )
        expected_vector_entries.append(
            {
                "field": vector_field_name(model),
                "model": asdict(model),
                "model_artifact_sha256": embedding_manifest[
                    "model_artifact_sha256"
                ],
                "encoder_runtime": embedding_manifest["encoder_runtime"],
                "embedding_manifest_sha256": file_facts(
                    embedding_manifest_path
                ).sha256,
                "embedding_manifest": _artifact(root, embedding_manifest_path),
            }
        )
    if (
        vector_entries != _json_dict({"entries": expected_vector_entries})["entries"]
        or manifest.get("vector_cache_bindings_verified") is not True
    ):
        raise DatasetIntegrityError("WANDS vector cache bindings differ")
    definition = wands_index_definition(index_spec, vector_models=vector_models)
    if manifest.get("index_definition_sha256") != canonical_sha256(definition):
        raise DatasetIntegrityError("WANDS index definition differs from its manifest")
    if manifest.get("index_write_block") != INDEX_WRITE_BLOCK_EVIDENCE:
        raise DatasetIntegrityError("WANDS index write block evidence differs")
    products_facts = file_facts(prepared_directory / "products.jsonl")
    manifest_products = cast(dict[str, Any], manifest["prepared_products"])
    if manifest_products.get("sha256") != products_facts.sha256:
        raise DatasetIntegrityError("WANDS index manifest references different product bytes")

    facts = collect_index_facts(client, index_spec.name)
    manifest_index = cast(dict[str, Any], manifest["index"])
    if asdict(facts) != manifest_index:
        raise DatasetIntegrityError("live WANDS index facts differ from its manifest")
    if facts.document_count != dataset_spec.expected_products:
        raise DatasetIntegrityError("live WANDS index has the wrong product count")

    mapping = cast(dict[str, Any], client.request("GET", f"/{index_spec.name}/_mapping"))
    live_mapping = cast(dict[str, Any], mapping[index_spec.name])["mappings"]
    if live_mapping != definition["mappings"]:
        raise DatasetIntegrityError("live WANDS mapping differs from the registered mapping")
    verify_live_index_settings(
        client,
        index=index_spec.name,
        definition=definition,
    )
    for model in vector_models:
        count = cast(
            dict[str, Any],
            client.request(
                "GET",
                f"/{index_spec.name}/_count",
                json_body={"query": {"exists": {"field": vector_field_name(model)}}},
            ),
        )
        if int(count.get("count", -1)) != dataset_spec.expected_products:
            raise DatasetIntegrityError(f"live WANDS index is missing {model.name} vectors")
    expected_documents = (
        iter_products_with_vectors(products_path, expected_vectors)
        if vector_models
        else iter_prepared_products(products_path)
    )
    index_content = verify_live_index_content(
        client,
        index=index_spec.name,
        expected_documents=expected_documents,
        expected_count=dataset_spec.expected_products,
        float32_fields=tuple(
            vector_field_name(model) for model in vector_models
        ),
    )
    if manifest.get("index_content") != index_content:
        raise DatasetIntegrityError("live WANDS index content differs from its manifest")
    quality_eligible = (
        wands_recorded_provenance_valid(
            root,
            provenance=manifest.get("benchmark_provenance"),
            completion_revision=manifest.get("completion_code_revision"),
        )
        and preparation_evidence.get("eligible_for_decision") is True
    )
    if (
        manifest.get("quality_evidence_eligible_for_decision") is not quality_eligible
        or manifest.get("latency_evidence_eligible_for_decision") is not False
    ):
        raise DatasetIntegrityError("WANDS index quality eligibility differs")
    return {
        "index": facts.name,
        "uuid": facts.uuid,
        "documents": facts.document_count,
        "segments": facts.segment_count,
        "deleted_documents": facts.deleted_document_count,
        "vector_models": [model.name for model in vector_models],
        "quality_evidence_eligible_for_decision": quality_eligible,
        "latency_evidence_eligible_for_decision": False,
    }
