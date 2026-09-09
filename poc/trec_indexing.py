from __future__ import annotations

import time
import tomllib
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from poc.config import ConfigError
from poc.datasets import DatasetIntegrityError, TrecProductSearchDatasetSpec, file_facts
from poc.index_content import verify_live_index_content
from poc.index_evidence import (
    INDEX_WRITE_BLOCK_EVIDENCE,
    lock_live_index_for_decisions,
    verify_live_index_settings,
)
from poc.indexing import WandsIndexSpec, wands_index_definition
from poc.manifest import canonical_sha256, collect_index_facts, read_json, write_json
from poc.os_client import OpenSearchClient
from poc.provenance import (
    collect_manifest_provenance,
    registered_opensearch_url,
    require_registered_opensearch_client,
    verify_decision_provenance,
)
from poc.trec_product_search import (
    iter_trec_products,
    verify_prepared_trec_product_search,
)


@dataclass(frozen=True, slots=True)
class TrecIndexSpec:
    name: str
    shards: int
    replicas: int


def load_trec_index_config(
    path: Path | str = Path("config/indexes.toml"),
) -> TrecIndexSpec:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    indexes = raw.get("indexes")
    values = (
        indexes.get("trec_product_search_2024")
        if isinstance(indexes, dict)
        else None
    )
    if raw.get("schema_version") != 1 or not isinstance(values, dict):
        raise ConfigError("index registry contains no TREC Product Search 2024 entry")
    prefix = values.get("name_prefix")
    shards = values.get("shards")
    replicas = values.get("replicas")
    if not isinstance(prefix, str) or not prefix:
        raise ConfigError("TREC index prefix must be a non-empty string")
    if not isinstance(shards, int) or isinstance(shards, bool) or shards <= 0:
        raise ConfigError("TREC index shards must be a positive integer")
    if not isinstance(replicas, int) or isinstance(replicas, bool) or replicas < 0:
        raise ConfigError("TREC index replicas must be a non-negative integer")
    return TrecIndexSpec(
        name=f"{prefix}-lexical-v1",
        shards=shards,
        replicas=replicas,
    )


def trec_lexical_index_definition(spec: TrecIndexSpec) -> dict[str, Any]:
    return wands_index_definition(
        WandsIndexSpec(name=spec.name, shards=spec.shards, replicas=spec.replicas)
    )


def iter_trec_index_documents(path: Path) -> Iterator[tuple[str, dict[str, str]]]:
    for document_id, document in iter_trec_products(path):
        yield document_id, {"product_id": document_id, **document}


def build_trec_lexical_index(
    client: OpenSearchClient,
    *,
    dataset_spec: TrecProductSearchDatasetSpec,
    index_spec: TrecIndexSpec,
    corpus_path: Path,
    preparation_path: Path,
    manifest_path: Path,
    expected_version: str,
    maximum_orphan_rate: float,
) -> dict[str, object]:
    root = manifest_path.parents[2]
    require_registered_opensearch_client(
        client,
        root / "config/benchmark-2.19.toml",
    )
    benchmark_provenance = collect_manifest_provenance(
        root,
        profile_path=root / "config/benchmark-2.19.toml",
        environment_path=(
            root / "results/environment/benchmark-profile-opensearch-2.19.json"
        ),
    )
    version = client.wait_until_ready(expected_version=expected_version)
    verify_prepared_trec_product_search(
        dataset_spec,
        corpus_path.parent,
        manifest_path.parents[2] / "data/prepared/trec-product-search-2024",
        preparation_path,
        maximum_orphan_rate=maximum_orphan_rate,
    )
    definition = trec_lexical_index_definition(index_spec)
    client.delete_index(index_spec.name)
    client.create_index(index_spec.name, definition)
    started = time.perf_counter()
    client.bulk_index(
        index_spec.name,
        iter_trec_index_documents(corpus_path),
        batch_size=500,
    )
    bulk_seconds = time.perf_counter() - started
    client.refresh(index_spec.name)
    merge_started = time.perf_counter()
    client.force_merge(index_spec.name)
    merge_seconds = time.perf_counter() - merge_started
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
        raise DatasetIntegrityError("TREC lexical index has the wrong product count")
    if facts.segment_count != 1 or facts.deleted_document_count != 0:
        raise DatasetIntegrityError("TREC lexical index must have one segment and no deletions")
    index_content = verify_live_index_content(
        client,
        index=index_spec.name,
        expected_documents=iter_trec_index_documents(corpus_path),
        expected_count=dataset_spec.expected_products,
    )
    quality_eligible = trec_index_quality_eligible(
        root,
        benchmark_provenance,
        preparation_verified=True,
        require_current_code_revision=True,
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
            "path": str(corpus_path.relative_to(manifest_path.parents[2])),
            "sha256": file_facts(corpus_path).sha256,
            "bytes": file_facts(corpus_path).bytes,
            "records": dataset_spec.expected_products,
        },
        "preparation_sha256": file_facts(preparation_path).sha256,
        "preparation_verified": True,
        "bulk_index_seconds": bulk_seconds,
        "bulk_documents_per_second": dataset_spec.expected_products / bulk_seconds,
        "force_merge_seconds": merge_seconds,
        "refresh_interval": "1s",
        "vector_state": "fields_not_added",
        "quality_evidence_eligible_for_decision": quality_eligible,
        "latency_evidence_eligible_for_decision": False,
        "benchmark_provenance": benchmark_provenance,
    }
    write_json(manifest_path, manifest)
    return manifest


def verify_trec_lexical_index(
    client: OpenSearchClient,
    *,
    dataset_spec: TrecProductSearchDatasetSpec,
    index_spec: TrecIndexSpec,
    corpus_path: Path,
    preparation_path: Path,
    manifest_path: Path,
    expected_version: str,
    maximum_orphan_rate: float,
) -> dict[str, Any]:
    root = manifest_path.parents[2]
    require_registered_opensearch_client(
        client,
        root / "config/benchmark-2.19.toml",
    )
    client.wait_until_ready(expected_version=expected_version)
    verify_prepared_trec_product_search(
        dataset_spec,
        corpus_path.parent,
        root / "data/prepared/trec-product-search-2024",
        preparation_path,
        maximum_orphan_rate=maximum_orphan_rate,
    )
    manifest = cast(dict[str, Any], read_json(manifest_path))
    verify_trec_lexical_manifest_metadata(
        manifest,
        dataset_spec=dataset_spec,
        expected_version=expected_version,
    )
    facts = collect_index_facts(client, index_spec.name)
    if facts.document_count != dataset_spec.expected_products:
        raise DatasetIntegrityError("live TREC lexical index count differs")
    if facts.segment_count != 1 or facts.deleted_document_count != 0:
        raise DatasetIntegrityError("live TREC lexical index segment state differs")
    if manifest.get("index") != asdict(facts):
        raise DatasetIntegrityError("TREC lexical index facts differ from manifest")
    definition = trec_lexical_index_definition(index_spec)
    if manifest.get("index_definition_sha256") != canonical_sha256(definition):
        raise DatasetIntegrityError("TREC lexical index definition hash differs")
    if manifest.get("index_write_block") != INDEX_WRITE_BLOCK_EVIDENCE:
        raise DatasetIntegrityError("TREC lexical index write block evidence differs")
    mapping_response = cast(
        dict[str, Any], client.request("GET", f"/{index_spec.name}/_mapping")
    )
    live_mapping = cast(dict[str, Any], mapping_response[index_spec.name])["mappings"]
    if live_mapping != definition["mappings"]:
        raise DatasetIntegrityError("live TREC lexical mapping differs")
    verify_live_index_settings(
        client,
        index=index_spec.name,
        definition=definition,
    )
    index_content = verify_live_index_content(
        client,
        index=index_spec.name,
        expected_documents=iter_trec_index_documents(corpus_path),
        expected_count=dataset_spec.expected_products,
    )
    if manifest.get("index_content") != index_content:
        raise DatasetIntegrityError("live TREC lexical index content differs")
    corpus = cast(dict[str, Any], manifest["source_corpus"])
    corpus_facts = file_facts(corpus_path)
    if corpus.get("sha256") != corpus_facts.sha256 or corpus.get("bytes") != corpus_facts.bytes:
        raise DatasetIntegrityError("TREC lexical source corpus differs")
    if manifest.get("preparation_sha256") != file_facts(preparation_path).sha256:
        raise DatasetIntegrityError("TREC lexical preparation artifact differs")
    provenance = manifest.get("benchmark_provenance")
    if (
        not isinstance(provenance, dict)
        or not trec_index_quality_eligible(
            manifest_path.parents[2],
            provenance,
            preparation_verified=True,
        )
        or manifest.get("preparation_verified") is not True
        or manifest.get("quality_evidence_eligible_for_decision") is not True
        or manifest.get("latency_evidence_eligible_for_decision") is not False
    ):
        raise DatasetIntegrityError("TREC lexical index provenance differs")
    return manifest


def verify_trec_lexical_manifest_metadata(
    manifest: dict[str, Any],
    *,
    dataset_spec: TrecProductSearchDatasetSpec,
    expected_version: str,
) -> None:
    source_corpus = manifest.get("source_corpus")
    expected = {
        "schema_version": 2,
        "dataset": "TREC Product Search 2024",
        "source_revision": dataset_spec.revision,
        "opensearch_version": expected_version,
        "refresh_interval": "1s",
        "vector_state": "fields_not_added",
        "index_write_block": INDEX_WRITE_BLOCK_EVIDENCE,
    }
    if any(manifest.get(key) != value for key, value in expected.items()) or (
        not isinstance(source_corpus, dict)
        or source_corpus.get("records") != dataset_spec.expected_products
    ):
        raise DatasetIntegrityError("TREC lexical index metadata differs")


def trec_index_quality_eligible(
    root: Path,
    provenance: dict[str, Any],
    *,
    preparation_verified: bool = False,
    require_current_code_revision: bool = False,
) -> bool:
    return preparation_verified and verify_decision_provenance(
        provenance,
        root=root,
        profile_path=root / "config/benchmark-2.19.toml",
        environment_path=(
            root / "results/environment/benchmark-profile-opensearch-2.19.json"
        ),
        require_current_code_revision=require_current_code_revision,
    )


def trec_opensearch_url() -> str:
    root = Path(__file__).resolve().parents[1]
    return registered_opensearch_url(
        root / "config/benchmark-2.19.toml",
        environment_variable="OPENSEARCH_HYBRID_TREC_OS_URL",
    )
