from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from poc.os_client import OpenSearchClient
from poc.provenance import collect_manifest_provenance

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class IndexFacts:
    name: str
    uuid: str
    document_count: int
    segment_count: int
    deleted_document_count: int


@dataclass(frozen=True, slots=True)
class RunManifest:
    schema_version: int
    created_at: str
    opensearch_version: str
    index: IndexFacts
    hnsw: dict[str, object]
    pagination_depth: int
    pipeline_id: str
    pipeline_sha256: str
    model_sha256: str
    encoder_runtime: str
    eligible_for_decision: bool
    declared_variable: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def collect_index_facts(client: OpenSearchClient, index: str) -> IndexFacts:
    index_info = cast(dict[str, Any], client.request("GET", f"/{index}"))
    settings = cast(dict[str, Any], cast(dict[str, Any], index_info[index])["settings"])
    uuid = str(cast(dict[str, Any], settings["index"])["uuid"])

    stats = cast(dict[str, Any], client.request("GET", f"/{index}/_stats/docs,segments"))
    primaries = cast(dict[str, Any], cast(dict[str, Any], stats["_all"])["primaries"])
    docs = cast(dict[str, Any], primaries["docs"])
    segments = cast(dict[str, Any], primaries["segments"])
    return IndexFacts(
        name=index,
        uuid=uuid,
        document_count=int(docs["count"]),
        segment_count=int(segments["count"]),
        deleted_document_count=int(docs["deleted"]),
    )


def build_run_manifest(
    client: OpenSearchClient,
    *,
    index: str,
    pipeline_id: str,
    pipeline_definition: object,
    model_sha256: str,
    pagination_depth: int,
    encoder_runtime: str,
    eligible_for_decision: bool,
    declared_variable: str,
    hnsw: dict[str, object],
) -> RunManifest:
    root = cast(dict[str, Any], client.request("GET", "/"))
    version = str(cast(dict[str, Any], root["version"])["number"])
    return RunManifest(
        schema_version=1,
        created_at=datetime.now(UTC).isoformat(),
        opensearch_version=version,
        index=collect_index_facts(client, index),
        hnsw=hnsw,
        pagination_depth=pagination_depth,
        pipeline_id=pipeline_id,
        pipeline_sha256=canonical_sha256(pipeline_definition),
        model_sha256=model_sha256,
        encoder_runtime=encoder_runtime,
        eligible_for_decision=eligible_for_decision,
        declared_variable=declared_variable,
    )


def write_json(path: Path, value: object) -> None:
    if path.name.endswith("manifest.json") and isinstance(value, dict):
        value = dict(value)
        if "benchmark_provenance" not in value:
            value["benchmark_provenance"] = collect_manifest_provenance(PROJECT_ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())
