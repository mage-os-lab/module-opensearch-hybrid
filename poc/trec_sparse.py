from __future__ import annotations

import json
import math
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, cast

from poc.config import ConfigError
from poc.indexing import WandsIndexSpec, wands_index_definition
from poc.trec_product_search import iter_trec_products


@dataclass(frozen=True, slots=True)
class TrecSparseSpec:
    document_model_name: str
    document_model_version: str
    document_model_format: str
    document_model_url: str
    document_model_sha256: str
    document_model_bytes: int
    query_tokenizer_name: str
    query_tokenizer_version: str
    query_tokenizer_format: str
    query_tokenizer_content_sha256: str
    query_tokenizer_content_bytes: int
    index_name: str
    text_field: str
    embedding_field: str
    shards: int
    replicas: int
    bulk_request_size: int
    expected_opensearch_version: str
    inference_batch_size: int
    maximum_token_length: int
    maximum_value_ratio: float
    minimum_preflight_documents_per_second: float


def load_trec_sparse_config(path: Path) -> TrecSparseSpec:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    document = raw.get("document_model")
    tokenizer = raw.get("query_tokenizer")
    index = raw.get("index")
    runtime = raw.get("runtime")
    if raw.get("schema_version") != 1 or not all(
        isinstance(value, dict) for value in (document, tokenizer, index, runtime)
    ):
        raise ConfigError("invalid TREC neural-sparse configuration")
    document = cast(dict[str, Any], document)
    tokenizer = cast(dict[str, Any], tokenizer)
    index = cast(dict[str, Any], index)
    runtime = cast(dict[str, Any], runtime)
    values = TrecSparseSpec(
        document_model_name=_required_string(document, "name"),
        document_model_version=_required_string(document, "version"),
        document_model_format=_required_string(document, "model_format"),
        document_model_url=_required_string(document, "package_url"),
        document_model_sha256=_required_string(document, "package_sha256"),
        document_model_bytes=_required_integer(document, "package_bytes", minimum=1),
        query_tokenizer_name=_required_string(tokenizer, "name"),
        query_tokenizer_version=_required_string(tokenizer, "version"),
        query_tokenizer_format=_required_string(tokenizer, "model_format"),
        query_tokenizer_content_sha256=_required_string(
            tokenizer, "content_sha256"
        ),
        query_tokenizer_content_bytes=_required_integer(
            tokenizer, "content_bytes", minimum=1
        ),
        index_name=_required_string(index, "name"),
        text_field=_required_string(index, "text_field"),
        embedding_field=_required_string(index, "embedding_field"),
        shards=_required_integer(index, "shards", minimum=1),
        replicas=_required_integer(index, "replicas", minimum=0),
        bulk_request_size=_required_integer(index, "bulk_request_size", minimum=1),
        expected_opensearch_version=_required_string(
            runtime, "expected_opensearch_version"
        ),
        inference_batch_size=_required_integer(
            runtime, "inference_batch_size", minimum=1
        ),
        maximum_token_length=_required_integer(
            document, "maximum_token_length", minimum=1
        ),
        maximum_value_ratio=_required_number(
            document, "maximum_value_ratio", minimum=0.0, maximum=1.0
        ),
        minimum_preflight_documents_per_second=_required_number(
            runtime, "minimum_preflight_documents_per_second", minimum=0.0
        ),
    )
    if values.document_model_format != "TORCH_SCRIPT":
        raise ConfigError("TREC sparse document model must use TorchScript")
    if values.query_tokenizer_format != "TORCH_SCRIPT":
        raise ConfigError("TREC sparse query tokenizer must use TorchScript")
    if len(values.query_tokenizer_content_sha256) != 64:
        raise ConfigError("TREC sparse query tokenizer SHA-256 is invalid")
    if len(values.document_model_sha256) != 64:
        raise ConfigError("TREC sparse document model SHA-256 is invalid")
    return values


def build_trec_sparse_index_definition(spec: TrecSparseSpec) -> dict[str, Any]:
    definition = wands_index_definition(
        WandsIndexSpec(name=spec.index_name, shards=spec.shards, replicas=spec.replicas)
    )
    definition["settings"]["index"]["knn"] = False
    properties = definition["mappings"]["properties"]
    properties[spec.text_field] = {"type": "text", "index": False}
    properties[spec.embedding_field] = {"type": "rank_features"}
    return definition


def build_trec_sparse_query(
    spec: TrecSparseSpec,
    text: str,
    *,
    model_id: str,
) -> dict[str, Any]:
    if not text.strip() or not model_id.strip():
        raise ValueError("TREC sparse query and tokenizer model ID must not be blank")
    return {
        "neural_sparse": {
            spec.embedding_field: {
                "query_text": text,
                "model_id": model_id,
            }
        }
    }


def trec_sparse_document_text(document: Mapping[str, object]) -> str:
    return ". ".join(
        str(document.get(field, "")).strip()
        for field in ("title", "description")
        if str(document.get(field, "")).strip()
    )


def prune_sparse_values(
    values: Mapping[str, float],
    *,
    maximum_value_ratio: float,
) -> dict[str, float]:
    positive = {
        str(token): float(value)
        for token, value in values.items()
        if math.isfinite(float(value)) and float(value) > 0.0
    }
    if not positive:
        return {}
    threshold = max(positive.values()) * maximum_value_ratio
    return {
        token: value for token, value in positive.items() if value >= threshold
    }


def iter_trec_precomputed_sparse_documents(
    corpus_path: Path,
    embeddings_path: Path,
    spec: TrecSparseSpec,
    *,
    maximum_records: int | None = None,
) -> Iterator[tuple[str, dict[str, object]]]:
    if maximum_records is not None and maximum_records <= 0:
        raise ValueError("TREC sparse maximum records must be positive")
    all_products = iter_trec_products(corpus_path)
    products = (
        all_products
        if maximum_records is None
        else islice(all_products, maximum_records)
    )
    with embeddings_path.open(encoding="utf-8") as handle:
        embeddings = (json.loads(line) for line in handle if line.strip())
        for product, raw_embedding in zip(products, embeddings, strict=True):
            document_id, document = product
            embedding = cast(dict[str, Any], raw_embedding)
            if str(embedding.get("product_id")) != document_id:
                raise ValueError(
                    f"TREC sparse embedding order differs at product {document_id}"
                )
            values = embedding.get("embedding")
            if not isinstance(values, dict) or not values:
                raise ValueError(
                    f"TREC sparse embedding is empty for product {document_id}"
                )
            yield document_id, {
                "product_id": document_id,
                **document,
                spec.text_field: trec_sparse_document_text(document),
                spec.embedding_field: values,
            }


def iter_sparse_checkpoint_lines(path: Path) -> Iterator[tuple[str, bytes]]:
    """Yield validated checkpoint IDs and exact bytes for safe hash resumption."""
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith(b"\n"):
                raise ValueError(
                    f"truncated sparse checkpoint line in {path}:{line_number}"
                )
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid sparse checkpoint JSON in {path}:{line_number}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"invalid sparse checkpoint record in {path}:{line_number}"
                )
            product_id = record.get("product_id")
            embedding = record.get("embedding")
            if not isinstance(product_id, str) or not product_id:
                raise ValueError(
                    f"missing sparse checkpoint product ID in {path}:{line_number}"
                )
            if not isinstance(embedding, dict) or not embedding:
                raise ValueError(
                    f"missing sparse checkpoint embedding in {path}:{line_number}"
                )
            if not all(
                isinstance(token, str)
                and token
                and isinstance(value, int | float)
                and not isinstance(value, bool)
                for token, value in embedding.items()
            ):
                raise ValueError(
                    f"invalid sparse checkpoint embedding in {path}:{line_number}"
                )
            yield product_id, line


def select_registered_model_identity(
    hits: list[dict[str, Any]],
    *,
    name: str,
    version: str,
) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    for hit in hits:
        source = hit.get("_source")
        if not isinstance(source, dict) or "chunk_number" in source:
            continue
        recorded_version = source.get("model_version", source.get("version"))
        if source.get("name") == name and recorded_version == version:
            matches.append(hit)
    if not matches:
        return None
    return min(
        matches,
        key=lambda hit: (
            0
            if isinstance(hit.get("_source"), dict)
            and hit["_source"].get("model_state") == "DEPLOYED"
            else 1,
            str(hit.get("_id", "")),
        ),
    )


def extract_sparse_prediction(response: Mapping[str, Any]) -> dict[str, float]:
    results = response.get("inference_results")
    if isinstance(results, list):
        for result in results:
            if not isinstance(result, dict):
                continue
            outputs = result.get("output")
            if not isinstance(outputs, list):
                continue
            for output in outputs:
                if not isinstance(output, dict):
                    continue
                data = output.get("dataAsMap")
                values = data.get("response") if isinstance(data, dict) else None
                if not isinstance(values, list):
                    continue
                for candidate in values:
                    if isinstance(candidate, dict) and candidate:
                        return {
                            str(token): float(weight)
                            for token, weight in candidate.items()
                        }
    raise ValueError("ML Commons sparse prediction contains no token map")


def _required_string(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"TREC sparse {key} must be a non-empty string")
    return value


def _required_integer(
    values: Mapping[str, Any], key: str, *, minimum: int
) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ConfigError(f"TREC sparse {key} must be an integer >= {minimum}")
    return value


def _required_number(
    values: Mapping[str, Any],
    key: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    value = values.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigError(f"TREC sparse {key} must be numeric")
    numeric = float(value)
    if numeric < minimum or (maximum is not None and numeric > maximum):
        raise ConfigError(f"TREC sparse {key} is outside the supported range")
    return numeric
