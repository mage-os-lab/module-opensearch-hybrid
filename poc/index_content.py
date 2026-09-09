from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import struct
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol, cast

from poc.datasets import DatasetIntegrityError

INDEX_CONTENT_DIGEST_ALGORITHM = "sha256-canonical-opensearch-source-v1"
INDEX_CONTENT_DOCUMENT_ORDER = "registered_exact_id_sequence"
INDEX_CONTENT_PAGE_SIZE = 500


class IndexContentClient(Protocol):
    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
    ) -> Any: ...


def verify_live_index_content(
    client: IndexContentClient,
    *,
    index: str,
    expected_documents: Iterable[tuple[str, Mapping[str, Any]]],
    expected_count: int,
    page_size: int = INDEX_CONTENT_PAGE_SIZE,
    float32_fields: tuple[str, ...] = (),
) -> dict[str, object]:
    """Bind every live ``_source`` value to its registered input stream.

    Each bounded multi-get page preserves the registered document ID sequence.
    A disk-backed uniqueness check avoids an in-memory corpus-sized set while
    still detecting missing, duplicate, reordered, or modified documents.
    """
    if expected_count <= 0 or page_size <= 0:
        raise ValueError("index content count and page size must be positive")
    normalized_float32_fields = tuple(sorted(set(float32_fields)))
    if any(not field for field in normalized_float32_fields):
        raise ValueError("index content float32 fields must not be blank")
    count_response = client.request("GET", f"/{index}/_count")
    if not isinstance(count_response, Mapping):
        raise DatasetIntegrityError("live index count response differs")
    live_count = count_response.get("count")
    if (
        not isinstance(live_count, int)
        or isinstance(live_count, bool)
        or live_count != expected_count
    ):
        raise DatasetIntegrityError(
            f"live index content count differs: expected {expected_count}, "
            f"found {live_count!r}"
        )

    expected_hasher = hashlib.sha256()
    live_hasher = hashlib.sha256()
    documents = 0
    page: list[tuple[str, Mapping[str, Any]]] = []
    with (
        tempfile.TemporaryDirectory(
            prefix="opensearch-hybrid-index-content-"
        ) as temporary,
        sqlite3.connect(Path(temporary) / "document-ids.sqlite3") as seen_ids,
    ):
        seen_ids.execute("CREATE TABLE ids (document_id TEXT PRIMARY KEY)")
        for document_id, source in expected_documents:
            if not isinstance(document_id, str) or not document_id:
                raise DatasetIntegrityError("expected index document ID is invalid")
            page.append((document_id, source))
            if len(page) == page_size:
                _record_unique_ids(seen_ids, page)
                _verify_content_page(
                    client,
                    index=index,
                        page=page,
                        expected_hasher=expected_hasher,
                        live_hasher=live_hasher,
                        float32_fields=normalized_float32_fields,
                )
                documents += len(page)
                page.clear()
        if page:
            _record_unique_ids(seen_ids, page)
            _verify_content_page(
                client,
                index=index,
                page=page,
                expected_hasher=expected_hasher,
                live_hasher=live_hasher,
                float32_fields=normalized_float32_fields,
            )
            documents += len(page)
    if documents != expected_count:
        raise DatasetIntegrityError(
            f"registered index content count differs: expected {expected_count}, "
            f"found {documents}"
        )
    expected_sha256 = expected_hasher.hexdigest()
    live_sha256 = live_hasher.hexdigest()
    if live_sha256 != expected_sha256:
        raise DatasetIntegrityError("live index content digest differs")
    return {
        "algorithm": INDEX_CONTENT_DIGEST_ALGORITHM,
        "document_order": INDEX_CONTENT_DOCUMENT_ORDER,
        "page_size": page_size,
        "documents": documents,
        "float32_fields": list(normalized_float32_fields),
        "sha256": live_sha256,
    }


def _record_unique_ids(
    connection: sqlite3.Connection,
    page: list[tuple[str, Mapping[str, Any]]],
) -> None:
    try:
        connection.executemany(
            "INSERT INTO ids (document_id) VALUES (?)",
            ((document_id,) for document_id, _ in page),
        )
    except sqlite3.IntegrityError as error:
        raise DatasetIntegrityError(
            "expected index document ID uniqueness differs"
        ) from error


def _verify_content_page(
    client: IndexContentClient,
    *,
    index: str,
    page: list[tuple[str, Mapping[str, Any]]],
    expected_hasher: Any,
    live_hasher: Any,
    float32_fields: tuple[str, ...],
) -> None:
    document_ids = [document_id for document_id, _ in page]
    response = client.request(
        "POST",
        f"/{index}/_mget",
        json_body={"ids": document_ids},
    )
    if not isinstance(response, Mapping):
        raise DatasetIntegrityError("live index multi-get response differs")
    documents = response.get("docs")
    if not isinstance(documents, list) or len(documents) != len(page):
        raise DatasetIntegrityError("live index multi-get page length differs")
    for (expected_id, expected_source), value in zip(page, documents, strict=True):
        if not isinstance(value, Mapping):
            raise DatasetIntegrityError("live index multi-get document differs")
        if value.get("_id") != expected_id:
            raise DatasetIntegrityError("live index multi-get response order differs")
        if value.get("found") is not True:
            raise DatasetIntegrityError(
                f"live index is missing expected document {expected_id}"
            )
        live_source = value.get("_source")
        if not isinstance(live_source, Mapping):
            raise DatasetIntegrityError(
                f"live index source is missing for document {expected_id}"
            )
        expected_line = _canonical_document_line(
            expected_id,
            expected_source,
            float32_fields=float32_fields,
        )
        live_line = _canonical_document_line(
            expected_id,
            cast(Mapping[str, Any], live_source),
            float32_fields=float32_fields,
        )
        expected_hasher.update(expected_line)
        live_hasher.update(live_line)
        if live_line != expected_line:
            raise DatasetIntegrityError(
                f"live index source differs for document {expected_id}"
            )


def _canonical_document_line(
    document_id: str,
    source: Mapping[str, Any],
    *,
    float32_fields: tuple[str, ...],
) -> bytes:
    try:
        normalized_source = dict(source)
        for field in float32_fields:
            if field in normalized_source:
                normalized_source[field] = _normalize_float32_value(
                    normalized_source[field]
                )
        return (
            json.dumps(
                {"_id": document_id, "_source": normalized_source},
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise DatasetIntegrityError(
            f"index source for document {document_id} is not canonical JSON"
        ) from error


def _normalize_float32_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _float32_hex(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_float32_hex(item) for item in value]
    raise DatasetIntegrityError("mapped float32 index source has an invalid shape")


def _float32_hex(value: object) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise DatasetIntegrityError("mapped float32 index source is not numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise DatasetIntegrityError("mapped float32 index source is not finite")
    try:
        return struct.pack(">f", numeric).hex()
    except (OverflowError, struct.error) as error:
        raise DatasetIntegrityError(
            "mapped float32 index source is outside finite storage range"
        ) from error
