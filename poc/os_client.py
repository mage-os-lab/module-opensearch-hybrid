from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from typing import Any, cast

import httpx


class OpenSearchError(RuntimeError):
    pass


class OpenSearchClient:
    def __init__(self, base_url: str = "http://127.0.0.1:9201", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def __enter__(self) -> OpenSearchClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        content: str | None = None,
        params: Mapping[str, str | int | bool] | None = None,
        allow_404: bool = False,
    ) -> Any:
        response = self._client.request(
            method,
            path,
            json=json_body,
            content=content,
            params=params,
            headers={"content-type": "application/x-ndjson"} if content is not None else None,
        )
        if allow_404 and response.status_code == 404:
            return None
        if response.is_error:
            raise OpenSearchError(
                f"{method} {path} returned {response.status_code}: {response.text[:2000]}"
            )
        if not response.content:
            return {}
        return response.json()

    def wait_until_ready(self, *, expected_version: str, timeout_seconds: float = 180.0) -> str:
        deadline = time.monotonic() + timeout_seconds
        last_error = "not contacted"
        while time.monotonic() < deadline:
            try:
                info = cast(dict[str, Any], self.request("GET", "/"))
                version = str(cast(dict[str, Any], info["version"])["number"])
                if version != expected_version:
                    raise OpenSearchError(
                        f"expected OpenSearch {expected_version}, connected to {version}"
                    )
                health = cast(
                    dict[str, Any],
                    self.request(
                        "GET",
                        "/_cluster/health",
                        params={"wait_for_status": "yellow", "timeout": "5s"},
                    ),
                )
                if not bool(health.get("timed_out")):
                    return version
                last_error = "cluster health request timed out"
            except (httpx.HTTPError, OpenSearchError, KeyError, TypeError) as exc:
                last_error = str(exc)
            time.sleep(1)
        raise OpenSearchError(f"OpenSearch did not become ready: {last_error}")

    def delete_index(self, index: str) -> None:
        self.request("DELETE", f"/{index}", allow_404=True)

    def create_index(self, index: str, definition: Mapping[str, Any]) -> None:
        self.request("PUT", f"/{index}", json_body=definition)

    def put_search_pipeline(self, pipeline: str, definition: Mapping[str, Any]) -> None:
        self.request("PUT", f"/_search/pipeline/{pipeline}", json_body=definition)

    def bulk_index(
        self,
        index: str,
        documents: Iterable[tuple[str, Mapping[str, Any]]],
        *,
        batch_size: int = 500,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("bulk batch size must be positive")
        batch: list[tuple[str, Mapping[str, Any]]] = []
        total = 0
        for document in documents:
            batch.append(document)
            if len(batch) == batch_size:
                self._send_bulk(index, batch)
                total += len(batch)
                batch.clear()
        if batch:
            self._send_bulk(index, batch)
            total += len(batch)
        if total == 0:
            raise ValueError("bulk request contains no documents")

    def _send_bulk(self, index: str, documents: list[tuple[str, Mapping[str, Any]]]) -> None:
        lines: list[str] = []
        for document_id, document in documents:
            lines.append(json.dumps({"index": {"_index": index, "_id": document_id}}))
            lines.append(json.dumps(document, separators=(",", ":"), sort_keys=True))
        response = cast(
            dict[str, Any], self.request("POST", "/_bulk", content="\n".join(lines) + "\n")
        )
        if response.get("errors"):
            failures = [
                item
                for item in cast(list[dict[str, Any]], response.get("items", []))
                if cast(dict[str, Any], item.get("index", {})).get("error")
            ]
            raise OpenSearchError(f"bulk indexing failed: {failures[:3]}")

    def refresh(self, index: str) -> None:
        self.request("POST", f"/{index}/_refresh")

    def force_merge(self, index: str) -> None:
        self.request(
            "POST",
            f"/{index}/_forcemerge",
            params={"max_num_segments": 1, "flush": "true"},
        )

    def update_index_settings(self, index: str, settings: Mapping[str, Any]) -> None:
        self.request("PUT", f"/{index}/_settings", json_body=settings)

    def search(
        self,
        index: str,
        request: Mapping[str, Any],
        *,
        pipeline: str | None = None,
        request_cache: bool | None = None,
    ) -> dict[str, Any]:
        params: dict[str, str | bool] = {}
        if pipeline:
            params["search_pipeline"] = pipeline
        if request_cache is not None:
            params["request_cache"] = request_cache
        return cast(
            dict[str, Any],
            self.request(
                "POST",
                f"/{index}/_search",
                json_body=request,
                params=params or None,
            ),
        )
