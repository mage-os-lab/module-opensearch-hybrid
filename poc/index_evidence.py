from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Protocol, cast

from poc.datasets import DatasetIntegrityError
from poc.manifest import canonical_sha256


class IndexSettingsClient(Protocol):
    def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str | int | bool] | None = None,
    ) -> Any: ...


class MutableIndexSettingsClient(IndexSettingsClient, Protocol):
    def update_index_settings(
        self,
        index: str,
        settings: Mapping[str, Any],
    ) -> None: ...


INDEX_WRITE_BLOCK_EVIDENCE: dict[str, object] = {
    "setting": "index.blocks.write",
    "value": True,
}
_PIPELINE_DIGEST_LENGTH = 16

_GENERATED_INDEX_SETTINGS = {
    "creation_date",
    "knn.derived_source",
    "merge",
    "provided_name",
    "replication",
    "uuid",
    "version",
}


def verify_live_index_settings(
    client: IndexSettingsClient,
    *,
    index: str,
    definition: Mapping[str, Any],
    expected_refresh_interval: str = "1s",
) -> None:
    response = cast(
        dict[str, Any],
        client.request(
            "GET",
            f"/{index}/_settings",
            params={"flat_settings": "false"},
        ),
    )
    index_response = response.get(index)
    if not isinstance(index_response, Mapping):
        raise DatasetIntegrityError("live index settings response differs")
    settings_response = index_response.get("settings")
    live_index = (
        settings_response.get("index")
        if isinstance(settings_response, Mapping)
        else None
    )
    if not isinstance(live_index, Mapping):
        raise DatasetIntegrityError("live index settings are missing")
    live_blocks = live_index.get("blocks")
    if (
        not isinstance(live_blocks, Mapping)
        or _normalize_settings(live_blocks.get("write")) != "true"
    ):
        raise DatasetIntegrityError("live index write block is missing")
    live_user_settings = {
        str(key): value
        for key, value in live_index.items()
        if key not in _GENERATED_INDEX_SETTINGS
    }

    definition_settings = definition.get("settings")
    if not isinstance(definition_settings, Mapping):
        raise DatasetIntegrityError("registered index settings are missing")
    definition_index = definition_settings.get("index")
    if not isinstance(definition_index, Mapping):
        raise DatasetIntegrityError("registered index core settings are missing")
    expected_user_settings = copy.deepcopy(dict(definition_index))
    expected_user_settings["refresh_interval"] = expected_refresh_interval
    expected_user_settings["blocks"] = {"write": True}
    analysis = definition_settings.get("analysis")
    if analysis is not None:
        expected_user_settings["analysis"] = copy.deepcopy(analysis)

    if _normalize_settings(live_user_settings) != _normalize_settings(
        expected_user_settings
    ):
        raise DatasetIntegrityError("live index settings or analysis differ")


def lock_live_index_for_decisions(
    client: MutableIndexSettingsClient,
    *,
    index: str,
) -> dict[str, object]:
    """Make a completed index read-only before evidence is collected."""
    if not index:
        raise ValueError("index name must not be blank")
    client.update_index_settings(
        index,
        {"index": {"blocks": {"write": True}}},
    )
    response = cast(
        dict[str, Any],
        client.request(
            "GET",
            f"/{index}/_settings",
            params={"flat_settings": "false"},
        ),
    )
    index_response = response.get(index)
    settings = (
        index_response.get("settings")
        if isinstance(index_response, Mapping)
        else None
    )
    live_index = settings.get("index") if isinstance(settings, Mapping) else None
    blocks = live_index.get("blocks") if isinstance(live_index, Mapping) else None
    if (
        not isinstance(blocks, Mapping)
        or _normalize_settings(blocks.get("write")) != "true"
    ):
        raise DatasetIntegrityError("live index write block was not applied")
    return dict(INDEX_WRITE_BLOCK_EVIDENCE)


def content_addressed_pipeline_id(
    prefix: str,
    definition: Mapping[str, Any],
) -> str:
    normalized_prefix = prefix.strip("-")
    if not normalized_prefix or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
        for character in normalized_prefix
    ):
        raise ValueError("search pipeline prefix is invalid")
    digest = canonical_sha256(definition)
    return f"{normalized_prefix}-sha256-{digest[:_PIPELINE_DIGEST_LENGTH]}"


def verify_live_search_pipeline(
    client: IndexSettingsClient,
    *,
    pipeline_id: str,
    expected_definition: Mapping[str, Any],
) -> None:
    expected_id = content_addressed_pipeline_id(
        pipeline_id.rsplit("-sha256-", 1)[0],
        expected_definition,
    )
    if pipeline_id != expected_id:
        raise DatasetIntegrityError("search pipeline identity differs from its content")
    response = client.request("GET", f"/_search/pipeline/{pipeline_id}")
    live_definition = (
        response.get(pipeline_id) if isinstance(response, Mapping) else None
    )
    if live_definition != expected_definition:
        raise DatasetIntegrityError("live search pipeline differs")


def _normalize_settings(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_settings(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list | tuple):
        return [_normalize_settings(item) for item in value]
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float) and not isinstance(value, bool):
        return str(value)
    return value
