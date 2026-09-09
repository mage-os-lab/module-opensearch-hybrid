from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from shutil import copytree
from typing import Any, cast

import pytest

from poc.config import ConfigError
from poc.datasets import DatasetIntegrityError, load_wands_config
from poc.end_to_end_latency import (
    LatencyProtocol,
    LatencyQuery,
    build_latency_request,
    latency_decision_status,
    load_latency_protocol,
    load_latency_workload,
    replay_latency_decision_gate,
    verify_latency_environment,
    verify_latency_identity,
    verify_latency_index_identity,
    verify_latency_measurements_from_raw,
    verify_latency_method_attribution,
    verify_latency_registered_config,
    verify_latency_replay_repeatability,
)
from poc.experiments import BM25Profile
from poc.manifest import canonical_sha256, write_json
from poc.neural_sparse import load_neural_sparse_spec
from poc.provenance import load_benchmark_profile
from poc.sku_slice import load_sku_slice_spec

ROOT = Path(__file__).resolve().parents[1]


def test_registered_latency_protocol_includes_required_target_and_facets() -> None:
    protocol = load_latency_protocol(ROOT / "config/experiments.toml")

    assert protocol.concurrency == (1, 4, 8)
    assert protocol.target_concurrency == 4
    assert protocol.p95_budget_ms == 200
    assert protocol.minimum_distinct_queries == 500
    assert protocol.include_layered_navigation_aggregations is True
    assert protocol.allow_query_vector_cache is False
    assert protocol.maximum_replay_relative_percentile_drift == 0.5


def test_latency_request_routes_known_items_around_semantic_fusion(
    bm25_selection: dict[str, Any],
) -> None:
    protocol = load_latency_protocol(ROOT / "config/experiments.toml")
    sparse = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")
    selection = bm25_selection
    profile = BM25Profile.from_mapping(
        cast(dict[str, Any], selection["selected_profile"])
    )

    known = build_latency_request(
        LatencyQuery("sku", "RW-123", "lexical_only"),
        profile=profile,
        sparse_spec=sparse,
        protocol=protocol,
    )
    general = build_latency_request(
        LatencyQuery("general", "comfortable running shoe", "hybrid"),
        profile=profile,
        sparse_spec=sparse,
        protocol=protocol,
    )

    assert "hybrid" not in known["query"]
    assert "hybrid" in general["query"]
    assert set(general["aggs"]) == {"brand", "category", "product_class"}


def test_latency_workload_loads_both_splits_and_routes_known_items(tmp_path: Path) -> None:
    prepared = tmp_path / "data/prepared/wands"
    prepared.mkdir(parents=True)
    for split in ("dev", "test"):
        (prepared / f"queries.{split}.jsonl").write_text(
            json.dumps({"query_id": "1", "query": f"{split} chair", "split": split}) + "\n"
        )
    (prepared / "queries.sku-slice.jsonl").write_text(
        json.dumps({
            "query_id": "1", "query": "SKU-001", "kind": "sku", "expected_document_id": "1"
        }) + "\n"
    )

    workload = load_latency_workload(tmp_path)

    assert [(query.query_id, query.text, query.route) for query in workload] == [
        ("wands-dev-1", "dev chair", "hybrid"),
        ("wands-test-1", "test chair", "hybrid"),
        ("known-1", "SKU-001", "lexical_only"),
    ]


def test_latency_environment_binds_host_resources_and_live_node() -> None:
    profile = load_benchmark_profile(ROOT / "config/benchmark.toml")
    plugins = [
        {"component": "opensearch-ml", "version": "3.8.0.0"},
        {"component": "opensearch-neural-search", "version": "3.8.0.0"},
    ]
    environment = {
        "schema_version": 2,
        "profile_id": profile.profile_id,
        "compose_files": list(profile.compose_files),
        "container_id": "a" * 64,
        "image": "opensearchproject/opensearch:3.8.0",
        "image_id": profile.container_image_id,
        "cpu_limit": profile.cpu_limit,
        "memory_limit_bytes": profile.memory_limit_bytes,
        "limits_verified": True,
        "architecture": "arm64",
        "latency_architecture_eligible": False,
        "opensearch": {
            "node_name": "a" * 12,
            "version": "3.8.0",
            "url": "http://127.0.0.1:9201",
            "plugins": plugins,
        },
    }
    live = {"name": "a" * 12, "version": {"number": "3.8.0"}}
    live_container = {
        "Id": "a" * 64,
        "Image": profile.container_image_id,
        "Config": {
            "Image": "opensearchproject/opensearch:3.8.0",
            "Labels": {
                "com.docker.compose.project.config_files": (
                    "/tmp/docker-compose.yml,/tmp/compose.bench.yml"
                )
            },
        },
        "HostConfig": {
            "NanoCpus": 8_000_000_000,
            "Memory": 16 * 1024**3,
        },
    }

    facts = verify_latency_environment(
        environment,
        profile=profile,
        host_architecture="arm64",
        live_root=live,
        live_container=live_container,
        container_architecture="arm64",
        live_plugins=list(reversed(plugins)),
    )

    assert facts["latency_architecture_eligible"] is False
    assert facts["container_id"] == "a" * 64

    environment["latency_architecture_eligible"] = True
    with pytest.raises(DatasetIntegrityError, match="architecture eligibility"):
        verify_latency_environment(
            environment,
            profile=profile,
            host_architecture="arm64",
            live_root=live,
            live_container=live_container,
            container_architecture="arm64",
            live_plugins=plugins,
        )

    environment["latency_architecture_eligible"] = False
    with pytest.raises(DatasetIntegrityError, match="host architecture"):
        verify_latency_environment(
            environment,
            profile=profile,
            host_architecture="x86_64",
            live_root=live,
            live_container=live_container,
            container_architecture="arm64",
            live_plugins=plugins,
        )

    environment["architecture"] = "arm64"
    live_container["HostConfig"]["NanoCpus"] = 0  # type: ignore[index]
    with pytest.raises(DatasetIntegrityError, match="live container"):
        verify_latency_environment(
            environment,
            profile=profile,
            host_architecture="arm64",
            live_root=live,
            live_container=live_container,
            container_architecture="arm64",
            live_plugins=plugins,
        )


def test_latency_environment_rejects_live_plugin_version_drift() -> None:
    profile = load_benchmark_profile(ROOT / "config/benchmark.toml")
    environment = {
        "schema_version": 2,
        "profile_id": profile.profile_id,
        "compose_files": list(profile.compose_files),
        "container_id": "a" * 64,
        "image": "opensearchproject/opensearch:3.8.0",
        "image_id": profile.container_image_id,
        "cpu_limit": profile.cpu_limit,
        "memory_limit_bytes": profile.memory_limit_bytes,
        "limits_verified": True,
        "architecture": "arm64",
        "latency_architecture_eligible": False,
        "opensearch": {
            "node_name": "a" * 12,
            "version": "3.8.0",
            "url": "http://127.0.0.1:9201",
            "plugins": [
                {"component": "opensearch-ml", "version": "3.8.0.0"}
            ],
        },
    }
    live = {"name": "a" * 12, "version": {"number": "3.8.0"}}
    live_container = {
        "Id": "a" * 64,
        "Image": profile.container_image_id,
        "Config": {
            "Image": "opensearchproject/opensearch:3.8.0",
            "Labels": {
                "com.docker.compose.project.config_files": (
                    "/tmp/docker-compose.yml,/tmp/compose.bench.yml"
                )
            },
        },
        "HostConfig": {
            "NanoCpus": 8_000_000_000,
            "Memory": 16 * 1024**3,
        },
    }

    with pytest.raises(DatasetIntegrityError, match="plugin inventory"):
        verify_latency_environment(
            environment,
            profile=profile,
            host_architecture="arm64",
            live_root=live,
            live_container=live_container,
            container_architecture="arm64",
            live_plugins=[
                {"component": "opensearch-ml", "version": "3.8.0.1"}
            ],
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema_version", 1),
        ("dataset", "WANDS"),
        ("method", "known_item_guarded_bm25"),
    ],
)
def test_latency_identity_requires_exact_schema_dataset_and_method(
    key: str,
    value: object,
) -> None:
    result: dict[str, object] = {
        "schema_version": 2,
        "dataset": "WANDS plus synthetic known-item latency workload",
        "method": "known_item_guarded_bm25_neural_sparse",
    }
    verify_latency_identity(result)

    result[key] = value
    with pytest.raises(DatasetIntegrityError, match="latency result identity"):
        verify_latency_identity(result)


def test_latency_decision_requires_verified_sku_index() -> None:
    eligible, reason = latency_decision_status(
        architecture_eligible=True,
        run_provenance_eligible=True,
        sku_index_eligible=False,
    )

    assert eligible is False
    assert reason == "synthetic SKU index is not latency decision-eligible"


def test_latency_requires_registered_index_and_config_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bm25_selection: dict[str, Any],
) -> None:
    copytree(ROOT / "config", tmp_path / "config")
    protocol = load_latency_protocol(tmp_path / "config/experiments.toml")
    sparse = load_neural_sparse_spec(tmp_path / "config/neural_sparse.toml")
    dataset = load_wands_config(tmp_path / "config/datasets.toml")
    sku = load_sku_slice_spec(tmp_path / "config/sku_slice.toml")
    selection = bm25_selection
    profile = BM25Profile.from_mapping(
        cast(dict[str, Any], selection["selected_profile"])
    )
    verified_selection = {
        **selection,
        "schema_version": 2,
        "quality_evidence_eligible_for_decision": True,
    }
    write_json(tmp_path / "results/wands/bm25-selection.json", verified_selection)
    monkeypatch.setattr(
        "poc.end_to_end_latency.verify_wands_bm25_selection",
        lambda *args, **kwargs: verified_selection,
        raising=False,
    )
    registered = verify_latency_registered_config(
        cast(Any, object()),
        root=tmp_path,
        dataset_spec=dataset,
        profile=profile,
        sparse_spec=sparse,
        sku_spec=sku,
        protocol=protocol,
    )

    bm25_selection = cast(dict[str, Any], registered["bm25_selection"])
    assert bm25_selection["path"] == (
        "results/wands/bm25-selection.json"
    )
    with pytest.raises(DatasetIntegrityError, match="registered BM25 profile"):
        verify_latency_registered_config(
            cast(Any, object()),
            root=tmp_path,
            dataset_spec=dataset,
            profile=replace(profile, tie_breaker=0.0),
            sparse_spec=sparse,
            sku_spec=sku,
            protocol=protocol,
        )
    with pytest.raises(DatasetIntegrityError, match="registered neural sparse"):
        verify_latency_registered_config(
            cast(Any, object()),
            root=tmp_path,
            dataset_spec=dataset,
            profile=profile,
            sparse_spec=replace(sparse, prune_ratio=0.2),
            sku_spec=sku,
            protocol=protocol,
        )
    with pytest.raises(DatasetIntegrityError, match="registered latency protocol"):
        verify_latency_registered_config(
            cast(Any, object()),
            root=tmp_path,
            dataset_spec=dataset,
            profile=profile,
            sparse_spec=sparse,
            sku_spec=sku,
            protocol=replace(protocol, pagination_depth=10),
        )

    with pytest.raises(DatasetIntegrityError, match="registered WANDS dataset"):
        verify_latency_registered_config(
            cast(Any, object()),
            root=tmp_path,
            dataset_spec=replace(dataset, expected_products=1),
            profile=profile,
            sparse_spec=sparse,
            sku_spec=sku,
            protocol=protocol,
        )
    with pytest.raises(DatasetIntegrityError, match="registered SKU slice"):
        verify_latency_registered_config(
            cast(Any, object()),
            root=tmp_path,
            dataset_spec=dataset,
            profile=profile,
            sparse_spec=sparse,
            sku_spec=replace(sku, index_name="custom-index"),
            protocol=protocol,
        )

    verify_latency_index_identity(sku.index_name, sku)
    with pytest.raises(DatasetIntegrityError, match="registered SKU index"):
        verify_latency_index_identity("custom-cheaper-index", sku)

    monkeypatch.setattr(
        "poc.end_to_end_latency.verify_wands_bm25_selection",
        lambda *args, **kwargs: {
            **verified_selection,
            "quality_evidence_eligible_for_decision": False,
        },
        raising=False,
    )
    with pytest.raises(DatasetIntegrityError, match="BM25 selection is not eligible"):
        verify_latency_registered_config(
            cast(Any, object()),
            root=tmp_path,
            dataset_spec=dataset,
            profile=profile,
            sparse_spec=sparse,
            sku_spec=sku,
            protocol=protocol,
        )


def test_latency_protocol_requires_layered_navigation_aggregations(
    tmp_path: Path,
) -> None:
    path = tmp_path / "experiments.toml"
    payload = (ROOT / "config/experiments.toml").read_text().replace(
        "include_layered_navigation_aggregations = true",
        "include_layered_navigation_aggregations = false",
    )
    path.write_text(payload)

    with pytest.raises(ConfigError, match="layered-navigation aggregations"):
        load_latency_protocol(path)


def test_latency_method_attribution_rejects_tampered_configuration() -> None:
    method_config: dict[str, object] = {
        "bm25_profile": {"name": "tuned", "fields": ("title", "description")},
        "neural_sparse_spec": {"model_name": "sparse"},
    }
    stored_config = json.loads(json.dumps(method_config))
    requests_sha256 = "a" * 64
    result = {
        "method_config": stored_config,
        "method_config_sha256": canonical_sha256(method_config),
        "request_definitions_sha256": requests_sha256,
    }

    verify_latency_method_attribution(
        result,
        expected_method_config=method_config,
        expected_requests_sha256=requests_sha256,
    )

    result["method_config"] = {**method_config, "cheaper_route": True}
    with pytest.raises(DatasetIntegrityError, match="method attribution"):
        verify_latency_method_attribution(
            result,
            expected_method_config=method_config,
            expected_requests_sha256=requests_sha256,
        )


def test_latency_measurements_are_recomputed_from_raw_samples() -> None:
    workload = [
        LatencyQuery("one", "first", "hybrid"),
        LatencyQuery("two", "second", "lexical_only"),
    ]
    protocol = LatencyProtocol(
        concurrency=(1,),
        target_concurrency=1,
        p95_budget_ms=200.0,
        minimum_distinct_queries=2,
        pagination_depth=100,
        include_layered_navigation_aggregations=True,
        allow_query_vector_cache=False,
        maximum_replay_relative_percentile_drift=0.5,
    )
    raw = {
        "cold": [
            {"query_id": "one", "route": "hybrid", "elapsed_ms": 10.0},
            {"query_id": "two", "route": "lexical_only", "elapsed_ms": 20.0},
        ],
        "warm": {
            "1": [
                {"query_id": "one", "route": "hybrid", "elapsed_ms": 5.0},
                {
                    "query_id": "two",
                    "route": "lexical_only",
                    "elapsed_ms": 15.0,
                },
            ]
        },
    }
    result = {
        "raw_samples": raw,
        "cold_measurement": {
            "samples": 2,
            "mean_ms": 15.0,
            "p50_ms": 15.0,
            "p95_ms": 19.5,
            "p99_ms": 19.9,
            "maximum_ms": 20.0,
        },
        "cold_measurement_by_route": {
            "hybrid": {
                "samples": 1,
                "mean_ms": 10.0,
                "p50_ms": 10.0,
                "p95_ms": 10.0,
                "p99_ms": 10.0,
                "maximum_ms": 10.0,
            },
            "lexical_only": {
                "samples": 1,
                "mean_ms": 20.0,
                "p50_ms": 20.0,
                "p95_ms": 20.0,
                "p99_ms": 20.0,
                "maximum_ms": 20.0,
            },
        },
        "measurements": {
            "1": {
                "samples": 2,
                "mean_ms": 10.0,
                "p50_ms": 10.0,
                "p95_ms": 14.5,
                "p99_ms": 14.9,
                "maximum_ms": 15.0,
            }
        },
        "measurements_by_route": {
            "1": {
                "hybrid": {
                    "samples": 1,
                    "mean_ms": 5.0,
                    "p50_ms": 5.0,
                    "p95_ms": 5.0,
                    "p99_ms": 5.0,
                    "maximum_ms": 5.0,
                },
                "lexical_only": {
                    "samples": 1,
                    "mean_ms": 15.0,
                    "p50_ms": 15.0,
                    "p95_ms": 15.0,
                    "p99_ms": 15.0,
                    "maximum_ms": 15.0,
                },
            }
        },
    }

    verify_latency_measurements_from_raw(
        result,
        workload=workload,
        protocol=protocol,
    )

    result["measurements"]["1"]["p95_ms"] = 1.0  # type: ignore[index]
    with pytest.raises(DatasetIntegrityError, match="raw samples"):
        verify_latency_measurements_from_raw(
            result,
            workload=workload,
            protocol=protocol,
        )


def test_latency_decision_gate_is_remeasured_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload = [
        LatencyQuery("one", "first", "hybrid"),
        LatencyQuery("two", "second", "lexical_only"),
    ]
    protocol = LatencyProtocol(
        concurrency=(1, 4),
        target_concurrency=4,
        p95_budget_ms=200.0,
        minimum_distinct_queries=2,
        pagination_depth=100,
        include_layered_navigation_aggregations=True,
        allow_query_vector_cache=False,
        maximum_replay_relative_percentile_drift=0.5,
    )
    requests = {query.query_id: {"query": query.text} for query in workload}
    elapsed_by_query = {"one": 90.0, "two": 110.0}
    measured: list[str] = []

    def measure(
        client: object,
        *,
        index: str,
        query: LatencyQuery,
        request: dict[str, Any],
        request_cache: bool,
    ) -> tuple[str, float]:
        del client, index, request, request_cache
        measured.append(query.query_id)
        return query.route, elapsed_by_query[query.query_id]

    class FakeClient:
        def request(
            self,
            method: str,
            path: str,
            *,
            params: dict[str, str] | None = None,
        ) -> dict[str, object]:
            assert method == "POST"
            assert path == "/fixture/_cache/clear"
            assert params == {
                "fielddata": "true",
                "query": "true",
                "request": "true",
            }
            return {}

    monkeypatch.setattr("poc.end_to_end_latency._measure_one", measure)
    replay = replay_latency_decision_gate(
        cast(Any, FakeClient()),
        index="fixture",
        workload=workload,
        requests=requests,
        protocol=protocol,
    )

    assert measured[:2] == ["one", "two"]
    assert sorted(measured[2:]) == ["one", "two"]
    assert replay["target_concurrency"] == 4
    assert replay["registered_p95_gate_passes"] is True
    elapsed_by_query["two"] = 250.0
    replay = replay_latency_decision_gate(
        cast(Any, FakeClient()),
        index="fixture",
        workload=workload,
        requests=requests,
        protocol=protocol,
    )
    assert replay["registered_p95_gate_passes"] is False


def test_latency_replay_rejects_materially_forged_stored_percentiles() -> None:
    stored = {
        "samples": 2,
        "p50_ms": 1.0,
        "p95_ms": 1.0,
        "p99_ms": 1.0,
    }
    live = {
        "samples": 2,
        "p50_ms": 90.0,
        "p95_ms": 100.0,
        "p99_ms": 110.0,
    }

    with pytest.raises(DatasetIntegrityError, match="repeatability"):
        verify_latency_replay_repeatability(
            stored,
            live,
            maximum_relative_drift=0.5,
        )
