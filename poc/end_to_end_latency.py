from __future__ import annotations

import json
import math
import platform
import subprocess
import time
import tomllib
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from poc.bm25 import load_prepared_queries, verify_wands_bm25_selection
from poc.config import ConfigError
from poc.datasets import (
    DatasetIntegrityError,
    WandsDatasetSpec,
    file_facts,
    load_wands_config,
)
from poc.experiments import BM25Profile, load_bm25_experiments
from poc.index_evidence import (
    content_addressed_pipeline_id,
    verify_live_search_pipeline,
)
from poc.indexing import load_wands_index_config
from poc.latency import summarize_latency_ms
from poc.manifest import (
    canonical_sha256,
    collect_index_facts,
    read_json,
    write_json,
)
from poc.neural_sparse import (
    NeuralSparseSpec,
    build_sparse_query,
    load_neural_sparse_spec,
)
from poc.os_client import OpenSearchClient
from poc.provenance import (
    BenchmarkProfile,
    ProvenanceError,
    collect_manifest_provenance,
    load_benchmark_profile,
    require_registered_opensearch_client,
    verify_container_resources,
    verify_decision_provenance,
)
from poc.search import build_normalization_pipeline, build_two_clause_hybrid_request
from poc.sku_slice import (
    SkuSliceSpec,
    build_known_item_lexical_query,
    load_sku_queries,
    load_sku_slice_spec,
)
from poc.sku_slice_indexing import (
    verify_sku_slice_index,
    verify_sku_slice_preparation,
)
from poc.wands import verify_prepared_wands

PIPELINE_DEFINITION = build_normalization_pipeline(lexical_weight=0.3)
PIPELINE_ID = content_addressed_pipeline_id(
    "opensearch-hybrid-wands-routed-sparse-latency-lw030-v1",
    PIPELINE_DEFINITION,
)
LATENCY_SCHEMA_VERSION = 2
LATENCY_DATASET = "WANDS plus synthetic known-item latency workload"
LATENCY_METHOD = "known_item_guarded_bm25_neural_sparse"


@dataclass(frozen=True, slots=True)
class LatencyProtocol:
    concurrency: tuple[int, ...]
    target_concurrency: int
    p95_budget_ms: float
    minimum_distinct_queries: int
    pagination_depth: int
    include_layered_navigation_aggregations: bool
    allow_query_vector_cache: bool
    maximum_replay_relative_percentile_drift: float


@dataclass(frozen=True, slots=True)
class LatencyQuery:
    query_id: str
    text: str
    route: str


def verify_latency_identity(result: Mapping[str, object]) -> None:
    if (
        result.get("schema_version") != LATENCY_SCHEMA_VERSION
        or result.get("dataset") != LATENCY_DATASET
        or result.get("method") != LATENCY_METHOD
    ):
        raise DatasetIntegrityError("latency result identity differs")


def verify_latency_index_identity(index: str, sku_spec: SkuSliceSpec) -> None:
    if index != sku_spec.index_name:
        raise DatasetIntegrityError("latency index differs from the registered SKU index")


def verify_latency_registered_config(
    client: OpenSearchClient,
    *,
    root: Path,
    dataset_spec: WandsDatasetSpec,
    profile: BM25Profile,
    sparse_spec: NeuralSparseSpec,
    sku_spec: SkuSliceSpec,
    protocol: LatencyProtocol,
) -> dict[str, object]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    datasets_path = root / "config/datasets.toml"
    registered_dataset = load_wands_config(datasets_path)
    if dataset_spec != registered_dataset:
        raise DatasetIntegrityError("latency caller differs from registered WANDS dataset")
    sku_slice_path = root / "config/sku_slice.toml"
    registered_sku = load_sku_slice_spec(sku_slice_path)
    if sku_spec != registered_sku:
        raise DatasetIntegrityError("latency caller differs from registered SKU slice spec")
    selection_path = root / "results/wands/bm25-selection.json"
    experiments_path = root / "config/experiments.toml"
    experiments = load_bm25_experiments(experiments_path)
    wands_index = load_wands_index_config(root / "config/indexes.toml")
    selection = verify_wands_bm25_selection(
        client,
        root=root,
        index=wands_index.name,
        experiments=experiments,
    )
    if selection.get("quality_evidence_eligible_for_decision") is not True:
        raise DatasetIntegrityError(
            "registered BM25 selection is not eligible for latency evidence"
        )
    selected_profile = selection.get("selected_profile")
    selected_profile_hash = canonical_sha256(selected_profile)
    if (
        selection.get("schema_version") != 2
        or selection.get("dataset") != "WANDS"
        or not isinstance(selected_profile, Mapping)
        or selection.get("selected_profile_sha256") != selected_profile_hash
        or canonical_sha256(profile.to_dict()) != selected_profile_hash
    ):
        raise DatasetIntegrityError("latency caller differs from registered BM25 profile")
    neural_sparse_path = root / "config/neural_sparse.toml"
    registered_sparse = load_neural_sparse_spec(neural_sparse_path)
    if sparse_spec != registered_sparse:
        raise DatasetIntegrityError("latency caller differs from registered neural sparse spec")
    registered_protocol = load_latency_protocol(experiments_path)
    if protocol != registered_protocol:
        raise DatasetIntegrityError("latency caller differs from registered latency protocol")
    return {
        "bm25_selection": _artifact(root, selection_path),
        "bm25_selection_quality_evidence_eligible_for_decision": True,
        "selected_bm25_profile": profile.to_dict(),
        "selected_bm25_profile_sha256": selected_profile_hash,
        "neural_sparse_config": _artifact(root, neural_sparse_path),
        "neural_sparse_spec": asdict(registered_sparse),
        "experiments_config": _artifact(root, experiments_path),
        "latency_protocol": asdict(registered_protocol),
        "datasets_config": _artifact(root, datasets_path),
        "wands_dataset_spec": asdict(registered_dataset),
        "sku_slice_config": _artifact(root, sku_slice_path),
        "sku_slice_spec": asdict(registered_sku),
    }


def latency_decision_status(
    *,
    architecture_eligible: bool,
    run_provenance_eligible: bool,
    sku_index_eligible: bool,
) -> tuple[bool, str | None]:
    if not architecture_eligible:
        return False, "registered latency architecture is x86_64"
    if not run_provenance_eligible:
        return False, "latency evidence requires clean committed source"
    if not sku_index_eligible:
        return False, "synthetic SKU index is not latency decision-eligible"
    return True, None


def normalize_plugin_inventory(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise DatasetIntegrityError("latency plugin inventory is missing")
    inventory: list[tuple[str, str]] = []
    for plugin in value:
        if not isinstance(plugin, Mapping):
            raise DatasetIntegrityError("latency plugin inventory schema differs")
        component = plugin.get("component")
        version = plugin.get("version")
        if (
            not isinstance(component, str)
            or not component
            or not isinstance(version, str)
            or not version
        ):
            raise DatasetIntegrityError("latency plugin inventory schema differs")
        inventory.append((component, version))
    if len(inventory) != len(set(inventory)):
        raise DatasetIntegrityError("latency plugin inventory contains duplicates")
    return [
        {"component": component, "version": version}
        for component, version in sorted(inventory)
    ]


def verify_latency_preparations(
    *,
    root: Path,
    dataset_spec: WandsDatasetSpec,
    sku_spec: SkuSliceSpec,
) -> dict[str, object]:
    prepared_directory = root / "data/prepared/wands"
    wands_manifest_path = prepared_directory / "manifest.json"
    wands_summary = verify_prepared_wands(
        dataset_spec,
        root / "data/raw/wands",
        prepared_directory,
    )
    sku_preparation_path = root / "results/wands/sku-slice/preparation.json"
    sku_preparation = verify_sku_slice_preparation(
        root=root,
        sku_spec=sku_spec,
        products_path=prepared_directory / "products.jsonl",
        preparation_path=sku_preparation_path,
    )
    return {
        "wands": {
            "manifest": _artifact(root, wands_manifest_path),
            "verification": wands_summary,
        },
        "sku_slice": {
            "manifest": _artifact(root, sku_preparation_path),
            "queries": sku_preparation["queries"],
            "qrels": sku_preparation["qrels"],
        },
    }


def latency_method_config(
    profile: BM25Profile,
    sparse_spec: NeuralSparseSpec,
) -> dict[str, object]:
    return {
        "bm25_profile": profile.to_dict(),
        "neural_sparse_spec": asdict(sparse_spec),
        "routing": {
            "hybrid": "bm25_plus_document_only_neural_sparse_minmax_lw030",
            "lexical_only": "known_item_guarded_bm25",
        },
    }


def verify_latency_method_attribution(
    result: dict[str, Any],
    *,
    expected_method_config: Mapping[str, object],
    expected_requests_sha256: str,
) -> None:
    expected_hash = canonical_sha256(expected_method_config)
    if (
        canonical_sha256(result.get("method_config")) != expected_hash
        or result.get("method_config_sha256") != expected_hash
        or result.get("request_definitions_sha256") != expected_requests_sha256
    ):
        raise DatasetIntegrityError("latency method attribution differs")


def verify_latency_measurements_from_raw(
    result: dict[str, Any],
    *,
    workload: list[LatencyQuery],
    protocol: LatencyProtocol,
) -> None:
    raw = result.get("raw_samples")
    if not isinstance(raw, dict):
        raise DatasetIntegrityError("latency raw samples are missing")
    cold_values, cold_routes = _validated_raw_latency_samples(
        raw.get("cold"),
        workload=workload,
    )
    expected_cold_routes = {
        route: summarize_latency_ms(
            [
                elapsed
                for sample_route, elapsed in cold_routes
                if sample_route == route
            ]
        )
        for route in ("hybrid", "lexical_only")
    }
    if (
        result.get("cold_measurement") != summarize_latency_ms(cold_values)
        or result.get("cold_measurement_by_route") != expected_cold_routes
    ):
        raise DatasetIntegrityError("latency cold summary differs from raw samples")
    warm = raw.get("warm")
    if not isinstance(warm, dict) or set(warm) != {
        str(value) for value in protocol.concurrency
    }:
        raise DatasetIntegrityError("latency warm raw samples differ")
    expected_measurements: dict[str, object] = {}
    expected_routes: dict[str, object] = {}
    for concurrency in protocol.concurrency:
        key = str(concurrency)
        values, routes = _validated_raw_latency_samples(
            warm[key],
            workload=workload,
        )
        expected_measurements[key] = summarize_latency_ms(values)
        expected_routes[key] = {
            route: summarize_latency_ms(
                [
                    elapsed
                    for sample_route, elapsed in routes
                    if sample_route == route
                ]
            )
            for route in ("hybrid", "lexical_only")
        }
    if (
        result.get("measurements") != expected_measurements
        or result.get("measurements_by_route") != expected_routes
    ):
        raise DatasetIntegrityError("latency summaries differ from raw samples")


def _validated_raw_latency_samples(
    value: object,
    *,
    workload: list[LatencyQuery],
) -> tuple[list[float], list[tuple[str, float]]]:
    if not isinstance(value, list) or len(value) != len(workload):
        raise DatasetIntegrityError("latency raw sample count differs")
    elapsed_values: list[float] = []
    routes: list[tuple[str, float]] = []
    for expected, record in zip(workload, value, strict=True):
        if not isinstance(record, dict) or set(record) != {
            "elapsed_ms",
            "query_id",
            "route",
        }:
            raise DatasetIntegrityError("latency raw sample schema differs")
        elapsed = record.get("elapsed_ms")
        if (
            record.get("query_id") != expected.query_id
            or record.get("route") != expected.route
            or not isinstance(elapsed, int | float)
            or isinstance(elapsed, bool)
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0.0
        ):
            raise DatasetIntegrityError("latency raw sample value differs")
        numeric = float(elapsed)
        elapsed_values.append(numeric)
        routes.append((expected.route, numeric))
    return elapsed_values, routes


def verify_latency_environment(
    environment: dict[str, Any],
    *,
    profile: BenchmarkProfile,
    host_architecture: str,
    live_root: dict[str, Any],
    live_container: dict[str, Any],
    container_architecture: str,
    live_plugins: object,
) -> dict[str, object]:
    if environment.get("schema_version") != 2:
        raise DatasetIntegrityError("latency environment schema differs")
    if (
        environment.get("profile_id") != profile.profile_id
        or environment.get("compose_files") != list(profile.compose_files)
        or environment.get("cpu_limit") != profile.cpu_limit
        or environment.get("memory_limit_bytes") != profile.memory_limit_bytes
        or environment.get("limits_verified") is not True
    ):
        raise DatasetIntegrityError("latency environment resource profile differs")
    try:
        live_resources = verify_container_resources(
            live_container,
            profile,
            architecture=container_architecture,
        )
    except ProvenanceError as error:
        raise DatasetIntegrityError(
            f"latency live container resources differ: {error}"
        ) from error
    container_config = cast(dict[str, Any], live_container.get("Config", {}))
    labels = cast(dict[str, str], container_config.get("Labels") or {})
    compose_label = labels.get("com.docker.compose.project.config_files", "")
    live_compose_files = tuple(
        Path(value).name for value in compose_label.split(",") if value.strip()
    )
    if live_compose_files != profile.compose_files:
        raise DatasetIntegrityError("latency live container Compose files differ")
    for key in (
        "container_id",
        "image",
        "image_id",
        "cpu_limit",
        "memory_limit_bytes",
        "architecture",
        "latency_architecture_eligible",
        "limits_verified",
    ):
        if environment.get(key) != live_resources.get(key):
            label = (
                "architecture eligibility"
                if key == "latency_architecture_eligible"
                else key.replace("_", " ")
            )
            raise DatasetIntegrityError(
                f"latency live container {label} differs"
            )
    recorded_architecture = _normalize_architecture(str(environment["architecture"]))
    actual_architecture = _normalize_architecture(host_architecture)
    if recorded_architecture != actual_architecture:
        raise DatasetIntegrityError("latency environment host architecture differs")
    expected_architecture_eligible = (
        actual_architecture == profile.latency_required_architecture
    )
    if (
        environment.get("latency_architecture_eligible")
        is not expected_architecture_eligible
    ):
        raise DatasetIntegrityError("latency environment architecture eligibility differs")
    container_id = environment.get("container_id")
    image = environment.get("image")
    opensearch = environment.get("opensearch")
    if (
        not isinstance(container_id, str)
        or len(container_id) < 12
        or not isinstance(image, str)
        or not image
        or not isinstance(opensearch, dict)
    ):
        raise DatasetIntegrityError("latency environment container identity is missing")
    live_version = cast(dict[str, Any], live_root.get("version", {})).get("number")
    node_name = opensearch.get("node_name")
    if (
        not isinstance(node_name, str)
        or not container_id.startswith(node_name)
        or live_root.get("name") != node_name
        or live_version != opensearch.get("version")
    ):
        raise DatasetIntegrityError("latency environment live OpenSearch identity differs")
    recorded_plugins = opensearch.get("plugins")
    normalized_recorded_plugins = normalize_plugin_inventory(recorded_plugins)
    if recorded_plugins != normalized_recorded_plugins:
        raise DatasetIntegrityError("latency environment plugin inventory schema differs")
    normalized_live_plugins = normalize_plugin_inventory(live_plugins)
    if normalized_recorded_plugins != normalized_live_plugins:
        raise DatasetIntegrityError("latency environment live plugin inventory differs")
    return {
        "profile_id": profile.profile_id,
        "compose_files": list(profile.compose_files),
        "container_id": container_id,
        "image": image,
        "image_id": live_resources["image_id"],
        "cpu_limit": profile.cpu_limit,
        "memory_limit_bytes": profile.memory_limit_bytes,
        "limits_verified": True,
        "host_architecture": actual_architecture,
        "latency_architecture_eligible": expected_architecture_eligible,
        "opensearch_node_name": node_name,
        "opensearch_version": live_version,
        "opensearch_plugins": normalized_live_plugins,
    }


def inspect_latency_container(environment: dict[str, Any]) -> tuple[dict[str, Any], str]:
    container_id = environment.get("container_id")
    if not isinstance(container_id, str) or len(container_id) < 12:
        raise DatasetIntegrityError("latency environment container identity is missing")
    try:
        inspect_result = subprocess.run(
            ["docker", "inspect", container_id],
            check=True,
            capture_output=True,
            text=True,
        )
        values = json.loads(inspect_result.stdout)
        architecture_result = subprocess.run(
            ["docker", "exec", container_id, "uname", "-m"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise DatasetIntegrityError(
            "latency live Docker inspection failed"
        ) from error
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
        raise DatasetIntegrityError(
            "latency live Docker inspection returned an invalid container"
        )
    architecture = architecture_result.stdout.strip()
    if not architecture:
        raise DatasetIntegrityError("latency live container architecture is missing")
    return cast(dict[str, Any], values[0]), architecture


def load_latency_protocol(path: Path) -> LatencyProtocol:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    decision = raw.get("decision")
    latency = raw.get("latency")
    if raw.get("schema_version") != 1 or not isinstance(
        decision, dict
    ) or not isinstance(latency, dict):
        raise ConfigError("experiment latency protocol is missing")
    concurrency = latency.get("concurrency")
    if (
        not isinstance(concurrency, list)
        or not concurrency
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in concurrency
        )
    ):
        raise ConfigError("latency concurrency must contain positive integers")
    target = decision.get("latency_concurrency")
    budget = decision.get("latency_p95_ms")
    minimum = decision.get("minimum_distinct_latency_queries")
    depth = latency.get("pagination_depth")
    aggregations = latency.get("include_layered_navigation_aggregations")
    cache = latency.get("allow_query_vector_cache")
    replay_drift = latency.get("maximum_replay_relative_percentile_drift")
    if not isinstance(target, int) or isinstance(target, bool) or target not in concurrency:
        raise ConfigError("latency target concurrency is invalid")
    if not isinstance(budget, int | float) or isinstance(budget, bool) or budget <= 0:
        raise ConfigError("latency p95 budget is invalid")
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum <= 0:
        raise ConfigError("latency distinct-query floor is invalid")
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 10:
        raise ConfigError("latency pagination depth is invalid")
    if aggregations is not True:
        raise ConfigError(
            "latency gate requires layered-navigation aggregations"
        )
    if not isinstance(cache, bool):
        raise ConfigError("latency aggregation or cache flags are invalid")
    if (
        not isinstance(replay_drift, int | float)
        or isinstance(replay_drift, bool)
        or not 0.0 <= replay_drift < 1.0
    ):
        raise ConfigError("latency replay percentile drift limit is invalid")
    return LatencyProtocol(
        concurrency=tuple(concurrency),
        target_concurrency=target,
        p95_budget_ms=float(budget),
        minimum_distinct_queries=minimum,
        pagination_depth=depth,
        include_layered_navigation_aggregations=aggregations,
        allow_query_vector_cache=cache,
        maximum_replay_relative_percentile_drift=float(replay_drift),
    )


def load_latency_workload(root: Path) -> list[LatencyQuery]:
    general: dict[str, str] = {}
    for split in ("dev", "test"):
        path = root / f"data/prepared/wands/queries.{split}.jsonl"
        for query_id, text in load_prepared_queries(
            path, expected_split=split
        ).items():
            general[f"wands-{split}-{query_id}"] = text
    known = load_sku_queries(
        root / "data/prepared/wands/queries.sku-slice.jsonl"
    )
    workload = [
        LatencyQuery(query_id, text, "hybrid")
        for query_id, text in sorted(general.items())
    ]
    workload.extend(
        LatencyQuery(f"known-{query_id}", query["query"], "lexical_only")
        for query_id, query in sorted(known.items())
    )
    return workload


def latency_workload_source_artifacts(root: Path) -> list[dict[str, object]]:
    return [
        _artifact(root, root / f"data/prepared/wands/queries.{split}.jsonl")
        for split in ("dev", "test")
    ] + [
        _artifact(root, root / "data/prepared/wands/queries.sku-slice.jsonl")
    ]


def build_latency_request(
    query: LatencyQuery,
    *,
    profile: BM25Profile,
    sparse_spec: NeuralSparseSpec,
    protocol: LatencyProtocol,
) -> dict[str, Any]:
    if query.route == "hybrid":
        request = build_two_clause_hybrid_request(
            profile.query(query.text),
            build_sparse_query(sparse_spec, query.text),
            size=10,
            pagination_depth=protocol.pagination_depth,
        )
    elif query.route == "lexical_only":
        request = {
            "size": 10,
            "_source": False,
            "track_scores": True,
            "query": build_known_item_lexical_query(query.text),
        }
    else:
        raise ValueError(f"unsupported latency query route {query.route}")
    if protocol.include_layered_navigation_aggregations:
        request["aggs"] = {
            "brand": {"terms": {"field": "brand.facet", "size": 20}},
            "category": {"terms": {"field": "category.facet", "size": 20}},
            "product_class": {
                "terms": {"field": "product_class.facet", "size": 20}
            },
        }
    return request


def run_routed_sparse_latency(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    dataset_spec: WandsDatasetSpec,
    profile: BM25Profile,
    sparse_spec: NeuralSparseSpec,
    sku_spec: SkuSliceSpec,
    protocol: LatencyProtocol,
) -> dict[str, object]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    benchmark_provenance = collect_manifest_provenance(root)
    verify_latency_index_identity(index, sku_spec)
    registered_configuration = verify_latency_registered_config(
        client,
        root=root,
        dataset_spec=dataset_spec,
        profile=profile,
        sparse_spec=sparse_spec,
        sku_spec=sku_spec,
        protocol=protocol,
    )
    environment_path = root / "results/environment/benchmark-profile.json"
    environment = cast(dict[str, Any], read_json(environment_path))
    benchmark_profile = load_benchmark_profile(root / "config/benchmark.toml")
    root_response = cast(dict[str, Any], client.request("GET", "/"))
    live_plugins = client.request(
        "GET", "/_cat/plugins?format=json&h=component,version"
    )
    architecture = _normalize_architecture(platform.machine())
    live_container, container_architecture = inspect_latency_container(environment)
    resource_profile = verify_latency_environment(
        environment,
        profile=benchmark_profile,
        host_architecture=architecture,
        live_root=root_response,
        live_container=live_container,
        container_architecture=container_architecture,
        live_plugins=live_plugins,
    )
    preparation_evidence = verify_latency_preparations(
        root=root,
        dataset_spec=dataset_spec,
        sku_spec=sku_spec,
    )
    index_manifest_path = root / "results/wands/sku-slice/index-manifest.json"
    index_manifest = verify_sku_slice_index(
        client,
        root=root,
        dataset_spec=dataset_spec,
        sparse_spec=sparse_spec,
        sku_spec=sku_spec,
        products_path=root / "data/prepared/wands/products.jsonl",
        embeddings_path=(
            root / "data/cache/neural-sparse/wands-doc-v3-distill.jsonl"
        ),
        preparation_path=root / "results/wands/sku-slice/preparation.json",
        precompute_manifest_path=(
            root / "results/wands/neural-sparse/precompute-manifest.json"
        ),
        manifest_path=index_manifest_path,
    )
    sku_index_eligible = (
        index_manifest.get("latency_evidence_eligible_for_decision") is True
    )
    workload = load_latency_workload(root)
    distinct_texts = {" ".join(query.text.casefold().split()) for query in workload}
    if len(distinct_texts) < protocol.minimum_distinct_queries:
        raise DatasetIntegrityError("latency workload has too few distinct queries")
    pipeline = build_normalization_pipeline(lexical_weight=0.3)
    client.put_search_pipeline(PIPELINE_ID, pipeline)
    verify_live_search_pipeline(
        client,
        pipeline_id=PIPELINE_ID,
        expected_definition=pipeline,
    )
    requests = {
        query.query_id: build_latency_request(
            query,
            profile=profile,
            sparse_spec=sparse_spec,
            protocol=protocol,
        )
        for query in workload
    }
    method_config = latency_method_config(profile, sparse_spec)
    requests_sha256 = canonical_sha256(requests)
    client.request(
        "POST",
        f"/{index}/_cache/clear",
        params={"fielddata": "true", "query": "true", "request": "true"},
    )
    cold_samples = [
        _measure_one(
            client,
            index=index,
            query=query,
            request=requests[query.query_id],
            request_cache=protocol.allow_query_vector_cache,
        )
        for query in workload
    ]
    cold_raw = [
        {
            "query_id": query.query_id,
            "route": route,
            "elapsed_ms": elapsed,
        }
        for query, (route, elapsed) in zip(workload, cold_samples, strict=True)
    ]
    for query in workload[:50]:
        _measure_one(
            client,
            index=index,
            query=query,
            request=requests[query.query_id],
            request_cache=protocol.allow_query_vector_cache,
        )
    measurements: dict[str, object] = {}
    route_samples: dict[str, dict[str, object]] = {}
    warm_raw: dict[str, list[dict[str, object]]] = {}
    for concurrency in protocol.concurrency:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            samples = list(
                executor.map(
                    lambda query: _measure_one(
                        client,
                        index=index,
                        query=query,
                        request=requests[query.query_id],
                        request_cache=protocol.allow_query_vector_cache,
                    ),
                    workload,
                )
            )
        measurements[str(concurrency)] = summarize_latency_ms(
            [elapsed for _, elapsed in samples]
        )
        route_samples[str(concurrency)] = {
            route: summarize_latency_ms(
                [elapsed for sample_route, elapsed in samples if sample_route == route]
            )
            for route in ("hybrid", "lexical_only")
        }
        warm_raw[str(concurrency)] = [
            {
                "query_id": query.query_id,
                "route": route,
                "elapsed_ms": elapsed,
            }
            for query, (route, elapsed) in zip(workload, samples, strict=True)
        ]
    verify_live_search_pipeline(
        client,
        pipeline_id=PIPELINE_ID,
        expected_definition=pipeline,
    )
    clean_committed_code = verify_decision_provenance(
        benchmark_provenance,
        root=root,
        profile_path=root / "config/benchmark.toml",
        environment_path=environment_path,
        require_current_code_revision=True,
    )
    architecture_eligible, ineligibility_reason = latency_decision_status(
        architecture_eligible=bool(
            resource_profile["latency_architecture_eligible"]
        ),
        run_provenance_eligible=clean_committed_code,
        sku_index_eligible=sku_index_eligible,
    )
    target = cast(dict[str, Any], measurements[str(protocol.target_concurrency)])
    p95 = float(target["p95_ms"])
    workload_source_files = latency_workload_source_artifacts(root)
    result: dict[str, object] = {
        "schema_version": LATENCY_SCHEMA_VERSION,
        "dataset": LATENCY_DATASET,
        "method": LATENCY_METHOD,
        "method_config": method_config,
        "method_config_sha256": canonical_sha256(method_config),
        "request_definitions_sha256": requests_sha256,
        "registered_configuration": registered_configuration,
        "opensearch_version": str(
            cast(dict[str, Any], root_response["version"])["number"]
        ),
        "index": asdict(collect_index_facts(client, index)),
        "index_manifest": {
            **_artifact(root, index_manifest_path),
            "schema_version": index_manifest["schema_version"],
            "latency_evidence_eligible_for_decision": sku_index_eligible,
        },
        "host_architecture": architecture,
        "latency_decision_eligible": architecture_eligible,
        "latency_ineligibility_reason": ineligibility_reason,
        "benchmark_provenance": benchmark_provenance,
        "implementation": _artifact(root, Path(__file__).resolve()),
        "benchmark_environment": {
            "artifact": _artifact(root, environment_path),
            "snapshot": environment,
        },
        "resource_profile": resource_profile,
        "protocol": {
            "concurrency": list(protocol.concurrency),
            "target_concurrency": protocol.target_concurrency,
            "p95_budget_ms": protocol.p95_budget_ms,
            "pagination_depth": protocol.pagination_depth,
            "include_layered_navigation_aggregations": (
                protocol.include_layered_navigation_aggregations
            ),
            "allow_query_vector_cache": protocol.allow_query_vector_cache,
            "maximum_replay_relative_percentile_drift": (
                protocol.maximum_replay_relative_percentile_drift
            ),
            "cold_cache_clear": ["fielddata", "query", "request"],
            "cold_definition": (
                "first complete workload pass after OpenSearch index-cache clear"
            ),
            "warm_definition": (
                "complete workload passes after 50 explicit warmup queries"
            ),
            "warmup_queries": 50,
        },
        "workload": {
            "queries": len(workload),
            "distinct_normalized_texts": len(distinct_texts),
            "minimum_distinct_queries": protocol.minimum_distinct_queries,
            "query_sha256": canonical_sha256(
                [
                    {"id": query.query_id, "text": query.text, "route": query.route}
                    for query in workload
                ]
            ),
            "source_files": workload_source_files,
            "preparation_manifests": preparation_evidence,
        },
        "pipeline_id": PIPELINE_ID,
        "pipeline": pipeline,
        "pipeline_sha256": canonical_sha256(pipeline),
        "cold_measurement": summarize_latency_ms(
            [elapsed for _, elapsed in cold_samples]
        ),
        "raw_samples": {"cold": cold_raw, "warm": warm_raw},
        "cold_measurement_by_route": {
            route: summarize_latency_ms(
                [
                    elapsed
                    for sample_route, elapsed in cold_samples
                    if sample_route == route
                ]
            )
            for route in ("hybrid", "lexical_only")
        },
        "measurements": measurements,
        "measurements_by_route": route_samples,
        "registered_p95_gate_passes": p95 <= protocol.p95_budget_ms,
        "supports_latency_decision_gate": (
            architecture_eligible and p95 <= protocol.p95_budget_ms
        ),
    }
    return result


def _measure_one(
    client: OpenSearchClient,
    *,
    index: str,
    query: LatencyQuery,
    request: dict[str, Any],
    request_cache: bool,
) -> tuple[str, float]:
    started = time.perf_counter()
    response = client.search(
        index,
        request,
        pipeline=PIPELINE_ID if query.route == "hybrid" else None,
        request_cache=request_cache,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    hits = cast(dict[str, Any], response.get("hits", {})).get("hits")
    if not isinstance(hits, list) or not hits:
        raise DatasetIntegrityError(f"latency query {query.query_id} returned no hits")
    if "aggregations" not in response:
        raise DatasetIntegrityError(
            f"latency query {query.query_id} returned no layered-navigation aggregations"
        )
    return query.route, elapsed_ms


def replay_latency_decision_gate(
    client: OpenSearchClient,
    *,
    index: str,
    workload: list[LatencyQuery],
    requests: Mapping[str, dict[str, Any]],
    protocol: LatencyProtocol,
) -> dict[str, object]:
    """Independently reproduce the registered warm p95 gate during verification."""
    if not workload:
        raise DatasetIntegrityError("latency verification workload is empty")
    client.request(
        "POST",
        f"/{index}/_cache/clear",
        params={"fielddata": "true", "query": "true", "request": "true"},
    )
    for query in workload[:50]:
        _measure_one(
            client,
            index=index,
            query=query,
            request=requests[query.query_id],
            request_cache=protocol.allow_query_vector_cache,
        )
    with ThreadPoolExecutor(max_workers=protocol.target_concurrency) as executor:
        samples = list(
            executor.map(
                lambda query: _measure_one(
                    client,
                    index=index,
                    query=query,
                    request=requests[query.query_id],
                    request_cache=protocol.allow_query_vector_cache,
                ),
                workload,
            )
        )
    measurement = summarize_latency_ms([elapsed for _, elapsed in samples])
    by_route = {
        route: summarize_latency_ms(
            [elapsed for sample_route, elapsed in samples if sample_route == route]
        )
        for route in ("hybrid", "lexical_only")
    }
    return {
        "target_concurrency": protocol.target_concurrency,
        "measurement": measurement,
        "measurement_by_route": by_route,
        "registered_p95_gate_passes": (
            float(measurement["p95_ms"]) <= protocol.p95_budget_ms
        ),
    }


def write_latency_result(root: Path, result: dict[str, object]) -> Path:
    architecture = str(result["host_architecture"])
    path = root / f"results/wands/latency/routed-sparse-{architecture}.json"
    write_json(path, result)
    return path


def verify_latency_result(
    client: OpenSearchClient,
    *,
    root: Path,
    index: str,
    dataset_spec: WandsDatasetSpec,
    profile: BM25Profile,
    sparse_spec: NeuralSparseSpec,
    sku_spec: SkuSliceSpec,
    protocol: LatencyProtocol,
    path: Path,
) -> dict[str, Any]:
    require_registered_opensearch_client(client, root / "config/benchmark.toml")
    result = cast(dict[str, Any], read_json(path))
    verify_latency_identity(result)
    verify_latency_index_identity(index, sku_spec)
    registered_configuration = verify_latency_registered_config(
        client,
        root=root,
        dataset_spec=dataset_spec,
        profile=profile,
        sparse_spec=sparse_spec,
        sku_spec=sku_spec,
        protocol=protocol,
    )
    if result.get("registered_configuration") != registered_configuration:
        raise DatasetIntegrityError("latency registered configuration differs")
    environment_path = root / "results/environment/benchmark-profile.json"
    environment = cast(dict[str, Any], read_json(environment_path))
    recorded_environment = cast(
        dict[str, Any], result.get("benchmark_environment", {})
    )
    if (
        recorded_environment.get("artifact") != _artifact(root, environment_path)
        or recorded_environment.get("snapshot") != environment
    ):
        raise DatasetIntegrityError("latency benchmark environment differs")
    benchmark_profile = load_benchmark_profile(root / "config/benchmark.toml")
    live_root = cast(dict[str, Any], client.request("GET", "/"))
    live_plugins = client.request(
        "GET", "/_cat/plugins?format=json&h=component,version"
    )
    live_container, container_architecture = inspect_latency_container(environment)
    resource_profile = verify_latency_environment(
        environment,
        profile=benchmark_profile,
        host_architecture=platform.machine(),
        live_root=live_root,
        live_container=live_container,
        container_architecture=container_architecture,
        live_plugins=live_plugins,
    )
    if result.get("resource_profile") != resource_profile:
        raise DatasetIntegrityError("latency recorded resource profile differs")
    if result.get("host_architecture") != resource_profile["host_architecture"]:
        raise DatasetIntegrityError("latency recorded host architecture differs")
    if result.get("opensearch_version") != resource_profile["opensearch_version"]:
        raise DatasetIntegrityError("latency recorded OpenSearch version differs")
    provenance = cast(dict[str, Any], result.get("benchmark_provenance", {}))
    clean_committed_code = verify_decision_provenance(
        provenance,
        root=root,
        profile_path=root / "config/benchmark.toml",
        environment_path=environment_path,
    )
    if not clean_committed_code:
        raise DatasetIntegrityError("latency benchmark provenance differs")
    if result.get("implementation") != _artifact(root, Path(__file__).resolve()):
        raise DatasetIntegrityError("latency benchmark implementation differs")
    workload = load_latency_workload(root)
    distinct_texts = {" ".join(query.text.casefold().split()) for query in workload}
    expected_hash = canonical_sha256(
        [
            {"id": query.query_id, "text": query.text, "route": query.route}
            for query in workload
        ]
    )
    recorded_workload = cast(dict[str, Any], result.get("workload"))
    preparation_evidence = verify_latency_preparations(
        root=root,
        dataset_spec=dataset_spec,
        sku_spec=sku_spec,
    )
    expected_source_files = latency_workload_source_artifacts(root)
    if (
        recorded_workload.get("queries") != len(workload)
        or recorded_workload.get("distinct_normalized_texts") != len(distinct_texts)
        or recorded_workload.get("query_sha256") != expected_hash
        or recorded_workload.get("source_files") != expected_source_files
        or recorded_workload.get("preparation_manifests") != preparation_evidence
    ):
        raise DatasetIntegrityError("latency workload differs")
    expected_requests = {
        query.query_id: build_latency_request(
            query,
            profile=profile,
            sparse_spec=sparse_spec,
            protocol=protocol,
        )
        for query in workload
    }
    verify_latency_method_attribution(
        result,
        expected_method_config=latency_method_config(profile, sparse_spec),
        expected_requests_sha256=canonical_sha256(expected_requests),
    )
    verify_latency_measurements_from_raw(
        result,
        workload=workload,
        protocol=protocol,
    )
    expected_protocol = {
        "concurrency": list(protocol.concurrency),
        "target_concurrency": protocol.target_concurrency,
        "p95_budget_ms": protocol.p95_budget_ms,
        "pagination_depth": protocol.pagination_depth,
        "include_layered_navigation_aggregations": (
            protocol.include_layered_navigation_aggregations
        ),
        "allow_query_vector_cache": protocol.allow_query_vector_cache,
        "maximum_replay_relative_percentile_drift": (
            protocol.maximum_replay_relative_percentile_drift
        ),
        "cold_cache_clear": ["fielddata", "query", "request"],
        "cold_definition": (
            "first complete workload pass after OpenSearch index-cache clear"
        ),
        "warm_definition": "complete workload passes after 50 explicit warmup queries",
        "warmup_queries": 50,
    }
    if result.get("protocol") != expected_protocol:
        raise DatasetIntegrityError("latency protocol differs")
    if result.get("index") != asdict(collect_index_facts(client, index)):
        raise DatasetIntegrityError("latency index facts differ")
    index_manifest_path = root / "results/wands/sku-slice/index-manifest.json"
    index_manifest = verify_sku_slice_index(
        client,
        root=root,
        dataset_spec=dataset_spec,
        sparse_spec=sparse_spec,
        sku_spec=sku_spec,
        products_path=root / "data/prepared/wands/products.jsonl",
        embeddings_path=(
            root / "data/cache/neural-sparse/wands-doc-v3-distill.jsonl"
        ),
        preparation_path=root / "results/wands/sku-slice/preparation.json",
        precompute_manifest_path=(
            root / "results/wands/neural-sparse/precompute-manifest.json"
        ),
        manifest_path=index_manifest_path,
    )
    sku_index_eligible = (
        index_manifest.get("latency_evidence_eligible_for_decision") is True
    )
    expected_index_manifest = {
        **_artifact(root, index_manifest_path),
        "schema_version": index_manifest["schema_version"],
        "latency_evidence_eligible_for_decision": sku_index_eligible,
    }
    if result.get("index_manifest") != expected_index_manifest:
        raise DatasetIntegrityError("latency index manifest differs")
    pipeline = build_normalization_pipeline(lexical_weight=0.3)
    if (
        result.get("pipeline_id") != PIPELINE_ID
        or result.get("pipeline") != pipeline
        or result.get("pipeline_sha256") != canonical_sha256(pipeline)
    ):
        raise DatasetIntegrityError("latency pipeline differs")
    try:
        verify_live_search_pipeline(
            client,
            pipeline_id=PIPELINE_ID,
            expected_definition=pipeline,
        )
    except DatasetIntegrityError as error:
        raise DatasetIntegrityError(f"live latency pipeline differs: {error}") from error
    route_counts = {
        "hybrid": sum(query.route == "hybrid" for query in workload),
        "lexical_only": sum(query.route == "lexical_only" for query in workload),
    }
    _verify_measurement(
        cast(dict[str, Any], result["cold_measurement"]),
        samples=len(workload),
    )
    cold_routes = cast(dict[str, Any], result["cold_measurement_by_route"])
    for route, count in route_counts.items():
        _verify_measurement(cast(dict[str, Any], cold_routes[route]), samples=count)
    measurements = cast(dict[str, Any], result["measurements"])
    measurement_routes = cast(dict[str, Any], result["measurements_by_route"])
    if set(measurements) != {str(value) for value in protocol.concurrency}:
        raise DatasetIntegrityError("latency concurrency measurements differ")
    for concurrency in protocol.concurrency:
        key = str(concurrency)
        _verify_measurement(
            cast(dict[str, Any], measurements[key]), samples=len(workload)
        )
        for route, count in route_counts.items():
            _verify_measurement(
                cast(dict[str, Any], measurement_routes[key][route]),
                samples=count,
            )
    eligible, ineligibility_reason = latency_decision_status(
        architecture_eligible=bool(
            resource_profile["latency_architecture_eligible"]
        ),
        run_provenance_eligible=clean_committed_code,
        sku_index_eligible=sku_index_eligible,
    )
    if (
        result.get("latency_decision_eligible") is not eligible
        or result.get("latency_ineligibility_reason") != ineligibility_reason
    ):
        raise DatasetIntegrityError("latency architecture eligibility differs")
    target = cast(dict[str, Any], measurements[str(protocol.target_concurrency)])
    passes = float(target["p95_ms"]) <= protocol.p95_budget_ms
    if (
        result.get("registered_p95_gate_passes") is not passes
        or result.get("supports_latency_decision_gate") is not (eligible and passes)
    ):
        raise DatasetIntegrityError("latency gate result differs")
    verified = dict(result)
    if eligible:
        live_replay = replay_latency_decision_gate(
            client,
            index=index,
            workload=workload,
            requests=expected_requests,
            protocol=protocol,
        )
        try:
            verify_live_search_pipeline(
                client,
                pipeline_id=PIPELINE_ID,
                expected_definition=pipeline,
            )
        except DatasetIntegrityError as error:
            raise DatasetIntegrityError(
                f"latency pipeline changed during live replay: {error}"
            ) from error
        if live_replay["registered_p95_gate_passes"] is not passes:
            raise DatasetIntegrityError(
                "latency stored gate does not reproduce in a fresh live measurement"
            )
        verify_latency_replay_repeatability(
            target,
            cast(Mapping[str, object], live_replay["measurement"]),
            maximum_relative_drift=(
                protocol.maximum_replay_relative_percentile_drift
            ),
        )
        live_routes = cast(
            Mapping[str, Mapping[str, object]],
            live_replay["measurement_by_route"],
        )
        stored_routes = cast(
            Mapping[str, Mapping[str, object]],
            measurement_routes[str(protocol.target_concurrency)],
        )
        for route in ("hybrid", "lexical_only"):
            verify_latency_replay_repeatability(
                stored_routes[route],
                live_routes[route],
                maximum_relative_drift=(
                    protocol.maximum_replay_relative_percentile_drift
                ),
            )
        live_replay["repeatability_verification"] = {
            "status": "passed",
            "maximum_relative_percentile_drift": (
                protocol.maximum_replay_relative_percentile_drift
            ),
            "percentiles": ["p50_ms", "p95_ms", "p99_ms"],
            "surfaces": ["overall", "hybrid", "lexical_only"],
        }
        completion_index_manifest = verify_sku_slice_index(
            client,
            root=root,
            dataset_spec=dataset_spec,
            sparse_spec=sparse_spec,
            sku_spec=sku_spec,
            products_path=root / "data/prepared/wands/products.jsonl",
            embeddings_path=(
                root / "data/cache/neural-sparse/wands-doc-v3-distill.jsonl"
            ),
            preparation_path=root / "results/wands/sku-slice/preparation.json",
            precompute_manifest_path=(
                root / "results/wands/neural-sparse/precompute-manifest.json"
            ),
            manifest_path=index_manifest_path,
        )
        if completion_index_manifest != index_manifest:
            raise DatasetIntegrityError(
                "latency index changed during live verification replay"
            )
        verified["live_verification_replay"] = live_replay
    return verified


def verify_latency_replay_repeatability(
    recorded: Mapping[str, object],
    live: Mapping[str, object],
    *,
    maximum_relative_drift: float,
) -> None:
    if not 0.0 <= maximum_relative_drift < 1.0:
        raise ValueError("latency replay drift limit must be in [0, 1)")
    if recorded.get("samples") != live.get("samples"):
        raise DatasetIntegrityError("latency replay repeatability sample count differs")
    for key in ("p50_ms", "p95_ms", "p99_ms"):
        recorded_value = float(cast(int | float, recorded[key]))
        live_value = float(cast(int | float, live[key]))
        if (
            not math.isfinite(recorded_value)
            or not math.isfinite(live_value)
            or recorded_value < 0.0
            or live_value < 0.0
        ):
            raise DatasetIntegrityError(
                f"latency replay repeatability has invalid {key} values"
            )
        scale = max(recorded_value, live_value)
        relative_drift = (
            0.0 if scale == 0.0 else abs(live_value - recorded_value) / scale
        )
        if relative_drift > maximum_relative_drift:
            raise DatasetIntegrityError(
                "latency replay repeatability differs for "
                f"{key}: recorded={recorded_value}, live={live_value}, "
                f"relative_drift={relative_drift:.6f}, "
                f"limit={maximum_relative_drift:.6f}"
            )


def _verify_measurement(measurement: dict[str, Any], *, samples: int) -> None:
    if measurement.get("samples") != samples:
        raise DatasetIntegrityError("latency sample count differs")
    percentiles = [
        float(measurement[key]) for key in ("p50_ms", "p95_ms", "p99_ms")
    ]
    if percentiles != sorted(percentiles) or any(value < 0 for value in percentiles):
        raise DatasetIntegrityError("latency percentile ordering differs")


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }


def _normalize_architecture(value: str) -> str:
    normalized = value.strip().lower()
    return {
        "aarch64": "arm64",
        "arm64": "arm64",
        "amd64": "x86_64",
        "x86_64": "x86_64",
    }.get(normalized, normalized)
