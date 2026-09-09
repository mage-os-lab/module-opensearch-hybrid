from __future__ import annotations

import base64
import binascii
import math
import platform
import re
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, Protocol, cast
from urllib.parse import urlsplit

from poc.manifest import canonical_sha256

REGISTERED_OPENSEARCH_VERSION = "3.8.0"
REGISTERED_INDEX_DOCUMENT_COUNT = 50_000
REGISTERED_RUNTIME_SYSTEM = "Linux"
REGISTERED_RUNTIME_ARCHITECTURES = frozenset({"amd64", "x86_64"})
CATALOG_EVIDENCE_SCOPES = frozenset({"synthetic_execution", "representative_merchant"})
REPRESENTATIVE_MERCHANT_SCOPE = "representative_merchant"


class HttpResponse(Protocol):
    def raise_for_status(self) -> None: ...

    def json(self) -> Any: ...


class HttpClient(Protocol):
    def get(self, url: str) -> HttpResponse: ...

    def post(self, url: str, *, json: Any) -> HttpResponse: ...


def capture_radial_similarity(
    client: HttpClient,
    specification: dict[str, Any],
    endpoint_url: str | None = None,
) -> dict[str, Any]:
    _validate_specification(specification)
    index = cast(str, specification["index"])
    identity = dict(cast(dict[str, Any], specification["identity"]))
    generation_id = int(specification["generation_id"])
    thresholds = [float(value) for value in cast(list[int | float], specification["thresholds"])]
    cases = cast(list[dict[str, Any]], specification["cases"])
    maximum_result_count = int(specification["maximum_result_count"])
    concurrency = int(specification["concurrency"])
    latency_samples = int(specification["latency_samples"])

    info = _json_response(client.get("/"), "OpenSearch root response")
    version = str(cast(dict[str, Any], info.get("version", {})).get("number", ""))
    if re.fullmatch(r"3\.8\.\d+(?:[-+][0-9A-Za-z.-]+)?", version) is None:
        raise ValueError("calibration capture requires OpenSearch 3.8.x")
    _validate_mapping(client, index, identity)
    eligible_filter = _eligible_filter(identity, generation_id)
    eligible_document_count = _count(client, index, eligible_filter)
    if eligible_document_count <= 0:
        raise ValueError("the eligible calibration universe is empty")
    index_document_count = _count(client, index, {"match_all": {}})

    judgment_identity = {
        "schema_version": 1,
        "split": specification["split"],
        "store_id": identity["store_id"],
        "use_case": identity["use_case"],
        "cases": cases,
    }
    identity["judgment_set_sha256"] = canonical_sha256(judgment_identity)
    output_thresholds = [{"min_score": threshold, "cases": []} for threshold in thresholds]
    for case in cases:
        seed_product_id = int(case["seed_product_id"])
        seed_vector = _seed_vector(
            client,
            index,
            identity,
            generation_id,
            seed_product_id,
        )
        for threshold, output_threshold in zip(thresholds, output_thresholds, strict=True):
            exact_ids = _exact_ids(
                client,
                index,
                identity,
                generation_id,
                seed_product_id,
                seed_vector,
                threshold,
                int(specification["exact_capture_limit"]),
            )
            radial_ids, latencies = _radial_measurements(
                client,
                index,
                identity,
                generation_id,
                seed_product_id,
                seed_vector,
                threshold,
                maximum_result_count,
                concurrency,
                latency_samples,
            )
            cast(list[dict[str, Any]], output_threshold["cases"]).append(
                {
                    "case_id": case["case_id"],
                    "seed_product_id": seed_product_id,
                    "slices": list(case["slices"]),
                    "exact_ids": exact_ids,
                    "exact_total": len(exact_ids),
                    "radial_ids": radial_ids,
                    "radial_latency_ms": latencies,
                    "unsafe_ids": list(case["unsafe_ids"]),
                }
            )

    review_product_ids = {
        int(case["seed_product_id"])
        for case in cases
    }
    for captured_threshold in output_thresholds:
        for case in cast(list[dict[str, Any]], captured_threshold["cases"]):
            review_product_ids.update(cast(list[int], case["radial_ids"]))

    provenance = {
        "captured_at": datetime.now(UTC).isoformat(),
        "opensearch_version": version,
        "physical_index": index,
        "generation_id": generation_id,
        "index_document_count": index_document_count,
        "eligible_document_count": eligible_document_count,
        "runtime_system": platform.system(),
        "runtime_architecture": platform.machine().lower(),
        "loopback_endpoint": _is_loopback_endpoint(endpoint_url),
        "exact_method": "knn_score_script_cosinesimil",
        "radial_method": "lucene_hnsw_min_score",
        "latency_samples_per_case": latency_samples,
        "catalog_evidence_scope": specification["catalog_evidence_scope"],
        "specification_sha256": canonical_sha256(specification),
    }
    qualification = radial_capture_qualification(
        str(specification["qualification_scope"]),
        provenance,
    )

    return {
        "schema_version": 1,
        "split": specification["split"],
        "storefront_limit": int(specification["storefront_limit"]),
        "maximum_result_count": maximum_result_count,
        "concurrency": concurrency,
        "identity": identity,
        "thresholds": output_thresholds,
        "review_catalog": _review_catalog(
            client,
            index,
            sorted(review_product_ids),
        ),
        "qualification": qualification,
        "capture_provenance": provenance,
    }


def radial_capture_qualification(
    scope: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    if scope not in ("smoke", "registered"):
        raise ValueError("the radial capture qualification scope is invalid")
    architecture = str(provenance.get("runtime_architecture", "")).lower()
    checks = {
        "registered_scope": scope == "registered",
        "opensearch_version": provenance.get("opensearch_version")
        == REGISTERED_OPENSEARCH_VERSION,
        "index_document_count": provenance.get("index_document_count")
        == REGISTERED_INDEX_DOCUMENT_COUNT,
        "runtime_system": provenance.get("runtime_system") == REGISTERED_RUNTIME_SYSTEM,
        "runtime_architecture": architecture in REGISTERED_RUNTIME_ARCHITECTURES,
        "loopback_endpoint": provenance.get("loopback_endpoint") is True,
    }
    decision_eligible = all(checks.values())
    catalog_evidence_scope = provenance.get("catalog_evidence_scope")
    return {
        "scope": scope,
        "catalog_evidence_scope": catalog_evidence_scope,
        "decision_eligible": decision_eligible,
        "merchant_decision_eligible": decision_eligible
        and catalog_evidence_scope == REPRESENTATIVE_MERCHANT_SCOPE,
        "requirements": {
            "opensearch_version": REGISTERED_OPENSEARCH_VERSION,
            "index_document_count": REGISTERED_INDEX_DOCUMENT_COUNT,
            "runtime_system": REGISTERED_RUNTIME_SYSTEM,
            "runtime_architecture": "amd64",
            "loopback_endpoint": True,
            "merchant_catalog_evidence_scope": REPRESENTATIVE_MERCHANT_SCOPE,
        },
        "checks": checks,
    }


def _is_loopback_endpoint(endpoint_url: str | None) -> bool:
    if endpoint_url is None:
        return False
    hostname = urlsplit(endpoint_url).hostname
    return hostname in ("127.0.0.1", "::1", "localhost")


def _review_catalog(
    client: HttpClient,
    index: str,
    product_ids: list[int],
) -> list[dict[str, str | int]]:
    source_fields = [
        "entity_id",
        "sku",
        "title",
        "product_class",
        "category",
        "brand",
    ]
    response = _json_response(
        client.post(
            f"/{index}/_mget",
            json={
                "docs": [
                    {
                        "_id": str(product_id),
                        "_source": source_fields,
                    }
                    for product_id in product_ids
                ],
            },
        ),
        "radial review catalog response",
    )
    documents = response.get("docs")
    if not isinstance(documents, list) or len(documents) != len(product_ids):
        raise ValueError("the radial review catalog response is incomplete")

    expected = set(product_ids)
    catalog: dict[int, dict[str, str | int]] = {}
    for document in documents:
        if not isinstance(document, dict) or document.get("found") is not True:
            raise ValueError("a radial review catalog product is unavailable")
        source = document.get("_source")
        if not isinstance(source, dict):
            raise ValueError("a radial review catalog product is invalid")
        product_id = source.get("entity_id")
        if (
            not isinstance(product_id, int)
            or isinstance(product_id, bool)
            or product_id not in expected
            or product_id in catalog
            or str(document.get("_id")) != str(product_id)
        ):
            raise ValueError("a radial review catalog product identity is invalid")
        context: dict[str, str | int] = {"product_id": product_id}
        for field in source_fields[1:]:
            value = source.get(field)
            if not isinstance(value, str):
                raise ValueError("a radial review catalog product field is invalid")
            context[field] = value
        catalog[product_id] = context
    if set(catalog) != expected:
        raise ValueError("the radial review catalog response is incomplete")

    return [catalog[product_id] for product_id in product_ids]


def _validate_specification(specification: dict[str, Any]) -> None:
    if specification.get("schema_version") != 1:
        raise ValueError("the radial capture specification schema is unsupported")
    if specification.get("split") not in ("development", "holdout"):
        raise ValueError("the radial capture split must be development or holdout")
    if specification.get("qualification_scope") not in ("smoke", "registered"):
        raise ValueError("the radial capture qualification scope is invalid")
    if specification.get("catalog_evidence_scope") not in CATALOG_EVIDENCE_SCOPES:
        raise ValueError("the radial capture catalog evidence scope is invalid")
    index = specification.get("index")
    if not isinstance(index, str) or re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,254}", index) is None:
        raise ValueError("the radial capture index name is invalid")
    if (
        not isinstance(specification.get("generation_id"), int)
        or specification["generation_id"] <= 0
    ):
        raise ValueError("the radial capture generation ID is invalid")
    storefront_limit = specification.get("storefront_limit")
    maximum_result_count = specification.get("maximum_result_count")
    if (
        not isinstance(storefront_limit, int)
        or not isinstance(maximum_result_count, int)
        or not 0 < storefront_limit <= maximum_result_count <= 20
    ):
        raise ValueError("the radial capture result bounds are invalid")
    concurrency = specification.get("concurrency")
    latency_samples = specification.get("latency_samples")
    if (
        concurrency != 4
        or not isinstance(latency_samples, int)
        or latency_samples < concurrency
        or latency_samples % concurrency != 0
    ):
        raise ValueError("radial latency capture requires complete concurrency-four waves")
    exact_capture_limit = specification.get("exact_capture_limit")
    if not isinstance(exact_capture_limit, int) or not 20 <= exact_capture_limit <= 10_000:
        raise ValueError("the exact capture limit is invalid")
    thresholds = specification.get("thresholds")
    if (
        not isinstance(thresholds, list)
        or not thresholds
        or any(
            (not isinstance(value, int) and not isinstance(value, float))
            or not 0.0 < float(value) <= 1.0
            for value in thresholds
        )
        or len({float(value) for value in thresholds}) != len(thresholds)
        or (specification["split"] == "holdout" and len(thresholds) != 1)
    ):
        raise ValueError("the radial threshold candidates are invalid")
    identity = specification.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("the radial capture identity is missing")
    for field in ("model_id", "model_revision", "similarity", "source_recipe_version", "use_case"):
        if not isinstance(identity.get(field), str) or not identity[field]:
            raise ValueError("the radial capture identity is incomplete")
    if identity["similarity"] != "cosine":
        raise ValueError("the first radial capture supports only cosine similarity")
    if not isinstance(identity.get("dimension"), int) or identity["dimension"] <= 0:
        raise ValueError("the radial capture dimension is invalid")
    if not isinstance(identity.get("store_id"), int) or identity["store_id"] <= 0:
        raise ValueError("the radial capture store ID is invalid")
    cases = specification.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("the radial capture requires judgment cases")
    case_ids: set[str] = set()
    seed_ids: set[int] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("every radial judgment case must be an object")
        case_id = case.get("case_id")
        seed_id = case.get("seed_product_id")
        slices = case.get("slices")
        unsafe_ids = case.get("unsafe_ids")
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ValueError("radial judgment case IDs must be unique")
        if not isinstance(seed_id, int) or seed_id <= 0 or seed_id in seed_ids:
            raise ValueError("radial seed product IDs must be unique positive integers")
        if (
            not isinstance(slices, list)
            or not slices
            or any(not isinstance(value, str) or not value for value in slices)
        ):
            raise ValueError("radial judgment slices are invalid")
        if (
            not isinstance(unsafe_ids, list)
            or any(not isinstance(value, int) or value <= 0 for value in unsafe_ids)
            or len(set(unsafe_ids)) != len(unsafe_ids)
        ):
            raise ValueError("radial unsafe product IDs are invalid")
        case_ids.add(case_id)
        seed_ids.add(seed_id)


def _validate_mapping(client: HttpClient, index: str, identity: dict[str, Any]) -> None:
    response = _json_response(client.get(f"/{index}/_mapping"), "index mapping")
    index_mapping = cast(dict[str, Any], response.get(index, {}))
    mappings = cast(dict[str, Any], index_mapping.get("mappings", {}))
    properties = cast(dict[str, Any], mappings.get("properties", {}))
    embedding = cast(dict[str, Any], properties.get("embedding", {}))
    method = cast(dict[str, Any], embedding.get("method", {}))
    if (
        embedding.get("type") != "knn_vector"
        or embedding.get("dimension") != identity["dimension"]
        or method.get("engine") != "lucene"
        or method.get("space_type") != "cosinesimil"
    ):
        raise ValueError("the radial capture index mapping is incompatible")


def _eligible_filter(identity: dict[str, Any], generation_id: int) -> dict[str, Any]:
    return {
        "bool": {
            "filter": [
                {"term": {"store_id": identity["store_id"]}},
                {"term": {"generation_id": generation_id}},
                {"term": {"model_revision": identity["model_revision"]}},
                {"term": {"embedding_eligible": True}},
                {"term": {"status": 1}},
                {"terms": {"visibility": [3, 4]}},
                {"term": {"is_salable": True}},
            ]
        }
    }


def _count(client: HttpClient, index: str, query: dict[str, Any]) -> int:
    response = _json_response(
        client.post(f"/{index}/_count", json={"query": query}),
        "eligible universe count",
    )
    count = response.get("count")
    if not isinstance(count, int) or count < 0:
        raise ValueError("the eligible universe count response is invalid")
    return count


def _seed_vector(
    client: HttpClient,
    index: str,
    identity: dict[str, Any],
    generation_id: int,
    seed_product_id: int,
) -> list[float]:
    response = _search_response(
        client.post(
            f"/{index}/_search",
            json={
                "size": 1,
                "track_total_hits": False,
                "_source": False,
                "fields": ["entity_id"],
                "docvalue_fields": [{"field": "embedding", "format": "binary"}],
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"store_id": identity["store_id"]}},
                            {"term": {"generation_id": generation_id}},
                            {"term": {"model_revision": identity["model_revision"]}},
                            {"term": {"entity_id": seed_product_id}},
                            {"term": {"embedding_eligible": True}},
                        ]
                    }
                },
            },
        ),
        "seed vector response",
    )
    hits = cast(dict[str, Any], response.get("hits", {})).get("hits")
    if not isinstance(hits, list) or len(hits) != 1 or not isinstance(hits[0], dict):
        raise ValueError("the radial seed vector is unavailable")
    encoded = cast(dict[str, Any], hits[0].get("fields", {})).get("embedding")
    if not isinstance(encoded, list) or len(encoded) != 1 or not isinstance(encoded[0], str):
        raise ValueError("the radial seed vector response is invalid")
    try:
        raw = base64.b64decode(encoded[0], validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("the radial seed vector Base64 is invalid") from error
    dimension = int(identity["dimension"])
    if len(raw) != dimension * 4:
        raise ValueError("the radial seed vector dimension is invalid")
    vector = list(struct.unpack(f"<{dimension}f", raw))
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("the radial seed vector contains a non-finite value")

    return vector


def _exact_ids(
    client: HttpClient,
    index: str,
    identity: dict[str, Any],
    generation_id: int,
    seed_product_id: int,
    seed_vector: list[float],
    min_score: float,
    exact_capture_limit: int,
) -> list[int]:
    eligible = _eligible_filter(identity, generation_id)
    cast(dict[str, Any], eligible["bool"])["must_not"] = [{"term": {"entity_id": seed_product_id}}]
    response = _search_response(
        client.post(
            f"/{index}/_search",
            json={
                "size": exact_capture_limit,
                "track_total_hits": True,
                "min_score": min_score,
                "_source": False,
                "fields": ["entity_id"],
                "sort": [{"_score": "desc"}, {"entity_id": "asc"}],
                "query": {
                    "script_score": {
                        "query": eligible,
                        "script": {
                            "lang": "knn",
                            "source": "knn_score",
                            "params": {
                                "field": "embedding",
                                "query_value": seed_vector,
                                "space_type": "cosinesimil",
                            },
                        },
                    }
                },
            },
        ),
        "exact similarity response",
    )
    hits_container = cast(dict[str, Any], response.get("hits", {}))
    total = hits_container.get("total")
    hits = hits_container.get("hits")
    if (
        not isinstance(total, dict)
        or total.get("relation") != "eq"
        or not isinstance(total.get("value"), int)
        or not isinstance(hits, list)
        or int(total["value"]) != len(hits)
    ):
        raise ValueError("the exact similarity membership capture is truncated")

    return _hit_ids(hits, min_score)


def _radial_measurements(
    client: HttpClient,
    index: str,
    identity: dict[str, Any],
    generation_id: int,
    seed_product_id: int,
    seed_vector: list[float],
    min_score: float,
    maximum_result_count: int,
    concurrency: int,
    latency_samples: int,
) -> tuple[list[int], list[float]]:
    eligible_filters = cast(
        list[dict[str, Any]],
        cast(dict[str, Any], _eligible_filter(identity, generation_id)["bool"])["filter"],
    )
    request = {
        "size": maximum_result_count,
        "track_total_hits": False,
        "_source": False,
        "fields": ["entity_id"],
        "sort": [{"_score": "desc"}, {"entity_id": "asc"}],
        "query": {
            "knn": {
                "embedding": {
                    "vector": seed_vector,
                    "min_score": min_score,
                    "filter": {
                        "bool": {
                            "filter": eligible_filters,
                            "must_not": [{"term": {"entity_id": seed_product_id}}],
                        }
                    },
                }
            }
        },
    }
    warm_ids, _warm_elapsed = _radial_once(client, index, request, min_score)
    samples: list[tuple[list[int], float]] = []
    for _wave in range(latency_samples // concurrency):
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            samples.extend(
                executor.map(
                    lambda _sample: _radial_once(client, index, request, min_score),
                    range(concurrency),
                )
            )
    radial_ids = warm_ids
    if any(ids != radial_ids for ids, _elapsed in samples):
        raise ValueError("radial similarity membership changed during capture")

    return radial_ids, [elapsed for _ids, elapsed in samples]


def _radial_once(
    client: HttpClient,
    index: str,
    request: dict[str, Any],
    min_score: float,
) -> tuple[list[int], float]:
    started_at = time.perf_counter_ns()
    response = _search_response(
        client.post(f"/{index}/_search", json=request),
        "radial similarity response",
    )
    elapsed_ms = max(0, time.perf_counter_ns() - started_at) / 1_000_000
    hits = cast(dict[str, Any], response.get("hits", {})).get("hits")
    if not isinstance(hits, list):
        raise ValueError("the radial similarity hits are invalid")

    return _hit_ids(hits, min_score), elapsed_ms


def _hit_ids(hits: list[Any], min_score: float) -> list[int]:
    product_ids: list[int] = []
    scores: list[float] = []
    for hit in hits:
        if not isinstance(hit, dict):
            raise ValueError("a similarity hit is invalid")
        fields = hit.get("fields")
        score = hit.get("_score")
        raw_id = fields.get("entity_id", [None])[0] if isinstance(fields, dict) else None
        if isinstance(raw_id, int) and not isinstance(raw_id, bool):
            product_id = raw_id
        elif isinstance(raw_id, str) and raw_id.isdigit():
            product_id = int(raw_id)
        else:
            raise ValueError("a similarity hit identity or score is invalid")
        if (
            isinstance(score, bool)
            or (not isinstance(score, int) and not isinstance(score, float))
            or not math.isfinite(float(score))
            or float(score) < min_score
        ):
            raise ValueError("a similarity hit identity or score is invalid")
        product_ids.append(product_id)
        scores.append(float(score))
    if len(product_ids) != len(set(product_ids)):
        raise ValueError("a similarity response contains duplicate product IDs")
    expected = sorted(
        zip(product_ids, scores, strict=True),
        key=lambda item: (-item[1], item[0]),
    )
    if product_ids != [product_id for product_id, _score in expected]:
        raise ValueError("a similarity response is not deterministically ordered")

    return product_ids


def _json_response(response: HttpResponse, label: str) -> dict[str, Any]:
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError(f"the {label} is not a JSON object")

    return cast(dict[str, Any], value)


def _search_response(response: HttpResponse, label: str) -> dict[str, Any]:
    value = _json_response(response, label)
    shards = value.get("_shards")
    if (
        value.get("timed_out") is not False
        or not isinstance(shards, dict)
        or shards.get("failed") != 0
    ):
        raise ValueError(f"the {label} is incomplete")

    return value
