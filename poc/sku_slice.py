from __future__ import annotations

import hashlib
import json
import math
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from poc.config import ConfigError
from poc.datasets import DatasetIntegrityError, file_facts
from poc.indexing import iter_prepared_products
from poc.manifest import write_json
from poc.neural_sparse import (
    NeuralSparseSpec,
    build_sparse_index_definition,
    iter_precomputed_sparse_documents,
)
from poc.trec import QrelRecord, identifier_sort_key, write_qrels


@dataclass(frozen=True, slots=True)
class SkuSliceSpec:
    dataset: str
    seed: str
    sku_prefix: str
    sku_queries: int
    exact_title_queries: int
    lexical_weight: float
    maximum_mrr_regression: float
    alpha: float
    index_name: str

    @property
    def query_count(self) -> int:
        return self.sku_queries + self.exact_title_queries


def load_sku_slice_spec(path: Path) -> SkuSliceSpec:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if raw.get("schema_version") != 1:
        raise ConfigError("unsupported SKU slice schema")
    spec = SkuSliceSpec(
        dataset=_required_string(raw, "dataset"),
        seed=_required_string(raw, "seed"),
        sku_prefix=_required_string(raw, "sku_prefix"),
        sku_queries=_required_integer(raw, "sku_queries"),
        exact_title_queries=_required_integer(raw, "exact_title_queries"),
        lexical_weight=_required_fraction(raw, "lexical_weight"),
        maximum_mrr_regression=_required_fraction(
            raw, "maximum_mrr_regression"
        ),
        alpha=_required_fraction(raw, "alpha"),
        index_name=_required_string(raw, "index_name"),
    )
    if "synthetic" not in spec.dataset.lower():
        raise ConfigError("SKU slice must disclose its synthetic provenance")
    return spec


def synthetic_sku(product_id: str, *, prefix: str) -> str:
    if not product_id or not prefix:
        raise ValueError("synthetic SKU product ID and prefix must not be blank")
    return f"{prefix}-{product_id}"


def build_sku_index_definition(
    sparse_spec: NeuralSparseSpec,
    sku_spec: SkuSliceSpec,
) -> dict[str, Any]:
    definition = build_sparse_index_definition(
        replace(sparse_spec, index_name=sku_spec.index_name),
        use_default_pipeline=False,
    )
    analysis = definition["settings"]["analysis"]
    analysis["normalizer"] = {
        "sku_normalizer": {
            "type": "custom",
            "filter": ["lowercase", "asciifolding"],
        }
    }
    definition["mappings"]["properties"]["sku"] = {
        "type": "keyword",
        "normalizer": "sku_normalizer",
    }
    for field in ("brand", "category", "product_class"):
        definition["mappings"]["properties"][field]["fields"]["facet"] = {
            "type": "keyword",
            "ignore_above": 256,
        }
    return definition


def build_known_item_lexical_query(query: str) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("known-item query must not be blank")
    return {
        "bool": {
            "should": [
                {"term": {"sku": {"value": query, "boost": 100.0}}},
                {
                    "match_phrase": {
                        "title": {"query": query, "boost": 20.0}
                    }
                },
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^8", "brand^2", "description"],
                        "type": "best_fields",
                        "operator": "and",
                    }
                },
            ],
            "minimum_should_match": 1,
        }
    }


def known_item_route(
    query: str,
    *,
    sku_prefix: str,
    lexical_top_title: str | None,
) -> str:
    normalized_query = " ".join(query.casefold().split())
    normalized_prefix = f"{sku_prefix.casefold()}-"
    if normalized_query.startswith(normalized_prefix):
        return "lexical_only_sku"
    normalized_title = (
        " ".join(lexical_top_title.casefold().split())
        if lexical_top_title is not None
        else ""
    )
    if normalized_query and normalized_query == normalized_title:
        return "lexical_only_exact_title"
    return "hybrid"


def prepare_sku_slice(
    *,
    root: Path,
    spec: SkuSliceSpec,
    products_path: Path,
    query_path: Path,
    qrels_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    records, qrels = build_sku_slice_records(products_path, spec)
    query_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = query_path.with_name(f".{query_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
            )
    temporary.replace(query_path)
    write_qrels(qrels_path, qrels)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset": spec.dataset,
        "provenance": "deterministic proxy because WANDS contains no merchant SKU field",
        "eligible_as_authentic_merchant_sku_evidence": False,
        "eligible_as_synthetic_known_item_regression_evidence": True,
        "seed": spec.seed,
        "source_products": _artifact(root, products_path),
        "counts": {
            "synthetic_sku": spec.sku_queries,
            "exact_title": spec.exact_title_queries,
            "total": spec.query_count,
        },
        "queries": _artifact(root, query_path),
        "qrels": _artifact(root, qrels_path),
    }
    write_json(manifest_path, manifest)
    return manifest


def build_sku_slice_records(
    products_path: Path,
    spec: SkuSliceSpec,
) -> tuple[list[dict[str, str]], list[QrelRecord]]:
    products = [
        (product_id, str(document.get("title", "")).strip())
        for product_id, document in iter_prepared_products(products_path)
    ]
    if len(products) < spec.query_count:
        raise DatasetIntegrityError("WANDS has too few products for the SKU slice")
    sku_candidates = sorted(
        products,
        key=lambda item: _selection_key(spec.seed, "sku", item[0]),
    )[: spec.sku_queries]
    title_candidates: list[tuple[str, str]] = []
    seen_titles: set[str] = set()
    for product_id, title in sorted(
        products,
        key=lambda item: _selection_key(spec.seed, "title", item[0]),
    ):
        normalized = " ".join(title.lower().split())
        if not normalized or normalized in seen_titles:
            continue
        seen_titles.add(normalized)
        title_candidates.append((product_id, title))
        if len(title_candidates) == spec.exact_title_queries:
            break
    if len(title_candidates) != spec.exact_title_queries:
        raise DatasetIntegrityError("WANDS has too few unique titles for the SKU slice")

    records: list[dict[str, str]] = []
    qrels: list[QrelRecord] = []
    for number, (product_id, _) in enumerate(sku_candidates, start=1):
        query_id = f"sku-{number:03d}"
        records.append(
            {
                "query_id": query_id,
                "query": synthetic_sku(product_id, prefix=spec.sku_prefix),
                "kind": "synthetic_sku",
                "expected_document_id": product_id,
            }
        )
        qrels.append(QrelRecord(query_id, product_id, 2))
    for number, (product_id, title) in enumerate(title_candidates, start=1):
        query_id = f"title-{number:03d}"
        records.append(
            {
                "query_id": query_id,
                "query": title,
                "kind": "exact_title",
                "expected_document_id": product_id,
            }
        )
        qrels.append(QrelRecord(query_id, product_id, 2))
    records.sort(key=lambda record: identifier_sort_key(record["query_id"]))
    return records, qrels


def iter_sku_sparse_documents(
    products_path: Path,
    embeddings_path: Path,
    sparse_spec: NeuralSparseSpec,
    sku_spec: SkuSliceSpec,
) -> Iterator[tuple[str, dict[str, Any]]]:
    for product_id, document in iter_precomputed_sparse_documents(
        products_path, embeddings_path, sparse_spec
    ):
        output = dict(document)
        output["sku"] = synthetic_sku(product_id, prefix=sku_spec.sku_prefix)
        yield product_id, output


def load_sku_queries(path: Path) -> dict[str, dict[str, str]]:
    queries: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = cast(dict[str, Any], json.loads(line))
            values = {
                key: str(record.get(key, "")).strip()
                for key in ("query_id", "query", "kind", "expected_document_id")
            }
            if not all(values.values()) or values["query_id"] in queries:
                raise DatasetIntegrityError(
                    f"invalid SKU slice query at {path}:{line_number}"
                )
            queries[values["query_id"]] = values
    if not queries:
        raise DatasetIntegrityError("SKU slice contains no queries")
    return queries


def assess_known_item_regression(
    *,
    baseline: dict[str, float],
    candidate: dict[str, float],
    maximum_mrr_regression: float,
    alpha: float,
) -> dict[str, object]:
    if not baseline or set(baseline) != set(candidate):
        raise ValueError("known-item paired scores must cover the same queries")
    if not 0.0 <= maximum_mrr_regression <= 1.0 or not 0.0 < alpha < 1.0:
        raise ValueError("known-item regression thresholds are invalid")
    deltas = {
        query_id: candidate[query_id] - baseline[query_id]
        for query_id in baseline
    }
    wins = sum(delta > 0.0 for delta in deltas.values())
    ties = sum(delta == 0.0 for delta in deltas.values())
    losses = sum(delta < 0.0 for delta in deltas.values())
    discordant = wins + losses
    p_value = (
        1.0
        if discordant == 0
        else sum(
            math.comb(discordant, count) for count in range(losses, discordant + 1)
        )
        / (2**discordant)
    )
    mean_delta = sum(deltas.values()) / len(deltas)
    significant_regression = mean_delta < 0.0 and p_value <= alpha
    within_margin = mean_delta >= -maximum_mrr_regression
    return {
        "queries": len(deltas),
        "baseline_mrr@10": sum(baseline.values()) / len(baseline),
        "candidate_mrr@10": sum(candidate.values()) / len(candidate),
        "paired_mean_delta": mean_delta,
        "maximum_allowed_regression": maximum_mrr_regression,
        "within_regression_margin": within_margin,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "test": "one_sided_exact_paired_sign_test_candidate_worse",
        "paired_sign_test_p_value": p_value,
        "alpha": alpha,
        "statistically_significant_regression": significant_regression,
        "passes_no_significant_regression_gate": (
            within_margin and not significant_regression
        ),
    }


def _selection_key(seed: str, kind: str, product_id: str) -> str:
    return hashlib.sha256(f"{seed}:{kind}:{product_id}".encode()).hexdigest()


def _required_string(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"SKU slice {key} must be a non-empty string")
    return value


def _required_integer(values: dict[str, Any], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"SKU slice {key} must be a positive integer")
    return value


def _required_fraction(values: dict[str, Any], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigError(f"SKU slice {key} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ConfigError(f"SKU slice {key} must be in [0, 1]")
    return result


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }
