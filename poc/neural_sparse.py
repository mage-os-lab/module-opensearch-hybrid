from __future__ import annotations

import importlib
import json
import math
import re
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from poc.config import ConfigError
from poc.datasets import DatasetIntegrityError, WandsDatasetSpec, file_facts
from poc.indexing import WandsIndexSpec, iter_prepared_products, wands_index_definition
from poc.manifest import canonical_sha256, read_json
from poc.provenance import verify_decision_provenance

WANDS_SPARSE_MAXIMUM_TOKEN_LENGTH = 256
WANDS_SPARSE_REPLAY_SAMPLE_SIZE = 256
WANDS_SPARSE_PRUNE_RATIO = 0.1


class SparseProbeClient(Protocol):
    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> Any: ...

    def search(self, index: str, body: dict[str, Any]) -> dict[str, Any]: ...


class WandsSparseEncoder(Protocol):
    runtime: str

    def encode(
        self,
        batch: list[tuple[str, str]],
    ) -> list[tuple[str, dict[str, float]]]: ...


@dataclass(frozen=True, slots=True)
class NeuralSparseSpec:
    model_name: str
    model_version: str
    model_format: str
    query_analyzer: str
    package_url: str
    package_sha256: str
    package_bytes: int
    model_content_sha256: str
    model_content_bytes: int
    tokenizer_content_sha256: str
    tokenizer_content_bytes: int
    index_name: str
    ingest_pipeline: str
    text_field: str
    embedding_field: str
    prune_type: str
    prune_ratio: float
    inference_batch_size: int
    bulk_request_size: int


def load_neural_sparse_spec(path: Path) -> NeuralSparseSpec:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if raw.get("schema_version") != 1:
        raise ConfigError("unsupported neural-sparse schema")
    model = raw.get("model")
    index = raw.get("index")
    if not isinstance(model, dict) or not isinstance(index, dict):
        raise ConfigError("neural-sparse model or index configuration is missing")
    strings = {
        "model_name": model.get("name"),
        "model_version": model.get("version"),
        "model_format": model.get("model_format"),
        "query_analyzer": model.get("query_analyzer"),
        "index_name": index.get("name"),
        "ingest_pipeline": index.get("ingest_pipeline"),
        "text_field": index.get("text_field"),
        "embedding_field": index.get("embedding_field"),
        "prune_type": index.get("prune_type"),
    }
    if any(not isinstance(value, str) or not value for value in strings.values()):
        raise ConfigError("neural-sparse string settings must be non-empty")
    if strings["model_format"] not in {"TORCH_SCRIPT", "ONNX"}:
        raise ConfigError("neural-sparse model format is unsupported")
    if strings["prune_type"] != "max_ratio":
        raise ConfigError("neural-sparse prune type must be max_ratio")
    model_content_sha256 = model.get("model_content_sha256")
    tokenizer_content_sha256 = model.get("tokenizer_content_sha256")
    model_content_bytes = model.get("model_content_bytes")
    tokenizer_content_bytes = model.get("tokenizer_content_bytes")
    package_url = model.get("package_url")
    package_sha256 = model.get("package_sha256")
    package_bytes = model.get("package_bytes")
    if (
        not isinstance(package_url, str)
        or not package_url.startswith("https://artifacts.opensearch.org/")
        or not isinstance(package_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", package_sha256) is None
        or not isinstance(package_bytes, int)
        or isinstance(package_bytes, bool)
        or package_bytes <= 0
        or not isinstance(model_content_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", model_content_sha256) is None
        or not isinstance(tokenizer_content_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", tokenizer_content_sha256) is None
        or not isinstance(model_content_bytes, int)
        or isinstance(model_content_bytes, bool)
        or model_content_bytes <= 0
        or not isinstance(tokenizer_content_bytes, int)
        or isinstance(tokenizer_content_bytes, bool)
        or tokenizer_content_bytes <= 0
    ):
        raise ConfigError("neural-sparse model artifact identity is invalid")
    prune_ratio = index.get("prune_ratio")
    if (
        not isinstance(prune_ratio, int | float)
        or isinstance(prune_ratio, bool)
        or not 0 <= float(prune_ratio) <= 1
    ):
        raise ConfigError("neural-sparse prune ratio must be in [0, 1]")
    inference_batch_size = index.get("inference_batch_size")
    bulk_request_size = index.get("bulk_request_size")
    if (
        not isinstance(inference_batch_size, int)
        or isinstance(inference_batch_size, bool)
        or inference_batch_size <= 0
        or not isinstance(bulk_request_size, int)
        or isinstance(bulk_request_size, bool)
        or bulk_request_size < inference_batch_size
    ):
        raise ConfigError("neural-sparse batch settings are invalid")
    return NeuralSparseSpec(
        model_name=str(strings["model_name"]),
        model_version=str(strings["model_version"]),
        model_format=str(strings["model_format"]),
        query_analyzer=str(strings["query_analyzer"]),
        package_url=package_url,
        package_sha256=package_sha256,
        package_bytes=package_bytes,
        model_content_sha256=model_content_sha256,
        model_content_bytes=model_content_bytes,
        tokenizer_content_sha256=tokenizer_content_sha256,
        tokenizer_content_bytes=tokenizer_content_bytes,
        index_name=str(strings["index_name"]),
        ingest_pipeline=str(strings["ingest_pipeline"]),
        text_field=str(strings["text_field"]),
        embedding_field=str(strings["embedding_field"]),
        prune_type=str(strings["prune_type"]),
        prune_ratio=float(prune_ratio),
        inference_batch_size=inference_batch_size,
        bulk_request_size=bulk_request_size,
    )


def verify_neural_sparse_model_artifacts(
    spec: NeuralSparseSpec,
    *,
    root: Path,
    model_path: Path,
    tokenizer_path: Path,
) -> dict[str, dict[str, object]]:
    expected = {
        "model_file": {
            "path": str(model_path.relative_to(root)),
            "sha256": spec.model_content_sha256,
            "bytes": spec.model_content_bytes,
        },
        "tokenizer_file": {
            "path": str(tokenizer_path.relative_to(root)),
            "sha256": spec.tokenizer_content_sha256,
            "bytes": spec.tokenizer_content_bytes,
        },
    }
    for label, path in (
        ("model_file", model_path),
        ("tokenizer_file", tokenizer_path),
    ):
        if not path.is_file():
            raise DatasetIntegrityError(f"missing pinned neural-sparse {label}: {path}")
        facts = file_facts(path)
        if (
            facts.sha256 != expected[label]["sha256"]
            or facts.bytes != expected[label]["bytes"]
        ):
            raise DatasetIntegrityError(
                f"neural-sparse {label} does not match registered bytes"
            )
    return expected


class TorchScriptWandsSparseEncoder:
    runtime = "torchscript_cpu"

    def __init__(
        self,
        *,
        spec: NeuralSparseSpec,
        model_path: Path,
        tokenizer_path: Path,
    ) -> None:
        if (
            spec.model_format != "TORCH_SCRIPT"
            or spec.prune_type != "max_ratio"
            or spec.prune_ratio != WANDS_SPARSE_PRUNE_RATIO
        ):
            raise DatasetIntegrityError(
                "unsupported WANDS sparse TorchScript or max-ratio recipe"
            )
        torch = importlib.import_module("torch")
        tokenizers = importlib.import_module("tokenizers")
        tokenizer: Any = tokenizers.Tokenizer.from_file(str(tokenizer_path))
        tokenizer.enable_truncation(max_length=WANDS_SPARSE_MAXIMUM_TOKEN_LENGTH)
        tokenizer.enable_padding()
        model = torch.jit.load(str(model_path), map_location="cpu").eval().float()
        output_tokens = tuple(
            tokenizer.id_to_token(index)
            for index in range(tokenizer.get_vocab_size())
        )
        if any(token is None for token in output_tokens):
            raise DatasetIntegrityError(
                "WANDS sparse tokenizer vocabulary contains unmapped token IDs"
            )
        self._spec = spec
        self._torch: Any = torch
        self._tokenizer: Any = tokenizer
        self._model: Any = model
        self._output_tokens = output_tokens

    def encode(
        self,
        batch: list[tuple[str, str]],
    ) -> list[tuple[str, dict[str, float]]]:
        if not batch:
            return []
        encoded = self._tokenizer.encode_batch([text for _, text in batch])
        inputs = {
            "input_ids": self._torch.tensor(
                [item.ids for item in encoded], dtype=self._torch.long
            ),
            "attention_mask": self._torch.tensor(
                [item.attention_mask for item in encoded], dtype=self._torch.long
            ),
        }
        with self._torch.inference_mode():
            raw = self._model(inputs)["output"].detach().cpu().numpy()
        values = np.asarray(raw)
        if values.shape != (len(batch), len(self._output_tokens)) or not np.isfinite(
            values
        ).all():
            raise DatasetIntegrityError("official WANDS sparse model returned invalid output")
        return [
            (product_id, self._prune_row(product_id, row))
            for (product_id, _), row in zip(batch, values, strict=True)
        ]

    def _prune_row(
        self,
        product_id: str,
        row: NDArray[Any],
    ) -> dict[str, float]:
        maximum = float(np.max(row))
        threshold = maximum * self._spec.prune_ratio
        indexes = np.flatnonzero((row > 0.0) & (row >= threshold))
        embedding = {
            cast(str, self._output_tokens[index]): float(row[index])
            for index in indexes
        }
        if not embedding:
            raise DatasetIntegrityError(
                f"official WANDS sparse model returned no tokens for {product_id}"
            )
        return embedding


def create_wands_sparse_encoder(
    *,
    spec: NeuralSparseSpec,
    model_path: Path,
    tokenizer_path: Path,
) -> WandsSparseEncoder:
    return TorchScriptWandsSparseEncoder(
        spec=spec,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )


def canonical_sparse_embedding_line(
    product_id: str,
    embedding: Mapping[str, float],
) -> str:
    return (
        json.dumps(
            {"product_id": product_id, "embedding": dict(embedding)},
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def wands_sparse_pruning_recipe(spec: NeuralSparseSpec) -> dict[str, object]:
    return {
        "type": spec.prune_type,
        "ratio": spec.prune_ratio,
        "implementation": "local_precomputed_exact",
    }


def wands_sparse_replay_contract(
    *,
    products_path: Path,
    dataset: WandsDatasetSpec,
    spec: NeuralSparseSpec,
) -> dict[str, object]:
    indexes = set(
        _spanning_indexes(
            dataset.expected_products,
            WANDS_SPARSE_REPLAY_SAMPLE_SIZE,
        )
    )
    product_ids: list[str] = []
    records = 0
    for position, (product_id, _) in enumerate(iter_prepared_products(products_path)):
        if position in indexes:
            product_ids.append(product_id)
        records += 1
    if records != dataset.expected_products or len(product_ids) != len(indexes):
        raise DatasetIntegrityError(
            "WANDS sparse deterministic replay sample is incomplete"
        )
    return {
        "method": "fixed_evenly_spaced_canonical_rows_including_endpoints",
        "sample_size": len(product_ids),
        "sample_product_ids_sha256": canonical_sha256(product_ids),
        "batch_size": spec.inference_batch_size,
        "pruning": wands_sparse_pruning_recipe(spec),
        "comparison": "exact_canonical_json_row",
    }


def verify_wands_sparse_precompute(
    *,
    root: Path,
    dataset: WandsDatasetSpec,
    spec: NeuralSparseSpec,
    products_path: Path,
    embeddings_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    value = read_json(manifest_path)
    if not isinstance(value, dict):
        raise DatasetIntegrityError("WANDS sparse precompute manifest must be an object")
    manifest = cast(dict[str, Any], value)
    if (
        spec.model_format != "TORCH_SCRIPT"
        or spec.prune_type != "max_ratio"
        or spec.prune_ratio != WANDS_SPARSE_PRUNE_RATIO
    ):
        raise DatasetIntegrityError(
            "WANDS sparse precompute does not use the registered max_ratio 0.1 recipe"
        )
    model_path = (
        root
        / "data/cache/models/neural-sparse/doc-v3-distill/"
        "opensearch-neural-sparse-encoding-doc-v3-distill.pt"
    )
    tokenizer_path = (
        root / "data/cache/models/neural-sparse/doc-v3-distill/tokenizer.json"
    )
    model_artifacts = verify_neural_sparse_model_artifacts(
        spec,
        root=root,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )
    embeddings_facts = file_facts(embeddings_path)
    replay_contract = wands_sparse_replay_contract(
        products_path=products_path,
        dataset=dataset,
        spec=spec,
    )
    expected_embeddings = {
        "path": str(embeddings_path.relative_to(root)),
        "sha256": embeddings_facts.sha256,
        "file_sha256": embeddings_facts.sha256,
        "bytes": embeddings_facts.bytes,
    }
    expected_metadata: dict[str, object] = {
        "schema_version": 2,
        "model": spec.model_name,
        "model_version": spec.model_version,
        **model_artifacts,
        "products_sha256": file_facts(products_path).sha256,
        "runtime": "torchscript_cpu",
        "batch_size": spec.inference_batch_size,
        "maximum_token_length": WANDS_SPARSE_MAXIMUM_TOKEN_LENGTH,
        "pruning": wands_sparse_pruning_recipe(spec),
        "deterministic_replay": replay_contract,
        "document_recipe": (
            "title.brand.category.product_class.features.description"
        ),
        "records": dataset.expected_products,
        "embeddings": expected_embeddings,
    }
    if any(
        manifest.get(key) != expected
        for key, expected in expected_metadata.items()
    ):
        raise DatasetIntegrityError(
            "WANDS sparse precompute metadata or registered artifacts differ"
        )

    semantic_records = 0
    replay_indexes = set(
        _spanning_indexes(
            dataset.expected_products,
            WANDS_SPARSE_REPLAY_SAMPLE_SIZE,
        )
    )
    replay_samples: list[tuple[str, str, dict[str, float]]] = []
    try:
        for position, (document_id, document) in enumerate(iter_precomputed_sparse_documents(
            products_path,
            embeddings_path,
            spec,
        )):
            embedding = document.get(spec.embedding_field)
            if not isinstance(embedding, dict) or not embedding:
                raise DatasetIntegrityError("WANDS sparse embedding is empty")
            for token, weight in embedding.items():
                if (
                    not isinstance(token, str)
                    or not token
                    or not isinstance(weight, int | float)
                    or isinstance(weight, bool)
                    or not math.isfinite(float(weight))
                    or float(weight) <= 0.0
                ):
                    raise DatasetIntegrityError(
                        "WANDS sparse embedding contains an invalid rank feature"
                    )
            maximum = max(float(weight) for weight in embedding.values())
            minimum_allowed = maximum * spec.prune_ratio
            if any(float(weight) < minimum_allowed for weight in embedding.values()):
                raise DatasetIntegrityError(
                    "WANDS sparse embedding does not follow max-ratio pruning"
                )
            if position in replay_indexes:
                replay_samples.append(
                    (
                        document_id,
                        str(document[spec.text_field]),
                        {str(token): float(weight) for token, weight in embedding.items()},
                    )
                )
            semantic_records += 1
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise DatasetIntegrityError(
            f"WANDS sparse embedding order or content differs: {error}"
        ) from error
    if semantic_records != dataset.expected_products:
        raise DatasetIntegrityError("WANDS sparse embedding record count differs")
    if len(replay_samples) != len(replay_indexes):
        raise DatasetIntegrityError("WANDS sparse replay sample is incomplete")
    encoder = create_wands_sparse_encoder(
        spec=spec,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )
    replayed: list[tuple[str, dict[str, float]]] = []
    replay_inputs = [
        (document_id, text) for document_id, text, _ in replay_samples
    ]
    for offset in range(0, len(replay_inputs), spec.inference_batch_size):
        replayed.extend(
            encoder.encode(
                replay_inputs[offset : offset + spec.inference_batch_size]
            )
        )
    expected_replay = [
        (document_id, embedding)
        for document_id, _, embedding in replay_samples
    ]
    replayed_rows = [
        canonical_sparse_embedding_line(document_id, embedding)
        for document_id, embedding in replayed
    ]
    expected_replay_rows = [
        canonical_sparse_embedding_line(document_id, embedding)
        for document_id, embedding in expected_replay
    ]
    if replayed_rows != expected_replay_rows:
        raise DatasetIntegrityError(
            "WANDS sparse deterministic sample re-encoding differs"
        )

    generation_provenance = manifest.get("generation_provenance")
    provenance_valid = isinstance(
        generation_provenance, Mapping
    ) and verify_decision_provenance(
        generation_provenance,
        root=root,
        profile_path=root / "config/benchmark.toml",
        environment_path=root / "results/environment/benchmark-profile.json",
        require_current_code_revision=True,
    )
    if (
        manifest.get("benchmark_provenance") != generation_provenance
        or manifest.get("generation_provenance_eligible_for_decision")
        is not provenance_valid
        or manifest.get("quality_evidence_eligible_for_decision")
        is not provenance_valid
    ):
        raise DatasetIntegrityError("WANDS sparse precompute provenance differs")
    manifest_facts = file_facts(manifest_path)
    return {
        "artifact": {
            "path": str(manifest_path.relative_to(root)),
            "sha256": manifest_facts.sha256,
            "bytes": manifest_facts.bytes,
        },
        "schema_version": 2,
        "records": dataset.expected_products,
        "semantic_records_verified": semantic_records,
        "sample_reencoding": {
            **replay_contract,
            "runtime": encoder.runtime,
            "status": "passed",
        },
        "model_artifacts": model_artifacts,
        "provenance_valid": provenance_valid,
        "eligible_for_decision": provenance_valid,
    }


def build_sparse_ingest_pipeline(
    spec: NeuralSparseSpec,
    *,
    model_id: str,
) -> dict[str, Any]:
    if not model_id:
        raise ValueError("neural-sparse model ID must not be empty")
    return {
        "description": "OpenSearch Hybrid WANDS document-only neural sparse ingestion",
        "processors": [
            {
                "sparse_encoding": {
                    "model_id": model_id,
                    "prune_type": spec.prune_type,
                    "prune_ratio": spec.prune_ratio,
                    "batch_size": spec.inference_batch_size,
                    "skip_existing": True,
                    "field_map": {spec.text_field: spec.embedding_field},
                }
            }
        ],
    }


def build_sparse_index_definition(
    spec: NeuralSparseSpec,
    *,
    use_default_pipeline: bool = True,
) -> dict[str, Any]:
    definition = wands_index_definition(
        WandsIndexSpec(name=spec.index_name, shards=1, replicas=0)
    )
    index_settings = definition["settings"]["index"]
    index_settings["knn"] = False
    if use_default_pipeline:
        index_settings["default_pipeline"] = spec.ingest_pipeline
    properties = definition["mappings"]["properties"]
    properties[spec.text_field] = {"type": "text", "index": False}
    properties[spec.embedding_field] = {"type": "rank_features"}
    return definition


def build_sparse_query(spec: NeuralSparseSpec, text: str) -> dict[str, Any]:
    if not text.strip():
        raise ValueError("neural-sparse query must not be blank")
    return {
        "neural_sparse": {
            spec.embedding_field: {
                "query_text": text,
                "analyzer": spec.query_analyzer,
            }
        }
    }


def validate_sparse_probe(
    *,
    expected_document_id: str,
    hit_ids: list[str],
    sparse_terms: set[str],
    analyzer_terms: set[str],
) -> list[str]:
    overlap = sorted(sparse_terms & analyzer_terms)
    if not sparse_terms or not analyzer_terms or not overlap:
        raise ValueError("sparse analyzer and document artifact have no token overlap")
    if expected_document_id not in hit_ids:
        raise ValueError("sparse query did not retrieve the expected document")
    return overlap


def collect_wands_sparse_tokenizer_probe(
    client: SparseProbeClient,
    *,
    root: Path,
    spec: NeuralSparseSpec,
    expected_document_id: str | None = None,
) -> dict[str, object]:
    embeddings_path = (
        root / "data/cache/neural-sparse/wands-doc-v3-distill.jsonl"
    )
    probe_document_id = expected_document_id
    sparse_terms: set[str] | None = None
    with embeddings_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise DatasetIntegrityError(
                    f"invalid sparse probe embedding at {embeddings_path}:{line_number}"
                ) from error
            if not isinstance(record, dict):
                raise DatasetIntegrityError(
                    f"invalid sparse probe embedding at {embeddings_path}:{line_number}"
                )
            record_id = str(record.get("product_id", ""))
            if probe_document_id is None:
                probe_document_id = record_id
            if not record_id or record_id != probe_document_id:
                continue
            embedding = record.get("embedding")
            if not isinstance(embedding, dict) or not embedding:
                raise DatasetIntegrityError("sparse probe embedding is empty")
            sparse_terms = {str(token) for token in embedding}
            break
    if not probe_document_id or sparse_terms is None:
        raise DatasetIntegrityError("deterministic sparse probe product is missing")

    source_value = client.request(
        "GET",
        f"/{spec.index_name}/_source/{probe_document_id}",
    )
    if not isinstance(source_value, dict):
        raise DatasetIntegrityError("neural-sparse probe source is not an object")
    source = cast(dict[str, Any], source_value)
    title = source.get("title")
    if not isinstance(title, str) or not title.strip():
        raise DatasetIntegrityError("neural-sparse probe source title is missing")
    analyzed_value = client.request(
        "POST",
        "/_analyze",
        json_body={"analyzer": spec.query_analyzer, "text": title},
    )
    if not isinstance(analyzed_value, dict) or not isinstance(
        analyzed_value.get("tokens"), list
    ):
        raise DatasetIntegrityError("neural-sparse analyzer response differs")
    analyzer_terms: set[str] = set()
    for token in cast(list[Any], analyzed_value["tokens"]):
        if not isinstance(token, dict) or not isinstance(token.get("token"), str):
            raise DatasetIntegrityError("neural-sparse analyzer token differs")
        analyzer_terms.add(str(token["token"]))

    search = client.search(
        spec.index_name,
        {
            "size": 5,
            "_source": ["product_id", "title"],
            "query": build_sparse_query(spec, title),
        },
    )
    hits_value = search.get("hits")
    if not isinstance(hits_value, dict) or not isinstance(
        hits_value.get("hits"), list
    ):
        raise DatasetIntegrityError("neural-sparse probe search response differs")
    hit_ids: list[str] = []
    for hit in cast(list[Any], hits_value["hits"]):
        if not isinstance(hit, dict) or not isinstance(hit.get("_id"), str):
            raise DatasetIntegrityError("neural-sparse probe hit differs")
        hit_ids.append(str(hit["_id"]))

    expected_search = client.search(
        spec.index_name,
        {
            "size": 1,
            "_source": ["product_id", "title"],
            "query": {
                "bool": {
                    "must": [build_sparse_query(spec, title)],
                    "filter": [{"ids": {"values": [probe_document_id]}}],
                }
            },
        },
    )
    expected_hits_value = expected_search.get("hits")
    if not isinstance(expected_hits_value, dict) or not isinstance(
        expected_hits_value.get("hits"), list
    ):
        raise DatasetIntegrityError("neural-sparse expected-document response differs")
    expected_hit_ids: list[str] = []
    for hit in cast(list[Any], expected_hits_value["hits"]):
        score = hit.get("_score") if isinstance(hit, dict) else None
        if (
            not isinstance(hit, dict)
            or not isinstance(hit.get("_id"), str)
            or not isinstance(score, int | float)
            or isinstance(score, bool)
            or not math.isfinite(float(score))
            or float(score) <= 0
        ):
            raise DatasetIntegrityError(
                "neural-sparse expected-document match differs"
            )
        expected_hit_ids.append(str(hit["_id"]))
    try:
        overlap = validate_sparse_probe(
            expected_document_id=probe_document_id,
            hit_ids=expected_hit_ids,
            sparse_terms=sparse_terms,
            analyzer_terms=analyzer_terms,
        )
    except ValueError as error:
        raise DatasetIntegrityError(str(error)) from error
    return {
        "document_id": probe_document_id,
        "artifact_sparse_term_count": len(sparse_terms),
        "analyzer_terms": sorted(analyzer_terms),
        "overlap": overlap,
        "probe_mode": "id_filtered_positive_match_with_top5_diagnostics",
        "expected_document_matched": True,
        "probe_hit_ids": hit_ids,
    }


def verify_wands_sparse_tokenizer_probe(
    client: SparseProbeClient,
    *,
    root: Path,
    spec: NeuralSparseSpec,
    index_evidence: Mapping[str, Any],
) -> dict[str, object]:
    path = root / "results/wands/neural-sparse/tokenizer-inspection.json"
    value = read_json(path)
    if not isinstance(value, dict):
        raise DatasetIntegrityError(
            "neural-sparse tokenizer inspection must be an object"
        )
    inspection = cast(dict[str, Any], value)
    live_probe = collect_wands_sparse_tokenizer_probe(
        client,
        root=root,
        spec=spec,
    )
    if any(inspection.get(key) != expected for key, expected in live_probe.items()):
        raise DatasetIntegrityError(
            "neural-sparse tokenizer inspection does not reproduce from live probe"
        )
    expected_quality = index_evidence.get("eligible_for_decision") is True
    if (
        inspection.get("schema_version") != 2
        or inspection.get("status") != "passed"
        or inspection.get("index_manifest") != index_evidence.get("artifact")
        or inspection.get("quality_evidence_eligible_for_decision")
        is not expected_quality
    ):
        raise DatasetIntegrityError("neural-sparse tokenizer inspection differs")
    return {
        "artifact": _artifact(root, path),
        "schema_valid": True,
        "status_passed": True,
        "index_manifest_bound": True,
        "live_probe_verified": True,
        "quality_declaration_consistent": True,
        "quality_declared": expected_quality,
        "eligible_for_decision": expected_quality,
        "ineligibility_reasons": [] if expected_quality else ["index_ineligible"],
    }


def sparse_document_text(document: dict[str, Any]) -> str:
    values = [
        document.get("title"),
        document.get("brand"),
        document.get("category"),
        document.get("product_class"),
        document.get("features"),
        document.get("description"),
    ]
    parts = [str(value).strip() for value in values if value is not None and str(value).strip()]
    return ". ".join(parts)


def iter_sparse_documents(
    path: Path,
    spec: NeuralSparseSpec,
) -> Iterator[tuple[str, dict[str, Any]]]:
    for document_id, document in iter_prepared_products(path):
        output = dict(document)
        output[spec.text_field] = sparse_document_text(document)
        yield document_id, output


def iter_precomputed_sparse_documents(
    products_path: Path,
    embeddings_path: Path,
    spec: NeuralSparseSpec,
) -> Iterator[tuple[str, dict[str, Any]]]:
    products = iter_prepared_products(products_path)
    with embeddings_path.open(encoding="utf-8") as handle:
        embeddings = (json.loads(line) for line in handle if line.strip())
        for product, embedding in zip(products, embeddings, strict=True):
            document_id, document = product
            if str(embedding.get("product_id")) != document_id:
                raise ValueError(
                    f"sparse embedding order differs at product {document_id}"
                )
            values = embedding.get("embedding")
            if not isinstance(values, dict) or not values:
                raise ValueError(f"sparse embedding is empty for product {document_id}")
            output = dict(document)
            output[spec.text_field] = sparse_document_text(document)
            output[spec.embedding_field] = values
            yield document_id, output


def select_registered_model(
    hits: list[dict[str, Any]],
    spec: NeuralSparseSpec,
) -> dict[str, Any] | None:
    matches = []
    for hit in hits:
        source = hit.get("_source")
        if not isinstance(source, dict):
            continue
        version = source.get("model_version", source.get("version"))
        if source.get("name") == spec.model_name and version == spec.model_version:
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


def _spanning_indexes(length: int, sample_size: int) -> list[int]:
    if length <= 0 or sample_size <= 0:
        raise ValueError("spanning sample dimensions must be positive")
    if length <= sample_size:
        return list(range(length))
    return [
        position * (length - 1) // (sample_size - 1)
        for position in range(sample_size)
    ]


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }
