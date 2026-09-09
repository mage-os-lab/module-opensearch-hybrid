from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from poc.datasets import DatasetIntegrityError, WandsDatasetSpec, file_facts
from poc.neural_sparse import (
    build_sparse_index_definition,
    build_sparse_ingest_pipeline,
    build_sparse_query,
    collect_wands_sparse_tokenizer_probe,
    load_neural_sparse_spec,
    select_registered_model,
    validate_sparse_probe,
    verify_neural_sparse_model_artifacts,
    verify_wands_sparse_precompute,
    verify_wands_sparse_tokenizer_probe,
    wands_sparse_replay_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def test_registered_doc_only_sparse_configuration_is_explicit() -> None:
    spec = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")

    assert spec.model_name.endswith("encoding-doc-v3-distill")
    assert spec.package_sha256 == (
        "89bfabeb3a1c8f462eb39d1692b2c366336e1a504948422621e0a89c4f8ca988"
    )
    assert spec.package_bytes == 268866133
    assert spec.model_content_sha256 == (
        "97e26ed0e46257618c08553ea3a19c753afe7f659a71d58b506a9bba05b6e563"
    )
    assert spec.model_content_bytes == 268143948
    assert spec.tokenizer_content_sha256 == (
        "91f1def9b9391fdabe028cd3f3fcc4efd34e5d1f08c3bf2de513ebb5911a1854"
    )
    assert spec.tokenizer_content_bytes == 711649
    assert spec.query_analyzer == "bert-uncased"
    assert spec.prune_ratio == 0.1
    assert spec.inference_batch_size == 64
    assert spec.bulk_request_size == 256


def test_sparse_model_artifacts_are_bound_to_registered_bytes(tmp_path: Path) -> None:
    spec = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")
    model = tmp_path / "model.pt"
    tokenizer = tmp_path / "tokenizer.json"
    model.write_bytes(b"model")
    tokenizer.write_bytes(b"tokenizer")
    fixture_spec = replace(
        spec,
        model_content_sha256=file_facts(model).sha256,
        model_content_bytes=file_facts(model).bytes,
        tokenizer_content_sha256=file_facts(tokenizer).sha256,
        tokenizer_content_bytes=file_facts(tokenizer).bytes,
    )

    evidence = verify_neural_sparse_model_artifacts(
        fixture_spec,
        root=tmp_path,
        model_path=model,
        tokenizer_path=tokenizer,
    )
    assert evidence["model_file"]["sha256"] == fixture_spec.model_content_sha256

    model.write_bytes(b"tampered")
    with pytest.raises(DatasetIntegrityError, match="registered bytes"):
        verify_neural_sparse_model_artifacts(
            fixture_spec,
            root=tmp_path,
            model_path=model,
            tokenizer_path=tokenizer,
        )


def test_wands_sparse_precompute_requires_current_registered_generator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = (
        tmp_path
        / "data/cache/models/neural-sparse/doc-v3-distill/"
        "opensearch-neural-sparse-encoding-doc-v3-distill.pt"
    )
    tokenizer = model.with_name("tokenizer.json")
    products = tmp_path / "data/prepared/wands/products.jsonl"
    embeddings = tmp_path / "data/cache/neural-sparse/wands-doc-v3-distill.jsonl"
    manifest = tmp_path / "results/wands/neural-sparse/precompute-manifest.json"
    for path in (model, tokenizer, products, embeddings, manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"model")
    tokenizer.write_bytes(b"tokenizer")
    products.write_text(json.dumps({"product_id": "1", "title": "shoe"}) + "\n")
    embeddings.write_text(
        json.dumps({"product_id": "1", "embedding": {"shoe": 1.0}}) + "\n"
    )
    spec = replace(
        load_neural_sparse_spec(ROOT / "config/neural_sparse.toml"),
        model_content_sha256=file_facts(model).sha256,
        model_content_bytes=file_facts(model).bytes,
        tokenizer_content_sha256=file_facts(tokenizer).sha256,
        tokenizer_content_bytes=file_facts(tokenizer).bytes,
        inference_batch_size=1,
    )
    dataset = WandsDatasetSpec(
        source="fixture",
        revision="a" * 40,
        expected_products=1,
        expected_queries=1,
        expected_judgments=1,
        expected_unique_pairs=1,
        expected_duplicate_pairs=0,
        expected_conflicting_pairs=0,
        license="fixture",
        primary_gain_mapping="fixture",
        duplicate_policy="fixture",
        split_seed="fixture",
        files={},
    )
    provenance = {"code_revision": {"git_commit": "a" * 40}}
    embedding_facts = file_facts(embeddings)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "model": spec.model_name,
                "model_version": spec.model_version,
                "model_file": {
                    "path": str(model.relative_to(tmp_path)),
                    "sha256": file_facts(model).sha256,
                    "bytes": file_facts(model).bytes,
                },
                "tokenizer_file": {
                    "path": str(tokenizer.relative_to(tmp_path)),
                    "sha256": file_facts(tokenizer).sha256,
                    "bytes": file_facts(tokenizer).bytes,
                },
                "products_sha256": file_facts(products).sha256,
                "runtime": "torchscript_cpu",
                "batch_size": 1,
                "maximum_token_length": 256,
                "pruning": {
                    "type": "max_ratio",
                    "ratio": spec.prune_ratio,
                    "implementation": "local_precomputed_exact",
                },
                "deterministic_replay": wands_sparse_replay_contract(
                    products_path=products,
                    dataset=dataset,
                    spec=spec,
                ),
                "document_recipe": (
                    "title.brand.category.product_class.features.description"
                ),
                "records": 1,
                "embeddings": {
                    "path": str(embeddings.relative_to(tmp_path)),
                    "sha256": embedding_facts.sha256,
                    "file_sha256": embedding_facts.sha256,
                    "bytes": embedding_facts.bytes,
                },
                "generation_provenance": provenance,
                "benchmark_provenance": provenance,
                "generation_provenance_eligible_for_decision": True,
                "quality_evidence_eligible_for_decision": True,
            }
        )
        + "\n"
    )
    requirements: list[bool] = []

    class FakeEncoder:
        runtime = "torchscript_cpu"

        def encode(
            self,
            batch: list[tuple[str, str]],
        ) -> list[tuple[str, dict[str, float]]]:
            return [(product_id, {"shoe": 1.0}) for product_id, _ in batch]

    def verify(*args: object, **kwargs: object) -> bool:
        requirements.append(bool(kwargs.get("require_current_code_revision")))
        return True

    monkeypatch.setattr("poc.neural_sparse.verify_decision_provenance", verify)
    monkeypatch.setattr(
        "poc.neural_sparse.create_wands_sparse_encoder",
        lambda **_kwargs: FakeEncoder(),
    )
    evidence = verify_wands_sparse_precompute(
        root=tmp_path,
        dataset=dataset,
        spec=spec,
        products_path=products,
        embeddings_path=embeddings,
        manifest_path=manifest,
    )

    assert evidence["semantic_records_verified"] == 1
    assert evidence["eligible_for_decision"] is True
    assert requirements == [True]


def test_wands_sparse_precompute_replays_official_encoder_not_rehashed_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = (
        tmp_path
        / "data/cache/models/neural-sparse/doc-v3-distill/"
        "opensearch-neural-sparse-encoding-doc-v3-distill.pt"
    )
    tokenizer = model.with_name("tokenizer.json")
    products = tmp_path / "data/prepared/wands/products.jsonl"
    embeddings = tmp_path / "data/cache/neural-sparse/wands-doc-v3-distill.jsonl"
    manifest = tmp_path / "results/wands/neural-sparse/precompute-manifest.json"
    for path in (model, tokenizer, products, embeddings, manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"model")
    tokenizer.write_bytes(b"tokenizer")
    products.write_text(json.dumps({"product_id": "1", "title": "shoe"}) + "\n")
    embeddings.write_text(
        json.dumps({"product_id": "1", "embedding": {"shoe": 2.0}}) + "\n"
    )
    spec = replace(
        load_neural_sparse_spec(ROOT / "config/neural_sparse.toml"),
        model_content_sha256=file_facts(model).sha256,
        model_content_bytes=file_facts(model).bytes,
        tokenizer_content_sha256=file_facts(tokenizer).sha256,
        tokenizer_content_bytes=file_facts(tokenizer).bytes,
        inference_batch_size=1,
    )
    dataset = WandsDatasetSpec(
        source="fixture",
        revision="a" * 40,
        expected_products=1,
        expected_queries=1,
        expected_judgments=1,
        expected_unique_pairs=1,
        expected_duplicate_pairs=0,
        expected_conflicting_pairs=0,
        license="fixture",
        primary_gain_mapping="fixture",
        duplicate_policy="fixture",
        split_seed="fixture",
        files={},
    )
    embedding_facts = file_facts(embeddings)
    provenance = {"code_revision": {"git_commit": "a" * 40}}
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "model": spec.model_name,
                "model_version": spec.model_version,
                "model_file": {
                    "path": str(model.relative_to(tmp_path)),
                    "sha256": file_facts(model).sha256,
                    "bytes": file_facts(model).bytes,
                },
                "tokenizer_file": {
                    "path": str(tokenizer.relative_to(tmp_path)),
                    "sha256": file_facts(tokenizer).sha256,
                    "bytes": file_facts(tokenizer).bytes,
                },
                "products_sha256": file_facts(products).sha256,
                "runtime": "torchscript_cpu",
                "batch_size": 1,
                "maximum_token_length": 256,
                "pruning": {
                    "type": "max_ratio",
                    "ratio": spec.prune_ratio,
                    "implementation": "local_precomputed_exact",
                },
                "deterministic_replay": wands_sparse_replay_contract(
                    products_path=products,
                    dataset=dataset,
                    spec=spec,
                ),
                "document_recipe": (
                    "title.brand.category.product_class.features.description"
                ),
                "records": 1,
                "embeddings": {
                    "path": str(embeddings.relative_to(tmp_path)),
                    "sha256": embedding_facts.sha256,
                    "file_sha256": embedding_facts.sha256,
                    "bytes": embedding_facts.bytes,
                },
                "generation_provenance": provenance,
                "benchmark_provenance": provenance,
                "generation_provenance_eligible_for_decision": True,
                "quality_evidence_eligible_for_decision": True,
            }
        )
        + "\n"
    )

    class FakeEncoder:
        runtime = "torchscript_cpu"

        def encode(
            self,
            batch: list[tuple[str, str]],
        ) -> list[tuple[str, dict[str, float]]]:
            return [(product_id, {"shoe": 1.0}) for product_id, _ in batch]

    monkeypatch.setattr(
        "poc.neural_sparse.create_wands_sparse_encoder",
        lambda **_kwargs: FakeEncoder(),
    )
    monkeypatch.setattr(
        "poc.neural_sparse.verify_decision_provenance",
        lambda *_args, **_kwargs: True,
    )

    with pytest.raises(DatasetIntegrityError, match="sample re-encoding"):
        verify_wands_sparse_precompute(
            root=tmp_path,
            dataset=dataset,
            spec=spec,
            products_path=products,
            embeddings_path=embeddings,
            manifest_path=manifest,
        )


class _SparseProbeClient:
    def __init__(self, *, expected_score: float = 2.5) -> None:
        self.expected_score = expected_score

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if method == "GET" and path.endswith("/_source/1"):
            return {"product_id": "1", "title": "Slow Cooker"}
        if method == "POST" and path == "/_analyze":
            assert json_body == {"analyzer": "bert-uncased", "text": "Slow Cooker"}
            return {"tokens": [{"token": "slow"}, {"token": "cooker"}]}
        raise AssertionError((method, path, json_body))

    def search(
        self,
        index: str,
        body: dict[str, object],
    ) -> dict[str, object]:
        assert index == "fixture-index"
        query = body["query"]
        assert query
        if isinstance(query, dict) and "bool" in query:
            assert query["bool"] == {
                "must": [
                    {
                        "neural_sparse": {
                            "sparse_embedding": {
                                "query_text": "Slow Cooker",
                                "analyzer": "bert-uncased",
                            }
                        }
                    }
                ],
                "filter": [{"ids": {"values": ["1"]}}],
            }
            return {
                "hits": {
                    "hits": [{"_id": "1", "_score": self.expected_score}]
                }
            }
        return {"hits": {"hits": [{"_id": "2"}, {"_id": "3"}]}}


def test_sparse_tokenizer_probe_is_reproduced_live(
    tmp_path: Path,
) -> None:
    embeddings = tmp_path / "data/cache/neural-sparse/wands-doc-v3-distill.jsonl"
    embeddings.parent.mkdir(parents=True)
    embeddings.write_text(
        json.dumps(
            {"product_id": "1", "embedding": {"slow": 1.0, "cooker": 0.5}}
        )
        + "\n"
    )
    spec = replace(
        load_neural_sparse_spec(ROOT / "config/neural_sparse.toml"),
        index_name="fixture-index",
    )
    client = _SparseProbeClient()
    live = collect_wands_sparse_tokenizer_probe(
        client,
        root=tmp_path,
        spec=spec,
    )
    assert live["probe_hit_ids"] == ["2", "3"]
    assert live["expected_document_matched"] is True
    index_manifest = tmp_path / "results/wands/neural-sparse/index-manifest.json"
    index_manifest.parent.mkdir(parents=True)
    index_manifest.write_text("{}\n")
    index_artifact = {
        "path": str(index_manifest.relative_to(tmp_path)),
        "sha256": file_facts(index_manifest).sha256,
        "bytes": file_facts(index_manifest).bytes,
    }
    inspection_path = (
        tmp_path / "results/wands/neural-sparse/tokenizer-inspection.json"
    )
    inspection_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                **live,
                "status": "passed",
                "index_manifest": index_artifact,
                "quality_evidence_eligible_for_decision": True,
            }
        )
        + "\n"
    )

    evidence = verify_wands_sparse_tokenizer_probe(
        client,
        root=tmp_path,
        spec=spec,
        index_evidence={
            "artifact": index_artifact,
            "eligible_for_decision": True,
        },
    )
    assert evidence["live_probe_verified"] is True
    assert evidence["eligible_for_decision"] is True

    inspection = json.loads(inspection_path.read_text())
    inspection["probe_hit_ids"] = ["2", "1"]
    inspection_path.write_text(json.dumps(inspection) + "\n")
    with pytest.raises(DatasetIntegrityError, match="live probe"):
        verify_wands_sparse_tokenizer_probe(
            client,
            root=tmp_path,
            spec=spec,
            index_evidence={
                "artifact": index_artifact,
                "eligible_for_decision": True,
            },
        )


def test_sparse_tokenizer_probe_rejects_nonpositive_expected_match(
    tmp_path: Path,
) -> None:
    embeddings = tmp_path / "data/cache/neural-sparse/wands-doc-v3-distill.jsonl"
    embeddings.parent.mkdir(parents=True)
    embeddings.write_text(
        json.dumps(
            {"product_id": "1", "embedding": {"slow": 1.0, "cooker": 0.5}}
        )
        + "\n"
    )
    spec = replace(
        load_neural_sparse_spec(ROOT / "config/neural_sparse.toml"),
        index_name="fixture-index",
    )

    with pytest.raises(DatasetIntegrityError, match="expected-document match"):
        collect_wands_sparse_tokenizer_probe(
            _SparseProbeClient(expected_score=0.0),
            root=tmp_path,
            spec=spec,
        )


def test_sparse_ingestion_and_query_use_compatible_doc_only_fields() -> None:
    spec = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")
    pipeline = build_sparse_ingest_pipeline(spec, model_id="model-1")
    definition = build_sparse_index_definition(spec)

    processor = pipeline["processors"][0]["sparse_encoding"]
    assert processor["field_map"] == {"sparse_text": "sparse_embedding"}
    assert processor["batch_size"] == 64
    assert processor["skip_existing"] is True
    assert definition["mappings"]["properties"]["sparse_embedding"] == {
        "type": "rank_features"
    }
    assert build_sparse_query(spec, "running shoe") == {
        "neural_sparse": {
            "sparse_embedding": {
                "query_text": "running shoe",
                "analyzer": "bert-uncased",
            }
        }
    }


def test_precomputed_sparse_index_does_not_apply_default_pipeline() -> None:
    spec = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")

    definition = build_sparse_index_definition(spec, use_default_pipeline=False)

    assert "default_pipeline" not in definition["settings"]["index"]
    assert definition["mappings"]["properties"]["sparse_embedding"] == {
        "type": "rank_features"
    }


def test_sparse_probe_requires_token_overlap_and_expected_hit() -> None:
    assert validate_sparse_probe(
        expected_document_id="1",
        hit_ids=["1", "2"],
        sparse_terms={"slow", "cook", "cooker"},
        analyzer_terms={"slow", "cooker"},
    ) == ["cooker", "slow"]

    try:
        validate_sparse_probe(
            expected_document_id="1",
            hit_ids=["2"],
            sparse_terms={"slow"},
            analyzer_terms={"slow"},
        )
    except ValueError as exc:
        assert "expected document" in str(exc)
    else:
        raise AssertionError("missing expected document should fail the sparse probe")


def test_registered_model_selection_prefers_deployed_exact_version() -> None:
    spec = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")
    models = [
        {
            "_id": "loading",
            "_source": {
                "name": spec.model_name,
                "model_version": spec.model_version,
                "model_state": "REGISTERED",
            },
        },
        {
            "_id": "ready",
            "_source": {
                "name": spec.model_name,
                "model_version": spec.model_version,
                "model_state": "DEPLOYED",
            },
        },
    ]

    selected = select_registered_model(models, spec)
    assert selected is not None
    assert selected["_id"] == "ready"
