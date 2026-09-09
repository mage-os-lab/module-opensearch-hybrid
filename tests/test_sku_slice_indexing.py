from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, TypedDict, cast

import pytest

from poc.datasets import DatasetIntegrityError, WandsDatasetSpec, file_facts
from poc.neural_sparse import NeuralSparseSpec, wands_sparse_replay_contract
from poc.sku_slice import SkuSliceSpec
from poc.sku_slice_indexing import build_sku_slice_index, verify_sku_slice_index


class SkuIndexFixture(TypedDict):
    root: Path
    dataset_spec: WandsDatasetSpec
    sparse_spec: NeuralSparseSpec
    sku_spec: SkuSliceSpec
    products_path: Path
    embeddings_path: Path
    preparation_path: Path
    precompute_manifest_path: Path
    manifest_path: Path


def test_sku_index_schema_v2_binds_start_provenance_and_verified_upstreams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    start_provenance = _provenance("a")
    collections: list[object] = []
    requirements: list[bool] = []

    def collect(*args: object, **kwargs: object) -> dict[str, object]:
        collections.append((args, kwargs))
        return start_provenance

    def verify(*args: object, **kwargs: object) -> bool:
        requirements.append(bool(kwargs.get("require_current_code_revision")))
        return True

    monkeypatch.setattr("poc.sku_slice_indexing.collect_manifest_provenance", collect)
    monkeypatch.setattr("poc.sku_slice_indexing.verify_decision_provenance", verify)
    _patch_sparse_replay(monkeypatch, verify)
    client = FakeOpenSearchClient(document_count=2)

    manifest = build_sku_slice_index(cast(Any, client), **fixture)

    assert manifest["schema_version"] == 2
    assert manifest["benchmark_provenance"] == start_provenance
    assert manifest["upstream_evidence_eligible_for_decision"] is True
    assert manifest["indexing_provenance_eligible_for_decision"] is True
    assert manifest["latency_evidence_eligible_for_decision"] is True
    assert len(collections) == 1
    assert requirements == [True, True]

    requirements.clear()
    verified = verify_sku_slice_index(cast(Any, client), **fixture)
    assert verified == manifest
    assert requirements == [True, False]

    client.documents["2"][fixture["sparse_spec"].embedding_field] = {
        "altered": 9.0
    }
    with pytest.raises(DatasetIntegrityError, match="source differs for document 2"):
        verify_sku_slice_index(cast(Any, client), **fixture)


def test_sku_index_rejects_legacy_sparse_precompute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    precompute = cast(
        dict[str, Any],
        json.loads(fixture["precompute_manifest_path"].read_text()),
    )
    precompute.pop("generation_provenance")
    precompute.pop("generation_provenance_eligible_for_decision")
    precompute.pop("quality_evidence_eligible_for_decision")
    precompute["schema_version"] = 1
    fixture["precompute_manifest_path"].write_text(
        json.dumps(precompute, indent=2, sort_keys=True) + "\n"
    )
    monkeypatch.setattr(
        "poc.sku_slice_indexing.collect_manifest_provenance",
        lambda *args, **kwargs: _provenance("b"),
    )
    monkeypatch.setattr(
        "poc.sku_slice_indexing.verify_decision_provenance",
        lambda *args, **kwargs: True,
    )
    _patch_sparse_replay(monkeypatch, lambda *args, **kwargs: True)

    with pytest.raises(
        DatasetIntegrityError,
        match="precompute metadata or registered artifacts differ",
    ):
        build_sku_slice_index(
            cast(Any, FakeOpenSearchClient(document_count=2)),
            **fixture,
        )


def test_sku_index_verifier_rejects_schema_and_live_index_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    monkeypatch.setattr(
        "poc.sku_slice_indexing.collect_manifest_provenance",
        lambda *args, **kwargs: _provenance("c"),
    )
    monkeypatch.setattr(
        "poc.sku_slice_indexing.verify_decision_provenance",
        lambda *args, **kwargs: True,
    )
    _patch_sparse_replay(monkeypatch, lambda *args, **kwargs: True)
    client = FakeOpenSearchClient(document_count=2)
    build_sku_slice_index(cast(Any, client), **fixture)
    manifest = cast(dict[str, Any], json.loads(fixture["manifest_path"].read_text()))
    manifest["schema_version"] = 1
    fixture["manifest_path"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(DatasetIntegrityError, match="SKU index manifest schema"):
        verify_sku_slice_index(cast(Any, client), **fixture)

    manifest["schema_version"] = 2
    fixture["manifest_path"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    client.uuid = "different-live-index"
    with pytest.raises(DatasetIntegrityError, match="live synthetic SKU index facts"):
        verify_sku_slice_index(cast(Any, client), **fixture)


def test_sku_index_verifier_rejects_live_analysis_or_setting_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    monkeypatch.setattr(
        "poc.sku_slice_indexing.collect_manifest_provenance",
        lambda *args, **kwargs: _provenance("f"),
    )
    monkeypatch.setattr(
        "poc.sku_slice_indexing.verify_decision_provenance",
        lambda *args, **kwargs: True,
    )
    _patch_sparse_replay(monkeypatch, lambda *args, **kwargs: True)
    client = FakeOpenSearchClient(document_count=2)
    build_sku_slice_index(cast(Any, client), **fixture)
    client.live_knn = "true"

    with pytest.raises(DatasetIntegrityError, match="live synthetic SKU settings"):
        verify_sku_slice_index(cast(Any, client), **fixture)


def test_sku_index_verifier_rejects_sparse_model_artifact_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    monkeypatch.setattr(
        "poc.sku_slice_indexing.collect_manifest_provenance",
        lambda *args, **kwargs: _provenance("2"),
    )
    monkeypatch.setattr(
        "poc.sku_slice_indexing.verify_decision_provenance",
        lambda *args, **kwargs: True,
    )
    _patch_sparse_replay(monkeypatch, lambda *args, **kwargs: True)
    client = FakeOpenSearchClient(document_count=2)
    build_sku_slice_index(cast(Any, client), **fixture)
    model_path = (
        fixture["root"]
        / "data/cache/models/neural-sparse/doc-v3-distill/"
        "opensearch-neural-sparse-encoding-doc-v3-distill.pt"
    )
    model_path.write_bytes(b"different model bytes")

    with pytest.raises(DatasetIntegrityError, match="registered bytes"):
        verify_sku_slice_index(cast(Any, client), **fixture)


def test_sku_preparation_rejects_rehashed_nondeterministic_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    query_path = fixture["root"] / "data/prepared/wands/queries.sku-slice.jsonl"
    records = [json.loads(line) for line in query_path.read_text().splitlines()]
    records[0]["query"] = "RW-arbitrary"
    query_path.write_text(
        "".join(
            json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
            for record in records
        )
    )
    preparation = cast(
        dict[str, Any], json.loads(fixture["preparation_path"].read_text())
    )
    preparation["queries"] = _artifact(fixture["root"], query_path)
    fixture["preparation_path"].write_text(
        json.dumps(preparation, indent=2, sort_keys=True) + "\n"
    )
    monkeypatch.setattr(
        "poc.sku_slice_indexing.collect_manifest_provenance",
        lambda *args, **kwargs: _provenance("1"),
    )
    monkeypatch.setattr(
        "poc.sku_slice_indexing.verify_decision_provenance",
        lambda *args, **kwargs: True,
    )

    with pytest.raises(DatasetIntegrityError, match="deterministic query content"):
        build_sku_slice_index(
            cast(Any, FakeOpenSearchClient(document_count=2)),
            **fixture,
        )


def _write_fixture(root: Path) -> SkuIndexFixture:
    products_path = root / "data/prepared/wands/products.jsonl"
    embeddings_path = root / "data/cache/neural-sparse/wands.jsonl"
    preparation_path = root / "results/wands/sku-slice/preparation.json"
    precompute_path = root / "results/wands/neural-sparse/precompute-manifest.json"
    manifest_path = root / "results/wands/sku-slice/index-manifest.json"
    model_path = (
        root
        / "data/cache/models/neural-sparse/doc-v3-distill/"
        "opensearch-neural-sparse-encoding-doc-v3-distill.pt"
    )
    tokenizer_path = model_path.with_name("tokenizer.json")
    for path in (
        products_path,
        embeddings_path,
        preparation_path,
        precompute_path,
        model_path,
        tokenizer_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"official model bytes")
    tokenizer_path.write_text('{"version":"1.0"}\n')
    products_path.write_text(
        json.dumps(
            {
                "product_id": "1",
                "title": "Trail shoe",
                "brand": "Rocket",
                "category": "Shoes",
                "product_class": "Footwear",
                "features": "Light",
                "description": "A trail shoe",
            }
        )
        + "\n"
        + json.dumps(
            {
                "product_id": "2",
                "title": "Road shoe",
                "brand": "Rocket",
                "category": "Shoes",
                "product_class": "Footwear",
                "features": "Fast",
                "description": "A road shoe",
            }
        )
        + "\n"
    )
    embeddings_path.write_text(
        json.dumps({"product_id": "1", "embedding": {"trail": 1.0}})
        + "\n"
        + json.dumps({"product_id": "2", "embedding": {"road": 1.0}})
        + "\n"
    )
    dataset_spec = WandsDatasetSpec(
        source="https://example.test/wands",
        revision="d" * 40,
        expected_products=2,
        expected_queries=1,
        expected_judgments=1,
        expected_unique_pairs=1,
        expected_duplicate_pairs=1,
        expected_conflicting_pairs=1,
        license="test",
        primary_gain_mapping="test",
        duplicate_policy="test",
        split_seed="test",
        files={},
    )
    sparse_spec = NeuralSparseSpec(
        model_name="official-sparse",
        model_version="1.0.0",
        model_format="TORCH_SCRIPT",
        query_analyzer="bert-uncased",
        package_url="https://artifacts.opensearch.org/fixture.zip",
        package_sha256="c" * 64,
        package_bytes=3,
        model_content_sha256=file_facts(model_path).sha256,
        model_content_bytes=file_facts(model_path).bytes,
        tokenizer_content_sha256=file_facts(tokenizer_path).sha256,
        tokenizer_content_bytes=file_facts(tokenizer_path).bytes,
        index_name="unused",
        ingest_pipeline="unused",
        text_field="sparse_text",
        embedding_field="sparse_embedding",
        prune_type="max_ratio",
        prune_ratio=0.1,
        inference_batch_size=2,
        bulk_request_size=10,
    )
    sku_spec = SkuSliceSpec(
        dataset="WANDS synthetic SKU and known-item proxy",
        seed="seed",
        sku_prefix="RW",
        sku_queries=1,
        exact_title_queries=1,
        lexical_weight=0.3,
        maximum_mrr_regression=0.01,
        alpha=0.05,
        index_name="sku-index",
    )
    query_path = root / "data/prepared/wands/queries.sku-slice.jsonl"
    qrels_path = root / "data/prepared/wands/qrels.sku-slice.trec"
    query_path.write_text(
        "".join(
            json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
            for record in (
                {
                    "query_id": "sku-001",
                    "query": "RW-2",
                    "kind": "synthetic_sku",
                    "expected_document_id": "2",
                },
                {
                    "query_id": "title-001",
                    "query": "Trail shoe",
                    "kind": "exact_title",
                    "expected_document_id": "1",
                },
            )
        )
    )
    qrels_path.write_text("sku-001 0 2 2\ntitle-001 0 1 2\n")
    preparation_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": sku_spec.dataset,
                "seed": sku_spec.seed,
                "eligible_as_authentic_merchant_sku_evidence": False,
                "eligible_as_synthetic_known_item_regression_evidence": True,
                "source_products": _artifact(root, products_path),
                "counts": {"synthetic_sku": 1, "exact_title": 1, "total": 2},
                "queries": _artifact(root, query_path),
                "qrels": _artifact(root, qrels_path),
            },
            indent=2,
        )
        + "\n"
    )
    embeddings = _artifact(root, embeddings_path)
    precompute_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "model": sparse_spec.model_name,
                "model_version": sparse_spec.model_version,
                "model_file": _artifact(root, model_path),
                "tokenizer_file": _artifact(root, tokenizer_path),
                "runtime": "torchscript_cpu",
                "batch_size": sparse_spec.inference_batch_size,
                "maximum_token_length": 256,
                "pruning": {
                    "type": sparse_spec.prune_type,
                    "ratio": sparse_spec.prune_ratio,
                    "implementation": "local_precomputed_exact",
                },
                "deterministic_replay": wands_sparse_replay_contract(
                    products_path=products_path,
                    dataset=dataset_spec,
                    spec=sparse_spec,
                ),
                "document_recipe": (
                    "title.brand.category.product_class.features.description"
                ),
                "records": 2,
                "products_sha256": file_facts(products_path).sha256,
                "embeddings": {**embeddings, "file_sha256": embeddings["sha256"]},
                "generation_provenance": _provenance("e"),
                "benchmark_provenance": _provenance("e"),
                "generation_provenance_eligible_for_decision": True,
                "quality_evidence_eligible_for_decision": True,
            },
            indent=2,
        )
        + "\n"
    )
    return SkuIndexFixture(
        root=root,
        dataset_spec=dataset_spec,
        sparse_spec=sparse_spec,
        sku_spec=sku_spec,
        products_path=products_path,
        embeddings_path=embeddings_path,
        preparation_path=preparation_path,
        precompute_manifest_path=precompute_path,
        manifest_path=manifest_path,
    )


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }


def _provenance(seed: str) -> dict[str, object]:
    return {
        "code_revision": {
            "git_commit": seed * 40,
            "source_tree_sha256": seed * 64,
            "source_dirty": False,
        },
        "benchmark_profile": {"profile_id": "test"},
        "benchmark_profile_sha256": seed * 64,
        "benchmark_environment": {"status": "test"},
        "generator_host": {"architecture": "test", "system": "test"},
    }


def _patch_sparse_replay(
    monkeypatch: pytest.MonkeyPatch,
    verify: object,
) -> None:
    class FakeEncoder:
        runtime = "torchscript_cpu"

        def encode(
            self,
            batch: list[tuple[str, str]],
        ) -> list[tuple[str, dict[str, float]]]:
            return [
                (product_id, {"trail" if product_id == "1" else "road": 1.0})
                for product_id, _ in batch
            ]

    monkeypatch.setattr("poc.neural_sparse.verify_decision_provenance", verify)
    monkeypatch.setattr(
        "poc.neural_sparse.create_wands_sparse_encoder",
        lambda **_kwargs: FakeEncoder(),
    )


class FakeOpenSearchClient:
    def __init__(self, *, document_count: int) -> None:
        self.document_count = document_count
        self.definition: dict[str, Any] | None = None
        self.documents: dict[str, dict[str, Any]] = {}
        self.uuid = "sku-index-uuid"
        self.live_knn = "false"
        self.write_block = False

    def wait_until_ready(self, *, expected_version: str) -> str:
        return expected_version

    def delete_index(self, index: str) -> None:
        return None

    def create_index(self, index: str, definition: object) -> None:
        self.definition = cast(dict[str, Any], definition)

    def bulk_index(
        self,
        index: str,
        documents: Any,
        *,
        batch_size: int,
    ) -> None:
        self.documents = {
            document_id: deepcopy(dict(document))
            for document_id, document in documents
        }
        assert list(self.documents) == ["1", "2"]

    def refresh(self, index: str) -> None:
        return None

    def force_merge(self, index: str) -> None:
        return None

    def update_index_settings(self, index: str, settings: object) -> None:
        if settings == {"index": {"blocks": {"write": True}}}:
            self.write_block = True
        return None

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: object | None = None,
        params: object | None = None,
    ) -> dict[str, object]:
        if path.endswith("/_count"):
            return {"count": self.document_count}
        if path.endswith("/_mget"):
            assert method == "POST"
            assert isinstance(json_body, dict)
            return {
                "docs": [
                    {
                        "_id": document_id,
                        "found": document_id in self.documents,
                        "_source": deepcopy(self.documents.get(document_id)),
                    }
                    for document_id in cast(list[str], json_body["ids"])
                ]
            }
        if path.endswith("/_stats/docs,segments"):
            return {
                "_all": {
                    "primaries": {
                        "docs": {"count": self.document_count, "deleted": 0},
                        "segments": {"count": 1},
                    }
                }
            }
        if path.endswith("/_mapping"):
            assert self.definition is not None
            return {"sku-index": {"mappings": self.definition["mappings"]}}
        if path.endswith("/_settings"):
            assert self.definition is not None
            analysis = cast(dict[str, Any], self.definition["settings"])["analysis"]
            return {
                "sku-index": {
                    "settings": {
                        "index": {
                            "analysis": analysis,
                            "knn": self.live_knn,
                            "number_of_replicas": "0",
                            "number_of_shards": "1",
                            "refresh_interval": "1s",
                            "blocks": {"write": "true"}
                            if self.write_block
                            else {},
                        }
                    }
                }
            }
        return {
            "sku-index": {"settings": {"index": {"uuid": self.uuid}}}
        }
