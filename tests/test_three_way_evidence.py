from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import poc.three_way_benchmark as benchmark_module
import poc.three_way_indexing as indexing_module
from poc.config import ModelSpec, load_model_registry
from poc.datasets import DatasetIntegrityError, WandsDatasetSpec, file_facts
from poc.experiments import load_bm25_experiments
from poc.neural_sparse import (
    NeuralSparseSpec,
    load_neural_sparse_spec,
    wands_sparse_replay_contract,
)
from poc.query_runtime import load_query_runtime_spec
from poc.three_way import ThreeWayCandidate, ThreeWaySpec, load_three_way_spec
from poc.three_way_benchmark import (
    derive_three_way_summary_eligibility,
    expected_three_way_artifact_keys,
    verify_registered_three_way_benchmark_inputs,
)
from poc.three_way_indexing import (
    build_three_way_index,
    build_three_way_index_definition,
    derive_three_way_index_eligibility,
    verify_registered_three_way_index_inputs,
    verify_three_way_index,
    verify_three_way_sparse_precompute,
)

ROOT = Path(__file__).resolve().parents[1]


def test_three_way_registered_index_inputs_reject_changed_index_identity() -> None:
    dataset = _registered_dataset()
    sparse = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")
    registered = load_three_way_spec(ROOT / "config/three_way.toml")
    changed = ThreeWaySpec(
        index_name="unregistered-index",
        dense_model=registered.dense_model,
        candidates=registered.candidates,
    )
    model = load_model_registry(ROOT / "config/models.toml")[
        registered.dense_model
    ]

    with pytest.raises(DatasetIntegrityError, match="three-way configuration"):
        verify_registered_three_way_index_inputs(
            root=ROOT,
            dataset=dataset,
            spec=changed,
            sparse=sparse,
            model=model,
        )


def test_three_way_sparse_precompute_is_semantically_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, sparse, products, embeddings, manifest = _sparse_fixture(tmp_path)
    requirements: list[bool] = []

    def verify(*args: object, **kwargs: object) -> bool:
        requirements.append(bool(kwargs.get("require_current_code_revision")))
        return True

    _patch_sparse_replay(monkeypatch, verify)

    evidence = verify_three_way_sparse_precompute(
        root=tmp_path,
        dataset=dataset,
        sparse=sparse,
        products_path=products,
        embeddings_path=embeddings,
        manifest_path=manifest,
    )

    assert evidence["schema_version"] == 2
    assert evidence["records"] == 2
    assert evidence["semantic_records_verified"] == 2
    assert evidence["eligible_for_decision"] is True
    assert requirements == [True]


def test_three_way_sparse_precompute_rejects_rehashed_wrong_document_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, sparse, products, embeddings, manifest = _sparse_fixture(tmp_path)
    embeddings.write_text(
        json.dumps({"product_id": "2", "embedding": {"trail": 1.0}})
        + "\n"
        + json.dumps({"product_id": "1", "embedding": {"road": 1.0}})
        + "\n"
    )
    value = json.loads(manifest.read_text())
    value["embeddings"] = {
        **_artifact(tmp_path, embeddings),
        "file_sha256": file_facts(embeddings).sha256,
    }
    manifest.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    _patch_sparse_replay(monkeypatch, lambda *args, **kwargs: True)

    with pytest.raises(DatasetIntegrityError, match="order"):
        verify_three_way_sparse_precompute(
            root=tmp_path,
            dataset=dataset,
            sparse=sparse,
            products_path=products,
            embeddings_path=embeddings,
            manifest_path=manifest,
        )


def test_three_way_index_eligibility_uses_strict_conjunction() -> None:
    assert derive_three_way_index_eligibility(
        provenance_valid=True,
        preparation_eligible=True,
        dense_cache_eligible=True,
        sparse_precompute_eligible=True,
        upstream_unchanged=True,
        live_index_clean=True,
    ) == (True, [])
    assert derive_three_way_index_eligibility(
        provenance_valid=True,
        preparation_eligible=True,
        dense_cache_eligible=True,
        sparse_precompute_eligible=True,
        upstream_unchanged=1,
        live_index_clean=True,
    ) == (False, ["three-way index upstream evidence changed during build"])


def test_three_way_index_binds_full_dense_and_sparse_live_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, sparse, products, embeddings, sparse_manifest = _sparse_fixture(
        tmp_path
    )
    spec = ThreeWaySpec(
        index_name="three-way-index",
        dense_model="dense-model",
        candidates=(ThreeWayCandidate("candidate", (0.3, 0.35, 0.35)),),
    )
    model = _dense_model()
    manifest_path = tmp_path / "results/wands/three-way/index-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "code_revision": {
            "git_commit": "a" * 40,
            "source_tree_sha256": "b" * 64,
            "source_dirty": False,
        }
    }
    upstream = {
        "preparation": {"eligible_for_decision": True},
        "dense_embedding_cache": {"eligible_for_decision": True},
        "sparse_precompute": {"eligible_for_decision": True},
        "eligible_for_decision": True,
        "ineligibility_reasons": [],
    }
    dense_ids = ["1", "2"]
    dense_vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    monkeypatch.setattr(
        indexing_module,
        "collect_manifest_provenance",
        lambda *args, **kwargs: provenance,
    )
    monkeypatch.setattr(
        indexing_module,
        "verify_registered_three_way_index_inputs",
        lambda **kwargs: {"fixture": True},
    )
    monkeypatch.setattr(
        indexing_module,
        "_three_way_index_upstream_evidence",
        lambda **kwargs: upstream,
    )
    monkeypatch.setattr(
        indexing_module,
        "load_cached_vectors",
        lambda **kwargs: (dense_ids, dense_vectors),
    )
    monkeypatch.setattr(
        indexing_module,
        "wands_completion_provenance_evidence",
        lambda *args, **kwargs: (True, provenance["code_revision"]),
    )
    monkeypatch.setattr(
        indexing_module,
        "wands_recorded_provenance_valid",
        lambda *args, **kwargs: True,
    )
    client = FakeThreeWayClient(index=spec.index_name, document_count=2)

    manifest = build_three_way_index(
        cast(Any, client),
        root=tmp_path,
        dataset=dataset,
        spec=spec,
        sparse=sparse,
        model=model,
        products_path=products,
        sparse_embeddings_path=embeddings,
        sparse_manifest_path=sparse_manifest,
        manifest_path=manifest_path,
    )

    assert cast(dict[str, Any], manifest["index_content"])["documents"] == 2
    assert client.exists_count_fields == ["embedding_dense-model"]
    assert "_source" not in build_three_way_index_definition(
        spec, sparse, model
    )["mappings"]
    assert verify_three_way_index(
        cast(Any, client),
        root=tmp_path,
        dataset=dataset,
        spec=spec,
        sparse=sparse,
        model=model,
        products_path=products,
        sparse_embeddings_path=embeddings,
        sparse_manifest_path=sparse_manifest,
        manifest_path=manifest_path,
    ) == manifest
    assert client.exists_count_fields == [
        "embedding_dense-model",
        "embedding_dense-model",
    ]

    sparse_vector = client.documents["2"].pop(sparse.embedding_field)
    with pytest.raises(DatasetIntegrityError, match="source differs for document 2"):
        verify_three_way_index(
            cast(Any, client),
            root=tmp_path,
            dataset=dataset,
            spec=spec,
            sparse=sparse,
            model=model,
            products_path=products,
            sparse_embeddings_path=embeddings,
            sparse_manifest_path=sparse_manifest,
            manifest_path=manifest_path,
        )
    client.documents["2"][sparse.embedding_field] = sparse_vector

    client.documents["2"]["embedding_dense-model"] = [0.25, 0.75]
    with pytest.raises(DatasetIntegrityError, match="source differs for document 2"):
        verify_three_way_index(
            cast(Any, client),
            root=tmp_path,
            dataset=dataset,
            spec=spec,
            sparse=sparse,
            model=model,
            products_path=products,
            sparse_embeddings_path=embeddings,
            sparse_manifest_path=sparse_manifest,
            manifest_path=manifest_path,
        )


def test_three_way_registered_benchmark_inputs_reject_runtime_drift() -> None:
    spec = load_three_way_spec(ROOT / "config/three_way.toml")
    sparse = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")
    model = load_model_registry(ROOT / "config/models.toml")[spec.dense_model]
    runtime = load_query_runtime_spec(ROOT / "config/query_runtime.toml")
    experiments = load_bm25_experiments(ROOT / "config/experiments.toml")

    with pytest.raises(DatasetIntegrityError, match="query runtime"):
        verify_registered_three_way_benchmark_inputs(
            root=ROOT,
            spec=spec,
            sparse=sparse,
            model=model,
            runtime=replace(runtime, batch_size=2),
            experiments=experiments,
        )


def test_three_way_artifact_contract_and_summary_eligibility_are_strict() -> None:
    spec = load_three_way_spec(ROOT / "config/three_way.toml")

    assert expected_three_way_artifact_keys(spec) == tuple(
        [f"dev/{candidate.name}" for candidate in spec.candidates]
        + ["test/selected"]
    )
    assert derive_three_way_summary_eligibility(
        provenance_valid=True,
        upstream_eligible=True,
        upstream_unchanged=True,
        upstream_artifact_bindings_valid=True,
        source_bindings_valid=True,
        held_out_artifact_eligible=True,
    ) == (True, [])
    assert derive_three_way_summary_eligibility(
        provenance_valid=True,
        upstream_eligible=1,
        upstream_unchanged=True,
        upstream_artifact_bindings_valid=True,
        source_bindings_valid=True,
        held_out_artifact_eligible=True,
    ) == (False, ["three-way upstream evidence is not decision eligible"])


def test_three_way_source_binding_rejects_forged_ineligibility_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = ThreeWayCandidate("candidate", (0.3, 0.35, 0.35))
    spec = ThreeWaySpec("three-way-index", "dense-model", (candidate,))
    provenance = {"code_revision": {"git_commit": "a" * 40}}
    upstream_file = tmp_path / "results/upstream.json"
    upstream_file.parent.mkdir(parents=True, exist_ok=True)
    upstream_file.write_text("{}\n")
    upstream = {
        "eligible_for_decision": True,
        "source": _artifact(tmp_path, upstream_file),
    }
    artifacts: dict[str, dict[str, object]] = {}
    for key, split, eligible, reason in (
        ("dev/candidate", "dev", False, "forged reason"),
        ("test/selected", "test", True, None),
    ):
        tag = f"fixture-{split}"
        manifest_path = tmp_path / f"runs/{tag}.manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "dataset": "WANDS",
                    "method": benchmark_module.THREE_WAY_METHOD,
                    "split": split,
                    "decision_scope": "tuning" if split == "dev" else "held_out",
                    "benchmark_provenance": provenance,
                    "benchmark_provenance_valid": True,
                    "completion_code_revision": provenance["code_revision"],
                    "upstream_evidence": upstream,
                    "upstream_artifact_bindings_valid_at_completion": True,
                    "selected_model_eligible": True,
                    "eligible_for_decision": eligible,
                    "ineligibility_reason": reason,
                },
                sort_keys=True,
            )
            + "\n"
        )
        artifacts[key] = {
            "manifest": str(manifest_path.relative_to(tmp_path)),
            "manifest_sha256": file_facts(manifest_path).sha256,
        }
    monkeypatch.setattr(
        benchmark_module,
        "wands_recorded_provenance_valid",
        lambda *args, **kwargs: True,
    )

    _, valid, reasons = benchmark_module._three_way_source_manifest_bindings(
        root=tmp_path,
        spec=spec,
        artifacts=artifacts,
        benchmark_provenance=provenance,
        upstream_evidence=upstream,
        selected_model_eligible=True,
    )

    assert valid is False
    assert reasons == [
        "three-way source manifest binding differs: dev/candidate"
    ]


def _registered_dataset() -> WandsDatasetSpec:
    from poc.datasets import load_wands_config

    return load_wands_config(ROOT / "config/datasets.toml")


def _sparse_fixture(
    root: Path,
) -> tuple[WandsDatasetSpec, NeuralSparseSpec, Path, Path, Path]:
    products = root / "data/prepared/wands/products.jsonl"
    embeddings = root / "data/cache/neural-sparse/wands-doc-v3-distill.jsonl"
    manifest = root / "results/wands/neural-sparse/precompute-manifest.json"
    model = (
        root
        / "data/cache/models/neural-sparse/doc-v3-distill/"
        "opensearch-neural-sparse-encoding-doc-v3-distill.pt"
    )
    tokenizer = model.with_name("tokenizer.json")
    for path in (products, embeddings, manifest, model, tokenizer):
        path.parent.mkdir(parents=True, exist_ok=True)
    products.write_text(
        json.dumps({"product_id": "1", "title": "Trail shoe"})
        + "\n"
        + json.dumps({"product_id": "2", "title": "Road shoe"})
        + "\n"
    )
    embeddings.write_text(
        json.dumps({"product_id": "1", "embedding": {"trail": 1.0}})
        + "\n"
        + json.dumps({"product_id": "2", "embedding": {"road": 1.0}})
        + "\n"
    )
    model.write_bytes(b"model")
    tokenizer.write_text('{"version":"1.0"}\n')
    dataset = WandsDatasetSpec(
        source="https://example.test/wands",
        revision="a" * 40,
        expected_products=2,
        expected_queries=1,
        expected_judgments=1,
        expected_unique_pairs=1,
        expected_duplicate_pairs=0,
        expected_conflicting_pairs=0,
        license="test",
        primary_gain_mapping="test",
        duplicate_policy="test",
        split_seed="test",
        files={},
    )
    sparse = NeuralSparseSpec(
        model_name="official-sparse",
        model_version="1.0.0",
        model_format="TORCH_SCRIPT",
        query_analyzer="bert-uncased",
        package_url="https://artifacts.opensearch.org/fixture.zip",
        package_sha256="c" * 64,
        package_bytes=3,
        model_content_sha256=file_facts(model).sha256,
        model_content_bytes=file_facts(model).bytes,
        tokenizer_content_sha256=file_facts(tokenizer).sha256,
        tokenizer_content_bytes=file_facts(tokenizer).bytes,
        index_name="sparse-index",
        ingest_pipeline="sparse-pipeline",
        text_field="sparse_text",
        embedding_field="sparse_embedding",
        prune_type="max_ratio",
        prune_ratio=0.1,
        inference_batch_size=2,
        bulk_request_size=10,
    )
    provenance: dict[str, Any] = {
        "code_revision": {
            "git_commit": "b" * 40,
            "source_tree_sha256": "c" * 64,
            "source_dirty": False,
        }
    }
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "model": sparse.model_name,
                "model_version": sparse.model_version,
                "model_file": _artifact(root, model),
                "tokenizer_file": _artifact(root, tokenizer),
                "products_sha256": file_facts(products).sha256,
                "runtime": "torchscript_cpu",
                "batch_size": sparse.inference_batch_size,
                "maximum_token_length": 256,
                "pruning": {
                    "type": sparse.prune_type,
                    "ratio": sparse.prune_ratio,
                    "implementation": "local_precomputed_exact",
                },
                "deterministic_replay": wands_sparse_replay_contract(
                    products_path=products,
                    dataset=dataset,
                    spec=sparse,
                ),
                "document_recipe": (
                    "title.brand.category.product_class.features.description"
                ),
                "records": dataset.expected_products,
                "embeddings": {
                    **_artifact(root, embeddings),
                    "file_sha256": file_facts(embeddings).sha256,
                },
                "generation_provenance": provenance,
                "benchmark_provenance": provenance,
                "generation_provenance_eligible_for_decision": True,
                "quality_evidence_eligible_for_decision": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return dataset, sparse, products, embeddings, manifest


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
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


def _dense_model() -> ModelSpec:
    return ModelSpec(
        name="dense-model",
        hf_id="fixture/dense-model",
        revision="d" * 40,
        dims=2,
        normalize=True,
        max_seq_length=32,
        bulk_dtype="float32",
        use_memory_efficient_attention=False,
        trust_remote_code=False,
        query_prefix="query: ",
        document_prefix="passage: ",
        query_template="{query}",
        document_template="{title} {description} {attributes}",
        license="fixture",
        decision_eligible=True,
        contamination="none",
    )


class FakeThreeWayClient:
    def __init__(self, *, index: str, document_count: int) -> None:
        self.index = index
        self.document_count = document_count
        self.definition: dict[str, Any] | None = None
        self.documents: dict[str, dict[str, Any]] = {}
        self.write_block = False
        self.exists_count_fields: list[str] = []

    def wait_until_ready(self, *, expected_version: str) -> str:
        return expected_version

    def delete_index(self, index: str) -> None:
        assert index == self.index

    def create_index(self, index: str, definition: object) -> None:
        assert index == self.index
        self.definition = cast(dict[str, Any], definition)

    def bulk_index(
        self,
        index: str,
        documents: Any,
        *,
        batch_size: int,
    ) -> None:
        del batch_size
        assert index == self.index
        self.documents = {
            document_id: deepcopy(dict(document))
            for document_id, document in documents
        }

    def refresh(self, index: str) -> None:
        assert index == self.index

    def force_merge(self, index: str) -> None:
        assert index == self.index

    def update_index_settings(self, index: str, settings: object) -> None:
        assert index == self.index
        if settings == {"index": {"blocks": {"write": True}}}:
            self.write_block = True

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: object | None = None,
        params: object | None = None,
    ) -> dict[str, object]:
        if path == f"/{self.index}/_count":
            if isinstance(json_body, dict) and "exists" in cast(
                dict[str, Any], json_body.get("query", {})
            ):
                exists = cast(dict[str, Any], json_body["query"])["exists"]
                field = cast(dict[str, str], exists)["field"]
                assert self.definition is not None
                properties = cast(
                    dict[str, Any],
                    cast(dict[str, Any], self.definition["mappings"])[
                        "properties"
                    ],
                )
                if cast(dict[str, Any], properties[field]).get("type") == (
                    "rank_features"
                ):
                    raise AssertionError(
                        "rank_features fields do not support exists queries"
                    )
                self.exists_count_fields.append(field)
            return {"count": self.document_count}
        if path == f"/{self.index}/_mget":
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
        if path == f"/{self.index}/_stats/docs,segments":
            return {
                "_all": {
                    "primaries": {
                        "docs": {"count": self.document_count, "deleted": 0},
                        "segments": {"count": 1},
                    }
                }
            }
        if path == f"/{self.index}/_mapping":
            assert self.definition is not None
            return {self.index: {"mappings": self.definition["mappings"]}}
        if path == f"/{self.index}/_settings":
            assert params == {"flat_settings": "false"}
            assert self.definition is not None
            definition_settings = cast(
                dict[str, Any], self.definition["settings"]
            )
            index_settings = deepcopy(
                cast(dict[str, Any], definition_settings["index"])
            )
            index_settings["refresh_interval"] = "1s"
            if self.write_block:
                index_settings["blocks"] = {"write": "true"}
            index_settings["analysis"] = deepcopy(
                definition_settings["analysis"]
            )
            return {
                self.index: {
                    "settings": {
                        "index": {
                            "uuid": "three-way-uuid",
                            "creation_date": "1",
                            **index_settings,
                        }
                    }
                }
            }
        if path == f"/{self.index}":
            return {
                self.index: {
                    "settings": {"index": {"uuid": "three-way-uuid"}}
                }
            }
        raise AssertionError((method, path))
