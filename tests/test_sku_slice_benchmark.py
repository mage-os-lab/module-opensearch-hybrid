from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

import poc.sku_slice_benchmark as benchmark_module
from poc.datasets import DatasetIntegrityError
from poc.neural_sparse import NeuralSparseSpec
from poc.sku_slice import SkuSliceSpec
from poc.sku_slice_benchmark import (
    derive_sku_benchmark_eligibility,
    run_sku_slice_benchmark,
    verify_sku_slice_summary,
)


class FakeCrossCheck:
    def to_dict(self) -> dict[str, str]:
        return {"status": "passed"}


class FakeClient:
    def __init__(self, index: str) -> None:
        self.index = index
        self.pipelines: dict[str, dict[str, Any]] = {}

    def put_search_pipeline(
        self, pipeline_id: str, definition: dict[str, Any]
    ) -> None:
        self.pipelines[pipeline_id] = definition

    def search(
        self,
        index: str,
        body: dict[str, Any],
        *,
        pipeline: str | None = None,
    ) -> dict[str, Any]:
        assert index == self.index
        if pipeline is not None:
            assert pipeline in self.pipelines
        encoded = json.dumps(body)
        exact_title = "Trail Shoe" in encoded
        document_id = "2" if exact_title else "1"
        title = "Trail Shoe" if exact_title else "SKU Product"
        return {
            "hits": {
                "hits": [
                    {
                        "_id": document_id,
                        "_score": 1.0,
                        "_source": {"title": title},
                    }
                ]
            }
        }

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: object | None = None,
        params: dict[str, object] | None = None,
    ) -> object:
        del json_body, params
        assert method == "GET"
        if path == "/":
            return {"version": {"number": "3.8.0"}}
        if path == f"/{self.index}":
            return {
                self.index: {
                    "settings": {"index": {"uuid": "fixture-sku-index"}}
                }
            }
        if path == f"/{self.index}/_stats/docs,segments":
            return {
                "_all": {
                    "primaries": {
                        "docs": {"count": 2, "deleted": 0},
                        "segments": {"count": 1},
                    }
                }
            }
        if path.startswith("/_search/pipeline/"):
            pipeline_id = path.rsplit("/", 1)[-1]
            return {pipeline_id: self.pipelines[pipeline_id]}
        raise AssertionError(path)


def test_sku_benchmark_eligibility_is_a_strict_conjunction() -> None:
    assert derive_sku_benchmark_eligibility(
        provenance_valid=True,
        start_upstream_eligible=True,
        completion_upstream_eligible=True,
        upstream_unchanged=True,
    ) == (True, [])
    eligible, reasons = derive_sku_benchmark_eligibility(
        provenance_valid=True,
        start_upstream_eligible="true",
        completion_upstream_eligible=True,
        upstream_unchanged=True,
    )
    assert eligible is False
    assert reasons == [
        "synthetic SKU start upstream evidence is not decision eligible"
    ]


def test_sku_summary_recomputes_live_gate_and_rejects_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sparse_spec = _sparse_spec()
    sku_spec = _sku_spec()
    prepared = tmp_path / "data/prepared/wands"
    prepared.mkdir(parents=True)
    (prepared / "queries.sku-slice.jsonl").write_text(
        json.dumps(
            {
                "query_id": "sku-001",
                "query": "RW-1",
                "kind": "synthetic_sku",
                "expected_document_id": "1",
            }
        )
        + "\n"
        + json.dumps(
            {
                "query_id": "title-001",
                "query": "Trail Shoe",
                "kind": "exact_title",
                "expected_document_id": "2",
            }
        )
        + "\n"
    )
    (prepared / "qrels.sku-slice.trec").write_text(
        "sku-001 0 1 2\ntitle-001 0 2 2\n"
    )
    evidence_directory = tmp_path / "results/wands/sku-slice"
    evidence_directory.mkdir(parents=True)
    (evidence_directory / "index-manifest.json").write_text(
        '{"schema_version":2}\n'
    )
    (evidence_directory / "preparation.json").write_text(
        '{"schema_version":1}\n'
    )
    upstream = {
        "index_manifest": {"sha256": "a" * 64},
        "eligible_for_decision": True,
    }
    provenance = {"code_revision": {"git_commit": "b" * 40}}
    monkeypatch.setattr(
        benchmark_module,
        "verify_registered_sku_benchmark_inputs",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        benchmark_module,
        "_sku_upstream_evidence",
        lambda *args, **kwargs: upstream,
    )
    monkeypatch.setattr(
        benchmark_module,
        "collect_manifest_provenance",
        lambda *args, **kwargs: provenance,
    )
    monkeypatch.setattr(
        benchmark_module,
        "wands_completion_provenance_evidence",
        lambda *args, **kwargs: (True, provenance["code_revision"]),
    )
    monkeypatch.setattr(
        benchmark_module,
        "wands_recorded_provenance_valid",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        benchmark_module,
        "verify_decision_provenance",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        benchmark_module,
        "cross_check_run",
        lambda *args, **kwargs: FakeCrossCheck(),
    )
    client = cast(Any, FakeClient(sku_spec.index_name))

    summary = run_sku_slice_benchmark(
        client,
        root=tmp_path,
        sparse_spec=sparse_spec,
        sku_spec=sku_spec,
    )

    assert summary["schema_version"] == 2
    assert summary["quality_evidence_eligible_for_decision"] is True
    assert summary["passes_registered_synthetic_gate"] is True
    assert verify_sku_slice_summary(
        client,
        root=tmp_path,
        sparse_spec=sparse_spec,
        sku_spec=sku_spec,
    ) == summary

    upstream_checks = iter(
        (
            upstream,
            {
                **upstream,
                "index_manifest": {"sha256": "c" * 64},
            },
        )
    )
    monkeypatch.setattr(
        benchmark_module,
        "_sku_upstream_evidence",
        lambda *args, **kwargs: next(upstream_checks),
    )
    with pytest.raises(DatasetIntegrityError, match="changed during decision replay"):
        verify_sku_slice_summary(
            client,
            root=tmp_path,
            sparse_spec=sparse_spec,
            sku_spec=sku_spec,
        )
    monkeypatch.setattr(
        benchmark_module,
        "_sku_upstream_evidence",
        lambda *args, **kwargs: upstream,
    )

    summary_path = tmp_path / "results/wands/sku-slice/summary.json"
    tampered = json.loads(summary_path.read_text())
    tampered["quality_evidence_eligible_for_decision"] = "true"
    summary_path.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n")
    with pytest.raises(DatasetIntegrityError, match="summary routing"):
        verify_sku_slice_summary(
            client,
            root=tmp_path,
            sparse_spec=sparse_spec,
            sku_spec=sku_spec,
        )


def _sparse_spec() -> NeuralSparseSpec:
    return NeuralSparseSpec(
        model_name="fixture-sparse",
        model_version="1.0.0",
        model_format="TORCH_SCRIPT",
        query_analyzer="bert-uncased",
        package_url="https://artifacts.opensearch.org/fixture.zip",
        package_sha256="a" * 64,
        package_bytes=1,
        model_content_sha256="b" * 64,
        model_content_bytes=1,
        tokenizer_content_sha256="c" * 64,
        tokenizer_content_bytes=1,
        index_name="unused",
        ingest_pipeline="unused",
        text_field="sparse_text",
        embedding_field="sparse_embedding",
        prune_type="max_ratio",
        prune_ratio=0.1,
        inference_batch_size=1,
        bulk_request_size=1,
    )


def _sku_spec() -> SkuSliceSpec:
    return SkuSliceSpec(
        dataset="WANDS synthetic SKU and known-item proxy",
        seed="fixture",
        sku_prefix="RW",
        sku_queries=1,
        exact_title_queries=1,
        lexical_weight=0.3,
        maximum_mrr_regression=0.01,
        alpha=0.05,
        index_name="fixture-sku-index",
    )
