from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from poc.datasets import DatasetIntegrityError
from poc.experiments import load_bm25_experiments
from poc.trec_sparse import (
    build_trec_sparse_index_definition,
    build_trec_sparse_query,
    extract_sparse_prediction,
    iter_sparse_checkpoint_lines,
    iter_trec_precomputed_sparse_documents,
    load_trec_sparse_config,
    prune_sparse_values,
    select_registered_model_identity,
)
from poc.trec_sparse_benchmark import (
    verify_registered_trec_significance_thresholds,
    verify_registered_trec_sparse_inputs,
    verify_trec_query_model_identity,
    verify_trec_sparse_summary_fields,
)

ROOT = Path(__file__).resolve().parents[1]


def test_trec_sparse_configuration_pins_official_2_19_model_pair() -> None:
    spec = load_trec_sparse_config(ROOT / "config/trec_neural_sparse.toml")

    assert spec.document_model_name.endswith("encoding-doc-v2-distill")
    assert spec.document_model_version == "1.0.0"
    assert spec.document_model_sha256 == (
        "86bab435d031edb2a6d921fd9ac317a7541d5d95666f642b606e7d0ebfb84358"
    )
    assert spec.query_tokenizer_name.endswith("tokenizer-v1")
    assert spec.query_tokenizer_version == "1.0.1"
    assert spec.query_tokenizer_content_sha256 == (
        "b3487da9c58ac90541b720f3b367084f271d280c7f3bdc3e6d9c9a269fb31950"
    )
    assert spec.query_tokenizer_content_bytes == 567691
    assert spec.expected_opensearch_version == "2.19.6"


def test_trec_sparse_query_uses_deployed_tokenizer_not_3_x_analyzer() -> None:
    spec = load_trec_sparse_config(ROOT / "config/trec_neural_sparse.toml")
    definition = build_trec_sparse_index_definition(spec)

    assert "default_pipeline" not in definition["settings"]["index"]
    assert definition["mappings"]["properties"]["sparse_embedding"] == {
        "type": "rank_features"
    }
    assert build_trec_sparse_query(spec, "running shoe", model_id="tokenizer-1") == {
        "neural_sparse": {
            "sparse_embedding": {
                "query_text": "running shoe",
                "model_id": "tokenizer-1",
            }
        }
    }


def test_sparse_max_ratio_pruning_matches_declared_threshold() -> None:
    assert prune_sparse_values(
        {"one": 10.0, "two": 1.0, "three": 0.999, "zero": 0.0},
        maximum_value_ratio=0.1,
    ) == {"one": 10.0, "two": 1.0}


def test_trec_sparse_documents_require_exact_corpus_embedding_order(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "collection.trec.gz"
    with gzip.open(corpus, "wt", encoding="utf-8") as handle:
        handle.write("doc-1\tTitle\tDescription\n")
    embeddings = tmp_path / "embeddings.jsonl"
    embeddings.write_text(
        json.dumps({"product_id": "wrong", "embedding": {"title": 1.0}}) + "\n"
    )
    spec = load_trec_sparse_config(ROOT / "config/trec_neural_sparse.toml")

    with pytest.raises(ValueError, match="order differs"):
        list(iter_trec_precomputed_sparse_documents(corpus, embeddings, spec))


def test_sparse_checkpoint_reader_preserves_exact_lines_and_rejects_truncation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "partial.jsonl"
    first = json.dumps(
        {"product_id": "doc-1", "embedding": {"shoe": 1.25}},
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"
    second = json.dumps(
        {"product_id": "doc-2", "embedding": {"blue": 0.75}},
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"
    checkpoint.write_text(first + second)

    assert list(iter_sparse_checkpoint_lines(checkpoint)) == [
        ("doc-1", first.encode()),
        ("doc-2", second.encode()),
    ]

    checkpoint.write_text(first + second.rstrip("\n"))
    with pytest.raises(ValueError, match="truncated"):
        list(iter_sparse_checkpoint_lines(checkpoint))


def test_trec_sparse_tokenizer_model_selection_ignores_chunks() -> None:
    hits = [
        {
            "_id": "chunk",
            "_source": {
                "name": "tokenizer",
                "model_version": "1.0.1",
                "chunk_number": 0,
            },
        },
        {
            "_id": "head",
            "_source": {
                "name": "tokenizer",
                "model_version": "1.0.1",
                "model_state": "DEPLOYED",
            },
        },
    ]

    selected = select_registered_model_identity(
        hits, name="tokenizer", version="1.0.1"
    )

    assert selected is not None
    assert selected["_id"] == "head"


def test_trec_sparse_prediction_requires_non_empty_token_map() -> None:
    response = {
        "inference_results": [
            {"output": [{"dataAsMap": {"response": [{"shoe": 1.25}]}}]}
        ]
    }

    assert extract_sparse_prediction(response) == {"shoe": 1.25}
    with pytest.raises(ValueError, match="token map"):
        extract_sparse_prediction({"inference_results": []})


def test_trec_sparse_summary_fields_reject_tampered_deltas() -> None:
    sparse_metrics = {"ndcg@10": 0.6}
    hybrid_metrics = {"ndcg@10": 0.7}
    baseline_metrics = {
        "tuned": {"ndcg@10": 0.5},
        "competent_integrator": {"ndcg@10": 0.45},
    }
    summary = {
        "quality_evidence_eligible_for_decision": True,
        "latency_evidence_eligible_for_decision": False,
        "latency_ineligibility_reason": "commodity x86 concurrency-4 run not recorded",
        "sparse_only_metrics": sparse_metrics,
        "hybrid_metrics": hybrid_metrics,
        "tuned_bm25_ndcg@10": 0.5,
        "sparse_minus_tuned_bm25_ndcg@10": 0.6 - 0.5,
        "hybrid_minus_tuned_bm25_ndcg@10": 0.7 - 0.5,
        "competent_integrator_bm25_ndcg@10": 0.45,
        "hybrid_minus_competent_integrator_bm25_ndcg@10": 0.7 - 0.45,
    }

    verify_trec_sparse_summary_fields(
        summary,
        sparse_metrics=sparse_metrics,
        hybrid_metrics=hybrid_metrics,
        baseline_metrics=baseline_metrics,
        quality_eligible=True,
    )

    summary["hybrid_minus_tuned_bm25_ndcg@10"] = 0.3
    with pytest.raises(DatasetIntegrityError, match="summary metrics"):
        verify_trec_sparse_summary_fields(
            summary,
            sparse_metrics=sparse_metrics,
            hybrid_metrics=hybrid_metrics,
            baseline_metrics=baseline_metrics,
            quality_eligible=True,
        )


def test_trec_query_model_identity_requires_the_registered_live_model() -> None:
    spec = load_trec_sparse_config(ROOT / "config/trec_neural_sparse.toml")
    live_model = {
        "algorithm": "SPARSE_TOKENIZE",
        "model_content_hash_value": spec.query_tokenizer_content_sha256,
        "model_content_size_in_bytes": spec.query_tokenizer_content_bytes,
        "model_format": spec.query_tokenizer_format,
        "model_state": "DEPLOYED",
        "name": spec.query_tokenizer_name,
    }
    artifact = {
        "status": "passed",
        "quality_evidence_eligible_for_decision": True,
        "model_id": "model-1",
        "registered_spec": {
            "name": spec.query_tokenizer_name,
            "version": spec.query_tokenizer_version,
            "format": spec.query_tokenizer_format,
        },
        "registered_model": live_model,
    }

    assert (
        verify_trec_query_model_identity(
            artifact,
            live_model=live_model,
            selected_model_id="model-1",
            spec=spec,
        )
        == "model-1"
    )

    live_model["model_state"] = "UNDEPLOYED"
    with pytest.raises(DatasetIntegrityError, match="query model identity"):
        verify_trec_query_model_identity(
            artifact,
            live_model=live_model,
            selected_model_id="model-1",
            spec=spec,
        )


def test_trec_sparse_requires_registered_configuration_and_thresholds() -> None:
    spec = load_trec_sparse_config(ROOT / "config/trec_neural_sparse.toml")
    experiments = load_bm25_experiments(ROOT / "config/experiments.toml")

    verify_registered_trec_sparse_inputs(
        root=ROOT,
        spec=spec,
        experiments=experiments,
    )
    with pytest.raises(DatasetIntegrityError, match="significance thresholds"):
        verify_registered_trec_significance_thresholds(
            root=ROOT,
            metric=experiments.primary_metric,
            alpha=1.0,
            minimum_delta=experiments.minimum_paired_delta,
        )
