from __future__ import annotations

from pathlib import Path

import numpy as np

from poc.config import ModelSpec
from poc.experiments import BM25Profile, load_bm25_experiments
from poc.rrf import (
    execute_exact_rrf_run,
    expected_rrf_artifact_keys,
    rrf_pipeline_id,
    select_rrf_rank_constant,
)

ROOT = Path(__file__).resolve().parents[1]


def _model() -> ModelSpec:
    return ModelSpec.from_mapping(
        "test_model",
        {
            "hf_id": "example/test",
            "revision": "1" * 40,
            "dims": 2,
            "normalize": True,
            "max_seq_length": 512,
            "bulk_dtype": "float32",
            "use_memory_efficient_attention": False,
            "trust_remote_code": False,
            "query_prefix": "",
            "document_prefix": "",
            "query_template": "{query}",
            "document_template": "{title}",
            "license": "Apache-2.0",
            "decision_eligible": True,
            "contamination": "none_known",
        },
    )


def test_rrf_selection_uses_registered_order_as_tie_break() -> None:
    constants = (1, 5, 10, 20, 60)
    metrics = {
        1: {"ndcg@10": 0.72},
        5: {"ndcg@10": 0.74},
        10: {"ndcg@10": 0.75},
        20: {"ndcg@10": 0.75},
        60: {"ndcg@10": 0.73},
    }

    assert select_rrf_rank_constant(constants, metrics, metric="ndcg@10") == 10


def test_rrf_pipeline_id_includes_model_and_rank_constant() -> None:
    pipeline_id = rrf_pipeline_id(_model(), 20)
    assert pipeline_id.startswith(
        "opensearch-hybrid-wands-test-model-rrf-k020-v1-sha256-"
    )


def test_rrf_artifact_keys_are_exactly_the_registered_grid_and_test_winner() -> None:
    experiments = load_bm25_experiments(ROOT / "config/experiments.toml")

    assert expected_rrf_artifact_keys(experiments) == {
        "dev/k001",
        "dev/k005",
        "dev/k010",
        "dev/k020",
        "dev/k060",
        "test/selected",
    }


def test_exact_rrf_run_uses_frozen_bm25_clause_and_registered_pipeline() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.pipelines: list[tuple[str, dict[str, object]]] = []
            self.requests: list[tuple[str, dict[str, object], str | None]] = []

        def put_search_pipeline(self, pipeline: str, definition: dict[str, object]) -> None:
            self.pipelines.append((pipeline, definition))

        def request(self, method: str, path: str) -> dict[str, object]:
            assert method == "GET"
            pipeline_id = path.rsplit("/", 1)[-1]
            return {
                pipeline_id: next(
                    definition
                    for current_id, definition in reversed(self.pipelines)
                    if current_id == pipeline_id
                )
            }

        def search(
            self,
            index: str,
            request: dict[str, object],
            *,
            pipeline: str | None = None,
        ) -> dict[str, object]:
            self.requests.append((index, request, pipeline))
            return {
                "hits": {
                    "hits": [
                        {"_id": "20", "_score": 0.9},
                        {"_id": "10", "_score": 0.8},
                    ]
                }
            }

    profile = BM25Profile.from_mapping(
        {
            "name": "frozen",
            "fields": ["title^8", "description"],
            "query_type": "best_fields",
            "operator": "or",
            "tie_breaker": 0.1,
        }
    )
    client = FakeClient()
    model = _model()

    records = execute_exact_rrf_run(
        client,  # type: ignore[arg-type]
        index="products",
        queries={"1": "coffee table"},
        query_vectors={"1": np.asarray([1.0, 0.0], dtype=np.float32)},
        profile=profile,
        model=model,
        rank_constant=20,
        tag="test-rrf",
    )

    expected_pipeline = rrf_pipeline_id(model, 20)
    assert client.pipelines[0][0] == expected_pipeline
    assert client.requests[0][2] == expected_pipeline
    request = client.requests[0][1]
    assert request["query"]["hybrid"]["queries"][0] == profile.query("coffee table")  # type: ignore[index]
    assert [(record.document_id, record.score) for record in records] == [
        ("20", 100.0),
        ("10", 99.0),
    ]
