from __future__ import annotations

import numpy as np

from poc.config import ModelSpec
from poc.dense import (
    encode_queries,
    execute_exact_dense_run,
    query_vectors_sha256,
    trec_score_for_rank,
)


class FakeBackend:
    device = "cpu"
    runtime = "fake"
    model_artifact_sha256 = "a" * 64

    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.texts.extend(texts)
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


def test_dense_query_encoder_applies_registered_query_template_and_prefix() -> None:
    model = ModelSpec.from_mapping(
        "test",
        {
            "hf_id": "example/test",
            "revision": "1" * 40,
            "dims": 2,
            "normalize": True,
            "max_seq_length": 512,
            "bulk_dtype": "float32",
            "use_memory_efficient_attention": False,
            "trust_remote_code": False,
            "query_prefix": "query: ",
            "document_prefix": "",
            "query_template": "{query}",
            "document_template": "{title}. {description}. {attributes}",
            "license": "Apache-2.0",
            "decision_eligible": True,
            "contamination": "none_known",
        },
    )
    backend = FakeBackend()

    vectors = encode_queries(backend, model=model, queries={"2": "chair", "1": "table"})

    assert backend.texts == ["query: table", "query: chair"]
    assert list(vectors) == ["1", "2"]
    assert vectors["1"].tolist() == [1.0, 0.0]


def test_dense_trec_score_is_deterministically_derived_from_engine_rank() -> None:
    assert trec_score_for_rank(1) == 100.0
    assert trec_score_for_rank(29) == 72.0
    assert trec_score_for_rank(100) == 1.0


def test_dense_run_canonicalizes_sub_micro_score_ties_by_product_id() -> None:
    class FakeClient:
        def search(self, index: str, body: dict[str, object]) -> dict[str, object]:
            assert body["size"] == 200
            return {
                "hits": {
                    "hits": [
                        {"_id": "20", "_score": 0.9000004},
                        {"_id": "10", "_score": 0.9000003},
                    ]
                }
            }

    model = ModelSpec.from_mapping(
        "test",
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

    records = execute_exact_dense_run(
        FakeClient(),  # type: ignore[arg-type]
        index="test",
        model=model,
        query_vectors={"1": np.asarray([1.0, 0.0], dtype=np.float32)},
        tag="test",
        score_tie_decimal_places=6,
        candidate_depth=200,
    )

    assert [record.document_id for record in records] == ["10", "20"]


def test_query_vector_hash_is_order_independent_and_byte_sensitive() -> None:
    first = {
        "2": np.asarray([0.0, 1.0], dtype=np.float32),
        "1": np.asarray([1.0, 0.0], dtype=np.float32),
    }
    reordered = {"1": first["1"], "2": first["2"]}
    changed = {"1": first["1"], "2": np.asarray([0.0, 0.9], dtype=np.float32)}

    assert query_vectors_sha256(first) == query_vectors_sha256(reordered)
    assert query_vectors_sha256(first) != query_vectors_sha256(changed)
