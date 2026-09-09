from __future__ import annotations

import numpy as np

from poc.query_correctness import select_self_retrieval_ids, top1_self_retrieval


def test_self_retrieval_uses_exact_similarity_and_product_id_tie_break() -> None:
    result = top1_self_retrieval(
        query_document_ids=["2", "1"],
        query_vectors=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        document_ids=["2", "3", "1"],
        document_vectors=np.asarray(
            [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        ),
    )

    assert result["top1_hits"] == 2
    assert result["top1_rate"] == 1.0
    assert result["failures"] == []


def test_self_retrieval_sample_is_deterministic_and_order_independent() -> None:
    first = select_self_retrieval_ids(["1", "2", "3", "4"], sample_size=3)
    second = select_self_retrieval_ids(["4", "2", "1", "3"], sample_size=3)

    assert first == second
    assert len(first) == 3
