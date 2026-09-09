from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

from poc.datasets import DatasetIntegrityError
from poc.trec import identifier_sort_key


def select_self_retrieval_ids(document_ids: list[str], *, sample_size: int) -> list[str]:
    if sample_size <= 0 or sample_size > len(document_ids):
        raise ValueError("self-retrieval sample size is outside the document set")
    if len(set(document_ids)) != len(document_ids):
        raise DatasetIntegrityError("self-retrieval document IDs are not unique")
    # Keep the frozen sampling salt so the project rename does not change benchmark queries.
    return sorted(
        document_ids,
        key=lambda document_id: (
            hashlib.sha256(
                f"rocket-search-self-retrieval:{document_id}".encode()
            ).digest(),
            identifier_sort_key(document_id),
        ),
    )[:sample_size]


def top1_self_retrieval(
    *,
    query_document_ids: list[str],
    query_vectors: NDArray[np.float32],
    document_ids: list[str],
    document_vectors: NDArray[np.float32],
    batch_size: int = 32,
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("self-retrieval batch size must be positive")
    if query_vectors.shape != (len(query_document_ids), document_vectors.shape[1]):
        raise DatasetIntegrityError("self-retrieval query vector shape differs")
    if document_vectors.shape[0] != len(document_ids) or document_vectors.ndim != 2:
        raise DatasetIntegrityError("self-retrieval document vector shape differs")
    if not np.isfinite(query_vectors).all() or not np.isfinite(document_vectors).all():
        raise DatasetIntegrityError("self-retrieval vectors contain non-finite values")
    if len(set(document_ids)) != len(document_ids):
        raise DatasetIntegrityError("self-retrieval document IDs are not unique")
    missing = sorted(set(query_document_ids) - set(document_ids), key=identifier_sort_key)
    if missing:
        raise DatasetIntegrityError(
            f"self-retrieval queries reference missing documents: {missing}"
        )

    order = sorted(
        range(len(document_ids)),
        key=lambda index: identifier_sort_key(document_ids[index]),
    )
    ordered_ids = [document_ids[index] for index in order]
    ordered_vectors = document_vectors[order]
    failures: list[dict[str, str]] = []
    for start in range(0, len(query_document_ids), batch_size):
        batch = query_vectors[start : start + batch_size]
        similarities = batch @ ordered_vectors.T
        top_indexes = np.argmax(similarities, axis=1)
        for offset, top_index in enumerate(top_indexes):
            expected = query_document_ids[start + offset]
            actual = ordered_ids[int(top_index)]
            if actual != expected:
                failures.append({"expected": expected, "actual": actual})
    hits = len(query_document_ids) - len(failures)
    return {
        "queries": len(query_document_ids),
        "top1_hits": hits,
        "top1_rate": hits / len(query_document_ids) if query_document_ids else 0.0,
        "failures": failures,
    }
