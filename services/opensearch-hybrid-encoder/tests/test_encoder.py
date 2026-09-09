import math
from typing import cast

import pytest

from mageos_opensearch_hybrid_encoder.encoder import (
    DIMENSION,
    MODEL_ID,
    MODEL_REVISION,
    RECIPE_VERSION,
    encode,
)
from mageos_opensearch_hybrid_encoder.server import EncoderRequestError, embed_request


def request(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "request_id": "request-1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dimension": DIMENSION,
        "recipe_version": RECIPE_VERSION,
        "encoder_identity_digest": MODEL_REVISION,
        "query": "red trail shoe",
    }
    payload.update(overrides)
    return payload


def test_encoder_is_deterministic_normalized_and_identity_bound() -> None:
    first = encode("red trail shoe")
    second = encode("red trail shoe")

    assert first == second
    assert len(first) == DIMENSION
    assert math.sqrt(sum(value * value for value in first)) == pytest.approx(1.0)


def test_query_response_echoes_exact_identity() -> None:
    response = embed_request(request(), documents=False)

    assert response["request_id"] == "request-1"
    assert response["model_id"] == MODEL_ID
    assert response["model_revision"] == MODEL_REVISION
    assert response["dimension"] == DIMENSION
    assert response["recipe_version"] == RECIPE_VERSION
    assert response["encoder_identity_digest"] == MODEL_REVISION
    assert response["production_eligible"] is False
    assert len(cast(list[list[float]], response["vectors"])) == 1


def test_document_batch_is_bounded() -> None:
    payload = request(query=None, documents=["document"] * 65)

    with pytest.raises(EncoderRequestError, match="1 to 64"):
        embed_request(payload, documents=True)


def test_model_mismatch_is_rejected() -> None:
    with pytest.raises(EncoderRequestError, match="not loaded"):
        embed_request(request(model_revision="wrong"), documents=False)
