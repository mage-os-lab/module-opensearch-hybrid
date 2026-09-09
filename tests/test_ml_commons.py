from __future__ import annotations

from typing import Any, cast

import onnx
from onnx import TensorProto, helper

from poc.ml_commons import (
    registration_body_for_version,
    wrap_prepooled_output_for_cls,
)


def test_opensearch_219_registration_omits_unsupported_none_pooling() -> None:
    source = {
        "name": "model",
        "model_config": {
            "pooling_mode": "none",
            "normalize_result": False,
            "embedding_dimension": 256,
        },
    }

    registration = registration_body_for_version(source, "2.19.6")

    assert registration["model_config"] == {
        "normalize_result": False,
        "embedding_dimension": 256,
    }
    assert cast(dict[str, Any], source["model_config"])["pooling_mode"] == "none"


def test_opensearch_3_registration_preserves_none_pooling() -> None:
    source = {"model_config": {"pooling_mode": "none"}}

    assert registration_body_for_version(source, "3.8.0") == source


def test_cls_compatibility_wrapper_adds_singleton_token_axis() -> None:
    inputs = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, ["batch", 256]
    )
    output = helper.make_tensor_value_info(
        "sentence_embedding", TensorProto.FLOAT, ["batch", 256]
    )
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["sentence_embedding"])],
        "already-pooled",
        [inputs],
        [output],
    )
    model = helper.make_model(graph)

    wrapped = wrap_prepooled_output_for_cls(model)

    onnx.checker.check_model(wrapped)
    dimensions = wrapped.graph.output[0].type.tensor_type.shape.dim
    assert [dimension.dim_value or dimension.dim_param for dimension in dimensions] == [
        "batch",
        1,
        256,
    ]
    assert wrapped.graph.node[-1].op_type == "Unsqueeze"
