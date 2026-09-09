from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import onnx
from onnx import TensorProto, helper


def registration_body_for_version(
    source: dict[str, Any],
    opensearch_version: str,
) -> dict[str, Any]:
    try:
        major = int(opensearch_version.split(".", maxsplit=1)[0])
    except ValueError as exc:
        raise ValueError("OpenSearch version must begin with a major number") from exc
    registration = deepcopy(source)
    if major < 3:
        model_config = cast(dict[str, Any], registration["model_config"])
        if model_config.get("pooling_mode") == "none":
            model_config.pop("pooling_mode")
    return registration


def wrap_prepooled_output_for_cls(model: onnx.ModelProto) -> onnx.ModelProto:
    wrapped = deepcopy(model)
    if len(wrapped.graph.output) != 1:
        raise ValueError("CLS compatibility wrapper requires exactly one model output")
    output = wrapped.graph.output[0]
    tensor_type = output.type.tensor_type
    dimensions = tensor_type.shape.dim
    if len(dimensions) != 2:
        raise ValueError("CLS compatibility wrapper requires a rank-2 pooled output")
    shape: list[str | int] = []
    for dimension in dimensions:
        if dimension.dim_param:
            shape.append(dimension.dim_param)
        else:
            shape.append(dimension.dim_value)
    wrapped_name = f"{output.name}_with_token_axis"
    axes_name = "opensearch_hybrid_cls_compat_axes"
    wrapped.graph.initializer.append(
        helper.make_tensor(axes_name, TensorProto.INT64, [1], [1])
    )
    wrapped.graph.node.append(
        helper.make_node(
            "Unsqueeze",
            [output.name, axes_name],
            [wrapped_name],
            name="opensearch_hybrid_cls_compat_unsqueeze",
        )
    )
    output.CopyFrom(
        helper.make_tensor_value_info(
            wrapped_name,
            tensor_type.elem_type,
            [shape[0], 1, shape[1]],
        )
    )
    onnx.checker.check_model(wrapped)
    return wrapped
