from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from poc.config import load_model_registry
from poc.model_runtime import create_bulk_backend

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    import onnx
    import onnxruntime  # type: ignore[import-untyped]
    import torch
    from onnxruntime.quantization import (  # type: ignore[import-untyped]
        QuantType,
        quantize_dynamic,
    )

    model = load_model_registry(ROOT / "config/models.toml")["arctic_embed_m_v2"]
    backend: Any = create_bulk_backend(root=ROOT, model=model, device="cpu")

    class QueryEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = backend._encoder

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            hidden_state = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                unpad_inputs=False,
                return_dict=False,
            )[0]
            return torch.nn.functional.normalize(
                hidden_state[:, 0, : model.dims],
                p=2,
                dim=1,
            )

    texts = [
        model.render_query("trail running shoe"),
        model.render_query("solid walnut writing desk"),
    ]
    tokens = backend._tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=model.max_seq_length,
        return_tensors="pt",
    )
    wrapper = QueryEncoder().eval()
    reference = wrapper(tokens["input_ids"], tokens["attention_mask"]).detach().numpy()
    with tempfile.TemporaryDirectory(prefix="opensearch-hybrid-arctic-onnx-") as temporary:
        path = Path(temporary) / "model.onnx"
        torch.onnx.export(
            wrapper,
            (tokens["input_ids"], tokens["attention_mask"]),
            str(path),
            input_names=["input_ids", "attention_mask"],
            output_names=["sentence_embedding"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "sentence_embedding": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
        exported = onnx.load(str(path), load_external_data=True)
        onnx.checker.check_model(exported)
        session = onnxruntime.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        candidate = session.run(
            ["sentence_embedding"],
            {
                "input_ids": tokens["input_ids"].numpy().astype(np.int64),
                "attention_mask": tokens["attention_mask"].numpy().astype(np.int64),
            },
        )[0]
        similarities = np.sum(reference * candidate, axis=1) / (
            np.linalg.norm(reference, axis=1) * np.linalg.norm(candidate, axis=1)
        )
        print(
            f"Arctic custom ONNX fp32: bytes={path.stat().st_size}, "
            f"min_cosine={float(np.min(similarities)):.9f}, "
            f"max_delta={float(np.max(np.abs(reference - candidate))):.9f}"
        )
        quantized_path = Path(temporary) / "model_qint8_arm64.onnx"
        quantize_dynamic(
            model_input=str(path),
            model_output=str(quantized_path),
            weight_type=QuantType.QInt8,
            per_channel=False,
        )
        quantized_session = onnxruntime.InferenceSession(
            str(quantized_path),
            providers=["CPUExecutionProvider"],
        )
        quantized = quantized_session.run(
            ["sentence_embedding"],
            {
                "input_ids": tokens["input_ids"].numpy().astype(np.int64),
                "attention_mask": tokens["attention_mask"].numpy().astype(np.int64),
            },
        )[0]
        quantized_similarities = np.sum(reference * quantized, axis=1) / (
            np.linalg.norm(reference, axis=1) * np.linalg.norm(quantized, axis=1)
        )
        print(
            f"Arctic custom ONNX int8: bytes={quantized_path.stat().st_size}, "
            f"min_cosine={float(np.min(quantized_similarities)):.9f}, "
            f"max_delta={float(np.max(np.abs(reference - quantized))):.9f}"
        )


if __name__ == "__main__":
    main()
