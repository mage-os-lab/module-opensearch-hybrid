from __future__ import annotations

import importlib
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from poc.bm25 import load_prepared_queries
from poc.config import ModelSpec, load_model_registry
from poc.manifest import write_json
from poc.model_runtime import compare_embedding_vectors, create_bulk_backend
from poc.provenance import collect_code_revision, collect_manifest_provenance
from poc.query_runtime import (
    inspect_int8_onnx_graph,
    load_query_runtime_spec,
    onnx_artifact_directory,
)
from poc.trec import identifier_sort_key

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "granite_embedding_english_r2"


class CalibrationReader:
    def __init__(self, batches: list[dict[str, np.ndarray]]) -> None:
        self._iterator = iter(batches)

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._iterator, None)


def main() -> None:
    benchmark_provenance = collect_manifest_provenance(ROOT)
    model = load_model_registry(ROOT / "config/models.toml")[MODEL_NAME]
    runtime = load_query_runtime_spec(ROOT / "config/query_runtime.toml")
    artifact_directory = onnx_artifact_directory(ROOT, model, runtime)
    dev_queries = load_prepared_queries(
        ROOT / "data/prepared/wands/queries.dev.jsonl",
        expected_split="dev",
    )
    test_queries = load_prepared_queries(
        ROOT / "data/prepared/wands/queries.test.jsonl",
        expected_split="test",
    )
    queries = {**dev_queries, **test_queries}
    query_ids = sorted(queries, key=identifier_sort_key)
    texts = [model.render_query(queries[query_id]) for query_id in query_ids]
    transformers = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        artifact_directory,
        local_files_only=True,
    )
    calibration_batches = _tokenize_batches(
        tokenizer,
        model,
        [model.render_query(dev_queries[query_id]) for query_id in sorted(dev_queries)],
    )
    reference_backend = create_bulk_backend(root=ROOT, model=model, device="cpu")
    reference = reference_backend.encode(texts)
    quantization = importlib.import_module("onnxruntime.quantization")
    shape_inference = importlib.import_module("onnxruntime.quantization.shape_inference")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="opensearch-hybrid-static-int8-") as temporary:
        temporary_root = Path(temporary)
        preprocessed_path = temporary_root / "model.preprocessed.onnx"
        quantized_path = temporary_root / "model.static-qdq-s8s8.onnx"
        shape_inference.quant_pre_process(
            artifact_directory / "onnx/model.onnx",
            preprocessed_path,
            auto_merge=True,
        )
        quantization.quantize_static(
            preprocessed_path,
            quantized_path,
            CalibrationReader(calibration_batches),
            quant_format=quantization.QuantFormat.QDQ,
            activation_type=quantization.QuantType.QInt8,
            weight_type=quantization.QuantType.QInt8,
            per_channel=True,
            op_types_to_quantize=["MatMul"],
        )
        vectors = _encode_onnx(
            quantized_path,
            tokenizer=tokenizer,
            model=model,
            texts=texts,
        )
        comparison = compare_embedding_vectors(reference, vectors)
        graph = inspect_int8_onnx_graph(quantized_path)
    result = {
        "schema_version": 2,
        "model": asdict(model),
        "runtime": asdict(runtime),
        "decision_scope": "diagnostic_static_quantization_probe",
        "eligible_for_decision": False,
        "benchmark_provenance": benchmark_provenance,
        "completion_code_revision": asdict(collect_code_revision(ROOT)),
        "calibration_split": "dev",
        "calibration_queries": len(dev_queries),
        "evaluation_queries": len(texts),
        "selection_surface": "embedding_parity_only_no_relevance_labels",
        "recipe": "static_qdq_s8s8_per_channel_matmul",
        "comparison": comparison,
        "graph": graph,
        "seconds_including_preprocess_and_quantization": time.perf_counter() - started,
    }
    write_json(
        ROOT / f"results/wands/query-runtime/{model.name}.static-quantization-probe.json",
        result,
    )
    print(
        f"{model.name} static QDQ S8S8: min="
        f"{comparison['minimum_cosine_similarity']:.9f}, "
        f"mean={comparison['mean_cosine_similarity']:.9f}, "
        f"max_abs={comparison['maximum_absolute_delta']:.9f}"
    )


def _tokenize_batches(
    tokenizer: Any,
    model: ModelSpec,
    texts: list[str],
    batch_size: int = 8,
) -> list[dict[str, np.ndarray]]:
    return [
        {
            name: np.asarray(value)
            for name, value in tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=model.max_seq_length,
                return_tensors="np",
            ).items()
            if name in {"input_ids", "attention_mask"}
        }
        for start in range(0, len(texts), batch_size)
    ]


def _encode_onnx(
    path: Path,
    *,
    tokenizer: Any,
    model: ModelSpec,
    texts: list[str],
    batch_size: int = 32,
) -> np.ndarray:
    onnxruntime = importlib.import_module("onnxruntime")
    session = onnxruntime.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    batches: list[np.ndarray] = []
    for inputs in _tokenize_batches(tokenizer, model, texts, batch_size):
        hidden_state = np.asarray(session.run(None, inputs)[0], dtype=np.float32)
        vectors = hidden_state[:, 0, : model.dims]
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        batches.append(vectors)
    return np.concatenate(batches)


if __name__ == "__main__":
    main()
