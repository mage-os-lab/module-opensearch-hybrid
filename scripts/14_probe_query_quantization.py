from __future__ import annotations

import argparse
import importlib
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from poc.bm25 import load_prepared_queries
from poc.config import WANDS_INT8_MODEL_NAMES, ModelSpec, load_model_registry
from poc.manifest import canonical_sha256, write_json
from poc.model_runtime import compare_embedding_vectors, create_bulk_backend
from poc.provenance import collect_code_revision, collect_manifest_provenance
from poc.query_runtime import (
    inspect_int8_onnx_graph,
    load_query_runtime_spec,
    onnx_artifact_directory,
)
from poc.trec import identifier_sort_key

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = WANDS_INT8_MODEL_NAMES
CANDIDATES = (
    ("arm64_pc1_s8", True, True, "all"),
    ("arm64_pc0_s8", False, True, "all"),
    ("arm64_pc1_u8", True, False, "all"),
    ("arm64_pc0_u8", False, False, "all"),
    ("arm64_pc1_s8_weight_matmuls", True, True, "weight"),
    ("arm64_pc1_s8_attention_projections", True, True, "attention"),
    ("arm64_pc1_s8_qkv_projections", True, True, "qkv"),
    ("arm64_pc1_s8_attention_output", True, True, "attention_output"),
    ("arm64_pc1_s8_mlp_projections", True, True, "mlp"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe int8 recipes without relevance labels")
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_provenance = collect_manifest_provenance(ROOT)
    model = load_model_registry(ROOT / "config/models.toml")[args.model]
    runtime = load_query_runtime_spec(ROOT / "config/query_runtime.toml")
    artifact_directory = onnx_artifact_directory(ROOT, model, runtime)
    queries: dict[str, str] = {}
    for split in ("dev", "test"):
        queries.update(
            load_prepared_queries(
                ROOT / f"data/prepared/wands/queries.{split}.jsonl",
                expected_split=split,
            )
        )
    query_ids = sorted(queries, key=identifier_sort_key)
    texts = [model.render_query(queries[query_id]) for query_id in query_ids]
    reference_backend = create_bulk_backend(root=ROOT, model=model, device="cpu")
    reference = reference_backend.encode(texts)
    transformers = importlib.import_module("transformers")
    optimum = importlib.import_module("optimum.onnxruntime")
    optimum_configuration = importlib.import_module("optimum.onnxruntime.configuration")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        artifact_directory,
        local_files_only=True,
    )
    quantizer = optimum.ORTQuantizer.from_pretrained(
        artifact_directory / "onnx",
        file_name="model.onnx",
    )
    weight_matmul_nodes = _weight_matmul_nodes(artifact_directory / "onnx/model.onnx")
    node_groups = {
        "all": None,
        "weight": weight_matmul_nodes,
        "attention": [name for name in weight_matmul_nodes if "/attn/W" in name],
        "qkv": [name for name in weight_matmul_nodes if "/attn/Wqkv/" in name],
        "attention_output": [name for name in weight_matmul_nodes if "/attn/Wo/" in name],
        "mlp": [name for name in weight_matmul_nodes if "/mlp/W" in name],
    }
    results: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix=f"opensearch-hybrid-{model.name}-quant-") as temporary:
        temporary_root = Path(temporary)
        for name, per_channel, symmetric_weights, node_group in CANDIDATES:
            started = time.perf_counter()
            selected_nodes = node_groups[node_group]
            configuration = optimum_configuration.AutoQuantizationConfig.arm64(
                is_static=False,
                per_channel=per_channel,
                use_symmetric_weights=symmetric_weights,
                nodes_to_quantize=selected_nodes,
            )
            candidate_directory = temporary_root / name
            quantizer.quantize(
                configuration,
                candidate_directory,
                file_suffix=name,
            )
            candidate_paths = list(candidate_directory.glob("*.onnx"))
            if len(candidate_paths) != 1:
                raise ValueError(
                    f"quantizer produced {len(candidate_paths)} ONNX files for {name}"
                )
            candidate_path = candidate_paths[0]
            vectors = encode_onnx(
                candidate_path,
                tokenizer=tokenizer,
                model=model,
                texts=texts,
            )
            comparison = compare_embedding_vectors(reference, vectors)
            results[name] = {
                "per_channel": per_channel,
                "symmetric_weights": symmetric_weights,
                "node_group": node_group,
                "quantized_node_count": (
                    len(selected_nodes) if selected_nodes is not None else None
                ),
                "comparison": comparison,
                "graph": inspect_int8_onnx_graph(candidate_path),
                "seconds_including_quantization": time.perf_counter() - started,
            }
            print(
                f"{model.name} {name}: min="
                f"{comparison['minimum_cosine_similarity']:.9f}, "
                f"mean={comparison['mean_cosine_similarity']:.9f}, "
                f"max_abs={comparison['maximum_absolute_delta']:.9f}"
            )
    write_json(
        ROOT / f"results/wands/query-runtime/{model.name}.quantization-probe.json",
        {
            "schema_version": 2,
            "model": asdict(model),
            "runtime": asdict(runtime),
            "decision_scope": "diagnostic_quantization_recipe_probe",
            "eligible_for_decision": False,
            "benchmark_provenance": benchmark_provenance,
            "completion_code_revision": asdict(collect_code_revision(ROOT)),
            "query_ids_sha256": canonical_sha256(query_ids),
            "sample_size": len(texts),
            "selection_surface": "embedding_parity_only_no_relevance_labels",
            "candidates": results,
        },
    )


def encode_onnx(
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
    input_names = {item.name for item in session.get_inputs()}
    batches: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=model.max_seq_length,
            return_tensors="np",
        )
        inputs = {name: np.asarray(value) for name, value in tokens.items() if name in input_names}
        hidden_state = np.asarray(session.run(None, inputs)[0], dtype=np.float32)
        vectors = hidden_state[:, 0, : model.dims]
        if model.normalize:
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        batches.append(vectors)
    return np.concatenate(batches)


def _weight_matmul_nodes(path: Path) -> list[str]:
    onnx = importlib.import_module("onnx")
    graph = onnx.load(str(path), load_external_data=False).graph
    initializers = {initializer.name for initializer in graph.initializer}
    nodes = [
        node.name
        for node in graph.node
        if node.op_type == "MatMul" and any(name in initializers for name in node.input)
    ]
    if not nodes or any(not name for name in nodes):
        raise ValueError("could not identify named weight MatMul nodes")
    return nodes


if __name__ == "__main__":
    main()
