from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from poc.bm25 import load_prepared_queries
from poc.config import WANDS_INT8_MODEL_NAMES, load_model_registry
from poc.query_runtime import (
    create_int8_query_backend,
    load_query_runtime_spec,
    verify_onnx_int8_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = WANDS_INT8_MODEL_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify dynamic-int8 ONNX query models")
    parser.add_argument("--model", action="append", choices=MODEL_NAMES, dest="models")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = load_model_registry(ROOT / "config/models.toml")
    runtime = load_query_runtime_spec(ROOT / "config/query_runtime.toml")
    queries = load_prepared_queries(
        ROOT / "data/prepared/wands/queries.dev.jsonl",
        expected_split="dev",
    )
    sample_texts = [queries[query_id] for query_id in list(queries)[:32]]
    for name in tuple(args.models or MODEL_NAMES):
        model = registry[name]
        manifest = verify_onnx_int8_artifact(root=ROOT, model=model, runtime=runtime)
        backend = create_int8_query_backend(root=ROOT, model=model, runtime=runtime)
        vectors = np.concatenate(
            [backend.encode([model.render_query(text)]) for text in sample_texts]
        )
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError(f"ONNX query vectors are not normalized for {name}")
        if backend.model_artifact_sha256 != manifest["artifact_sha256"]:
            raise ValueError(f"ONNX backend loaded an unexpected artifact for {name}")
        print(
            f"verified {name}: {len(vectors)} query vectors, "
            f"runtime={backend.runtime}"
        )


if __name__ == "__main__":
    main()
