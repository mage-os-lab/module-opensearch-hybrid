from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

from poc.config import WANDS_INT8_MODEL_NAMES, load_model_registry
from poc.provenance import collect_manifest_provenance
from poc.query_runtime import build_onnx_int8_artifact, load_query_runtime_spec

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = WANDS_INT8_MODEL_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pinned dynamic-int8 ONNX query models")
    parser.add_argument("--model", action="append", choices=MODEL_NAMES, dest="models")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_provenance = collect_manifest_provenance(ROOT)
    registry = load_model_registry(ROOT / "config/models.toml")
    runtime = load_query_runtime_spec(ROOT / "config/query_runtime.toml")
    for name in tuple(args.models or MODEL_NAMES):
        manifest = build_onnx_int8_artifact(
            root=ROOT,
            model=registry[name],
            runtime=runtime,
            generation_provenance=generation_provenance,
        )
        quantized = cast(dict[str, Any], manifest["quantized_file"])
        print(
            f"built {name} {runtime.precision}/{runtime.quantization_config}: "
            f"{quantized['bytes']} bytes, sha256={quantized['sha256']}"
        )


if __name__ == "__main__":
    main()
