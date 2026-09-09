from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from poc.config import WANDS_INT8_MODEL_NAMES, load_model_registry
from poc.int8_dense import (
    build_query_runtime_quality_artifact,
    verify_query_runtime_quality_artifact,
)
from poc.manifest import read_json, write_json
from poc.provenance import collect_manifest_provenance

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    benchmark_provenance = collect_manifest_provenance(ROOT)
    registry = load_model_registry(ROOT / "config/models.toml")
    fp32_summary = cast(dict[str, Any], read_json(ROOT / "results/wands/dense-summary.json"))
    int8_summary = cast(
        dict[str, Any], read_json(ROOT / "results/wands/int8-dense-summary.json")
    )
    fp32_models = cast(dict[str, dict[str, dict[str, Any]]], fp32_summary["models"])
    int8_models = cast(dict[str, dict[str, Any]], int8_summary["models"])

    for model_name in WANDS_INT8_MODEL_NAMES:
        fp32 = fp32_models[model_name]["dev"]
        int8 = cast(dict[str, Any], int8_models[model_name]["splits"])["dev"]
        model = registry[model_name]
        result = build_query_runtime_quality_artifact(
            root=ROOT,
            model=model,
            fp32_entry=fp32,
            int8_entry=int8,
            benchmark_provenance=benchmark_provenance,
        )
        output = ROOT / f"results/wands/query-runtime/{model_name}.quality-guard.json"
        write_json(output, result)
        verify_query_runtime_quality_artifact(
            root=ROOT,
            model=model,
            fp32_entry=fp32,
            int8_entry=int8,
        )
        stability = cast(dict[str, Any], result["ranking_stability"])
        print(
            f"{model_name}: {result['status']}; "
            f"NDCG@10 loss={result['ndcg@10_loss']:.6f}, "
            f"mean top-10 overlap={float(stability['mean_overlap']):.4f}"
        )


if __name__ == "__main__":
    main()
