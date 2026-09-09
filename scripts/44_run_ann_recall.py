from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from poc.ann_recall import run_wands_ann_recall
from poc.config import load_model_registry
from poc.experiments import load_bm25_experiments
from poc.indexing import load_wands_index_config
from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url
from poc.query_runtime import load_query_runtime_spec
from poc.three_way import load_three_way_spec

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    three_way = load_three_way_spec(ROOT / "config/three_way.toml")
    model = load_model_registry(ROOT / "config/models.toml")[three_way.dense_model]
    runtime = load_query_runtime_spec(ROOT / "config/query_runtime.toml")
    experiments = load_bm25_experiments(ROOT / "config/experiments.toml")
    index = load_wands_index_config(ROOT / "config/indexes.toml")
    url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_OS_URL",
    )
    with OpenSearchClient(url, timeout=300) as client:
        result = run_wands_ann_recall(
            client,
            root=ROOT,
            index=index.name,
            model=model,
            runtime=runtime,
            minimum_recall=experiments.minimum_ann_recall_at_100,
            ef_search_grid=experiments.ann_ef_search_grid,
        )
    held_out = cast(dict[str, Any], result["held_out_test"])
    recall = cast(dict[str, Any], held_out["recall"])
    print(
        f"ANN ef_search={result['selected_ef_search']}; "
        f"held-out mean recall@100={recall['mean_recall@100']:.6f}; "
        f"passes={recall['meets_mean_recall_floor']}"
    )


if __name__ == "__main__":
    main()
