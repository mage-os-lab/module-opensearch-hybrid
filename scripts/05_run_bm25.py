from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from poc.bm25 import run_wands_bm25_benchmark
from poc.datasets import load_wands_config
from poc.experiments import load_bm25_experiments
from poc.indexing import load_wands_index_config, verify_wands_lexical_index
from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    dataset_spec = load_wands_config(ROOT / "config/datasets.toml")
    index_spec = load_wands_index_config(ROOT / "config/indexes.toml")
    experiments = load_bm25_experiments(ROOT / "config/experiments.toml")
    url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_OS_URL",
    )
    with OpenSearchClient(url, timeout=120) as client:
        verify_wands_lexical_index(
            client,
            dataset_spec=dataset_spec,
            index_spec=index_spec,
            prepared_directory=ROOT / "data/prepared/wands",
            manifest_path=ROOT / "results/wands/index-manifest.json",
        )
        selection = run_wands_bm25_benchmark(
            client,
            root=ROOT,
            index=index_spec.name,
            experiments=experiments,
        )
    selected = cast(dict[str, Any], selection["selected_profile"])
    held_out = cast(dict[str, Any], selection["held_out_test_metrics"])
    tuned = cast(dict[str, float], held_out["tuned"])
    competent = cast(dict[str, float], held_out["competent_integrator"])
    print(
        f"selected BM25 profile on dev: {selected['name']} "
        f"from {selection['bm25_tuning_budget']} candidates"
    )
    print(
        f"held-out tuned BM25: NDCG@10={tuned['ndcg@10']:.6f}, "
        f"MRR exact@10={tuned['mrr_exact@10']:.6f}, judged@10={tuned['judged@10']:.6f}"
    )
    print(
        "held-out competent-integrator BM25: "
        f"NDCG@10={competent['ndcg@10']:.6f}, "
        f"MRR exact@10={competent['mrr_exact@10']:.6f}, "
        f"judged@10={competent['judged@10']:.6f}"
    )


if __name__ == "__main__":
    main()
