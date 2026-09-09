from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from poc.bm25 import (
    verify_wands_bm25_selection,
)
from poc.datasets import load_wands_config
from poc.experiments import BM25Profile, load_bm25_experiments
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
        selection = verify_wands_bm25_selection(
            client,
            root=ROOT,
            index=index_spec.name,
            experiments=experiments,
        )
        selected = BM25Profile.from_mapping(
            cast(dict[str, Any], selection["selected_profile"])
        )
        artifacts = cast(dict[str, dict[str, Any]], selection["artifacts"])

    print(
        f"verified BM25 selection {selected.name}: {len(artifacts)} artifacts, "
        "all registered dev and held-out arms reran byte-identically"
    )


if __name__ == "__main__":
    main()
