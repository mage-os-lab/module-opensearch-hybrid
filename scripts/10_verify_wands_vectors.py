from __future__ import annotations

from pathlib import Path
from typing import cast

from poc.datasets import load_wands_config
from poc.indexing import load_wands_index_config, verify_wands_lexical_index
from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    dataset_spec = load_wands_config(ROOT / "config/datasets.toml")
    index_spec = load_wands_index_config(ROOT / "config/indexes.toml")
    url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_OS_URL",
    )
    with OpenSearchClient(url, timeout=120) as client:
        summary = verify_wands_lexical_index(
            client,
            dataset_spec=dataset_spec,
            index_spec=index_spec,
            prepared_directory=ROOT / "data/prepared/wands",
            manifest_path=ROOT / "results/wands/index-manifest.json",
        )
    models = cast(list[str], summary["vector_models"])
    print(
        f"verified WANDS vector index {summary['uuid']}: {summary['documents']} products, "
        f"models={','.join(models)}, "
        f"quality_eligible={summary['quality_evidence_eligible_for_decision']}"
    )


if __name__ == "__main__":
    main()
