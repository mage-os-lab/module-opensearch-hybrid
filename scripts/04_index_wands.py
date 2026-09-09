from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from poc.datasets import load_wands_config
from poc.indexing import build_wands_lexical_index, load_wands_index_config
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
        manifest = build_wands_lexical_index(
            client,
            dataset_spec=dataset_spec,
            index_spec=index_spec,
            prepared_directory=ROOT / "data/prepared/wands",
            manifest_path=ROOT / "results/wands/index-manifest.json",
        )
    index = cast(dict[str, Any], manifest["index"])
    print(
        f"indexed WANDS: {index['document_count']} products, {index['segment_count']} segment, "
        f"uuid={index['uuid']}, "
        f"quality_eligible={manifest['quality_evidence_eligible_for_decision']}"
    )


if __name__ == "__main__":
    main()
