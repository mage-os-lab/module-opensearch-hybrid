from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from poc.config import WANDS_FINALIST_MODEL_NAMES, load_model_registry
from poc.datasets import load_wands_config
from poc.indexing import build_wands_vector_index, load_wands_index_config
from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = WANDS_FINALIST_MODEL_NAMES


def main() -> None:
    dataset_spec = load_wands_config(ROOT / "config/datasets.toml")
    index_spec = load_wands_index_config(ROOT / "config/indexes.toml")
    registry = load_model_registry(ROOT / "config/models.toml")
    models = tuple(registry[name] for name in MODEL_NAMES)
    url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_OS_URL",
    )
    with OpenSearchClient(url, timeout=300) as client:
        manifest = build_wands_vector_index(
            client,
            dataset_spec=dataset_spec,
            index_spec=index_spec,
            models=models,
            root=ROOT,
            prepared_directory=ROOT / "data/prepared/wands",
            manifest_path=ROOT / "results/wands/index-manifest.json",
        )
    facts = cast(dict[str, Any], manifest["index"])
    print(
        f"indexed WANDS vectors: {facts['document_count']} products, "
        f"{len(models)} models, {facts['segment_count']} segment, "
        f"quality_eligible={manifest['quality_evidence_eligible_for_decision']}"
    )


if __name__ == "__main__":
    main()
