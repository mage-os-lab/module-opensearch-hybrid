from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from poc.datasets import load_trec_product_search_config
from poc.experiments import load_bm25_experiments
from poc.trec_product_search import prepare_trec_product_search

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    spec = load_trec_product_search_config(ROOT / "config/datasets.toml")
    experiments = load_bm25_experiments(ROOT / "config/experiments.toml")
    result = prepare_trec_product_search(
        spec,
        ROOT / "data/raw/trec-product-search-2024",
        ROOT / "data/prepared/trec-product-search-2024",
        ROOT / "results/trec-product-search/preparation.json",
        maximum_orphan_rate=experiments.maximum_trec_orphan_rate,
    )
    counts = cast(dict[str, Any], result["counts"])
    print(
        f"prepared TREC Product Search: {counts['products']} products, "
        f"{counts['queries']} queries, "
        f"orphan rate={float(cast(Any, result['orphan_rate'])):.6f}"
    )


if __name__ == "__main__":
    main()
