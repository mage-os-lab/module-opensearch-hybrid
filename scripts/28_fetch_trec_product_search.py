from __future__ import annotations

import argparse
from pathlib import Path

from poc.datasets import fetch_trec_product_search, load_trec_product_search_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch the pinned TREC Product Search 2024 sources"
    )
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args()
    spec = load_trec_product_search_config(ROOT / "config/datasets.toml")
    statuses = fetch_trec_product_search(
        spec,
        ROOT / "data/raw/trec-product-search-2024",
        repair=args.repair,
    )
    print(
        "TREC Product Search sources verified: "
        + ", ".join(f"{name}={status}" for name, status in statuses.items())
    )


if __name__ == "__main__":
    main()
