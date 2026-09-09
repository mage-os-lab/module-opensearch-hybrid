from __future__ import annotations

from pathlib import Path

from poc.datasets import (
    load_trec_product_search_config,
    verify_trec_product_search_raw,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    spec = load_trec_product_search_config(ROOT / "config/datasets.toml")
    facts = verify_trec_product_search_raw(
        spec,
        ROOT / "data/raw/trec-product-search-2024",
    )
    print(
        "verified TREC Product Search raw sources: "
        + ", ".join(f"{name}={value.bytes} bytes" for name, value in facts.items())
    )


if __name__ == "__main__":
    main()
