from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from poc.datasets import load_wands_config
from poc.wands import prepare_wands

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    spec = load_wands_config(ROOT / "config/datasets.toml")
    manifest = prepare_wands(
        spec,
        ROOT / "data/raw/wands",
        ROOT / "data/prepared/wands",
    )
    raw_counts = cast(dict[str, Any], manifest["raw_counts"])
    annotations = cast(dict[str, Any], manifest["annotations"])
    split = cast(dict[str, Any], manifest["split"])
    print(
        f"prepared WANDS: {raw_counts['products']} products, {raw_counts['queries']} queries, "
        f"{annotations['unique_pairs']} unique qrels"
    )
    print(
        f"split: {split['dev_queries']} dev / {split['test_queries']} test; "
        f"conflicts consolidated: {annotations['conflicting_pairs']}"
    )


if __name__ == "__main__":
    main()
