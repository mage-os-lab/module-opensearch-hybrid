from __future__ import annotations

from pathlib import Path

from poc.datasets import load_wands_config
from poc.wands import verify_prepared_wands

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    spec = load_wands_config(ROOT / "config/datasets.toml")
    raw = ROOT / "data/raw/wands"
    prepared = ROOT / "data/prepared/wands"
    summary = verify_prepared_wands(spec, raw, prepared)

    print(
        f"verified prepared WANDS: {summary['products']} products, {summary['queries']} queries, "
        f"{summary['qrels']} qrels, byte-identical rebuild"
    )


if __name__ == "__main__":
    main()
