from __future__ import annotations

import argparse
from pathlib import Path

from poc.datasets import fetch_wands, load_wands_config, verify_wands_raw

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch immutable WANDS source files")
    parser.add_argument("--repair", action="store_true", help="replace an invalid existing file")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    spec = load_wands_config(root / "config/datasets.toml")
    destination = root / "data/raw/wands"
    statuses = fetch_wands(spec, destination, repair=args.repair)
    facts = verify_wands_raw(spec, destination)
    for name in ("product", "query", "label"):
        print(f"{name}: {statuses[name]}, {facts[name].bytes} bytes, sha256={facts[name].sha256}")


if __name__ == "__main__":
    main()
