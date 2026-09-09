from __future__ import annotations

from pathlib import Path

from poc.datasets import load_wands_config, verify_wands_raw

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    spec = load_wands_config(ROOT / "config/datasets.toml")
    facts = verify_wands_raw(spec, ROOT / "data/raw/wands")
    print(
        f"verified WANDS {spec.revision}: {sum(item.bytes for item in facts.values())} bytes, "
        f"{len(facts)} pinned files"
    )


if __name__ == "__main__":
    main()
