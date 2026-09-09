from __future__ import annotations

from pathlib import Path

from poc.sku_slice import load_sku_slice_spec, prepare_sku_slice

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    spec = load_sku_slice_spec(ROOT / "config/sku_slice.toml")
    manifest = prepare_sku_slice(
        root=ROOT,
        spec=spec,
        products_path=ROOT / "data/prepared/wands/products.jsonl",
        query_path=ROOT / "data/prepared/wands/queries.sku-slice.jsonl",
        qrels_path=ROOT / "data/prepared/wands/qrels.sku-slice.trec",
        manifest_path=ROOT / "results/wands/sku-slice/preparation.json",
    )
    print(f"prepared {manifest['counts']} synthetic SKU and known-item queries")


if __name__ == "__main__":
    main()
