from __future__ import annotations

import argparse
from pathlib import Path

from poc.config import WANDS_FINALIST_MODEL_NAMES, load_model_registry
from poc.embedding_cache import verify_embedding_cache

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = WANDS_FINALIST_MODEL_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify WANDS embedding caches")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        choices=DEFAULT_MODELS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = load_model_registry(ROOT / "config/models.toml")
    names = tuple(args.models or DEFAULT_MODELS)
    products_path = ROOT / "data/prepared/wands/products.jsonl"
    for name in names:
        model = registry[name]
        verify_embedding_cache(root=ROOT, products_path=products_path, model=model)
        print(f"verified embedding cache: {name}, {model.dims} dimensions")


if __name__ == "__main__":
    main()
