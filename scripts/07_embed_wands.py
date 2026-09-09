from __future__ import annotations

import argparse
from pathlib import Path

from poc.config import WANDS_FINALIST_MODEL_NAMES, load_model_registry
from poc.embedding_cache import build_embedding_cache
from poc.model_runtime import (
    create_bulk_backend,
    release_device_memory,
    select_bulk_device,
)
from poc.provenance import collect_manifest_provenance

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = WANDS_FINALIST_MODEL_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build revision-pinned WANDS embedding caches")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        choices=DEFAULT_MODELS,
    )
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_provenance = collect_manifest_provenance(ROOT)
    registry = load_model_registry(ROOT / "config/models.toml")
    names = tuple(args.models or DEFAULT_MODELS)
    device = select_bulk_device(args.device)
    products_path = ROOT / "data/prepared/wands/products.jsonl"
    print(f"embedding {len(names)} WANDS model(s) on {device}", flush=True)
    for name in names:
        model = registry[name]
        print(
            f"loading {name} at {model.revision} with max_seq_length={model.max_seq_length}",
            flush=True,
        )
        backend = create_bulk_backend(root=ROOT, model=model, device=device)
        summary = build_embedding_cache(
            root=ROOT,
            products_path=products_path,
            model=model,
            backend=backend,
            batch_size=args.batch_size,
            shard_size=args.shard_size,
            generation_provenance=generation_provenance,
        )
        print(
            f"{name}: {summary.document_count} documents, {summary.encoded_vectors} encoded, "
            f"{summary.cache_hits} cache hits, {summary.shard_count} shards",
            flush=True,
        )
        del backend
        release_device_memory()


if __name__ == "__main__":
    main()
