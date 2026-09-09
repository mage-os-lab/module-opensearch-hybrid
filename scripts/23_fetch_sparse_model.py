from __future__ import annotations

import argparse
from pathlib import Path

from poc.model_packages import (
    PinnedZipMember,
    PinnedZipPackage,
    fetch_and_extract_pinned_zip,
)
from poc.neural_sparse import load_neural_sparse_spec

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_RELATIVE_PATH = Path("data/cache/models/neural-sparse/doc-v3-distill.zip")
MODEL_RELATIVE_PATH = Path(
    "data/cache/models/neural-sparse/doc-v3-distill/"
    "opensearch-neural-sparse-encoding-doc-v3-distill.pt"
)
TOKENIZER_RELATIVE_PATH = Path(
    "data/cache/models/neural-sparse/doc-v3-distill/tokenizer.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and extract the pinned WANDS neural-sparse model package"
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="replace invalid existing package or extracted files",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    spec = load_neural_sparse_spec(root / "config/neural_sparse.toml")
    statuses = fetch_and_extract_pinned_zip(
        PinnedZipPackage(
            url=spec.package_url,
            sha256=spec.package_sha256,
            bytes=spec.package_bytes,
        ),
        package_path=root / PACKAGE_RELATIVE_PATH,
        members=(
            PinnedZipMember(
                name=MODEL_RELATIVE_PATH.name,
                destination=root / MODEL_RELATIVE_PATH,
                sha256=spec.model_content_sha256,
                bytes=spec.model_content_bytes,
            ),
            PinnedZipMember(
                name=TOKENIZER_RELATIVE_PATH.name,
                destination=root / TOKENIZER_RELATIVE_PATH,
                sha256=spec.tokenizer_content_sha256,
                bytes=spec.tokenizer_content_bytes,
            ),
        ),
        repair=args.repair,
    )
    print(
        "WANDS neural-sparse model package verified: "
        + ", ".join(f"{name}={status}" for name, status in statuses.items())
    )


if __name__ == "__main__":
    main()
