from __future__ import annotations

import argparse
from pathlib import Path

from poc.model_packages import (
    PinnedZipMember,
    PinnedZipPackage,
    fetch_and_extract_pinned_zip,
)
from poc.trec_sparse import load_trec_sparse_config

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_RELATIVE_PATH = Path("data/cache/models/neural-sparse/doc-v2-distill.zip")
MODEL_RELATIVE_PATH = Path(
    "data/cache/models/neural-sparse/doc-v2-distill/"
    "opensearch-neural-sparse-encoding-doc-v2-distill.pt"
)
TOKENIZER_RELATIVE_PATH = Path(
    "data/cache/models/neural-sparse/doc-v2-distill/tokenizer.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and extract the pinned TREC neural-sparse document model"
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="replace invalid existing package or extracted files",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    spec = load_trec_sparse_config(root / "config/trec_neural_sparse.toml")
    statuses = fetch_and_extract_pinned_zip(
        PinnedZipPackage(
            url=spec.document_model_url,
            sha256=spec.document_model_sha256,
            bytes=spec.document_model_bytes,
        ),
        package_path=root / PACKAGE_RELATIVE_PATH,
        members=(
            PinnedZipMember(
                name=MODEL_RELATIVE_PATH.name,
                destination=root / MODEL_RELATIVE_PATH,
            ),
            PinnedZipMember(
                name=TOKENIZER_RELATIVE_PATH.name,
                destination=root / TOKENIZER_RELATIVE_PATH,
            ),
        ),
        repair=args.repair,
    )
    print(
        "TREC neural-sparse document model package verified: "
        + ", ".join(f"{name}={status}" for name, status in statuses.items())
    )


if __name__ == "__main__":
    main()
