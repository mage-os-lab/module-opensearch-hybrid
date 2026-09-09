from __future__ import annotations

import argparse
from pathlib import Path

from poc.module_package import (
    build_module_package,
    load_manifest,
    verify_archive,
    verify_module_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify the reproducible Mage-OS module package"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build a package ZIP and canonical manifest")
    build.add_argument("--module-root", type=Path, required=True)
    build.add_argument("--archive", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="verify a module root and optional ZIP")
    verify.add_argument("--module-root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--archive", type=Path)

    args = parser.parse_args()
    if args.command == "build":
        manifest = build_module_package(args.module_root, args.archive, args.manifest)
        print(
            f"wrote {args.archive} and {args.manifest}; "
            f"files={manifest['payload']['file_count']}; "
            f"payload_sha256={manifest['payload']['sha256']}; "
            f"archive_sha256={manifest['archive']['sha256']}"
        )
        return

    manifest = load_manifest(args.manifest)
    verify_module_payload(args.module_root, manifest)
    if args.archive is not None:
        verify_archive(args.archive, manifest)
    print(
        f"verified {args.module_root}; files={manifest['payload']['file_count']}; "
        f"payload_sha256={manifest['payload']['sha256']}"
    )


if __name__ == "__main__":
    main()
