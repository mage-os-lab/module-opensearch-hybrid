from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    Path("Makefile"),
    Path("poc/benchmark_source_identity.py"),
    Path("poc/module_vector_ingestion.py"),
    Path("scripts/47_benchmark_module_vector_ingestion.py"),
    Path("tests/test_module_vector_ingestion_benchmark.py"),
    Path("etc/opensearch_hybrid_contract.json"),
)


def collect_source_identity(
    root: Path = ROOT,
    source_paths: tuple[Path, ...] = SOURCE_PATHS,
) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *map(str, source_paths)],
        cwd=root,
        check=False,
    )
    untracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            *map(str, source_paths),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "git_commit": commit,
        "source_tree_dirty": diff.returncode != 0 or bool(untracked),
        "untracked_source_paths": untracked,
        "file_sha256": source_hashes(root, source_paths),
        "verification_method": "git_worktree",
    }


def load_source_identity(
    root: Path,
    source_paths: tuple[Path, ...],
    manifest_path: Path,
) -> dict[str, Any]:
    loaded = json.loads(manifest_path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError("the source identity manifest must be a JSON object")

    return verify_source_identity(root, source_paths, loaded)


def verify_source_identity(
    root: Path,
    source_paths: tuple[Path, ...],
    identity: dict[str, Any],
) -> dict[str, Any]:
    commit = identity.get("git_commit")
    dirty = identity.get("source_tree_dirty")
    untracked = identity.get("untracked_source_paths")
    if (
        not isinstance(commit, str)
        or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit) is None
        or dirty is not False
        or untracked != []
    ):
        raise ValueError("the source identity must describe committed source bytes")
    recorded_hashes = identity.get("file_sha256")
    if not isinstance(recorded_hashes, dict):
        raise ValueError("the source identity must contain file hashes")
    expected_paths = {str(path) for path in source_paths}
    if set(recorded_hashes) != expected_paths:
        raise ValueError("the source identity file set does not match the benchmark source set")
    actual_hashes = source_hashes(root, source_paths)
    for path, actual_hash in actual_hashes.items():
        recorded_hash = recorded_hashes.get(path)
        if not isinstance(recorded_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", recorded_hash):
            raise ValueError(f"the recorded source hash is invalid for {path}")
        if not hmac.compare_digest(recorded_hash, actual_hash):
            raise ValueError(f"the source hash does not match for {path}")
    manifest_sha256 = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return {
        **identity,
        "verification_method": "external_manifest_file_hashes",
        "manifest_sha256": manifest_sha256,
    }


def source_hashes(root: Path, source_paths: tuple[Path, ...]) -> dict[str, str]:
    return {
        str(path): hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in source_paths
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Attest committed module vector benchmark source bytes without runtime dependencies"
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identity = collect_source_identity()
    if identity["source_tree_dirty"]:
        parser.error("source identity export requires committed benchmark source bytes")
    if args.output.exists():
        parser.error("source identity output already exists")
    args.output.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}; git_commit={identity['git_commit']}")


if __name__ == "__main__":
    main()
