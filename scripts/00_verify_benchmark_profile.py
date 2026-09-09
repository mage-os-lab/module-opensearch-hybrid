from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from poc.manifest import write_json
from poc.provenance import (
    BenchmarkProfile,
    benchmark_environment_snapshots_match,
    collect_live_benchmark_environment,
    load_benchmark_profile,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = ROOT / "results/environment/benchmark-profile.json"


def persist_benchmark_environment(
    output_path: Path,
    environment: Mapping[str, Any],
    *,
    profile: BenchmarkProfile,
) -> bool:
    """Write changed live facts, preserving an equivalent artifact byte-for-byte."""

    try:
        existing = json.loads(output_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        existing = None
    if isinstance(existing, Mapping) and benchmark_environment_snapshots_match(
        existing,
        environment,
        profile=profile,
    ):
        return False
    write_json(output_path, dict(environment))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a live benchmark container profile")
    parser.add_argument("--profile", type=Path, default=ROOT / "config/benchmark.toml")
    parser.add_argument("--project")
    parser.add_argument("--expected-version")
    parser.add_argument("--url")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    profile = load_benchmark_profile(args.profile)
    supplied_values = {
        "Compose project": (args.project, profile.compose_project),
        "OpenSearch version": (args.expected_version, profile.opensearch_version),
        "OpenSearch URL": (args.url, profile.opensearch_url),
    }
    for label, (supplied, registered) in supplied_values.items():
        if supplied is not None and supplied != registered:
            parser.error(
                f"{label} override {supplied!r} differs from registered value {registered!r}"
            )

    environment = collect_live_benchmark_environment(ROOT, profile)
    changed = persist_benchmark_environment(
        args.output,
        environment,
        profile=profile,
    )
    action = "wrote" if changed else "preserved"
    print(f"Verified benchmark profile {profile.profile_id}; {action} {args.output}")


if __name__ == "__main__":
    main()
