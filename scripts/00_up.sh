#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_root/.uv-cache}"

docker compose -f docker-compose.yml -f compose.bench.yml up -d --wait opensearch
uv run --frozen python scripts/00_verify.py
uv run --frozen python scripts/00_verify_benchmark_profile.py
