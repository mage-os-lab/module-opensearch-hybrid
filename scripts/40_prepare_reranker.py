from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url
from poc.reranker import (
    load_reranker_spec,
    load_wands_reranker_evidence_context,
    prepare_wands_reranker,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    spec = load_reranker_spec(ROOT / "config/reranker.toml")
    url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_OS_URL",
    )
    with OpenSearchClient(url, timeout=300) as client:
        evidence_context = load_wands_reranker_evidence_context(
            root=ROOT,
            client=client,
        )
        manifest = prepare_wands_reranker(
            root=ROOT,
            spec=spec,
            evidence_context=evidence_context,
            local_files_only=args.local_files_only,
        )
    latency = cast(dict[str, Any], manifest["standalone_rerank50_latency"])
    print(
        f"prepared {manifest['unique_pairs']} reranker pair scores; "
        f"rerank50 p95={latency['p95_ms']:.3f} ms"
    )


if __name__ == "__main__":
    main()
