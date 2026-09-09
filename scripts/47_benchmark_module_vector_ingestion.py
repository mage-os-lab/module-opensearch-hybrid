from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from poc.benchmark_source_identity import (
    SOURCE_PATHS,
    collect_source_identity,
    load_source_identity,
)
from poc.manifest import write_json
from poc.module_vector_ingestion import (
    BenchmarkConfig,
    assert_decision_environment,
    run_benchmark,
)
from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark numeric-array and Base64 module vector ingestion on OpenSearch 3.8"
    )
    parser.add_argument("--url")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=10_000)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--warmup-documents", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--decision-run", action="store_true")
    parser.add_argument("--opensearch-image", default="")
    parser.add_argument("--topology", choices=("same-host", "remote"), default="remote")
    parser.add_argument("--source-identity", type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    config = BenchmarkConfig(
        document_count=args.documents,
        dimension=args.dimension,
        batch_sizes=tuple(args.batch_sizes),
        trials=args.trials,
        warmup_documents=args.warmup_documents,
        seed=args.seed,
        decision_run=args.decision_run,
    )
    source_identity = (
        load_source_identity(ROOT, SOURCE_PATHS, args.source_identity)
        if args.source_identity is not None
        else collect_source_identity()
    )
    if args.decision_run and source_identity["source_tree_dirty"]:
        parser.error("a decision run requires committed benchmark source bytes")
    try:
        assert_decision_environment(
            decision_run=args.decision_run,
            topology=args.topology,
            machine=platform.machine(),
            system=platform.system(),
            opensearch_image=args.opensearch_image,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.output.exists() and not args.replace:
        parser.error("output already exists; use --replace only for an explicitly disposable run")

    registered_url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_OS_URL",
    )
    if args.decision_run and args.url is not None and args.url.rstrip("/") != registered_url:
        parser.error("a decision run requires the registered OpenSearch endpoint")
    opensearch_url = args.url.rstrip("/") if args.url is not None else registered_url

    contract = json.loads(
        (ROOT / "etc/opensearch_hybrid_contract.json").read_text()
    )
    with OpenSearchClient(opensearch_url, timeout=300) as client:
        result = run_benchmark(
            client,
            config=config,
            contract=contract,
            source_identity=source_identity,
            opensearch_image=args.opensearch_image or "unrecorded",
            topology=args.topology,
        )
    write_json(args.output, result)
    print(
        f"wrote {args.output}; selected_batch_size={result['selected_batch_size']}; "
        f"eligible_for_decision={str(result['eligible_for_decision']).lower()}"
    )
    if args.decision_run and not result["eligible_for_decision"]:
        raise SystemExit(1)
if __name__ == "__main__":
    main()
