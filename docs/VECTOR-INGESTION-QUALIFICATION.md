# Vector ingestion qualification

This qualification compares the module's numeric JSON array and little-endian float32 Base64 vector wire formats on the exact OpenSearch 3.8 mapping. It measures transport and indexing behavior only. It does not establish search relevance or transfer OpenSearch project benchmark claims to this module.

## Fixed conditions

- 256 float32 dimensions
- deterministic normalized vectors and identical non-vector document fields
- the module's frozen mapping and index settings
- external greater-than-or-equal document versions
- clean isolated indices for every arm and trial
- the immutable OpenSearch 3.8 image digest recorded in the evidence
- alternating arm order across repeated trials
- same-host execution for a decision run

The harness records serialized request bytes, encoding time, documents per second, bulk request p50, p95, and p99, build and validated-readiness time, worker CPU and Linux RSS, final counts, mapping and contract digests, index statistics, node heap and CPU snapshots, rejected writes, source-file hashes, and runtime identity. The benchmark calls OpenSearch bulk transport directly, so it records retry and dead-letter counts as zero alongside `transport_mode=direct_bulk_no_queue`; queue retry behavior remains covered by the module lifecycle fixtures rather than inferred from this transport benchmark.

## Source checkout

Run the commands below from the full [source repository](https://github.com/mage-os-lab/module-opensearch-hybrid/blob/main/docs/DEVELOPMENT.md), which contains the Makefile and benchmark scripts. They are not part of the installed Composer module.

## Development smoke

Start the pinned OpenSearch fixture, then run:

```bash
make module-vector-ingestion-smoke
```

The smoke runs 10,000 documents against batch sizes 16, 32, and 64. It selects the batch size with the best lower throughput across the two arms. The selection is development evidence only and is not eligible for a release decision.

## Decision run

Commit the benchmark source, use the batch size selected by the smoke, and run at least five paired trials over 50,000 documents:

```bash
make module-vector-ingestion-decision MODULE_VECTOR_BENCHMARK_BATCH_SIZE=64
```

The decision command refuses dirty benchmark source bytes, a mutable OpenSearch image reference, remote topology, fewer than 50,000 documents, fewer than five trials, or more than one batch size.

For an isolated container that intentionally has no Git binary, export the source identity from the clean host checkout, mount both the checkout and manifest read-only, and pass `--source-identity` to the benchmark command:

```bash
python3 -m poc.benchmark_source_identity --output /evidence/source-identity.json
```

The container re-hashes the complete registered source set before the run. It rejects an incomplete manifest, a dirty or uncommitted identity, an invalid commit ID, or any byte mismatch. The result records both the source commit and the manifest digest.

The registered gates are:

- complete document and vector counts with zero unexplained failures
- at least 60% smaller serialized payload for the Base64 arm
- no more than 5% median throughput regression
- no more than 5% median bulk p95 latency regression

The evidence file remains ineligible if any gate fails. A payload improvement is useful even when inference or indexing dominates full generation time, but it must be reported as a local observation rather than a universal multiplier.

## Failure and cleanup behavior

Every arm uses a unique module-owned qualification index. The harness deletes the index in a `finally` boundary after success or failure. It never changes a production generation, alias, search pipeline, readiness latch, activation record, or Magento configuration.

Preserve a failed decision artifact for diagnosis. Do not use `--replace` for retained evidence.

The current registered decision and its retained artifact are documented in [`VECTOR-INGESTION-RESULTS.md`](VECTOR-INGESTION-RESULTS.md).
