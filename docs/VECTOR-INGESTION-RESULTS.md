# Vector ingestion decision result

## Decision

Keep little-endian float32 Base64 as the production document-vector wire format and keep batch size 64 for the current 256-dimension contract. The registered same-host Linux amd64 decision run passed every gate.

This is a local module observation, not a universal OpenSearch performance claim. It covers deterministic vector serialization, transport, indexing, refresh, and count validation. It does not include encoder inference or the Magento queue lifecycle.

## Registered result

| Measure | Numeric arrays | Base64 | Observed comparison |
| --- | ---: | ---: | ---: |
| Median serialized bytes | 305,421,021 | 107,087,268 | 64.94% smaller |
| Median documents per second | 2,168.24 | 3,005.89 | 1.3863 ratio |
| Median bulk p95 | 29.292 ms | 27.454 ms | 0.9373 ratio |
| Median full-build wall time | 23.060 s | 16.634 s | 27.87% lower |
| Median worker CPU time | 6.251 s | 1.291 s | 79.35% lower |

Each arm completed five 50,000-document trials after a 1,000-document warmup. Every trial finished with 50,000 documents, 50,000 vectors, zero failed bulk items, zero rejected OpenSearch writes, zero direct-transport retries, and zero dead letters. Peak benchmark-worker RSS was 154,140,672 bytes. OpenSearch reported an 8 GiB maximum heap, with after-trial heap use ranging from 875,144,080 to 5,171,267,616 bytes.

## Gate result

| Gate | Required | Result |
| --- | --- | --- |
| Complete and error-free | Exact document/vector counts and no unexplained failures | Pass |
| Payload reduction | At least 60% | Pass at 64.94% |
| Throughput regression | No more than 5% | Pass; Base64 was faster in this run |
| Bulk p95 regression | No more than 5% | Pass; Base64 p95 was lower in this run |

## Environment and provenance

- OpenSearch 3.8.0, Lucene 10.5.0
- immutable image `opensearchproject/opensearch:3.8.0@sha256:bcc1797519726ceb6d651d4a3e60b7c30da91793914a8dfe75fd441d4f641509`
- Debian 13 Linux amd64, AMD Ryzen 7 7700, 16 logical CPUs, 128 GiB host memory
- same-host Docker network topology
- committed benchmark source `e08d05dae2f47099d81a0346fb6951759c6c4e34`
- source verification `external_manifest_file_hashes`
- retained artifact SHA-256 `4bc9ac88aa09ed0a2979b01f87626a176911ae66d31245bd33193c65860e00d4`

The complete retained artifact is [`results/module-opensearch-hybrid/base64-ingestion-decision-e08d05d.json`](https://github.com/rocketweb/module-opensearch-hybrid/blob/e5c511d820a09c2b1b91a1fc1b1d95e7e07884dd/results/module-opensearch-hybrid/base64-ingestion-decision-e08d05d.json).

## Scope limits

The result supports the wire-format decision only. End-to-end production qualification still needs the Magento full-build fixture with the real encoder, queue handoff, retry accounting, inactive-build refresh suppression, validation, and activation gates. Any larger dimension, mapping, shard layout, hardware profile, OpenSearch version, or batch-size change requires a new registered run.
