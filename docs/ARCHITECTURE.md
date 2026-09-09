# Architecture and retrieval contract

The Magento module owns catalog integration, search routing, generation state, and
operator controls. OpenSearch owns the native index, module generation indices,
and score fusion. A separate encoder produces query and document vectors.

## Query path

1. The wrapper checks the store's configuration, active generation, readiness,
   accepted result contract, request name, sort, and page size.
2. Native OpenSearch supplies the top 100 relevance candidates, their lexical
   scores, totals, and aggregations. Exact SKU and exact-title safeguards retain
   native ordering.
3. The authenticated encoder produces a query vector using the generation's sealed
   model identity. OpenSearch combines dense scores with the native candidate
   scores using the frozen normalization and weights.
4. The adapter checks the result contract and returns the requested page. Crossing
   pages append the matching native tail; deep pages use the same native sort.

Hybrid reranking cannot recover a product outside the native candidate set.
Unsupported requests and failures use the native adapter. Circuit-open requests
bypass the failing hybrid path.

The machine-readable contract is [opensearch_hybrid_contract.json](../etc/opensearch_hybrid_contract.json).
It binds the model revision, 256-dimensional cosine vectors, source recipe,
mapping, pipeline, fusion weights, and result behavior. Merchant search attribute
weights remain native catalog-search configuration. Changing a bound contract
requires a validated replacement generation and explicit acceptance.

## Catalog changes and generations

Product saves and deletes record changes inside the product transaction. After
commit, Mage-OS Async Events hands work to separate correctness and embedding
queues. Native indexing and hybrid processing acknowledge the same journal
revisions; readiness stays closed while either side is behind.

Correctness work updates price, inventory, salability, and deletion state without
waiting behind embedding work. An unchanged semantic source hash permits vector
reuse. Semantic changes invalidate stale vectors and enqueue encoding. Active and
building generations receive changes concurrently, with revision-aware coalescing
and external OpenSearch revisions protecting against stale writes.

Each generation binds its store scope, physical index, encoder endpoint and
identity, and contract digests. Changes to store scope, inventory topology,
configuration, or retrieval contracts invalidate affected generations. Reconciliation
can prepare a replacement under the existing verified encoder identity; it does
not validate, accept, or activate that replacement automatically.

## Build, activation, and recovery

Full builds use resumable batches with an outstanding-work cap. Inactive-build
refresh suppression is restored before validation and activation. Validation checks
coverage, index and encoder identity, current scope, and journal readiness.

Build, activation, rollback, and cleanup previews describe their exact targets.
Confirmation tokens bind mutations to that state and reject stale previews.
Activation records the accepted contract. Native rollback and retained-generation
rollback remain separate choices. Cleanup protects active and retained generations
and preserves activation history and shared services.

See [OPERATIONS.md](OPERATIONS.md) for commands and [UPGRADE.md](UPGRADE.md) for
backup, compatibility, and restore procedures.

## Encoder and observability

The encoder runs outside the Magento request process. Its sealed identity binds
the model artifacts, runtime versions, deployment digest, and qualification
evidence. Query and document responses must match that identity. Deterministic
fake mode is ineligible for ordinary operator activation.

Module telemetry records bounded route labels and timings without raw query text
or exception messages. The operations snapshot reports readiness, queue health,
generation coverage, and audits. Monitoring templates cover Magento, the encoder,
OpenSearch, and RabbitMQ; collectors and alert delivery require deployment setup.

[Radial product similarity](RADIAL-SIMILARITY.md) is experimental, uses an independent
circuit, and has no accepted threshold profiles by default.
