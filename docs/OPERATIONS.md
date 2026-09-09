# OpenSearch Hybrid operations

This module starts disabled and keeps every store on native OpenSearch until a production generation passes validation and is explicitly activated. Treat the native adapter as the safe state during installation, maintenance, and incidents.

The first support contract is OpenSearch `>=3.8.0,<3.9.0`. Doctor reports the detected cluster version. Generation creation stops before writing generation state when the active cluster is outside this range; native Magento search remains the exit path.

## Required services

Run these consumers under the deployment process supervisor:

```text
event.trigger.consumer
event.retry.consumer
mageos.opensearch_hybrid.correctness.consumer
mageos.opensearch_hybrid.embedding.consumer
```

Reserve independent process capacity for the correctness consumer. Embedding backlog must not block native and hybrid watermark acknowledgements.

The deployment also requires:

- Mage-OS 3.4.x with Mage-OS Async Events 4.0.8 or newer in major version 4.
- RabbitMQ 4.1.
- OpenSearch 3.8.x.
- A sealed production encoder selected by immutable deployment artifact digest.
- External monitoring for query latency, fallback rate, encoder availability, OpenSearch requests, and RabbitMQ consumer health.

## Installation baseline

Keep the module disabled while installing and compiling it:

```bash
bin/magento module:enable MageOS_AsyncEvents MageOS_OpenSearchHybrid
bin/magento setup:upgrade
bin/magento setup:di:compile
bin/magento setup:db:status
bin/magento cache:clean config
```

Configure the included stores, encoder endpoint, allowed encoder hosts, encrypted encoder token, timeouts, and retained-generation window before enabling the module. Do not put the encoder token in source control or operator output. The default same-host routes and identity-addressed Compose flow are documented in [Same-host encoder deployment](https://github.com/mage-os-lab/module-opensearch-hybrid/blob/main/services/opensearch-hybrid-encoder/deploy/README.md).

Enable bounded Mage-OS Async Events trace cleanup. This module treats disabled cleanup or retention over 90 days as an operational alert. A 30-day starting point is:

```bash
bin/magento config:set system/async_events/subscriber_log_cleanup_cron 1
bin/magento config:set system/async_events/subscriber_log_cron_delete_period 30
bin/magento cache:clean config
```

Install or reconcile the module-owned subscription for each included store:

```bash
bin/magento mage-os:opensearch-hybrid:events:install \
  --store=<store-id> \
  --dry-run
bin/magento mage-os:opensearch-hybrid:events:install \
  --store=<store-id> \
  --confirm=<preview-token>
bin/magento mage-os:opensearch-hybrid:reconcile --store=<store-id>
```

Review the subscription preview before applying it. It reports the canonical create or repair fields and active duplicate IDs, but includes only a SHA-256 digest of each verification token. The confirmed command locks and re-reads the rows, so drift after preview invalidates the token without writing. Repository and resource-model guards protect both the submitted metadata and persisted ownership, preventing a caller from evading the guard by clearing or replacing metadata first. Setup and scheduled reconciliation continue to repair module-owned state automatically.

Reconciliation also checks for a pending generation invalidation created by a confirmed Admin configuration save or an explicit noninteractive configuration write. For an enabled and included store that remains in hybrid mode, it may create one replacement production generation using only the sealed encoder identity already bound to the store's active generation. A per-store database advisory lock and captured journal boundary prevent duplicate candidates. The encoder at the current configured endpoint must still prove that exact identity before any generation row is written.

The replacement remains an ordinary unaccepted candidate. Reconciliation never validates, accepts, activates, rolls back, or cleans up a generation. A changed encoder identity still requires the explicit `build --encoder-identity` workflow. Failed automatic creation is reported as `FAILED_TO_CREATE`, causes the CLI command to fail, logs only the exception class, and leaves native fallback in force.

Then confirm the disabled-state baseline:

```bash
bin/magento mage-os:opensearch-hybrid:status --store=<store-id>
bin/magento mage-os:opensearch-hybrid:metrics --store=<store-id>
bin/magento mage-os:opensearch-hybrid:doctor
```

`doctor` returns a failing exit code for unhealthy dependencies, high or critical operational alerts, an incompatible active generation, a radial feature that cannot establish a safe state, and unsafe trace retention. An intentionally disabled or enabled-but-uncalibrated radial feature remains a safe fallback state. The metrics command emits a JSON snapshot from durable module and Async Events state. Subscription metrics retain total successful and failed delivery-attempt counts, plus `trace_unresolved_failure_count`. Only a UUID trace whose latest attempt remains failed produces `async_event_handoff_failure`; a later successful retry preserves the failure evidence without keeping health red.

Status schema version 2 adds `capabilities` and per-store `similarity` records. Capabilities distinguish the supported current OpenSearch runtime from the exact OpenSearch 3.8.0 Base64 qualification run. Similarity distinguishes `DISABLED`, `ENABLED_UNCALIBRATED`, `READY`, and `FAILED_CLOSED`, and exposes only server-controlled profile IDs.

## Build and activation

First verify the exact production encoder identity approved for this release:

```bash
bin/magento mage-os:opensearch-hybrid:encoder:identity \
  --expect=<sealed-encoder-identity-sha256>
```

Preview the store-scoped production generation with the same pin:

```bash
bin/magento mage-os:opensearch-hybrid:build \
  --store=<store-id> \
  --encoder-identity=<sealed-encoder-identity-sha256> \
  --dry-run
```

The JSON preview is read-only. It verifies the encoder identity and supported OpenSearch version, captures the store-scope fingerprint, counts the current eligible catalog, and reports total batches, the concurrent outstanding cap, initial admission when consumers make no progress, target vectors, vector-only raw and Base64 bytes, expected lifecycle rows, current generations, physical-index creation, and identical current and planned activation routes. The byte estimates are not a disk forecast because they exclude document source, OpenSearch graph and replica overhead, and database row overhead. The preview also emits a confirmation token bound to all of that current state.

After reviewing that current impact, create the inactive production generation with the same pin:

```bash
bin/magento mage-os:opensearch-hybrid:build \
  --store=<store-id> \
  --encoder-identity=<sealed-encoder-identity-sha256> \
  --confirm=<preview-token>
```

Apply recomputes the preview and rejects a stale token before creating database or OpenSearch state. A changed catalog count, store scope, generation set, activation route, contract, configuration, or encoder deployment requires another review. OpenSearch support is revalidated during apply.

If an operator stops seeding, pause and resume one exact generation:

```bash
bin/magento mage-os:opensearch-hybrid:build --pause=<generation-id>
bin/magento mage-os:opensearch-hybrid:build --resume=<generation-id>
```

New inactive generations persist their intended build and normal refresh intervals, then disable periodic OpenSearch refresh while full-build document work is incomplete. Pause and resume preserve that state. Validation restores the normal interval before its explicit refresh and live checks. A failed restore leaves the generation ineligible for activation and keeps the store on native search. The `status` progress rows expose `build_refresh_interval`, `normal_refresh_interval`, and `refresh_state` for diagnosis.

Review status and metrics. Validation without contract acceptance reports the exact current result-contract digest. Review that contract, then repeat validation with the approved digest:

```bash
bin/magento mage-os:opensearch-hybrid:validate --generation=<generation-id>
bin/magento mage-os:opensearch-hybrid:validate \
  --generation=<generation-id> \
  --accept-result-contract=<result-contract-sha256>
```

Activation is a separate two-step mutation. Preview the current state and target, review the JSON, then apply only with that preview's exact token:

```bash
bin/magento mage-os:opensearch-hybrid:activate \
  --store=<store-id> \
  --generation=<generation-id> \
  --dry-run
bin/magento mage-os:opensearch-hybrid:activate \
  --store=<store-id> \
  --generation=<generation-id> \
  --confirm=<preview-token>
```

The same operation is available under System > Tools > OpenSearch Hybrid for Admin roles with `MageOS_OpenSearchHybrid::activate`. Admin activation and rollback require a fresh preview that shows the current state, exact target generation, coverage, validation result, physical index, and encoder identity. Apply requires both the state-bound preview token and a typed generation ID or `native`.

After activation, verify `status`, `metrics`, `doctor`, queue health, and the supported storefront request path. Store activation is not acceptance of the release. Record storefront and monitoring evidence separately.

The default storefront relevance check must cover the actual Mage-OS request shape, including its MSI `is_out_of_stock ASC, position ASC` prefix when out-of-stock-to-bottom is enabled. Confirm the rendered storefront remains semantically ordered, every salable hit precedes every out-of-stock hit in the bounded hybrid result, and an explicit name or price sort still routes to native search.

The GraphQL check must cover the `products(search: ...)` relevance request and prove that it reaches the query encoder while the circuit remains closed. Confirm native totals and aggregations remain coherent. Explicit GraphQL name and price sorts are outside the hybrid contract and must route to native search. Encoder failure must also return the GraphQL request through native search within the deployment deadline.

## Routine checks

Collect these commands on a schedule and during deploy verification:

```bash
bin/magento mage-os:opensearch-hybrid:status
bin/magento mage-os:opensearch-hybrid:metrics
bin/magento mage-os:opensearch-hybrid:doctor
```

For a same-host node-exporter textfile collector, use the packaged
[`deploy/monitoring`](../deploy/monitoring/README.md) templates. The metrics command retains JSON as its
default output and emits bounded OpenMetrics only with `--format=openmetrics`. Add `--query-window=300`
to derive fixed-cardinality route counts, fallback ratios, and latency p95 values from a bounded tail of
the dedicated query log. `--query-max-bytes` defaults to 5 MiB and must remain between 64 KiB and 50 MiB.
The query snapshot reports whether that bounded tail covers the full requested window.

The same directory includes a review-only Grafana import template for all packaged Magento, encoder,
OpenSearch, RabbitMQ, and alert series. Importing it does not provision Prometheus, load rules, configure
notifications, or activate a missing scrape. Treat an empty panel as missing evidence until the exact target
and series are confirmed in Prometheus.

Alert on any high or critical item in the metrics payload. The durable snapshot covers readiness, coverage, journal lag, outbox age and dead work, lifecycle subscription drift, terminal Async Events handoff failures, trace retention, activation history, cleanup history, and generation identity. Investigate retained failed-attempt counts as history, but page on `trace_unresolved_failure_count` and the corresponding high alert.

The deployment monitor must also provide:

- Query route counts and lexical fallback rate. The packaged collector covers catalog-search events when
  its bounded query window is enabled.
- Query encoder, hybrid OpenSearch, and end-to-end search latency percentiles. The packaged collector
  covers per-window p95 values for controlled catalog-search request and component identities.
- Encoder availability, production identity, runtime errors, and concurrency saturation. The packaged
  Prometheus rules cover these after the environment configures the authenticated encoder scrape example.
- Encoder timeouts and catalog-search circuit-open duration. These remain environment-owned alerting work.
- OpenSearch target and cluster health, JVM heap, data-path disk use, circuit-breaker trips, and shard
  allocation restrictions. The packaged Prometheus rules cover these after the OpenSearch 3.8-compatible
  exporter is installed under a reviewed maintenance window and its authenticated TLS scrape is configured.
- RabbitMQ target availability, required processing queues, consumer presence, oldest-message age,
  dead-letter work, and redelivery bursts. The packaged Prometheus rules cover these after the environment
  enables RabbitMQ 4.1's built-in exporter and configures the authenticated TLS aggregate and bounded
  detailed scrapes. Plugin enablement, broker restart, TLS, credentials, and network routing remain separate
  maintenance actions.
- Document encoding throughput and source-hash mismatch count.
- Reconciliation repair count.

These metrics cannot be proven from Magento durable state alone.

Each storefront search emits one `OpenSearch Hybrid query telemetry.` info event to `var/log/opensearch-hybrid-query.log`. Its structured context uses schema version 1 and includes the request name, store ID, route, bounded reason, generation ID when applicable, total duration, component durations, outcomes, and error classes. Native eligibility reasons distinguish disabled configuration, excluded stores, closed readiness, unsupported request names, pages outside the top-100 contract, and unsupported sorts. It never includes raw query text, response documents, or exception messages. The opt-in bounded OpenMetrics snapshot exports only controlled identities, counts, fallback ratios, and p95 values; it omits generation IDs and error classes. Monitoring export failures are isolated from storefront behavior.

Each radial similarity request emits a separate `OpenSearch Hybrid radial similarity telemetry.` event to the same dedicated log. Derive request count, latency, result-count distribution, empty rate, fallback reasons, and circuit state from this event. Radial failures use a similarity-only circuit and cannot open the catalog-search circuit.

## Incident response

When search correctness or availability is uncertain, return the affected store to native search first:

```bash
bin/magento mage-os:opensearch-hybrid:rollback \
  --store=<store-id> \
  --dry-run
bin/magento mage-os:opensearch-hybrid:rollback \
  --store=<store-id> \
  --confirm=<preview-token>
```

This is an explicit activation-state change. Confirm `activation_mode` is `NATIVE`, the readiness latch is closed, and the supported storefront request path uses native OpenSearch.

Then:

1. Capture `status`, `metrics`, `doctor`, consumer status, encoder deployment identity, and OpenSearch health.
2. Stop or scale down embedding workers if they amplify the incident. Keep correctness processing independent when its dependencies are healthy.
3. Reconcile one store if module-owned work or the lifecycle subscription is stale.
4. Inspect dead work. Replay only one reviewed job with its exact UUID.
5. Build and validate a replacement generation when scope, contract, encoder, mapping, or pipeline identity changed.
6. Activate only after validation and storefront acceptance pass.

Manual replay is approval-gated:

```bash
bin/magento mage-os:opensearch-hybrid:retry \
  --job=<job-uuid> \
  --confirm=<job-uuid>
```

A retained generation can be restored only when it still passes the current contract, scope, identity, and readiness checks:

```bash
bin/magento mage-os:opensearch-hybrid:rollback \
  --store=<store-id> \
  --generation=<generation-id> \
  --dry-run
bin/magento mage-os:opensearch-hybrid:rollback \
  --store=<store-id> \
  --generation=<generation-id> \
  --confirm=<preview-token>
```

## Generation cleanup

Do not delete retained generations directly from MySQL or OpenSearch. Preview one exact generation:

```bash
bin/magento mage-os:opensearch-hybrid:cleanup \
  --store=<store-id> \
  --generation=<generation-id> \
  --dry-run
```

The preview lists exact database row counts, the physical index, shared pipeline, preserved activation history, and external encoder artifacts. Active, ready, building, catching-up, and rollback-retained generations do not receive a confirmation token.

After review and approval, apply only the token from the unchanged preview:

```bash
bin/magento mage-os:opensearch-hybrid:cleanup \
  --store=<store-id> \
  --generation=<generation-id> \
  --confirm=<preview-token>
```

Cleanup removes the generation's physical index and cascading module data. It preserves activation history, the shared search pipeline, and external encoder release artifacts. Keep encoder artifacts and identity evidence at least as long as any generation that references them.
