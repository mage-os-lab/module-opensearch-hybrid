# OpenSearch Hybrid upgrade runbook

An upgrade must preserve a verified native-search exit and must not make an old hybrid generation compatible by assumption. A generation is queryable only while its store scope and all current retrieval-contract digests still match.

## Before the change

For every included store, capture:

```bash
bin/magento mage-os:opensearch-hybrid:status --store=<store-id>
bin/magento mage-os:opensearch-hybrid:metrics --store=<store-id>
bin/magento mage-os:opensearch-hybrid:doctor
```

Record the active generation ID, physical index, pipeline ID, encoder identity digest, scope digest, contract digests, native and hybrid watermarks, and retained rollback generations.

Prepare the normal Magento database backup and an OpenSearch snapshot that includes every referenced physical index. Verify both backup targets and the restore procedure. Preserve the deployed encoder artifact digest, actual image reference, sealed identity manifest, artifacts, and qualification evidence.

If the maintenance requires a known lexical-only state, explicitly roll each affected store back to native search before changing code:

```bash
bin/magento mage-os:opensearch-hybrid:rollback \
  --store=<store-id> \
  --dry-run
bin/magento mage-os:opensearch-hybrid:rollback \
  --store=<store-id> \
  --confirm=<preview-token>
```

Confirm the storefront request path is native before continuing.

## Install during a Mage-OS 3.3 to 3.4 migration

This module supports Mage-OS 3.4.x and must not be installed into the running 3.3 application before the platform upgrade transaction. Capture the approved `composer.json`, `composer.lock`, `app/etc/config.php`, database backup, OpenSearch snapshot, and deployment artifact manifest as one pre-change release point. Verify the restore procedure before changing the dependency graph.

While maintenance mode is enabled, use Mage-OS's root-update command to stage the 3.4 product metapackage, then stage this module and Mage-OS Async Events without resolving either request separately:

```bash
composer require-commerce \
  mage-os/product-community-edition=<approved-3.4-version> \
  --no-update \
  --force-root-updates
composer require \
  mage-os/module-opensearch-hybrid:<approved-module-version> \
  mage-os/mageos-async-events:^4.0.8 \
  --no-update
composer update --with-all-dependencies
```

When an approved local package extraction is supplied through a Composer path repository, set `COMPOSER_MIRROR_PATH_REPOS=1` on the `composer update` command so the installed module is a copy of the reviewed payload rather than a source symlink. Do not use a path repository for a normal tagged package from the approved public repository.

Before running database setup, verify that `composer.lock` contains the exact approved Mage-OS 3.4 product version, module version, and Async Events version. Then enable both modules and apply the database and DI changes:

```bash
bin/magento module:enable MageOS_AsyncEvents MageOS_OpenSearchHybrid
bin/magento setup:upgrade --keep-generated
bin/magento setup:db:status
bin/magento setup:di:compile
bin/magento cache:clean
```

Keep the module disabled at the configuration level and preserve native search routing until post-change verification passes. If dependency resolution, schema application, DI compilation, or catalog-preservation verification fails, keep maintenance mode enabled and restore the complete pre-change release point. Do not attempt to reverse declarative-schema changes with a Composer downgrade alone.

## Apply the upgrade

Use the repository and deployment system's approved release procedure. The relevant Magento steps are:

```bash
bin/magento maintenance:enable
bin/magento setup:upgrade --keep-generated
bin/magento setup:di:compile
bin/magento setup:db:status
bin/magento cache:clean
```

Start the required consumers through the process supervisor. Reconcile module-owned state and run health checks:

```bash
bin/magento mage-os:opensearch-hybrid:reconcile --all-stores
bin/magento mage-os:opensearch-hybrid:metrics
bin/magento mage-os:opensearch-hybrid:doctor
```

When the module's contract, mapping, pipeline, recipe, encoder endpoint, store definition, or generation-affecting configuration changed, the previous generation must remain unavailable to hybrid query routing. `active_generation_contract_drift` or a scope mismatch is evidence that a validated replacement is required, not a condition to override.

Result contract v5 makes the first 100 reranked results and the native tail one coherent order. A crossing page includes the end of the reranked prefix plus the native tail. Deep pages use matching native relevance and entity-ID tie-breaking, including availability-first sorting when requested by the storefront. Supported page sizes are 1 through 100; larger pages delegate entirely to native search. GraphQL relevance remains supported, and explicit name or price sorts remain native.

A v4 or older generation must be rolled back to native and replaced. The result-contract and full-contract digests change; the model, mapping, and pipeline definitions do not. Rebuild and validate the replacement, then explicitly accept its current digest. Do not copy old acceptance or edit stored digests. Re-run relevance, complete pagination (including 12-, 24-, and 36-result pages), totals/facets, availability ordering, and latency checks before treating the new generation as qualified. Historical v4 benchmark results remain historical evidence.

## Replace an incompatible generation

Verify the approved encoder identity, build a new generation, and accept the exact current result contract:

```bash
bin/magento mage-os:opensearch-hybrid:encoder:identity \
  --expect=<sealed-encoder-identity-sha256>
bin/magento mage-os:opensearch-hybrid:build \
  --store=<store-id> \
  --encoder-identity=<sealed-encoder-identity-sha256> \
  --dry-run
bin/magento mage-os:opensearch-hybrid:build \
  --store=<store-id> \
  --encoder-identity=<sealed-encoder-identity-sha256> \
  --confirm=<preview-token>
bin/magento mage-os:opensearch-hybrid:validate \
  --generation=<new-generation-id> \
  --accept-result-contract=<current-result-contract-sha256>
```

Review generation coverage, journal lag, watermarks, dead work, consumer health, encoder identity, and storefront acceptance. Then activate with a separate approval:

```bash
bin/magento mage-os:opensearch-hybrid:activate \
  --store=<store-id> \
  --generation=<new-generation-id> \
  --dry-run
bin/magento mage-os:opensearch-hybrid:activate \
  --store=<store-id> \
  --generation=<new-generation-id> \
  --confirm=<preview-token>
```

Do not delete the previous physical index during the upgrade. Keep it through the configured rollback-retention window and until the new generation passes the deployment's acceptance period.

## Complete the change

Disable maintenance mode through the normal release procedure, then verify:

- `setup:db:status` reports no pending declarative-schema changes.
- `doctor` succeeds.
- `metrics` has no high or critical alerts.
- All required consumers are supervised and processing.
- The encoder reports the exact approved identity.
- The active generation is current for scope and contract.
- Native and hybrid watermarks cover the required boundary.
- Supported storefront and GraphQL requests pass exact-SKU, title, pagination, sort, totals, and aggregation acceptance.
- Encoder outage and deadline tests route safely to native search.
- External latency, fallback, encoder, OpenSearch, and RabbitMQ monitors are healthy.

Record committed, deployed, live-verified, and accepted states separately.

## Roll back the upgrade

If hybrid behavior is suspect, return affected stores to native search first:

```bash
bin/magento mage-os:opensearch-hybrid:rollback \
  --store=<store-id> \
  --dry-run
bin/magento mage-os:opensearch-hybrid:rollback \
  --store=<store-id> \
  --confirm=<preview-token>
```

Verify native storefront behavior. Then follow the deployment system's code rollback and database compatibility procedure. Do not restore a retained generation unless the current code accepts its exact contract, scope, encoder identity, coverage, and readiness state.

When a database restore is required, restore the matching Magento database, OpenSearch snapshot, code version, configuration, and encoder artifact set as one reviewed recovery point. A database-only or index-only restore can split generation identity from its physical data.

After recovery, run `setup:db:status`, `status`, `metrics`, `doctor`, consumer checks, and storefront acceptance again.

## Retire old generations

After the retention and acceptance periods pass, use the cleanup preview and exact token flow documented in [OPERATIONS.md](OPERATIONS.md). Never remove the index, database row, or encoder artifacts independently.
