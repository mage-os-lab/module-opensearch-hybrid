# OpenSearch Hybrid

> Note: this module requires upgrading OpenSearch to 3.8 from August 2026

OpenSearch Hybrid adds semantic reranking to Mage-OS catalog search. It reranks the
native OpenSearch top 100 relevance candidates while preserving native totals,
facets, filters, and catalog-search attribute weights. Native OpenSearch remains
available when hybrid search is disabled, unavailable, or unsuitable for a request.

This is the canonical [Mage-OS Lab repository](https://github.com/mage-os-lab/module-opensearch-hybrid).
It includes the Magento module at the repository root, a customer-hosted [encoder service](services/opensearch-hybrid-encoder/README.md), and
the tools used to test and qualify them. Installation leaves hybrid search disabled.
Each store requires a validated generation and explicit activation.

## Requirements

| Component | Supported target |
| --- | --- |
| Mage-OS | 3.5.x |
| PHP | 8.4 or 8.5 |
| OpenSearch | 3.8.x |
| Mage-OS Async Events | 4.0.8 or newer within major version 4 |
| Queue transport | RabbitMQ 4.1 |
| Encoder | Qualified Linux amd64 CPU runtime with the frozen Arctic Embed M v2 model |

The release target is Mage-OS 3.5.x. The current Composer constraints and CI fixtures
still use Mage-OS 3.4.x; use that platform for evaluation until 3.5 compatibility
has been updated and validated.

The module package alone does not include model weights or a qualified encoder
image. See [current limitations](docs/LIMITATIONS.md) before
evaluating a deployment.

## Installation

Register this repository in an isolated Mage-OS evaluation installation:

```bash
composer config repositories.mageos-opensearch-hybrid vcs https://github.com/mage-os-lab/module-opensearch-hybrid
composer require mage-os/module-opensearch-hybrid:dev-main mage-os/mageos-async-events:^4.0.8
bin/magento module:enable MageOS_AsyncEvents MageOS_OpenSearchHybrid
bin/magento setup:upgrade
bin/magento setup:di:compile
```

`dev-main` tracks ongoing development. For an existing store, follow the backup
and compatibility checks in [UPGRADE.md](docs/UPGRADE.md) before changing its
dependencies or database. Contributors can clone this repository and register its
root as a Composer `path` repository instead.

Configure the module under Catalog > OpenSearch Hybrid. Selecting the global
`mageos_opensearch_hybrid` search engine installs the wrapper; the module Enabled
flag and per-store activation are separate controls.

Deploy the encoder using the [same-host deployment guide](services/opensearch-hybrid-encoder/deploy/README.md)
and verify its [identity](docs/ENCODER-GENERATION.md) before building a generation.
Run the following consumers under the deployment's process supervisor:

```text
event.trigger.consumer
event.retry.consumer
mageos.opensearch_hybrid.correctness.consumer
mageos.opensearch_hybrid.embedding.consumer
```

Reserve independent process capacity for correctness work so an embedding backlog
cannot delay it. The [operations guide](docs/OPERATIONS.md) covers configuration,
generation builds, activation, monitoring, and rollback. For a disposable source
review, use the [Lab evaluation guide](docs/LAB-EVALUATION.md).

The Composer package is `mage-os/module-opensearch-hybrid`; the Magento component is
`MageOS_OpenSearchHybrid`. The encoder runs outside Magento and performs no model
downloads during requests.

## Search behavior

- Storefront relevance searches and relevance-sorted GraphQL product searches can
  use hybrid reranking. Exact SKU and exact-title safeguards retain native results.
- A page crossing position 100 combines the reranked prefix with the matching native
  tail. Later pages use native search with the same ordering. Page sizes above 100
  and unsupported explicit sorts use native search throughout.
- Storefront availability ordering keeps salable products first when requested.
  Native totals and aggregations remain authoritative.
- Catalog changes flow through separate correctness and embedding queues. An
  incompatible or incomplete generation causes native fallback until it is ready.
- Activation, rollback, and cleanup use previews and confirmations tied to the
  current store state. Retained generations provide a rollback path.

The pagination contract assumes unchanged catalog state and route availability
between requests. Product-seed radial similarity is
[experimental and disabled by default](docs/RADIAL-SIMILARITY.md).

## Documentation

- [Architecture and retrieval contract](docs/ARCHITECTURE.md)
- [Operations and incident response](docs/OPERATIONS.md)
- [Upgrades and rollback](docs/UPGRADE.md)
- [Encoder identity and generation setup](docs/ENCODER-GENERATION.md)
- [Monitoring](deploy/monitoring/README.md)
- [Current limitations](docs/LIMITATIONS.md)
- [Vector ingestion qualification](docs/VECTOR-INGESTION-QUALIFICATION.md)
- [Development, testing, and packaging](docs/DEVELOPMENT.md)

## License

OpenSearch Hybrid is licensed under the [Open Software License 3.0 (OSL-3.0)](LICENSE.txt).
Existing component notices and third-party licenses continue to apply, including
the module's OSL-3.0 and AFL-3.0 notices and the separate licenses for datasets and
model weights. See the module's [trademark notices](TRADEMARKS.md).
