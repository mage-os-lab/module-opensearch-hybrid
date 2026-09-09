# Experimental radial catalog preparation

This developer fixture supports the [radial calibration protocol](../../docs/RADIAL-CALIBRATION-PROTOCOL.md). It does not qualify a merchant threshold.

`prepare-registered-catalog.php` creates an isolated, deterministic 50,000-product latency fixture on one existing Magento store website. It does not install the module, change included stores, build or activate a generation, register a threshold profile, or publish evidence.

The fixture uses only SKUs beginning with `mageos-radial-registered-`. Apply and rollback both recompute current store, website, inventory-source, product-count, and fixture-identity state. A state-derived SHA-256 confirmation token is required for either write. A stale token writes nothing. Any radial capture against this fixture must declare `catalog_evidence_scope: synthetic_execution`; the evaluator permits technical ANN and latency eligibility but rejects merchant-threshold eligibility for that scope.

The preparer inserts simple visible, enabled, salable products in 250-product transactions. Names, materials, product families, rooms, prices, weights, and description lengths vary deterministically. The dataset is appropriate for the registered 50,000-document latency and ANN execution profile. It is synthetic and cannot by itself satisfy representative merchant relevance, brand, popularity, configurable-parent, or held-out judgment gates.

## Before apply

Use a store website that is isolated from the accepted storefront. Confirm that its existing eligible-product count is below 50,000. Take and verify a full Magento database backup. Retain the plan JSON beside that backup. Confirm sufficient database, OpenSearch, RabbitMQ, and encoder capacity for the later production generation build.

Preview the exact impact:

```bash
php dev/qualification/prepare-registered-catalog.php \
  /var/www/html plan 1
```

The initial store-1 plan should report the resolved website and MSI source, zero prefixed products, the current unrelated eligible-product count, the number of products to create, the exact database tables, reindexers, rollback prefix, and a `confirmation_token`.

Apply only the unchanged plan:

```bash
php dev/qualification/prepare-registered-catalog.php \
  /var/www/html apply 1 <exact-plan-token>
```

The apply path obtains a per-store database advisory lock, recomputes the plan, refuses drift, resumes a complete existing prefix after interruption, and reindexes inventory, legacy stock, product price, and catalog search. Success requires exactly 50,000 eligible products for the target store. Apply also refuses a prefix whose rows are incomplete, ineligible, or non-contiguous.

## Rollback

Preview the inverse from current state:

```bash
php dev/qualification/prepare-registered-catalog.php \
  /var/www/html rollback-preview 1
```

Apply the exact inverse token:

```bash
php dev/qualification/prepare-registered-catalog.php \
  /var/www/html rollback 1 <exact-rollback-token>
```

Rollback deletes MSI rows, legacy stock rows, and catalog entities only for the reserved prefix. Magento foreign keys remove the associated website and EAV rows. The same four indexers run afterward, and success requires zero remaining prefixed products. Unlike apply, rollback preview remains available when the prefix is incomplete or the store has grown beyond 50,000 eligible products. Its state records `fixture_consistent` and bounded `fixture_inconsistencies`, and the confirmation token binds that exact recovery impact. Keep the database backup until the product count, native storefront, and any later module-generation cleanup are independently verified.
