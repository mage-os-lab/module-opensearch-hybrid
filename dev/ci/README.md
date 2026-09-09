# Mage-OS 3.4 install and upgrade CI fixtures

These fixtures install the public `mage-os/project-community-edition` package and verify the module against MySQL 8.4, RabbitMQ 4.1, and OpenSearch 3.8. Fresh Mage-OS 3.4.0 installs run on PHP 8.4 and 8.5. A separate PHP 8.4 lane creates a Mage-OS 3.3.0 store and durable catalog state, then adds this 3.4-only module to the same Composer transaction that upgrades the graph and database to Mage-OS 3.4.0. The fixtures have no Hyva or private Composer repository dependency.

The upgrade path changes `mage-os/product-community-edition` to 3.4.0 before Composer update and `setup:upgrade`. It uses the installed Mage-OS root-update plugin's `require-commerce` command so the root package constraints advance with the product metapackage. It verifies the exact locked platform package before and after the upgrade. The pre-upgrade product and catalog identity must survive. After the module's schema is installed, a deterministic generation must build and a product save must create both correctness and embedding outbox work before the module unit suite runs. This lane does not claim that the module supports Mage-OS 3.3. See the official [Mage-OS 3.4.0 release and upgrade notes](https://mage-os.org/releases/2026-08-11-mage-os-3-4-0-release/) and [Mage-OS 3.3.0 release notes](https://mage-os.org/releases/2026-08-05-mage-os-3-3-0-release/).

The GitHub lanes first build the deterministic module ZIP, extract it outside the source subtree, verify every extracted byte against the canonical package manifest, and provide that extraction as the Composer path repository. The path repository is mirrored into the fixture instead of symlinked. Magento's template path validator, declarative schema, DI compilation, storefront, GraphQL, and PHPUnit checks therefore exercise the verified package payload rather than the source directory. Local runs default to the source subtree unless `MAGEOS_MODULE_INSTALL_ROOT` points at an independently verified extraction.

Both the fresh-install and platform-upgrade runners enforce the module-owned `phpcs.xml.dist` against that exact package payload. The blocking gate applies the Mage-OS Magento2 standard to runtime, presentation, configuration, and unit-test source while treating warnings as review debt. Standalone `dev` bootstrap and qualification programs are excluded from this runtime gate because their required `require`, direct output, and process exit behavior is independently exercised by the fixture. This exclusion does not apply to deployable module code.

After Composer resolves the fixture, the runner generates a reproducible CycloneDX 1.6 JSON SBOM from
the actual installed dependency graph. The official Composer generator manifest is isolated and locked under
`dev/sbom/tool`. CI copies that manifest and lock file into a runner-owned temporary tool directory, where it
installs all generator dependencies. Magento therefore never scans or packages the generator's third-party
classes, and the generator process can inspect the fixture's real `vendor` repository without an environment
override. The generator is not a module runtime dependency. A repository verifier requires exact resolved
versions, the module's dependency edges, and a root-to-module edge before CI retains each PHP lane's SBOM
for 30 days. Before verification, it removes environment-local path-repository references and rewrites the
JSON canonically; identical resolved inputs therefore produce identical retained bytes without exposing the
runner checkout path. This is the Mage-OS fixture boundary only. The production encoder image needs its own
SBOM, container scan, and signed release binding.

The production-build gate injects an already-verified identity client at the service boundary. It proves a rejected identity writes no generation row, while an accepted operator pin freezes the exact model, endpoint, and identity digest into a non-fake generation before seeding. It never substitutes the deterministic encoder for production inference.

After installation and DI compilation, the runner creates a deterministic generation and saves a real product through `ProductRepositoryInterface`. It verifies that the journal, priority job, embedding job, and closed readiness latch are committed while scheduled native indexing remains pending. It then runs native catalog-search indexing and the correctness-priority consumer, proving both applied watermarks reach the save revision while 500 embedding messages remain queued.

The runner next deletes the same product through `ProductRepositoryInterface`, runs native indexing and the priority consumer again, and verifies the tombstone, native and hybrid acknowledgements, deletion watermarks, missing repository entity, and supersession of stale embedding work. Each correctness job must complete within 20 seconds without consuming the embedding backlog.

Direct source-item capture expands affected SKUs through composite parents. The reservation fixture exercises the real `inventory.reservations.updateSalabilityStatus` consumer and requires a stock-scoped priority refresh, a final unsalable OpenSearch document after a sales reservation, restored generation readiness, and unchanged semantic embedding identity and hashes.

Inventory topology acceptance reassigns the base website to a custom stock, disables and re-enables a linked source, then adds and removes a second source link. It requires each affected generation to fail closed, freezes stock/source topology into the scope digest, reindexes MSI before rebuilding, and validates a replacement generation against the final topology.

The active-readiness drill points the store at that deterministic validated generation, applies a priority-only price change, and observes the intermediate state after Magento's native price index acknowledges it. Hybrid eligibility must remain closed until the independent correctness consumer completes the same journal revision, after which the exact active generation must become readable again.

The storefront drill then serves the installed Luma application over HTTP. Healthy GraphQL and non-exact Luma relevance searches must both return the expected product through the query encoder, including Luma's position-only default sort. Exact SKU and exact-title searches must return the expected product without opening the circuit. With an unreachable private encoder endpoint, a non-exact query must meet the bounded response deadline while opening the circuit and returning the native result, and a GraphQL product query must continue returning the same SKU through native fallback.

Later probes cover bounded full price-reindex fan-out, update-on-save catalog price rules, and all four Magento price dimension modes. Each path must refresh every writable generation, complete native and hybrid journal acknowledgements, restore READY validation, and preserve unchanged semantic vectors.

The final operational probes prove that an unverified production encoder cannot create a generation. The retained-generation drill completes the accepted non-fake generation by injecting a deterministic document encoder only at the service boundary, validates and activates it, snapshots the generation, progress, embedding bytes, and outbox state, then performs an exact-token native rollback and exact-token retained-generation restoration. The rollback set must be byte-for-byte unchanged after restoration, and the store is returned to the earlier deterministic generation before later probes. This verifies control-plane preservation and does not qualify the fixture encoder for production inference.

The installed CLI drill independently obtains state-bound JSON previews through the real activation and rollback commands. It proves a stale activation token changes neither the store pointer nor the audit table, then applies activation, native rollback, and retained-generation restoration using only each current preview token. All three writes must record the `cli` actor before the fixture restores the later-probe baseline.

The installed subscription CLI drill separately requires a state-bound preview for operator repair. It proves implicit mutation and stale tokens write nothing, repairs an event-name drift, disables only the newly active duplicate, preserves the verification token, and restores the fixture baseline. It also attempts repository save and resource-model delete operations after replacing the in-memory ownership metadata, proving the guard checks persisted ownership before either mutation.

Cleanup is exercised as a separate two-step approval boundary. It previews exact database and OpenSearch impact, refuses active and rollback-retained generations, rejects stale confirmation tokens without writing an audit, removes only an eligible generation and its physical index, preserves every activation-history row and shared external resource, and records the completed cleanup independently of the deleted generation.

The Admin configuration probe renders the in-form generation-impact controls, refuses an included-store change without confirmation before any configuration or journal write, applies the exact current preview, rejects its token after state changes, and restores the default effective store coverage only with a newly computed token. Public previews contain value digests rather than raw configuration values.

The opt-in large-catalog qualification expands an otherwise completed disposable fixture to exactly 10,000 visible, salable products. Fixture-only catalog, MSI source, and legacy stock bridge rows are inserted in bounded transactions, then the normal Magento inventory, price, and catalog-search indexers run before the module builds, processes, and validates a deterministic generation. The probe refuses to start when any writable generation exists, requires exact complete coverage with no failures, and relies on the ordinary validation service to prove exact OpenSearch document count and contract identity. It then serves the application on loopback and uses real GraphQL HTTP requests to prove exact totals for the deterministic qualification slice, stable sorted pagination, price filtering, price dimensions, and coherent aggregations. It is available from manual GitHub workflow dispatch and is intentionally excluded from routine pull-request runs.

The same disposable runner has exact-target characterization modes for 100,000 and 1,000,000 products. These modes require a fresh fixture with no prior `mageos-hybrid-large-` products, an exact `scale-<target>` confirmation, a new absolute evidence path, and the manifest for the independently verified package payload. The retained JSON binds Mage-OS, PHP, OpenSearch, the package and module Composer digests, retrieval-contract, catalog, generation, timing, memory, and GraphQL facts. A separate verifier rejects incomplete or inconsistent evidence. The fixture uses deterministic fake vectors, so the artifact explicitly remains ineligible for production encoder or infrastructure capacity guidance.

The service ports bind only to loopback. Credentials are fixed, local, CI-only values and must never be reused outside this disposable fixture. The runner refuses to overwrite an existing Mage-OS path.

From the repository root:

```bash
mkdir /private/tmp/mageos-opensearch-hybrid-sbom-tool
cp dev/sbom/tool/composer.json \
  dev/sbom/tool/composer.lock \
  /private/tmp/mageos-opensearch-hybrid-sbom-tool/
composer --working-dir=/private/tmp/mageos-opensearch-hybrid-sbom-tool install \
  --no-dev --no-interaction --no-progress --prefer-dist
docker compose -f dev/ci/compose.yaml up -d --wait
MAGEOS_FIXTURE_ROOT=/private/tmp/mageos-opensearch-hybrid-ci \
  MAGEOS_SBOM_TOOL_ROOT=/private/tmp/mageos-opensearch-hybrid-sbom-tool \
  dev/ci/run.sh
MAGEOS_FIXTURE_ROOT=/private/tmp/mageos-opensearch-hybrid-upgrade-ci \
  MAGEOS_SBOM_TOOL_ROOT=/private/tmp/mageos-opensearch-hybrid-sbom-tool \
  dev/ci/run-upgrade.sh
MAGEOS_FIXTURE_ROOT=/private/tmp/mageos-opensearch-hybrid-ci \
  dev/ci/run-large-catalog.sh
docker compose -f dev/ci/compose.yaml down -v

mkdir /private/tmp/mageos-opensearch-hybrid-scale-package
uv run --frozen python scripts/50_build_module_package.py build \
  --module-root module-opensearch-hybrid \
  --archive /private/tmp/mageos-opensearch-hybrid-scale-package/module.zip \
  --manifest /private/tmp/mageos-opensearch-hybrid-scale-package/manifest.json
mkdir /private/tmp/mageos-opensearch-hybrid-scale-package/module
unzip -q /private/tmp/mageos-opensearch-hybrid-scale-package/module.zip \
  -d /private/tmp/mageos-opensearch-hybrid-scale-package/module
uv run --frozen python scripts/50_build_module_package.py verify \
  --module-root /private/tmp/mageos-opensearch-hybrid-scale-package/module \
  --archive /private/tmp/mageos-opensearch-hybrid-scale-package/module.zip \
  --manifest /private/tmp/mageos-opensearch-hybrid-scale-package/manifest.json
MAGEOS_CI_MYSQL_TMPFS_SIZE=8g \
  MAGEOS_CI_RABBITMQ_TMPFS_SIZE=1g \
  MAGEOS_CI_OPENSEARCH_TMPFS_SIZE=8g \
  MAGEOS_CI_OPENSEARCH_JAVA_OPTS='-Xms2g -Xmx2g' \
  docker compose -f dev/ci/compose.yaml up -d --wait
MAGEOS_FIXTURE_ROOT=/private/tmp/mageos-opensearch-hybrid-scale-ci \
  MAGEOS_LARGE_CATALOG_PRODUCTS=100000 \
  MAGEOS_SCALE_CONFIRMATION=scale-100000 \
  MAGEOS_MODULE_INSTALL_ROOT=/private/tmp/mageos-opensearch-hybrid-scale-package/module \
  /private/tmp/mageos-opensearch-hybrid-scale-package/module/dev/ci/install-scale-fixture.sh
MAGEOS_FIXTURE_ROOT=/private/tmp/mageos-opensearch-hybrid-scale-ci \
  MAGEOS_LARGE_CATALOG_PRODUCTS=100000 \
  MAGEOS_SCALE_CONFIRMATION=scale-100000 \
  MAGEOS_SCALE_EVIDENCE_PATH=/private/tmp/mageos-opensearch-hybrid-scale-100000.json \
  MAGEOS_MODULE_INSTALL_ROOT=/private/tmp/mageos-opensearch-hybrid-scale-package/module \
  MAGEOS_MODULE_PACKAGE_MANIFEST=/private/tmp/mageos-opensearch-hybrid-scale-package/manifest.json \
  /private/tmp/mageos-opensearch-hybrid-scale-package/module/dev/ci/run-large-catalog.sh
docker compose -f dev/ci/compose.yaml down -v
```

The large-catalog commands are optional. The 100,000 and 1,000,000 modes need a separately provisioned disposable fixture; do not point them at an accepted or shared test store. The Compose defaults remain intentionally small for routine CI. The example raises the temporary MySQL and OpenSearch filesystems to 8 GiB and OpenSearch heap to 2 GiB for the first 100,000-product run. Size the 1,000,000-product run from observed 100,000-product high-water marks instead of assuming those values are sufficient. The runner applies a 12 GiB PHP CLI memory limit to the one-process 1,000,000-product characterization by default and records the effective limit in its evidence; `MAGEOS_LARGE_CATALOG_PHP_MEMORY_LIMIT` can set another explicit positive whole number of MiB or GiB. This is harness capacity, not a production consumer recommendation.

The service images are pinned by digest, and the GitHub workflow uses immutable action commit SHAs. Digest and action updates require the same compatibility checks as a dependency update. `make workflow-check` runs the pinned actionlint release and must pass for workflow changes. Runner-owned temporary paths are exported from a step through `GITHUB_ENV`, after the job starts; the `runner` context is not valid in job-level `env` expressions.
