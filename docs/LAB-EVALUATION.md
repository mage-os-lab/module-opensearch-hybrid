# Lab evaluation guide

This guide covers source evaluation on Mage-OS 3.4, PHP 8.4 or 8.5, and OpenSearch 3.8.x. The checkout provides the module, encoder source, and deterministic integration fixtures. Real semantic search also requires a qualified encoder and matching model artifacts. Read the [current limitations](../docs/LIMITATIONS.md) before interpreting fixture results as deployment evidence.

## Review and run the source

Use Python 3.13, uv, PHP 8.4 or 8.5, Composer, Go, and Docker. PHP must have the extensions required by the public Mage-OS Composer package. Run from the repository root:

```bash
uv sync --frozen --extra models --extra evaluation
make check
make workflow-check
make module-package
make module-package-verify
```

`make check` runs the encoder tests, both Ruff and mypy checks, and the complete root Python suite. It needs local socket access for its disposable HTTP servers. The OpenSearch integration test is opt-in and otherwise reports a skip. CI runs that integration test separately against its disposable OpenSearch service, plus the public fresh-install and upgrade fixtures.

For installed Magento behavior, follow the [fresh-install fixture instructions](../dev/ci/README.md). They create a new disposable Mage-OS installation, compile DI, exercise lifecycle and queue behavior, and run Luma and GraphQL HTTP checks. Inspect the loopback ports before starting Compose. Use new temporary installation and SBOM directories; the runner refuses to overwrite an existing installation. The fixed database and queue credentials belong only to this disposable fixture.

The fixture injects deterministic encoders for controlled behavioral assertions. Its isolated activation probes do not qualify real inference. The ordinary `--fake` build and fake HTTP service remain ineligible for operator activation. Do not change identity flags or database acceptance fields to turn a fake generation into a production candidate.

## Understand the current result contract

Version `native-candidate-score-rerank-v5` reranks at most 100 native candidates. It normalizes supported relevance and storefront position aliases to native relevance with a stable entity-ID tie-break. A requested availability-first sort also applies when selecting native candidates and the native tail. Native totals and facets remain authoritative.

For page sizes up to 100, a page crossing position 100 combines the end of the reranked prefix with the native tail. Later pages use the same native candidate sort. Page sizes above 100 and unsupported explicit sorts use native search throughout. This guarantees a coherent ordering across the boundary for unchanged catalog state and a healthy hybrid route; it does not create a cross-request snapshot when products or route availability change.

Existing v4 or older generations must be replaced, validated, and explicitly accepted under the new digest. See the [upgrade procedure](../docs/UPGRADE.md). Keep native search available while doing so. Earlier v4 measurements do not establish relevance or latency for v5.

## Evaluate real semantic relevance

A real inference evaluation requires an operator-supplied sealed Linux amd64 encoder, matching model artifacts, runtime versions, identity manifest, and qualification evidence. Follow the [encoder qualification guide](../services/opensearch-hybrid-encoder/README.md) and [generation guide](../docs/ENCODER-GENERATION.md). These are the source build and qualification path until a reviewed immutable image and complete bundle are available.

The module starts disabled. Build and validate a new generation, inspect the exact result-contract digest, and use the normal activation preview and confirmation. Assess relevance against independent merchant judgments and measure latency, mixed-load capacity, recovery, and monitoring in the target deployment. Current qualification limits are listed in [LIMITATIONS.md](../docs/LIMITATIONS.md).
