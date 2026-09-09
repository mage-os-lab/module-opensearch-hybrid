# Development, testing, and packaging

Use the full source repository for the commands below. The Composer module has
no runtime dependency on the encoder source or Python contributor tools. Run packaging and qualification commands from this checkout.

## Local checks

Use Python 3.13, uv, PHP 8.4 or 8.5, Composer, Go, and Docker. PHP needs the
extensions required by the public Mage-OS Composer package.

```bash
uv sync --frozen --group dev --extra models --extra evaluation
make check
make workflow-check
```

`make check` runs the encoder tests, Ruff, mypy, and the root Python suite. The
disposable HTTP tests require local socket access. The OpenSearch integration test
is opt-in locally and runs against a disposable service in CI. `workflow-check`
runs the pinned actionlint validator and is required when changing a workflow or
workflow fixture.

Follow the [fresh-install and upgrade fixture instructions](../dev/ci/README.md)
for installed Magento behavior, including DI compilation, database schema, catalog
changes, consumers, Luma, and GraphQL requests. The fixtures use deterministic
encoders; real inference requires separate qualification.

## Module package

```bash
make module-package
make module-package-verify
```

The output is a Composer ZIP with `composer.json` at its root and an external JSON
manifest recording every packaged path, byte count, mode, and SHA-256. The builder
uses fixed ZIP metadata, omits local caches, refuses symlinks and outputs inside
the module payload, and rejects runtime dependencies on contributor tooling.
Existing artifacts are never overwritten. CI retains the ZIP and manifest together.

## Repository and Composer installation

This is the canonical product repository. The Magento module and `composer.json`
are at its root. `services/` contains the companion encoder; `poc/`, `scripts/`,
`tests/`, and `config/` contain contributor and qualification tools.

Clone this repository for development:

```bash
git clone https://github.com/mage-os-lab/module-opensearch-hybrid.git
cd module-opensearch-hybrid
```

Composer and Git archives omit the Python tooling, encoder source, caches, and
benchmark data. The reproducible module ZIP retains the PHP tests, installation
fixtures, and operator documentation. Build output belongs in the ignored `dist/`
directory. No benchmark datasets, run outputs, or historical research notes are
included in this repository.

## Qualification tools

- [Encoder build and qualification](../services/opensearch-hybrid-encoder/README.md)
- [Vector transport qualification](../docs/VECTOR-INGESTION-QUALIFICATION.md)
- [Experimental radial calibration](../docs/RADIAL-CALIBRATION-PROTOCOL.md)

Benchmark manifests bind evidence to source hashes, runtime identity, and registered
conditions. Preserve the exact source and protocol documents with retained evidence.
Changing source files changes the evidence identity; old measurements do not become
current by updating their labels or manifests.
