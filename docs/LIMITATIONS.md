# Current limitations

The supported target is Mage-OS 3.4.x, PHP 8.4 or 8.5, and OpenSearch 3.8.x.
Installation, source tests, and deterministic fixtures do not establish production
readiness for a merchant deployment.

## Retrieval and pagination

- Hybrid search reranks at most 100 native candidates. It does not expand recall
  beyond those candidates; native totals and facets remain authoritative.
- Unsupported explicit sorts and page sizes above 100 use native search. Deep pages
  use the native tail with the same candidate ordering.
- Pagination assumes unchanged catalog state and route availability between
  requests. It does not create a snapshot across requests.
- Generations built for v4 or older result contracts require replacement, validation,
  and explicit acceptance under v5. See [UPGRADE.md](UPGRADE.md).

## Encoder and deployment evidence

The module ZIP contains no model weights or qualified encoder image. Real semantic
search requires a sealed Linux amd64 encoder with matching artifacts, runtime
versions, identity, and qualification evidence. The source build and verification
procedure is documented in [ENCODER-GENERATION.md](ENCODER-GENERATION.md).

A reviewed immutable encoder image, complete public evidence bundle, production
image scan, and signature bundle are not supplied with this source tree. The
packaged tooling supports generating and verifying them; tooling alone is not
release evidence. Naming and distribution requirements remain documented in
[TRADEMARKS.md](../TRADEMARKS.md).

## Relevance, capacity, and recovery

End-to-end relevance and latency requalification for the current v5 request builder
remains outstanding. Earlier v4 measurements do not validate its candidate-sort
and pagination changes. No conversion or revenue improvement is established.

Deterministic integration fixtures verify behavior under controlled conditions.
They do not establish production inference capacity, mixed-load latency, or
real-encoder rollback reliability. The optional 100,000 and 1,000,000-product
characterization modes are test tools, not supported production capacity claims.
Independent merchant judgments and deployment-specific load and recovery checks
are still needed before making those claims.

Monitoring configurations are templates. Exporter installation, credentials,
scrape routes, dashboard setup, and alert delivery require verification in the
target environment.

## Experimental radial similarity

Product-seed radial similarity is disabled by default and has an empty threshold
registry. It returns the caller's fallback until a representative merchant
calibration is accepted. Synthetic catalog fixtures cannot qualify a merchant
threshold. See [RADIAL-SIMILARITY.md](RADIAL-SIMILARITY.md).
