# Experimental radial product similarity

This feature has no accepted threshold profiles by default and is not part of the qualified catalog-search behavior.

Product-seed similarity is isolated from catalog quick search and disabled by default. The service accepts only a server-controlled threshold profile ID and a bounded caller-declared fallback. It never accepts an index name, model identity, raw query DSL, or caller-selected score threshold.

An enabled request still fails closed to the declared fallback unless all of these conditions hold:

- the store has an active generation with an open readiness latch and accepted result contract
- the threshold profile is registered, accepted, scoped to the store, and exactly matches the generation model identity and current similarity contract
- the seed product has a vector in that exact generation
- OpenSearch returns a complete, valid radial response before the configured deadline

The radial request uses Lucene `min_score`, caps the response independently, excludes the seed, requires the exact store and generation identity, and filters to enabled, search-visible, salable, embedding-eligible documents. Results are stabilized by score descending and entity ID ascending. Raw vectors are used only inside the service and are never returned.

The threshold registry is intentionally empty until the calibration protocol passes its development and held-out gates. Enabling the feature before that point returns the caller's fallback with `threshold_profile_unavailable`.

Radial requests use a circuit that is independent from catalog quick search. An OpenSearch failure on a product recommendation can therefore fail that surface closed without routing catalog search away from an otherwise healthy hybrid generation. The service emits one bounded `radial_similarity` event to `var/log/opensearch-hybrid-query.log` with store, generation, profile, latency, result count, empty-result state, fallback reason, circuit state, and error class. It never records a vector, product payload, raw query, response document, or exception message.

Status and doctor report one of four radial states for every module-owned store state:

- `DISABLED`: the store-scoped feature flag is off
- `ENABLED_UNCALIBRATED`: the feature is enabled but no accepted profile applies
- `READY`: an accepted profile matches the active generation and the radial circuit is closed
- `FAILED_CLOSED`: the active generation, profile configuration, or radial circuit is unsafe

The status capability record also exposes the exact OpenSearch version, supported range, and the registered Base64 wire-format qualification identity. A newer 3.8 patch can be supported without being described as the exact benchmarked 3.8.0 runtime.

The frozen evidence format and human selection boundary are documented in [`RADIAL-CALIBRATION-PROTOCOL.md`](RADIAL-CALIBRATION-PROTOCOL.md).

The request contract follows the OpenSearch [`min_score` radial search](https://docs.opensearch.org/latest/vector-search/specialized-operations/radial-search-knn/) API. The module's pinned OpenSearch 3.8 integration lane executes the filtered request against both numeric-array and Base64-built indices and requires identical results.
