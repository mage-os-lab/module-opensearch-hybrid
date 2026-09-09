# Experimental radial similarity calibration

Run the calibration commands from the full [source repository](https://github.com/mage-os-lab/module-opensearch-hybrid/blob/main/docs/DEVELOPMENT.md). Its `config/` and `scripts/` directories are outside the Composer module.

No similarity threshold is included or accepted by default. Threshold work is split into development selection and held-out verification, and neither stage can auto-register a profile.

## Frozen identity

Every bound schema-2 capture binds the full model ID and revision, dimension, similarity function, source recipe, store, use case, exact schema-1 technical-capture SHA-256, and completed judgment-set SHA-256. A change to any field, technical result, candidate pool, or merchant label creates a different calibration identity.

The judgment set must cover head and tail categories, sparse and dense categories, configurable parents and simple children, short and long descriptions, missing descriptions, brand-heavy and feature-heavy products, in-stock and out-of-stock examples, and deliberately unrelated near-neighbor traps. Each case carries explicit slice labels. Every returned candidate must receive exactly one merchant label: `substitute`, `acceptable`, `irrelevant`, or `unsafe`.

## Capture contract

For every frozen `min_score` candidate and seed case, capture:

- the complete exact cosine threshold membership in score order
- the radial result IDs in score order
- exact membership total so truncation is detectable
- at least four concurrency samples for radial latency
- representative slice labels and unsafe product IDs
- a hard maximum result count of 20 or less

The technical capture is schema 1 and cannot nominate a threshold. It proves approximate radial membership against an exact cosine query and creates the candidate pool. It also captures a bounded review catalog for every seed and returned candidate containing only product ID, SKU, title, product class, category, and brand. Descriptions and vectors are excluded. A separate worksheet presents that human-readable context without scores, ranks, or threshold membership. Binding a fully completed worksheet to the exact technical-capture SHA-256 produces the schema-2 evidence accepted by the evaluator.

The binder refuses an incomplete label, a changed capture, extra or missing candidates, altered case metadata, changed review context, or an invalid label. The reviewer may edit only `label` and `notes`. The evaluator refuses unbound quality evidence, truncated exact membership, duplicate or invalid IDs, incomplete identity, an oversized radial response, or insufficient latency samples. ANN false positives describe radial-versus-exact execution differences only. Merchant false positives describe judged `irrelevant` or `unsafe` results and are reported separately.

## Registered gates

- mean recall at the storefront limit at least 0.97
- no slice below 0.90 recall
- radial p95 at concurrency 4 no greater than 100 ms on the registered 50,000-product Linux amd64 profile
- bounded result count under the broadest accepted threshold
- no merchant-identified unsafe hit
- complete merchant labels for every candidate returned by any development threshold

Merchant precision, recall against the judged candidate pool, normalized gain, false-positive rate, result-count distribution, and empty-result rate are always reported even though the first protocol does not assign them independent numeric gates. Selecting among technically eligible development thresholds remains a human decision.

Every specification declares `qualification_scope` as either `smoke` or `registered`. Smoke captures exercise the tooling but can never populate `technical_eligible_candidates` or `eligible_candidates`, even when every observed quality and latency check passes. A registered capture is decision-eligible only when the capture itself observes OpenSearch 3.8.0, exactly 50,000 indexed documents, Linux amd64, and a loopback OpenSearch endpoint. The evaluator recomputes those checks from captured provenance and refuses altered qualification evidence.

Every specification also declares `catalog_evidence_scope` as either `synthetic_execution` or `representative_merchant`. Synthetic evidence may populate `technical_eligible_candidates` after the registered ANN and latency gates pass, but it can never populate `eligible_candidates`, even if a completed worksheet is later bound to it. Only a registered capture declared `representative_merchant` can pass the separate merchant-catalog gate. This declaration is retained in capture provenance and included in the tamper-checked qualification evidence.

## Evaluation

Prepare a reviewed technical-capture specification from `config/radial-calibration.example.json`. The example is smoke-scoped, synthetic, and its threshold is format-only rather than a recommended or accepted value. Replace the physical index, generation, store, identity, cases, slice labels, and known unsafe traps with the frozen development case set. Change `qualification_scope` to `registered` only for the frozen 50,000-document Linux amd64 run executed through loopback on the OpenSearch host. Use `catalog_evidence_scope: representative_merchant` only for a reviewed catalog and case set that satisfy the merchant-representativeness protocol; the deterministic execution fixture must remain `synthetic_execution`. The capture tool derives a provisional case-set digest instead of accepting one from the caller. Binding completed merchant labels later replaces it with the exact merchant judgment-set digest used by a threshold profile.

Run the capture on the same Linux amd64 host as OpenSearch so measured time does not include an unregistered network topology:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run --frozen python \
  scripts/49_capture_radial_similarity_calibration.py \
  --url http://127.0.0.1:9200 \
  --specification /evidence/radial-development-specification.json \
  --output /evidence/radial-development-capture.json
```

The capture tool verifies OpenSearch 3.8.x, the Lucene cosine mapping, generation and model-revision filters, the complete exact `knn_score` threshold membership, deterministic radial membership, complete concurrency-four latency waves, and complete review metadata for the exact captured product IDs. It refuses an exact response whose total exceeds the captured hit list. Seed vectors are held only in process memory and requests; they are never written to the output.

Create the score-blinded worksheet:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run --frozen python \
  scripts/51_prepare_radial_similarity_judgments.py prepare \
  --capture /evidence/radial-development-capture.json \
  --output /evidence/radial-development-judgments.json
```

A merchant reviewer fills every `label` and may add `notes` without changing the capture SHA, cases, seeds, slices, candidate IDs, or review context. Bind that completed file back to the exact technical capture:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run --frozen python \
  scripts/51_prepare_radial_similarity_judgments.py bind \
  --capture /evidence/radial-development-capture.json \
  --judgments /evidence/radial-development-judgments.json \
  --output /evidence/radial-development-bound.json
```

Evaluate the immutable capture separately:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run --frozen python \
  scripts/48_evaluate_radial_similarity_calibration.py \
  --input /evidence/radial-development-bound.json \
  --output /evidence/radial-evaluation.json
```

The report leaves `accepted_profile` null and sets `requires_human_selection=true`. Unbound schema-1 technical captures always have an empty `eligible_candidates` list even when their ANN and latency gates pass. Development schema-2 evidence may nominate a threshold for review. Held-out evidence may verify only that already-frozen candidate. Viewing held-out results never authorizes changing the threshold, registering a profile, enabling a store, adding a storefront adapter, or releasing the module.
