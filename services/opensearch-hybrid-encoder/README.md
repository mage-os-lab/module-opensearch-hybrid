# Mage-OS OpenSearch Hybrid encoder

This is the companion encoder service for `mage-os/module-opensearch-hybrid`. It is designed to run on the existing Magento host beside OpenSearch. Magento can still use a separately hosted endpoint when that fits the deployment better.

The service has two modes:

- `fake` is a dependency-free deterministic test service. Every response declares `production_eligible: false`.
- `production` loads the frozen Snowflake Arctic model through ONNX Runtime. Startup fails unless the manifest, artifact bytes, Linux amd64 platform, runtime package versions, deployment digest, and qualification evidence all match.

This source tree provides `Dockerfile.production`, the Linux amd64 dependency lock, and deployment tools. It does not include a qualified encoder image or model weights. Build and qualify the exact artifact set before connecting a real generation. See the module's [current limitations](https://github.com/mage-os-lab/module-opensearch-hybrid/blob/main/docs/LIMITATIONS.md).

## Local fake service

Run the HTTP contract locally without model dependencies:

```bash
PYTHONPATH=src python -m mageos_opensearch_hybrid_encoder.server
```

The fake service binds to `127.0.0.1:8080`. Leave both tokens unset for loopback-only development, or set different `ENCODER_API_TOKEN` and `ENCODER_METRICS_TOKEN` values to protect both surfaces. Configuring only one token or reusing one value is rejected. The fake image remains pinned separately in `Dockerfile` and can never load a production manifest.

## Production contract

Production mode requires all of these values:

```text
ENCODER_MODE=production
ENCODER_IDENTITY_MANIFEST=/run/encoder/identity.json
ENCODER_ARTIFACT_ROOT=/srv/encoder/artifacts
ENCODER_EXPECTED_DEPLOYMENT_DIGEST=sha256:<deployment-digest>
ENCODER_API_TOKEN_FILE=/run/secrets/encoder_api_token
ENCODER_METRICS_TOKEN_FILE=/run/secrets/encoder_metrics_token
```

The service reads the tokens once at startup. Each token file must be an absolute, regular, non-symlink file containing one bounded value without whitespace. Production refuses to start without separate, different API and metrics tokens, so a monitoring collector cannot invoke embedding routes.

The frozen serving runtime is Python 3.13 with NumPy, ONNX Runtime, and Hugging Face Tokenizers. Sentence Transformers and PyTorch are not installed in the serving image because they do not execute production requests. Model export provenance stays in the sealed model configuration and qualification evidence.

Query requests use the same float32 ONNX artifact as document requests. Query batches remain fixed at 1 and document batches may contain up to 64 values. The service loads one shared `CPUExecutionProvider` session with sequential execution, one ONNX intra-op thread, and one ONNX inter-op thread. The service permits four concurrent model calls by default, so four calls can use four CPU cores without nested thread multiplication. HTTP admission has eight additional bounded connection slots so health and metrics remain usable while inference is saturated. Inference and connection overload return HTTP 503 with `{"error":"overloaded"}`. Admission never waits for an inference slot. Each connection has a five-second absolute deadline to finish its headers and body, as well as an inactivity timeout; trickling bytes does not extend that deadline. Expired connections close and release capacity. If every HTTP slot is occupied, health also receives a bounded overload response until capacity returns.

## Metrics

`GET /metrics` returns authenticated OpenMetrics 1.0 text using the independently scoped metrics bearer token. It reports fixed-cardinality request outcomes, encoded text counts, cumulative request-duration histogram buckets, current in-flight requests, the configured inference concurrency limit, and the exact loaded encoder identity. The in-flight request metric includes admitted embedding handlers, so it can briefly exceed the inference limit while overload is rejected. It never includes request IDs, source text, query text, vectors, exception messages, client addresses, or arbitrary error classes. Runtime failures return the bounded `503 {"error":"runtime_unavailable"}` response and increment only the `runtime_error` outcome.

The endpoint follows the official [OpenMetrics 1.0 text contract](https://prometheus.io/docs/specs/om/open_metrics_spec/) and ends every response with `# EOF`. It is served on the encoder's existing private route and must not be published independently. Production collectors can use Prometheus `authorization.credentials_file` with the existing token file; the [Prometheus scrape configuration](https://prometheus.io/docs/prometheus/latest/configuration/configuration/) documents that file-backed bearer-token setting.

## Artifact qualification

Qualification uses the same tokenizer, ONNX session settings, rendering, normalization, and runtime class as production serving. It does not use the fake encoder or a second benchmark implementation.

First capture fixed reference vectors from the exact candidate bytes on a trusted preparation host. Resolve symlinks before passing artifact paths because qualification accepts only regular files:

```bash
PYTHONPATH=src python -m mageos_opensearch_hybrid_encoder.qualification capture-reference \
  --query-model /absolute/path/to/query.onnx \
  --document-model /absolute/path/to/document.onnx \
  --tokenizer /absolute/path/to/tokenizer.json \
  --output /absolute/new/path/reference-vectors.json
```

Copy the same three artifact bytes and the reference file to the protected Linux amd64 builder. Verify cross-platform parity, then measure batch-1 query encoding with four simultaneous callers:

```bash
PYTHONPATH=src python -m mageos_opensearch_hybrid_encoder.qualification verify-amd64 \
  --query-model /absolute/path/to/query.onnx \
  --document-model /absolute/path/to/document.onnx \
  --tokenizer /absolute/path/to/tokenizer.json \
  --reference /absolute/path/to/reference-vectors.json \
  --output /absolute/new/path/amd64-parity.json

PYTHONPATH=src python -m mageos_opensearch_hybrid_encoder.qualification measure-latency \
  --query-model /absolute/path/to/query.onnx \
  --document-model /absolute/path/to/document.onnx \
  --tokenizer /absolute/path/to/tokenizer.json \
  --warmup-samples 20 \
  --measured-samples 100 \
  --p95-budget-ms 200 \
  --output /absolute/new/path/latency-concurrency-four.json
```

The harness records exact artifact hashes, platform, package versions, raw latency samples, and thresholds. It refuses non-amd64 decision evidence, artifact drift, runtime-version drift, vector drift, invalid protocols, and existing output paths. These files satisfy only the `amd64_parity`, `reference_vectors`, and `latency_concurrency_four` identity roles. The WANDS quality guard, self-retrieval evidence, and end-to-end Magento and OpenSearch gates remain separate requirements.

After deploying the sealed image, run the authenticated HTTP gate from the same network route Magento will use. This measures 4 simultaneous batch-1 query calls, validates every response identity and vector, and retains 100 raw samples. The token is read from a file and is never included in output:

```bash
PYTHONPATH=src python -m mageos_opensearch_hybrid_encoder.http_qualification \
  --endpoint http://mageos-hybrid-encoder-<identity-prefix>:8080 \
  --token-file /run/secrets/encoder_api_token \
  --expected-identity-digest <64-character-identity-digest> \
  --expected-deployment-digest sha256:<deployment-digest> \
  --route-label docker_network_identity_alias \
  --warmup-samples 20 \
  --measured-samples 100 \
  --p95-budget-ms 200 \
  --output /absolute/new/path/http-latency-concurrency-four.json
```

Keep this deployment-bound HTTP evidence in the release bundle. The direct runtime latency artifact remains part of the sealed artifact tree because it is created before the final image digest exists.

## Build boundary

The production image cannot be built from Git alone. Place the exact qualified model, tokenizer, configuration, and evidence files under `release/artifacts/` using the same relative paths that will appear in the release manifest. That directory is ignored by Git but included in the production Docker build context.

Regenerate the hash-locked Linux amd64 requirements only as a reviewed dependency update:

```bash
make encoder-production-lock
```

Build on Linux amd64 or through an amd64 BuildKit builder. The Dockerfile defaults to the qualified digest-pinned base image. A reviewed requalification may override it with another digest-qualified base. The output name is an operator choice and is not part of the encoder identity:

```bash
docker buildx build \
  --platform linux/amd64 \
  --file services/opensearch-hybrid-encoder/Dockerfile.production \
  --build-arg PYTHON_BASE_IMAGE=python:<version>-slim-trixie@sha256:<digest> \
  --build-arg SOURCE_REVISION=<git-commit> \
  --build-arg SOURCE_URL=<approved-public-source-url> \
  --tag registry.example/namespace/opensearch-hybrid-encoder:<release> \
  services/opensearch-hybrid-encoder
```

Set `SOURCE_URL` to the public source repository for the exact build revision. Keep it aligned with the source and destination recorded in the release evidence.

Resolve the immutable deployment digest, then finalize `identity-manifest.example.json` with that digest and every artifact hash. A registry can provide that digest after a push. A registry-free installation can export and load the image, bind the local content-addressed image ID, and use `ENCODER_PULL_POLICY=never`. Seal the manifest against the exact bytes included in the image:

```bash
PYTHONPATH=src python -m mageos_opensearch_hybrid_encoder.identity \
  /absolute/path/to/identity.json \
  /absolute/path/to/release/artifacts \
  --output /absolute/new/path/sealed-identity.json
```

The source manifest is mounted into the container after the deployment digest exists. This avoids a circular digest while still binding the deployed artifact, model bytes, runtime, and evidence into one canonical encoder identity. Registry hostnames, repository names, and credentials are deployment transport details and are not sealed into the identity.

Stage the qualified release bytes under these canonical paths before the final image build:

```text
release/artifacts/
  models/model.onnx
  tokenizer/tokenizer.json
  model/config.json
  evidence/amd64-parity.json
  evidence/quality-guard.json
  evidence/self-retrieval.json
  evidence/reference-vectors.json
  evidence/latency-concurrency-four.json
```

After the image build resolves the immutable local image ID or registry digest, build the schema 2 manifest. The release command validates every artifact hash, the frozen Arctic configuration, Linux amd64 evidence, exact runtime versions, WANDS quality eligibility, self-retrieval eligibility, and the concurrency-four budget before it emits a manifest:

```bash
PYTHONPATH=src python -m mageos_opensearch_hybrid_encoder.release \
  --artifact-root /absolute/path/to/release/artifacts \
  --deployment-digest sha256:<deployment-digest> \
  --base-artifact-digest sha256:<base-image-digest> \
  --output /absolute/new/path/identity.json
```

Then seal the emitted manifest against the same read-only artifact tree. Existing output paths are never overwritten.

## Registry-neutral SBOM, scan, and checksum binding

No container registry is required to build or verify the encoder release evidence. Build the final Linux amd64 image as a single-platform OCI archive and keep that exact archive through review, signing, installation, and rollback:

```bash
docker buildx build \
  --platform linux/amd64 \
  --file services/opensearch-hybrid-encoder/Dockerfile.production \
  --build-arg PYTHON_BASE_IMAGE=python:<version>-slim-trixie@sha256:<digest> \
  --build-arg SOURCE_REVISION=<git-commit> \
  --build-arg SOURCE_URL=<approved-public-source-url> \
  --output type=oci,dest=/absolute/new/path/encoder.oci.tar \
  services/opensearch-hybrid-encoder
```

Inspect the archive before creating the schema 2 identity. The command rejects multiple platforms, non-OCI manifests, unsafe archive entries, invalid blob sizes, and any digest mismatch. Copy its `manifest_digest` into `deployment.artifact_digest` when building and sealing the encoder identity:

```bash
PYTHONPATH=services/opensearch-hybrid-encoder/src \
  python -m mageos_opensearch_hybrid_encoder.supply_chain inspect-image \
  --oci-archive /absolute/path/encoder.oci.tar
```

The exact Linux amd64 Syft, Grype, and Cosign release artifacts and their official SHA-256 values are locked in `supply-chain-tools.json`. Download those named artifacts from the recorded URLs, verify their SHA-256 values before execution, and do not substitute an unreviewed tool version.

Generate both the machine catalog and the retained CycloneDX 1.6 SBOM from the OCI archive. Grype reads that exact Syft catalog. Write Grype JSON through standard output so credentials cannot be copied by an output-file configuration path:

```bash
syft oci-archive:/absolute/path/encoder.oci.tar \
  --source-name mageos-opensearch-hybrid-encoder \
  -o syft-json=/absolute/path/encoder.syft.json \
  -o cyclonedx-json@1.6=/absolute/path/encoder.cdx.json

grype sbom:/absolute/path/encoder.syft.json --show-suppressed -o json \
  > /absolute/path/encoder.grype.json
```

Normalize the two scanner files before retaining them. This removes host-local scanner configuration and cache paths without removing packages, findings, database provenance, or image binding. The manifest builder refuses unnormalized evidence, a database older than 24 hours, suppressed findings, and any High or Critical vulnerability:

```bash
PYTHONPATH=services/opensearch-hybrid-encoder/src \
  python -m mageos_opensearch_hybrid_encoder.supply_chain normalize \
  --syft-json /absolute/path/encoder.syft.json \
  --grype-json /absolute/path/encoder.grype.json

PYTHONPATH=services/opensearch-hybrid-encoder/src \
  python -m mageos_opensearch_hybrid_encoder.supply_chain build \
  --oci-archive /absolute/path/encoder.oci.tar \
  --sealed-identity /absolute/path/sealed-identity.json \
  --syft-json /absolute/path/encoder.syft.json \
  --cyclonedx-json /absolute/path/encoder.cdx.json \
  --grype-json /absolute/path/encoder.grype.json \
  --tool-lock services/opensearch-hybrid-encoder/supply-chain-tools.json \
  --manifest /absolute/new/path/supply-chain.manifest.json
```

Run the same command with `verify` instead of `build` before signing. It rebuilds the canonical manifest from every retained input and rejects drift. The detached Sigstore bundle avoids a circular digest. Choose the certificate identity and issuer only when the final public release workflow is approved:

```bash
cosign sign-blob \
  --bundle /absolute/new/path/supply-chain.manifest.sigstore.json \
  /absolute/path/supply-chain.manifest.json

cosign verify-blob \
  --bundle /absolute/path/supply-chain.manifest.sigstore.json \
  --certificate-identity '<approved-release-identity>' \
  --certificate-oidc-issuer '<approved-oidc-issuer>' \
  /absolute/path/supply-chain.manifest.json
```

The implemented verifier prepares the registry-neutral evidence boundary. Production eligibility still requires the real qualified artifact tree, its exact OCI archive, a passing current scan, the approved signing identity, the resulting Sigstore bundle, and independent review of those retained bytes.

## Same-host deployment

The files under `deploy/` keep each encoder identity in its own Compose project. They support two routes:

- `docker-network` attaches the encoder to an existing Magento Docker network under `mageos-hybrid-encoder-<identity-prefix>`. It publishes no host port.
- `loopback` binds one unique port to `127.0.0.1` for Magento running directly on the host.

Both routes require content-addressed image bytes, but neither requires GHCR or any other registry. Operators can use a digest-qualified image from a registry or a previously loaded local image ID with pulls disabled. Each replacement gets a new DNS alias or loopback port, so the old endpoint can remain available for retained generations and rollback.

See [deploy/README.md](deploy/README.md) for the exact installation flow. Module identity verification and generation setup are documented in [ENCODER-GENERATION.md](https://github.com/mage-os-lab/module-opensearch-hybrid/blob/main/docs/ENCODER-GENERATION.md).
