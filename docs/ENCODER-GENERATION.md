# Production encoder generation

The module verifies a sealed encoder identity before creating a generation. The [companion source and build tools](https://github.com/mage-os-lab/module-opensearch-hybrid/blob/main/services/opensearch-hybrid-encoder/README.md) contain the ONNX runtime, same-host deployment package, and deterministic fake mode. They are separate from the Composer module. Run the service commands below from that source checkout. Real inference requires an immutable Linux amd64 Arctic artifact and matching qualification evidence.

## Deployment baseline

The first production identity uses:

- Linux amd64 containers on the existing Magento host, with 8 CPUs and a 16 GiB memory limit as the starting qualification profile.
- A content-addressed OCI image supplied by any digest-capable registry or loaded directly on the host.
- A protected Linux amd64 build and qualification runner. Public pull-request jobs must not execute on the persistent operator host.
- A random API token mounted from a root-owned host file and stored through Magento's encrypted configuration field.
- A public release evidence bundle and `/var/lib/mageos-opensearch-hybrid/releases/<identity>` on the Magento host for retained operator evidence.
- One Compose project and endpoint per encoder identity. Docker-hosted Magento uses an identity-specific alias on its existing network. Bare-metal Magento uses an identity-specific loopback port.

The Magento module still accepts a separately hosted private endpoint. Same-host Compose is the default installation path, not a hard runtime dependency.

GHCR is not required. Registry hostnames and repository names are transport configuration, not canonical identity fields. Do not paste registry credentials or API tokens into tickets, chat, build logs, manifests, or source control.

## Frozen identity contract

The current production identity accepts only:

- Model `Snowflake/snowflake-arctic-embed-m-v2.0` at revision `95c2741480856aa9666782eb4afe11959938017f`.
- A 256-dimensional cosine vector with L2 normalization after truncation.
- Recipe `mageos-v1` with the frozen query and document rendering contract.
- Linux amd64 using `CPUExecutionProvider`.
- One shared float32 model artifact for query and document encoding.
- Query batch size 1 and maximum document batch size 64.
- Immutable deployment artifact and base-artifact SHA-256 digests.
- Exact byte hashes and sizes for every model, tokenizer, configuration, evidence, vector, and latency artifact.

Changing any frozen value requires a reviewed contract change and full requalification. It is not a manifest-edit shortcut.

## Required artifacts

Start from `services/opensearch-hybrid-encoder/identity-manifest.example.json`. The artifact root must contain regular, non-symlink files for these exact roles:

- `document_model`
- `query_model`
- `tokenizer`
- `model_config`
- `amd64_parity`
- `quality_guard`
- `self_retrieval`
- `reference_vectors`
- `latency_concurrency_four`

Every manifest record includes a relative path, lowercase SHA-256, and positive byte size. Qualification is valid only when all five evidence roles are present and the manifest status is `qualified`.

## Build and qualify

The production image build must:

1. Run on or target Linux amd64.
2. Pin the base image by digest and record its digest.
3. Pin Python, NumPy, ONNX Runtime, Tokenizers, and all transitive runtime dependencies.
4. Include all model and tokenizer bytes in the immutable image or a matching read-only artifact volume.
5. Perform no model, tokenizer, code, or package downloads at runtime.
6. Run the parity, quality guard, self-retrieval, reference-vector, and concurrency-four latency checks under the production execution provider.
7. Store the raw evidence and hashes with the release.
8. Resolve the deployment digest and deploy by that digest, either from a registry or a local image loaded with pulls disabled.

The manifest is finalized only after the image and every evidence artifact exist. Record exact runtime versions, deployment artifact digest, base-artifact digest, artifact hashes, and sizes. Record the actual image reference separately in operator evidence. `services/opensearch-hybrid-encoder/requirements-production.lock` is the hash-locked Linux amd64 dependency resolution and changes only through reviewed requalification.

Seal the manifest against the read-only artifact directory:

```bash
cd services/opensearch-hybrid-encoder
PYTHONPATH=src python -m mageos_opensearch_hybrid_encoder.identity \
  /run/encoder/identity.json \
  /srv/encoder/artifacts \
  --output /absolute/new/path/sealed-identity.json
```

Store the emitted `encoder_identity_digest`, sealed output, source manifest, deployment digest, actual image reference, artifact set, and evidence as one public release record. Keep a matching host copy under `/var/lib/mageos-opensearch-hybrid/releases/<identity>`. Any canonical byte or field change produces a different identity. Moving the same OCI digest between repositories does not.

## Production service behavior

The production service calls `seal_identity()` during startup. Startup fails if the manifest, artifacts, runtime claims, installed versions, deployment digests, or qualification evidence do not match. It also refuses production mode without a bounded token loaded from an absolute regular non-symlink file.

The service uses the exact bound tokenizer and one shared float32 ONNX model for queries and documents. Its shared session requires only `CPUExecutionProvider`, the exact `input_ids` and `attention_mask` inputs, and the exact `sentence_embedding` output. The runtime applies the frozen query prefix, validates shape and finite values, and L2-normalizes every 256-dimensional vector.

It serves the exact sealed identity from `GET /v1/identity` and includes the same identity in every query and document embedding response. Request bodies and concurrency are bounded, query text is not logged, bearer authorization uses a constant-time comparison, and identity-bearing responses send `Cache-Control: no-store`.

The deterministic fake server cannot load a manifest and cannot become production eligible through configuration.

## Side-by-side deployment

Do not replace the old encoder endpoint in place while a generation still references it. Deploy each identity as its own Compose project and keep the old image, endpoint, release directory, and evidence available through the rollback-retention window.

Use the deployment package under `services/opensearch-hybrid-encoder/deploy`:

- Docker-hosted Magento attaches the encoder to the existing external network with no host port. The alias is `mageos-hybrid-encoder-<first-12-identity-characters>`.
- Bare-metal Magento binds a unique port to `127.0.0.1`. The exact loopback host must be present in the module allowlist.
- Replacement identities get a new alias or port. The old route stays in place until no retained generation references it.

For each replacement:

1. Deploy the new image by digest without directing Magento to it.
2. Verify health and `GET /v1/identity` from the Magento network path.
3. Update the allowed endpoint and secret through the deployment's approved configuration path.
4. Verify the exact identity digest from Magento.
5. Build and validate a new generation.
6. Activate only after storefront and monitoring acceptance.
7. Retain the old endpoint, image, artifacts, and generation for rollback.
8. Remove them only after the generation cleanup preview is approved and the retention period has passed.

## Magento verification and generation

Verify the exact operator-approved digest before creating any database state:

```bash
bin/magento mage-os:opensearch-hybrid:encoder:identity \
  --expect=<sealed-encoder-identity-sha256>
```

Preview the production generation with the same digest. Review the current catalog workload, vector-only byte estimates, current generations, and zero activation changes:

```bash
bin/magento mage-os:opensearch-hybrid:build \
  --store=<store-id> \
  --encoder-identity=<sealed-encoder-identity-sha256> \
  --dry-run
```

Create the inactive production generation with the same digest only after the impact is approved:

```bash
bin/magento mage-os:opensearch-hybrid:build \
  --store=<store-id> \
  --encoder-identity=<sealed-encoder-identity-sha256> \
  --confirm=<preview-token>
```

Mage-OS recomputes the complete build preview and rejects a stale confirmation before opening the generation transaction. The store-scope digest, endpoint, model revision, and identity digest verified by that current preview are immutable generation fields. Resume and reconciliation recheck the stored endpoint against the stored pin before sending more documents.

Then follow validation and activation in [OPERATIONS.md](OPERATIONS.md). No endpoint-discovered digest is approval by itself.

## Release evidence

Keep this exact bundle:

- Source commit and dependency lock files.
- Build invocation, runner architecture, and runtime version output.
- Actual image reference, pull policy, immutable deployment digest, and base-artifact digest.
- Source identity manifest and sealed identity output.
- Every named artifact with size and SHA-256.
- Raw parity, quality, self-retrieval, reference-vector, and latency results.
- Authenticated concurrency-four HTTP latency evidence bound to the deployed digest, sealed identity, and Magento network route.
- Magento identity verification output.
- Generation ID, store ID, validation report, result-contract acceptance, activation audit, and storefront acceptance.
- Rollback and retention owner.

Do not call the encoder production-ready until this bundle exists and the exact deployed digest passes the Magento identity gate.
