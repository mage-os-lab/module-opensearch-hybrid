# Same-host encoder deployment

This deployment keeps the encoder beside an existing Magento installation without opening it to the public network. Every sealed identity gets its own Compose project and endpoint. Do not replace an endpoint in place while a retained generation still references it.

## Release files

Keep one directory per full encoder identity digest:

```text
/var/lib/mageos-opensearch-hybrid/releases/<identity-digest>/
  identity.json
  sealed-identity.json
  encoder.oci.tar
  encoder.syft.json
  encoder.cdx.json
  encoder.grype.json
  supply-chain.manifest.json
  supply-chain.manifest.sigstore.json
  evidence/
```

The deployment image contains the model and evidence bytes used at runtime. The host release directory keeps the source manifest, sealed output, exact OCI archive, normalized scanner evidence, checksum manifest, signature bundle, and operator evidence. Verify the detached Sigstore bundle against the approved release identity before loading the image. Keep both the loaded image bytes and this directory as long as any Magento generation references the identity.

Create a host token without printing it:

```bash
install -d -m 0700 /etc/mageos-opensearch-hybrid
umask 077
openssl rand -hex 32 > /etc/mageos-opensearch-hybrid/encoder-token
openssl rand -hex 32 > /etc/mageos-opensearch-hybrid/encoder-metrics-token
chown 65532:65532 /etc/mageos-opensearch-hybrid/encoder-token \
  /etc/mageos-opensearch-hybrid/encoder-metrics-token
chmod 0400 /etc/mageos-opensearch-hybrid/encoder-token \
  /etc/mageos-opensearch-hybrid/encoder-metrics-token
```

Copy `encoder.env.example` outside the repository, fill in the exact release paths and digests, and keep the file private. `ENCODER_RUN_UID` and `ENCODER_RUN_GID` must match the token file owner.

Choose one content-addressed image source:

- Registry: set `ENCODER_IMAGE` to any digest-qualified reference such as `registry.example/namespace/opensearch-hybrid-encoder@sha256:<digest>` and use `ENCODER_PULL_POLICY=always` or `missing`.
- Local: load the image on the Magento host, set `ENCODER_IMAGE=sha256:<local-image-id>`, and use `ENCODER_PULL_POLICY=never`.

In both cases, set `ENCODER_DEPLOYMENT_DIGEST` to the same digest used by `ENCODER_IMAGE` and record it in the schema 2 manifest as `deployment.artifact_digest`. A registry account is optional. Repository names and credentials are not part of the canonical encoder identity.

Docker Compose uses bind mounts for file-backed secrets and does not remap their ownership. Preflight checks both host owners and permissions and refuses identical API and metrics values because an unreadable or shared token would weaken the service boundary. See the [Docker Compose secrets reference](https://docs.docker.com/reference/compose-file/services/#secrets).

## Route from Magento

Use `ENCODER_ROUTING_MODE=docker-network` when Magento runs in Docker. Set:

```text
MAGENTO_DOCKER_NETWORK=<existing-external-network>
ENCODER_NETWORK_ALIAS=mageos-hybrid-encoder-<first-12-identity-characters>
```

The Magento endpoint is:

```text
http://mageos-hybrid-encoder-<first-12-identity-characters>:8080
```

Use `ENCODER_ROUTING_MODE=loopback` when PHP and Magento run directly on the host. Select an unused port such as `18080`. The encoder binds only to:

```text
http://127.0.0.1:18080
```

Use a new alias or port for every replacement identity. Do not reuse the old route until its rollback retention has ended and cleanup has been approved.

## Preflight and deploy

The preflight checks the canonical manifest digest, content-addressed deployment digest, pull policy, identity-specific route, token permissions, and release paths without printing the token:

```bash
python3 deploy/preflight.py /absolute/path/to/encoder.env
```

Review the output, then deploy the exact release:

```bash
deploy/deploy.sh /absolute/path/to/encoder.env
```

The container runs as UID and GID `65532`, drops all capabilities, uses a read-only root filesystem, mounts the token and manifest read-only, publishes no port in Docker-network mode, and starts with 8 CPUs and a 16 GiB memory limit by default. Tune those limits only through a new qualified runtime profile.

## Connect Magento

In Catalog > OpenSearch Hybrid > Encoder Deployment:

1. Set the identity-specific endpoint.
2. Add the exact DNS alias or loopback host to Explicitly Allowed Hosts.
3. Paste the same API token into the encrypted API Token field.
4. Save, clear config cache, and verify the exact approved identity digest.

For Docker routing, allowlist the full `mageos-hybrid-encoder-<prefix>` alias. For loopback routing, allowlist exactly `127.0.0.1`. Link-local, metadata, unspecified, multicast, and unlisted loopback addresses remain blocked.

```bash
bin/magento cache:clean config
bin/magento mage-os:opensearch-hybrid:encoder:identity \
  --expect=<sealed-encoder-identity-sha256>
```

Identity verification is read-only. Building a generation, activation, rollback, cleanup, and removal of an old encoder remain separate operator actions.

## Scrape encoder metrics

The authenticated `/metrics` endpoint is available on the same identity-specific Docker alias or loopback route. Keep the target private and configure the collector with the existing token file instead of copying the token into a monitoring configuration literal:

```yaml
scrape_configs:
  - job_name: mageos-opensearch-hybrid-encoder
    metrics_path: /metrics
    authorization:
      type: Bearer
      credentials_file: /run/secrets/encoder_metrics_token
    static_configs:
      - targets:
          - mageos-hybrid-encoder-<identity-prefix>:8080
```

Use the identity-specific target so replacement and retained encoder deployments remain independently observable. The module's query telemetry, OpenSearch, RabbitMQ, database, and process metrics still require environment-owned collectors and alert rules.
