# Same-host monitoring collector

The module exposes its durable operations snapshot as JSON by default and as bounded OpenMetrics on request:

```bash
bin/magento mage-os:opensearch-hybrid:metrics --format=openmetrics
bin/magento mage-os:opensearch-hybrid:metrics --store=<store-id> --format=openmetrics
bin/magento mage-os:opensearch-hybrid:metrics \
  --format=openmetrics \
  --query-window=300 \
  --query-max-bytes=5242880
```

The OpenMetrics output contains controlled operational identities only. It omits physical index names,
model revisions, contract digests, alert details, raw queries, documents, vectors, and credentials.

Query telemetry is opt-in. `--query-window` accepts 60 through 3600 seconds and reads no more than the
explicit `--query-max-bytes` limit from the end of `var/log/opensearch-hybrid-query.log`. The default byte
limit is 5 MiB and the accepted range is 64 KiB through 50 MiB. The snapshot emits only controlled store,
request, route, reason, component, and outcome labels, plus window counts, fallback ratios, and p95
durations. It drops malformed or unknown identities and never emits generation IDs, raw query text,
exception classes, response data, or log lines. A completeness gauge remains closed when the bounded tail
cannot prove that it covers the whole requested time window.

`collect-openmetrics.sh` writes a complete snapshot atomically for a node-exporter textfile collector. It
does not replace the last successful file unless the Magento command succeeds and emits the OpenMetrics
end marker. Configure these environment variables for the service account that owns Magento CLI reads:

- `MAGEOS_ROOT`: absolute Magento installation root for a direct PHP CLI deployment.
- `MAGEOS_CLI`: optional absolute path to an executable Magento CLI wrapper. When set, the collector
  invokes this executable directly with the Magento command arguments and does not use `MAGEOS_ROOT`
  or `PHP_BINARY`.
- `OPENMETRICS_TEXTFILE_DIRECTORY`: node-exporter textfile collector directory.
- `OPENMETRICS_STORE_ID`: optional exact positive store ID. Omit it to collect every store.
- `OPENMETRICS_QUERY_WINDOW_SECONDS`: optional 60 through 3600 second query-telemetry window. Omit it to
  retain durable-state-only collection.
- `OPENMETRICS_QUERY_MAX_BYTES`: bounded query-log tail, from 65536 through 52428800 bytes. The default is
  5242880 when query telemetry is enabled.
- `PHP_BINARY`: optional PHP CLI path. The default is `php`.

`MAGEOS_CLI` supports containerized Magento installations without evaluating a shell command from an
environment variable. A reviewed, root-owned wrapper can bind one exact container and working directory:

```bash
#!/usr/bin/env bash
set -euo pipefail
exec /usr/bin/docker exec \
  --user www-data \
  --workdir /var/www/html \
  farm-example-php-1 \
  php bin/magento "$@"
```

Treat Docker daemon access as root-equivalent. Do not add the Magento web user to the Docker group merely
to run this collector. Choose an environment-owned supervisor account, a narrowly reviewed privilege
boundary, or run the direct collector inside the PHP container with a mounted textfile directory.

For a host-side container wrapper, copy
`mageos-opensearch-hybrid-metrics-container.service.example` and replace its supervisor user, supervisor
group, wrapper, textfile-directory, store-ID, and collector-script placeholders. The supervisor identity
must already have the environment-approved container-runtime access. Keep the wrapper root-owned and bind
it to one exact Magento container, user, and working directory.

Install a reviewed copy of the collector outside the release directory, replace the three `@@...@@`
placeholders in the example systemd service, and use the example timer only after confirming the service
user can read Magento configuration and atomically write the collector file. The timer performs no
generation, routing, queue, or configuration mutation. The example service bounds each collection attempt
to 45 seconds and preserves the prior complete textfile snapshot when collection fails.

Load `prometheus-alert-rules.yml` only after validating the rules with the environment's pinned Prometheus
version. These rules cover the durable Magento surface, the opt-in bounded query snapshot, the companion
encoder, OpenSearch, and the module-owned RabbitMQ queues. Query alerts require at least 20 requests in the
current window before evaluating
the five-percent fallback and 200 ms hybrid-p95 thresholds. Encoder alerts cover target availability, a
missing or non-production identity, a runtime-error rate above one percent with at least 20 requests in five
minutes, and sustained use of at least 90 percent of configured concurrency. OpenSearch alerts cover target
availability, red or sustained-yellow cluster status, JVM heap, data-path disk use, circuit-breaker trips,
and restricted shard allocation. RabbitMQ alerts cover scrape availability, required processing queues,
consumer presence, oldest processing work, dead-letter work, and redelivery bursts. Keep separate collectors
and alert rules for database availability and process supervision.

`prometheus-encoder-scrape.yml.example` is a standalone-valid example for the authenticated encoder
endpoint. Merge its scrape job into the environment-owned Prometheus configuration and replace the example
`127.0.0.1:18080` target with one reviewed private route for each active, building, or rollback-retained
encoder identity. The endpoint is not published by the companion Compose template, so the environment must
provide a loopback binding, private reverse proxy, or Prometheus container network route. Do not expose it
publicly.

The example reads the independently scoped metrics bearer token from
`/run/secrets/encoder_metrics_token`. Mount or provision that exact file for the Prometheus service instead
of copying the token into YAML, command arguments, or environment literals. Never use the encoder API token
as the metrics credential. Encoder alerts remain inactive until a target with the exact
`mageos-opensearch-hybrid-encoder` job name is configured.

`prometheus-opensearch-scrape.yml.example` is the secure starting point for OpenSearch 3.8. The official
OpenSearch Prometheus Exporter plugin version `3.8.0.0` exposes `/_prometheus/metrics`; its immutable ZIP is
published at
`https://github.com/opensearch-project/opensearch-prometheus-exporter/releases/download/3.8.0.0/prometheus-exporter-3.8.0.0.zip`
with SHA-256 `c9e74cacd000299627be8a3b132600fa21e6b4e7912f0e6e3409528d3d44cd6a`.
Download and verify that artifact through the environment's ordinary dependency process. Installing or
removing it on every OpenSearch node, restarting the cluster, and changing OpenSearch security remain
separate maintenance actions and are never performed by this Magento module.

The scrape example uses HTTPS, a trusted CA file, a dedicated monitoring username, and a file-backed
password. Replace the loopback target and example username with environment-owned values. Grant only the
permissions required for the metrics endpoint, keep anonymous OpenSearch access disabled, and never set
`insecure_skip_verify`. The exporter enables detailed index metrics by default; set
`prometheus.indices: false` in `opensearch.yml` unless reviewed index-level series are required, because
physical-index labels can create high cardinality. OpenSearch alerts remain inactive until a target with
the exact `mageos-opensearch-hybrid-opensearch` job name is configured.

`prometheus-rabbitmq-scrape.yml.example` uses RabbitMQ 4.1's built-in `rabbitmq_prometheus` plugin. It
scrapes aggregate broker metrics from `/metrics` and a bounded set of queue metric families from
`/metrics/detailed`. The detailed scrape is limited to queue depth, consumer count, delivery, and oldest-head
metrics for one configured virtual host. RabbitMQ's detailed endpoint filters by metric family and virtual
host, not by queue name, so replace the example `/` virtual host and review the resulting series before
activation. Do not expand the family list without a cardinality review.

The example expects TLS on the private `127.0.0.1:15692` target, a trusted CA file, a dedicated monitoring
username, and a file-backed password. Enable `rabbitmq_prometheus`, configure the listener certificate, and
set `prometheus.authentication.enabled = true` through the environment's normal maintenance process. Grant
the monitoring identity only the permissions needed to observe the selected virtual host. Do not expose an
anonymous or cleartext metrics listener. Plugin enablement, broker restart, TLS, user creation, secret
provisioning, and network routing are separate environment-owned actions and are never performed by this
Magento module.

The packaged alerts bind to the exact `mageos.opensearch_hybrid.embedding` and
`mageos.opensearch_hybrid.correctness_priority` processing queues and their `.dead` queues. The initial
thresholds page on a processing queue without consumers, correctness-priority work older than two minutes,
embedding work at least fifteen minutes old, any dead-letter work, and five or more redeliveries in five
minutes. Review these starting thresholds against the deployment's worker capacity and service objectives.
RabbitMQ alerts remain inactive until both example job names are configured.

`grafana-dashboard.json` is a non-editable import template for the packaged monitoring surface. Review the
PromQL, import the JSON through the environment-owned Grafana change process, and map its single
`DS_PROMETHEUS` input to the approved Prometheus data source. Its fixed panels cover durable readiness and
work, query routes and latency, encoder identity and runtime, OpenSearch health and capacity, RabbitMQ queue
state, and firing packaged alerts. The dashboard contains no endpoint, user, credential, notification route,
or mutable control.

An empty panel does not prove health. It usually means that the matching environment-owned scrape job is not
active, its job name differs from the packaged contract, or no series exists in the selected time range.
Confirm targets and alert evaluation directly in Prometheus before relying on the dashboard. Importing this
file does not install a data source, load alert rules, configure notifications, enable any service exporter,
or change Magento routing.

`prometheus.yml.example` is a minimal same-host Prometheus configuration for the loopback-only
node-exporter deployment. It scrapes `127.0.0.1:9100` every 30 seconds and evaluates the module alert rules
on the same interval. Copy the rules to the configured environment-owned path, then validate both files
with the exact Prometheus version approved for that environment:

```bash
PROMTOOL=/absolute/path/to/promtool \
PROMTOOL_SHA256=<approved-lowercase-sha256> \
./validate-prometheus-config.sh \
  /etc/prometheus/rules/mageos-opensearch-hybrid.yml \
  /etc/prometheus/prometheus.yml \
  /etc/prometheus/rules/mageos-opensearch-hybrid.test.yml
```

The validator rejects a relative or non-executable tool path and verifies the binary digest before running
`promtool --version`, `promtool check rules`, `promtool check config`, and the supplied healthy/failure
rule tests. Omit the three file arguments only after staging the example's absolute `rule_files` target;
full config validation deliberately fails when a referenced rule file is absent. Validate the encoder,
OpenSearch, and RabbitMQ scrape examples separately only after their file-backed credentials and CA files
exist at the configured paths.

A Prometheus service on another host cannot reach this loopback target. Keep node-exporter bound to
loopback unless the environment owner approves a separately authenticated and firewalled collection path.
The example intentionally does not configure retention, remote write, dashboard provisioning, notification
routes, or credentials because those remain environment-owned decisions.
