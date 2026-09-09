from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = (
    REPOSITORY_ROOT

    / "deploy"
    / "monitoring"
    / "collect-openmetrics.sh"
)
ALERT_RULES = COLLECTOR.with_name("prometheus-alert-rules.yml")
SYSTEMD_SERVICE = COLLECTOR.with_name("mageos-opensearch-hybrid-metrics.service.example")
CONTAINER_SYSTEMD_SERVICE = COLLECTOR.with_name(
    "mageos-opensearch-hybrid-metrics-container.service.example"
)
PROMETHEUS_CONFIG = COLLECTOR.with_name("prometheus.yml.example")
ENCODER_PROMETHEUS_CONFIG = COLLECTOR.with_name("prometheus-encoder-scrape.yml.example")
OPENSEARCH_PROMETHEUS_CONFIG = COLLECTOR.with_name("prometheus-opensearch-scrape.yml.example")
RABBITMQ_PROMETHEUS_CONFIG = COLLECTOR.with_name("prometheus-rabbitmq-scrape.yml.example")
GRAFANA_DASHBOARD = COLLECTOR.with_name("grafana-dashboard.json")
MONITORING_README = COLLECTOR.with_name("README.md")
PROMETHEUS_VALIDATOR = COLLECTOR.with_name("validate-prometheus-config.sh")
PROMETHEUS_RULE_TESTS = COLLECTOR.with_name("prometheus-alert-rules.test.yml")
QUERY_METRICS_SERVICE = (
    REPOSITORY_ROOT

    / "Model"
    / "Operations"
    / "QueryTelemetryMetricsService.php"
)
MODULE_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "module-opensearch-hybrid.yml"


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    magento_root = tmp_path / "magento"
    (magento_root / "bin").mkdir(parents=True)
    (magento_root / "bin" / "magento").write_text("fixture\n", encoding="utf-8")
    textfile_directory = tmp_path / "textfile"
    textfile_directory.mkdir()
    fake_php = tmp_path / "fake-php"
    fake_php.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "arguments_path = os.environ.get('FAKE_ARGUMENTS_PATH', '')\n"
        "if arguments_path:\n"
        "    Path(arguments_path).write_text('\\n'.join(sys.argv[1:]) + '\\n', encoding='utf-8')\n"
        "sys.stdout.write(os.environ.get('FAKE_OPENMETRICS', ''))\n"
        "raise SystemExit(int(os.environ.get('FAKE_EXIT_CODE', '0')))\n",
        encoding="utf-8",
    )
    fake_php.chmod(0o755)

    return magento_root, textfile_directory, fake_php


def _run(
    magento_root: Path,
    textfile_directory: Path,
    fake_php: Path,
    metrics: str,
    **extra_environment: str,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "MAGEOS_ROOT": str(magento_root),
            "OPENMETRICS_TEXTFILE_DIRECTORY": str(textfile_directory),
            "PHP_BINARY": str(fake_php),
            "FAKE_OPENMETRICS": metrics,
            **extra_environment,
        }
    )

    return subprocess.run(
        [str(COLLECTOR)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def _query_log_line(timestamp: str, context: dict[str, object]) -> str:
    encoded = json.dumps(context, separators=(",", ":"))
    return (
        f"[{timestamp}] mageos_opensearch_hybrid_query.INFO: "
        f"OpenSearch Hybrid query telemetry. {encoded} []\n"
    )


def _summarize_query_log(tmp_path: Path, log: Path) -> dict[str, object]:
    php = shutil.which("php")
    assert php is not None
    runner = tmp_path / "summarize-query-log.php"
    runner.write_text(
        "<?php\n"
        "declare(strict_types=1);\n"
        "require $argv[1];\n"
        "$service = new "
        "MageOS\\OpenSearchHybrid\\Model\\Operations\\QueryTelemetryMetricsService();\n"
        "echo json_encode($service->summarize(\n"
        "    $argv[2],\n"
        "    (int)$argv[3],\n"
        "    (int)$argv[4],\n"
        "    (int)$argv[5]\n"
        "), JSON_THROW_ON_ERROR);\n",
        encoding="utf-8",
    )
    now = int(datetime(2026, 8, 31, 21, 5, tzinfo=UTC).timestamp())
    result = subprocess.run(
        [
            php,
            str(runner),
            str(QUERY_METRICS_SERVICE),
            str(log),
            "300",
            "65536",
            str(now),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    return cast(dict[str, object], json.loads(result.stdout))


def test_collector_atomically_writes_a_complete_snapshot(tmp_path: Path) -> None:
    magento_root, textfile_directory, fake_php = _fixture(tmp_path)
    metrics = "metric_name 1\n# EOF\n"

    result = _run(magento_root, textfile_directory, fake_php, metrics)

    target = textfile_directory / "mageos_opensearch_hybrid.prom"
    assert result.returncode == 0
    assert target.read_text(encoding="utf-8") == metrics
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert list(textfile_directory.glob(".mageos_opensearch_hybrid.prom.*")) == []


def test_collector_retains_the_previous_snapshot_when_output_is_incomplete(tmp_path: Path) -> None:
    magento_root, textfile_directory, fake_php = _fixture(tmp_path)
    target = textfile_directory / "mageos_opensearch_hybrid.prom"
    target.write_text("previous 1\n# EOF\n", encoding="utf-8")

    result = _run(magento_root, textfile_directory, fake_php, "incomplete 1\n")

    assert result.returncode == 1
    assert target.read_text(encoding="utf-8") == "previous 1\n# EOF\n"
    assert "retaining the previous collector file" in result.stderr


def test_collector_rejects_output_after_the_end_marker(tmp_path: Path) -> None:
    magento_root, textfile_directory, fake_php = _fixture(tmp_path)
    target = textfile_directory / "mageos_opensearch_hybrid.prom"
    target.write_text("previous 1\n# EOF\n", encoding="utf-8")

    result = _run(
        magento_root,
        textfile_directory,
        fake_php,
        "metric_name 1\n# EOF\nunexpected trailing output\n",
    )

    assert result.returncode == 1
    assert target.read_text(encoding="utf-8") == "previous 1\n# EOF\n"


def test_collector_retains_the_previous_snapshot_when_magento_fails(tmp_path: Path) -> None:
    magento_root, textfile_directory, fake_php = _fixture(tmp_path)
    target = textfile_directory / "mageos_opensearch_hybrid.prom"
    target.write_text("previous 1\n# EOF\n", encoding="utf-8")

    result = _run(
        magento_root,
        textfile_directory,
        fake_php,
        "partial 1\n",
        FAKE_EXIT_CODE="1",
    )

    assert result.returncode == 1
    assert target.read_text(encoding="utf-8") == "previous 1\n# EOF\n"
    assert list(textfile_directory.glob(".mageos_opensearch_hybrid.prom.*")) == []


def test_collector_rejects_an_invalid_store_scope(tmp_path: Path) -> None:
    magento_root, textfile_directory, fake_php = _fixture(tmp_path)

    result = _run(
        magento_root,
        textfile_directory,
        fake_php,
        "metric_name 1\n# EOF\n",
        OPENMETRICS_STORE_ID="all",
    )

    assert result.returncode == 2
    assert "positive integer" in result.stderr
    assert not (textfile_directory / "mageos_opensearch_hybrid.prom").exists()


def test_collector_supports_an_executable_magento_cli_wrapper(tmp_path: Path) -> None:
    _, textfile_directory, fake_cli = _fixture(tmp_path)
    arguments_path = tmp_path / "arguments.txt"
    environment = os.environ.copy()
    environment.pop("MAGEOS_ROOT", None)
    environment.pop("PHP_BINARY", None)
    environment.update(
        {
            "MAGEOS_CLI": str(fake_cli),
            "OPENMETRICS_TEXTFILE_DIRECTORY": str(textfile_directory),
            "OPENMETRICS_STORE_ID": "1",
            "FAKE_OPENMETRICS": "metric_name 1\n# EOF\n",
            "FAKE_ARGUMENTS_PATH": str(arguments_path),
        }
    )

    result = subprocess.run(
        [str(COLLECTOR)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0
    assert (textfile_directory / "mageos_opensearch_hybrid.prom").read_text(
        encoding="utf-8"
    ) == "metric_name 1\n# EOF\n"
    assert arguments_path.read_text(encoding="utf-8").splitlines() == [
        "mage-os:opensearch-hybrid:metrics",
        "--format=openmetrics",
        "--store=1",
    ]


def test_collector_passes_only_bounded_query_window_arguments(tmp_path: Path) -> None:
    _, textfile_directory, fake_cli = _fixture(tmp_path)
    arguments_path = tmp_path / "arguments.txt"
    environment = os.environ.copy()
    environment.pop("MAGEOS_ROOT", None)
    environment.pop("PHP_BINARY", None)
    environment.update(
        {
            "MAGEOS_CLI": str(fake_cli),
            "OPENMETRICS_TEXTFILE_DIRECTORY": str(textfile_directory),
            "OPENMETRICS_QUERY_WINDOW_SECONDS": "300",
            "OPENMETRICS_QUERY_MAX_BYTES": "5242880",
            "FAKE_OPENMETRICS": "metric_name 1\n# EOF\n",
            "FAKE_ARGUMENTS_PATH": str(arguments_path),
        }
    )

    result = subprocess.run(
        [str(COLLECTOR)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0
    assert arguments_path.read_text(encoding="utf-8").splitlines() == [
        "mage-os:opensearch-hybrid:metrics",
        "--format=openmetrics",
        "--query-window=300",
        "--query-max-bytes=5242880",
    ]


def test_collector_rejects_an_unbounded_query_window(tmp_path: Path) -> None:
    magento_root, textfile_directory, fake_php = _fixture(tmp_path)

    result = _run(
        magento_root,
        textfile_directory,
        fake_php,
        "metric_name 1\n# EOF\n",
        OPENMETRICS_QUERY_WINDOW_SECONDS="3601",
    )

    assert result.returncode == 2
    assert "60 through 3600" in result.stderr
    assert not (textfile_directory / "mageos_opensearch_hybrid.prom").exists()


def test_query_telemetry_summary_is_bounded_and_privacy_safe(tmp_path: Path) -> None:
    log = tmp_path / "opensearch-hybrid-query.log"
    base_context: dict[str, object] = {
        "schema_version": 1,
        "event": "query_route",
        "store_id": 1,
        "request_name": "quick_search_container",
        "generation_id": 91,
        "raw_query": "private merchant query",
    }
    log.write_text(
        _query_log_line(
            "2026-08-31T20:00:00.000000+00:00",
            {
                **base_context,
                "route": "hybrid",
                "reason": "eligible",
                "duration_ms": 999,
                "components": {},
            },
        )
        + "malformed telemetry line\n"
        + _query_log_line(
            "2026-08-31T21:04:30.000000+00:00",
            {
                **base_context,
                "route": "hybrid",
                "reason": "eligible",
                "duration_ms": 100,
                "components": {
                    "native_candidates": {"outcome": "success", "duration_ms": 10},
                    "query_encoder": {"outcome": "success", "duration_ms": 20},
                    "hybrid_opensearch": {"outcome": "success", "duration_ms": 30},
                },
            },
        )
        + _query_log_line(
            "2026-08-31T21:04:40.000000+00:00",
            {
                **base_context,
                "route": "native_fallback",
                "reason": "hybrid_error",
                "duration_ms": 300,
                "error_class": "SensitiveExceptionName",
                "components": {"query_encoder": {"outcome": "failure", "duration_ms": 25}},
            },
        )
        + _query_log_line(
            "2026-08-31T21:04:50.000000+00:00",
            {
                **base_context,
                "request_name": "attacker_controlled_name",
                "route": "attacker_controlled_route",
                "reason": "attacker_controlled_reason",
                "duration_ms": 1,
                "components": {},
            },
        ),
        encoding="utf-8",
    )

    snapshot = _summarize_query_log(tmp_path, log)

    assert snapshot["window_seconds"] == 300
    assert snapshot["log_available"] is True
    assert snapshot["window_complete"] is True
    assert snapshot["records_rejected"] == 1
    assert snapshot["requests"] == [
        {
            "store_id": 1,
            "request_name": "quick_search_container",
            "requests": 2,
            "fallback_ratio": 0.5,
        }
    ]
    assert snapshot["latency"] == [
        {
            "store_id": 1,
            "request_name": "quick_search_container",
            "route": "hybrid",
            "observations": 1,
            "p95_duration_ms": 100,
        },
        {
            "store_id": 1,
            "request_name": "quick_search_container",
            "route": "native_fallback",
            "observations": 1,
            "p95_duration_ms": 300,
        },
    ]
    encoded = json.dumps(snapshot)
    assert "private merchant query" not in encoded
    assert "SensitiveExceptionName" not in encoded
    assert "generation_id" not in encoded
    assert "attacker_controlled" not in encoded


def test_query_telemetry_summary_reports_an_incomplete_truncated_window(tmp_path: Path) -> None:
    log = tmp_path / "opensearch-hybrid-query.log"
    context: dict[str, object] = {
        "schema_version": 1,
        "event": "query_route",
        "store_id": 1,
        "request_name": "graphql_product_search",
        "route": "hybrid",
        "reason": "eligible",
        "duration_ms": 80,
        "components": {},
    }
    log.write_text(
        ("discarded filler\n" * 5000)
        + _query_log_line("2026-08-31T21:04:50.000000+00:00", context),
        encoding="utf-8",
    )

    snapshot = _summarize_query_log(tmp_path, log)

    assert snapshot["bytes_scanned"] == 65536
    assert snapshot["window_complete"] is False
    assert snapshot["requests"] == [
        {
            "store_id": 1,
            "request_name": "graphql_product_search",
            "requests": 1,
            "fallback_ratio": 0,
        }
    ]


def test_alert_rules_detect_missing_metrics_and_preserve_durable_severity() -> None:
    rules = ALERT_RULES.read_text(encoding="utf-8")
    stale_rule = rules.split("- alert: MageOSOpenSearchHybridMetricsStale", 1)[1].split(
        "- alert:", 1
    )[0]
    durable_rule = rules.split("- alert: MageOSOpenSearchHybridDurableAlert", 1)[1].split(
        "- alert:", 1
    )[0]

    assert "absent(mageos_opensearch_hybrid_snapshot_generated_timestamp_seconds)" in stale_rule
    assert "severity:" not in durable_rule


def test_alert_rules_cover_query_staleness_completeness_fallback_and_latency() -> None:
    rules = ALERT_RULES.read_text(encoding="utf-8")
    stale_rule = rules.split("- alert: MageOSOpenSearchHybridQueryMetricsStale", 1)[1].split(
        "- alert:", 1
    )[0]

    assert "MageOSOpenSearchHybridQueryMetricsStale" in rules
    assert "MageOSOpenSearchHybridQueryWindowIncomplete" in rules
    assert "MageOSOpenSearchHybridQueryFallbackRateHigh" in rules
    assert "MageOSOpenSearchHybridQueryLatencyHigh" in rules
    assert "mageos_opensearch_hybrid_query_window_requests >= 20" in rules
    assert "mageos_opensearch_hybrid_query_window_fallback_ratio > 0.05" in rules
    assert "mageos_opensearch_hybrid_query_window_p95_duration_milliseconds" in rules
    assert "> 200" in rules
    assert "mageos_opensearch_hybrid_query_collection_enabled == 1" in stale_rule


def test_alert_rules_cover_encoder_availability_identity_errors_and_saturation() -> None:
    rules = ALERT_RULES.read_text(encoding="utf-8")

    assert "MageOSOpenSearchHybridEncoderUnavailable" in rules
    assert "MageOSOpenSearchHybridEncoderIdentityMissing" in rules
    assert "MageOSOpenSearchHybridEncoderNotProductionEligible" in rules
    assert "MageOSOpenSearchHybridEncoderRuntimeErrorRateHigh" in rules
    assert "MageOSOpenSearchHybridEncoderSaturated" in rules
    assert 'up{job="mageos-opensearch-hybrid-encoder"} == 0' in rules
    assert 'outcome="runtime_error"' in rules
    assert "mageos_opensearch_hybrid_encoder_concurrency_limit" in rules


def test_encoder_scrape_example_uses_private_file_backed_credentials() -> None:
    config = yaml.safe_load(ENCODER_PROMETHEUS_CONFIG.read_text(encoding="utf-8"))
    scrape = config["scrape_configs"][0]

    assert scrape["job_name"] == "mageos-opensearch-hybrid-encoder"
    assert scrape["metrics_path"] == "/metrics"
    assert scrape["scheme"] == "http"
    assert scrape["follow_redirects"] is False
    assert scrape["authorization"] == {
        "type": "Bearer",
        "credentials_file": "/run/secrets/encoder_metrics_token",
    }
    assert scrape["static_configs"] == [{"targets": ["127.0.0.1:18080"]}]
    assert "credentials" not in scrape["authorization"]


def test_alert_rules_cover_opensearch_health_capacity_and_failures() -> None:
    rules = ALERT_RULES.read_text(encoding="utf-8")

    assert "MageOSOpenSearchHybridOpenSearchUnavailable" in rules
    assert "MageOSOpenSearchHybridOpenSearchClusterRed" in rules
    assert "MageOSOpenSearchHybridOpenSearchClusterYellow" in rules
    assert "MageOSOpenSearchHybridOpenSearchJVMHeapHigh" in rules
    assert "MageOSOpenSearchHybridOpenSearchDiskHigh" in rules
    assert "MageOSOpenSearchHybridOpenSearchCircuitBreakerTrips" in rules
    assert "MageOSOpenSearchHybridOpenSearchShardAllocationRestricted" in rules
    assert 'opensearch_cluster_status{job="mageos-opensearch-hybrid-opensearch"} == 2' in rules
    assert (
        'opensearch_jvm_mem_heap_used_percent{job="mageos-opensearch-hybrid-opensearch"} > 85'
        in rules
    )
    assert (
        'opensearch_circuitbreaker_tripped_count{job="mageos-opensearch-hybrid-opensearch"}[5m]'
        in rules
    )


def test_opensearch_scrape_example_uses_tls_and_file_backed_credentials() -> None:
    config = yaml.safe_load(OPENSEARCH_PROMETHEUS_CONFIG.read_text(encoding="utf-8"))
    scrape = config["scrape_configs"][0]

    assert scrape["job_name"] == "mageos-opensearch-hybrid-opensearch"
    assert scrape["metrics_path"] == "/_prometheus/metrics"
    assert scrape["scheme"] == "https"
    assert scrape["follow_redirects"] is False
    assert scrape["basic_auth"] == {
        "username": "mageos_hybrid_monitor",
        "password_file": "/run/secrets/opensearch_metrics_password",
    }
    assert scrape["tls_config"] == {
        "ca_file": "/run/secrets/opensearch_metrics_ca.pem",
        "min_version": "TLS12",
    }
    assert scrape["static_configs"] == [{"targets": ["127.0.0.1:9200"]}]
    assert "password" not in scrape["basic_auth"]


def test_alert_rules_cover_module_owned_rabbitmq_queues() -> None:
    rules = ALERT_RULES.read_text(encoding="utf-8")

    assert "MageOSOpenSearchHybridRabbitMQUnavailable" in rules
    assert "MageOSOpenSearchHybridRabbitMQConsumerMissing" in rules
    assert "MageOSOpenSearchHybridRabbitMQPriorityWorkOld" in rules
    assert "MageOSOpenSearchHybridRabbitMQDeadLettered" in rules
    assert "MageOSOpenSearchHybridRabbitMQRedeliveryBurst" in rules
    assert "mageos.opensearch_hybrid.embedding" in rules
    assert "mageos.opensearch_hybrid.correctness_priority" in rules
    assert "rabbitmq_detailed_queue_head_message_timestamp" in rules
    assert "rabbitmq_detailed_queue_messages_redelivered_total" in rules


def test_rabbitmq_scrape_example_limits_detailed_queue_families() -> None:
    config = yaml.safe_load(RABBITMQ_PROMETHEUS_CONFIG.read_text(encoding="utf-8"))
    aggregate, detailed = config["scrape_configs"]

    assert aggregate["job_name"] == "mageos-opensearch-hybrid-rabbitmq"
    assert aggregate["metrics_path"] == "/metrics"
    assert detailed["job_name"] == "mageos-opensearch-hybrid-rabbitmq-queues"
    assert detailed["metrics_path"] == "/metrics/detailed"
    assert detailed["params"] == {
        "family": [
            "queue_coarse_metrics",
            "queue_consumer_count",
            "queue_delivery_metrics",
            "queue_metrics",
        ],
        "vhost": ["/"],
    }
    for scrape in (aggregate, detailed):
        assert scrape["scheme"] == "https"
        assert scrape["follow_redirects"] is False
        assert scrape["basic_auth"] == {
            "username": "mageos_hybrid_monitor",
            "password_file": "/run/secrets/rabbitmq_metrics_password",
        }
        assert scrape["tls_config"] == {
            "ca_file": "/run/secrets/rabbitmq_metrics_ca.pem",
            "min_version": "TLS12",
        }
        assert scrape["static_configs"] == [{"targets": ["127.0.0.1:15692"]}]
        assert "password" not in scrape["basic_auth"]


def test_grafana_dashboard_is_a_safe_importable_template() -> None:
    dashboard = json.loads(GRAFANA_DASHBOARD.read_text(encoding="utf-8"))

    assert dashboard["uid"] == "mageos-opensearch-hybrid"
    assert dashboard["title"] == "Mage-OS OpenSearch Hybrid"
    assert dashboard["editable"] is False
    assert dashboard["refresh"] == "30s"
    assert dashboard["time"] == {"from": "now-6h", "to": "now"}
    assert dashboard["schemaVersion"] >= 39
    assert dashboard["version"] == 1
    assert dashboard["templating"]["list"] == [
        {
            "name": "DS_PROMETHEUS",
            "label": "Prometheus",
            "type": "datasource",
            "query": "prometheus",
            "current": {},
            "hide": 0,
            "refresh": 1,
        }
    ]

    panels = [panel for panel in dashboard["panels"] if panel["type"] != "row"]
    assert len({panel["id"] for panel in dashboard["panels"]}) == len(dashboard["panels"])
    assert all(
        panel["datasource"] == {"type": "prometheus", "uid": "${DS_PROMETHEUS}"} for panel in panels
    )
    assert "password" not in json.dumps(dashboard).lower()
    assert "bearer" not in json.dumps(dashboard).lower()


def test_grafana_dashboard_covers_each_packaged_monitoring_surface() -> None:
    dashboard = json.loads(GRAFANA_DASHBOARD.read_text(encoding="utf-8"))
    expressions = "\n".join(
        target["expr"] for panel in dashboard["panels"] for target in panel.get("targets", [])
    )

    for metric in (
        "mageos_opensearch_hybrid_readiness",
        "mageos_opensearch_hybrid_alert",
        "mageos_opensearch_hybrid_work_jobs",
        "mageos_opensearch_hybrid_query_window_routes",
        "mageos_opensearch_hybrid_query_window_fallback_ratio",
        "mageos_opensearch_hybrid_query_window_p95_duration_milliseconds",
        "mageos_opensearch_hybrid_query_component_window_p95_duration_milliseconds",
        "mageos_opensearch_hybrid_encoder_identity_info",
        "mageos_opensearch_hybrid_encoder_requests_total",
        "mageos_opensearch_hybrid_encoder_request_duration_seconds_bucket",
        "mageos_opensearch_hybrid_encoder_inflight_requests",
        "opensearch_cluster_status",
        "opensearch_jvm_mem_heap_used_percent",
        "opensearch_fs_path_available_bytes",
        "opensearch_circuitbreaker_tripped_count",
        "rabbitmq_detailed_queue_consumers",
        "rabbitmq_detailed_queue_head_message_timestamp",
        "rabbitmq_detailed_queue_messages",
        "rabbitmq_detailed_queue_messages_redelivered_total",
        'ALERTS{alertname=~"MageOSOpenSearchHybrid.*"',
    ):
        assert metric in expressions

    encoded = json.dumps(dashboard)
    assert "{{encoder_identity_digest}}" in encoded
    assert "{{deployment_digest}}" not in encoded

    readme = MONITORING_README.read_text(encoding="utf-8")
    assert "grafana-dashboard.json" in readme
    assert "import" in readme.lower()


def test_module_workflow_runs_the_monitoring_contract_tests() -> None:
    workflow = MODULE_WORKFLOW.read_text(encoding="utf-8")

    assert "  pull_request:\n  push:\n" in workflow
    assert "run: make check" in workflow
    makefile = (REPOSITORY_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "check: encoder-check" in makefile
    assert "uv run --frozen --extra models --extra evaluation pytest" in makefile


def test_systemd_collector_has_a_bounded_execution_time() -> None:
    service = SYSTEMD_SERVICE.read_text(encoding="utf-8")

    assert "TimeoutStartSec=45s" in service
    assert "Environment=OPENMETRICS_QUERY_WINDOW_SECONDS=300" in service
    assert "Environment=OPENMETRICS_QUERY_MAX_BYTES=5242880" in service


def test_container_systemd_collector_uses_an_external_cli_wrapper() -> None:
    service = CONTAINER_SYSTEMD_SERVICE.read_text(encoding="utf-8")

    assert "Environment=MAGEOS_CLI=@@MAGENTO_CLI_WRAPPER@@" in service
    assert "User=@@SUPERVISOR_USER@@" in service
    assert "Group=@@SUPERVISOR_GROUP@@" in service
    assert "TimeoutStartSec=45s" in service
    assert "NoNewPrivileges=true" in service
    assert "docker exec" not in service


def test_prometheus_example_scrapes_only_the_same_host_exporter() -> None:
    config = yaml.safe_load(PROMETHEUS_CONFIG.read_text(encoding="utf-8"))

    assert config["global"] == {
        "scrape_interval": "30s",
        "evaluation_interval": "30s",
    }
    assert config["rule_files"] == ["/etc/prometheus/rules/mageos-opensearch-hybrid.yml"]
    assert config["scrape_configs"] == [
        {
            "job_name": "mageos-opensearch-hybrid",
            "scrape_interval": "30s",
            "scrape_timeout": "10s",
            "scheme": "http",
            "static_configs": [{"targets": ["127.0.0.1:9100"]}],
        }
    ]


def test_prometheus_validator_uses_a_digest_pinned_tool(tmp_path: Path) -> None:
    arguments_path = tmp_path / "promtool-arguments.txt"
    promtool = tmp_path / "promtool"
    promtool.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$*" >>"${PROMTOOL_ARGUMENTS_PATH}"\n'
        "if [[ \"${1:-}\" == '--version' ]]; then printf '%s\\n' 'promtool fixture'; fi\n",
        encoding="utf-8",
    )
    promtool.chmod(0o755)
    digest = hashlib.sha256(promtool.read_bytes()).hexdigest()
    environment = os.environ.copy()
    environment.update(
        {
            "PROMTOOL": str(promtool),
            "PROMTOOL_SHA256": digest,
            "PROMTOOL_ARGUMENTS_PATH": str(arguments_path),
        }
    )

    result = subprocess.run(
        [str(PROMETHEUS_VALIDATOR)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0
    assert result.stdout == "promtool fixture\n"
    assert arguments_path.read_text(encoding="utf-8").splitlines() == [
        "--version",
        f"check rules {ALERT_RULES}",
        f"check config {PROMETHEUS_CONFIG}",
        f"test rules {PROMETHEUS_RULE_TESTS.name}",
    ]


def test_prometheus_validator_rejects_an_unpinned_tool(tmp_path: Path) -> None:
    promtool = tmp_path / "promtool"
    promtool.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    promtool.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "PROMTOOL": str(promtool),
            "PROMTOOL_SHA256": "0" * 64,
        }
    )

    result = subprocess.run(
        [str(PROMETHEUS_VALIDATOR)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "digest does not match" in result.stderr
