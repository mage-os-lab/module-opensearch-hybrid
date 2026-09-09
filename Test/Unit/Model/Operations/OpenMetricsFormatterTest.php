<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Operations;

class OpenMetricsFormatterTest extends \PHPUnit\Framework\TestCase
{
    public function testFormatsDurableSnapshotAsBoundedOpenMetrics(): void
    {
        $snapshot = [
            'schema_version' => 1,
            'generated_at' => '2026-08-27T12:00:00+00:00',
            'readiness' => [[
                'store_id' => 2,
                'activation_mode' => 'HYBRID',
                'active_generation_id' => 4,
                'ready' => true,
                'required_watermark' => 444,
                'native_watermark' => 444,
                'hybrid_watermark' => 444,
                'native_lag' => 0,
                'hybrid_lag' => 0,
                'state_age_seconds' => 12,
                'cache_version' => 9,
            ]],
            'generations' => [[
                'generation_id' => 4,
                'store_id' => 2,
                'state' => 'READY',
                'active_for_store' => true,
                'coverage_total' => 50000,
                'coverage_complete' => 50000,
                'coverage_failed' => 0,
                'coverage_percent' => 100.0,
                'journal_lag' => 0,
                'contract_current' => true,
                'seeding_state' => 'COMPLETED',
                'refresh_state' => 'RESTORED',
            ]],
            'work' => [[
                'store_id' => 2,
                'lane' => 'EMBEDDING',
                'state' => 'DEAD',
                'jobs' => 2,
                'attempts' => 5,
                'manual_replays' => 1,
                'oldest_age_seconds' => 30,
            ]],
            'subscriptions' => [[
                'subscription_id' => 7,
                'store_id' => 2,
                'healthy' => true,
                'enabled' => true,
                'trace_count' => 15,
                'trace_success_count' => 14,
                'trace_failure_count' => 1,
                'trace_unresolved_failure_count' => 0,
                'oldest_trace_age_seconds' => 60,
            ]],
            'async_event_trace_cleanup' => [
                'enabled' => true,
                'retention_days' => 30,
                'bounded' => true,
                'oldest_module_trace_age_seconds' => 60,
            ],
            'activation_events' => [[
                'store_id' => 2,
                'mode' => 'HYBRID',
                'status' => 'COMPLETED',
                'events' => 3,
            ]],
            'cleanup_events' => [[
                'store_id' => 2,
                'status' => 'COMPLETED',
                'events' => 1,
            ]],
            'alerts' => [[
                'severity' => 'high',
                'code' => 'dead_letter_nonzero',
                'store_id' => 2,
                'generation_id' => null,
                'detail' => 'Do not export this detail.',
            ]],
        ];

        $output = (new \MageOS\OpenSearchHybrid\Model\Operations\OpenMetricsFormatter())->format($snapshot);

        self::assertStringContainsString(
            'mageos_opensearch_hybrid_readiness{store_id="2",activation_mode="HYBRID"} 1',
            $output
        );
        self::assertStringContainsString(
            'mageos_opensearch_hybrid_generation_coverage_total{store_id="2",generation_id="4",state="READY"} 50000',
            $output
        );
        self::assertStringContainsString(
            'mageos_opensearch_hybrid_work_jobs{store_id="2",lane="EMBEDDING",state="DEAD"} 2',
            $output
        );
        self::assertStringContainsString(
            'mageos_opensearch_hybrid_subscription_healthy{store_id="2",subscription_id="7"} 1',
            $output
        );
        self::assertStringContainsString(
            'mageos_opensearch_hybrid_alert{severity="high",code="dead_letter_nonzero",store_id="2",generation_id="none"} 1',
            $output
        );
        self::assertStringNotContainsString('Do not export this detail.', $output);
        self::assertStringNotContainsString('physical_index', $output);
        self::assertStringEndsWith("# EOF\n", $output);
    }

    public function testEscapesLabelValues(): void
    {
        $snapshot = [
            'schema_version' => 1,
            'generated_at' => '2026-08-27T12:00:00+00:00',
            'readiness' => [],
            'generations' => [],
            'work' => [[
                'store_id' => 2,
                'lane' => "EMBED\\DING\nnext",
                'state' => 'WAIT"ING',
                'jobs' => 1,
                'attempts' => 1,
                'manual_replays' => 0,
                'oldest_age_seconds' => null,
            ]],
            'subscriptions' => [],
            'async_event_trace_cleanup' => [],
            'activation_events' => [],
            'cleanup_events' => [],
            'alerts' => [],
        ];

        $output = (new \MageOS\OpenSearchHybrid\Model\Operations\OpenMetricsFormatter())->format($snapshot);

        self::assertStringContainsString(
            'lane="EMBED\\\\DING\\nnext",state="WAIT\\"ING"',
            $output
        );
    }

    public function testAggregatesAlertsWithTheSameControlledIdentity(): void
    {
        $alert = [
            'severity' => 'high',
            'code' => 'dead_letter_nonzero',
            'store_id' => 2,
            'generation_id' => null,
            'detail' => 'Lane-specific detail is deliberately omitted.',
        ];
        $snapshot = [
            'schema_version' => 1,
            'generated_at' => '2026-08-27T12:00:00+00:00',
            'readiness' => [],
            'generations' => [],
            'work' => [],
            'subscriptions' => [],
            'async_event_trace_cleanup' => [],
            'activation_events' => [],
            'cleanup_events' => [],
            'alerts' => [$alert, $alert],
        ];

        $output = (new \MageOS\OpenSearchHybrid\Model\Operations\OpenMetricsFormatter())->format($snapshot);
        $sample = 'mageos_opensearch_hybrid_alert{severity="high",code="dead_letter_nonzero",'
            . 'store_id="2",generation_id="none"}';

        self::assertSame(1, substr_count($output, $sample));
        self::assertStringContainsString($sample . ' 2', $output);
    }

    public function testFormatsBoundedQueryTelemetryWithoutSensitiveFields(): void
    {
        $snapshot = [
            'schema_version' => 1,
            'generated_at' => '2026-08-31T21:05:00+00:00',
            'alerts' => [],
        ];
        $querySnapshot = [
            'schema_version' => 1,
            'generated_timestamp_seconds' => 1788210300,
            'window_seconds' => 300,
            'log_available' => true,
            'window_complete' => true,
            'bytes_scanned' => 2048,
            'records_rejected' => 1,
            'routes' => [[
                'store_id' => 1,
                'request_name' => 'quick_search_container',
                'route' => 'native_fallback',
                'reason' => 'hybrid_error',
                'requests' => 2,
            ]],
            'requests' => [[
                'store_id' => 1,
                'request_name' => 'quick_search_container',
                'requests' => 20,
                'fallback_ratio' => 0.1,
            ]],
            'latency' => [[
                'store_id' => 1,
                'request_name' => 'quick_search_container',
                'route' => 'hybrid',
                'observations' => 18,
                'p95_duration_ms' => 120.5,
            ]],
            'components' => [[
                'store_id' => 1,
                'request_name' => 'quick_search_container',
                'component' => 'query_encoder',
                'outcome' => 'success',
                'observations' => 18,
                'p95_duration_ms' => 20.5,
            ]],
            'raw_query' => 'never export this',
        ];

        $output = (new \MageOS\OpenSearchHybrid\Model\Operations\OpenMetricsFormatter())
            ->format($snapshot, $querySnapshot);

        self::assertStringContainsString(
            'mageos_opensearch_hybrid_query_collection_enabled 1',
            $output
        );
        self::assertStringContainsString(
            'mageos_opensearch_hybrid_query_window_fallback_ratio'
                . '{store_id="1",request_name="quick_search_container"} 0.1',
            $output
        );
        self::assertStringContainsString(
            'mageos_opensearch_hybrid_query_window_p95_duration_milliseconds'
                . '{store_id="1",request_name="quick_search_container",route="hybrid"} 120.5',
            $output
        );
        self::assertStringContainsString(
            'mageos_opensearch_hybrid_query_component_window_p95_duration_milliseconds'
                . '{store_id="1",request_name="quick_search_container",component="query_encoder",outcome="success"} 20.5',
            $output
        );
        self::assertStringNotContainsString('never export this', $output);
        self::assertStringEndsWith("# EOF\n", $output);
    }
}
