<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Operations;

class OpenMetricsFormatter
{
    private const PREFIX = 'mageos_opensearch_hybrid_';

    public function format(array $snapshot, ?array $querySnapshot = null): string
    {
        $lines = [];
        $this->family($lines, 'snapshot_schema_version', 'Operations snapshot schema version.');
        $this->sample($lines, 'snapshot_schema_version', [], $this->number($snapshot['schema_version'] ?? 0));
        $this->family($lines, 'snapshot_generated_timestamp_seconds', 'Operations snapshot generation time.');
        $generatedAt = strtotime((string)($snapshot['generated_at'] ?? ''));
        if ($generatedAt !== false) {
            $this->sample($lines, 'snapshot_generated_timestamp_seconds', [], $generatedAt);
        }

        $this->readiness($lines, $this->rows($snapshot, 'readiness'));
        $this->generations($lines, $this->rows($snapshot, 'generations'));
        $this->work($lines, $this->rows($snapshot, 'work'));
        $this->subscriptions($lines, $this->rows($snapshot, 'subscriptions'));
        $this->traceCleanup($lines, $snapshot['async_event_trace_cleanup'] ?? []);
        $this->events($lines, $this->rows($snapshot, 'activation_events'), 'activation');
        $this->events($lines, $this->rows($snapshot, 'cleanup_events'), 'cleanup');
        $this->alerts($lines, $this->rows($snapshot, 'alerts'));
        $this->family($lines, 'query_collection_enabled', 'Whether bounded query telemetry collection is enabled.');
        $this->sample($lines, 'query_collection_enabled', [], $this->boolean($querySnapshot !== null));
        if ($querySnapshot !== null) {
            $this->queryTelemetry($lines, $querySnapshot);
        }
        $lines[] = '# EOF';

        return implode("\n", $lines) . "\n";
    }

    private function readiness(array &$lines, array $rows): void
    {
        foreach ([
            'readiness' => 'Whether the store hybrid readiness latch is open.',
            'active_generation' => 'Current active hybrid generation ID, or zero for native routing.',
            'required_watermark' => 'Required correctness journal watermark.',
            'native_watermark' => 'Native search correctness watermark.',
            'hybrid_watermark' => 'Hybrid search correctness watermark.',
            'native_lag' => 'Native search correctness lag.',
            'hybrid_lag' => 'Hybrid search correctness lag.',
            'readiness_state_age_seconds' => 'Age of the durable readiness state.',
            'readiness_cache_version' => 'Durable readiness cache version.',
        ] as $name => $help) {
            $this->family($lines, $name, $help);
        }
        foreach ($rows as $row) {
            $labels = [
                'store_id' => $row['store_id'] ?? 0,
                'activation_mode' => $row['activation_mode'] ?? 'UNKNOWN',
            ];
            $this->sample($lines, 'readiness', $labels, $this->boolean($row['ready'] ?? false));
            $this->sample($lines, 'active_generation', $labels, $this->number($row['active_generation_id'] ?? 0));
            foreach ([
                'required_watermark' => 'required_watermark',
                'native_watermark' => 'native_watermark',
                'hybrid_watermark' => 'hybrid_watermark',
                'native_lag' => 'native_lag',
                'hybrid_lag' => 'hybrid_lag',
                'readiness_cache_version' => 'cache_version',
            ] as $metric => $field) {
                $this->sample($lines, $metric, $labels, $this->number($row[$field] ?? 0));
            }
            $this->optionalSample($lines, 'readiness_state_age_seconds', $labels, $row['state_age_seconds'] ?? null);
        }
    }

    private function generations(array &$lines, array $rows): void
    {
        foreach ([
            'generation_active' => 'Whether the generation is active for its store.',
            'generation_coverage_total' => 'Total products in the generation coverage boundary.',
            'generation_coverage_complete' => 'Products with completed generation coverage.',
            'generation_coverage_failed' => 'Products with failed generation coverage.',
            'generation_coverage_percent' => 'Completed generation coverage percentage.',
            'generation_journal_lag' => 'Generation change-journal lag.',
            'generation_contract_current' => 'Whether generation contracts match current code.',
            'generation_refresh_restored' => 'Whether the normal index refresh interval is restored.',
        ] as $name => $help) {
            $this->family($lines, $name, $help);
        }
        foreach ($rows as $row) {
            $labels = [
                'store_id' => $row['store_id'] ?? 0,
                'generation_id' => $row['generation_id'] ?? 0,
                'state' => $row['state'] ?? 'UNKNOWN',
            ];
            $this->sample($lines, 'generation_active', $labels, $this->boolean($row['active_for_store'] ?? false));
            foreach ([
                'generation_coverage_total' => 'coverage_total',
                'generation_coverage_complete' => 'coverage_complete',
                'generation_coverage_failed' => 'coverage_failed',
                'generation_coverage_percent' => 'coverage_percent',
                'generation_journal_lag' => 'journal_lag',
            ] as $metric => $field) {
                $this->sample($lines, $metric, $labels, $this->number($row[$field] ?? 0));
            }
            $this->sample($lines, 'generation_contract_current', $labels, $this->boolean($row['contract_current'] ?? false));
            $this->sample(
                $lines,
                'generation_refresh_restored',
                $labels,
                $this->boolean(($row['refresh_state'] ?? null) === 'RESTORED')
            );
        }
    }

    private function work(array &$lines, array $rows): void
    {
        foreach ([
            'work_jobs' => 'Durable jobs by lane and state.',
            'work_attempts' => 'Durable job attempts by lane and state.',
            'work_manual_replays' => 'Manual job replays by lane and state.',
            'work_oldest_age_seconds' => 'Age of the oldest durable job by lane and state.',
        ] as $name => $help) {
            $this->family($lines, $name, $help);
        }
        foreach ($rows as $row) {
            $labels = [
                'store_id' => $row['store_id'] ?? 0,
                'lane' => $row['lane'] ?? 'UNKNOWN',
                'state' => $row['state'] ?? 'UNKNOWN',
            ];
            foreach ([
                'work_jobs' => 'jobs',
                'work_attempts' => 'attempts',
                'work_manual_replays' => 'manual_replays',
            ] as $metric => $field) {
                $this->sample($lines, $metric, $labels, $this->number($row[$field] ?? 0));
            }
            $this->optionalSample($lines, 'work_oldest_age_seconds', $labels, $row['oldest_age_seconds'] ?? null);
        }
    }

    private function subscriptions(array &$lines, array $rows): void
    {
        foreach ([
            'subscription_healthy' => 'Whether the module-owned Async Events subscription is healthy.',
            'subscription_enabled' => 'Whether the module-owned Async Events subscription is enabled.',
            'subscription_trace_total' => 'Retained Async Events delivery traces.',
            'subscription_trace_success_total' => 'Retained successful Async Events delivery attempts.',
            'subscription_trace_failure_total' => 'Retained failed Async Events delivery attempts.',
            'subscription_trace_unresolved_failure_total' => 'Async Events traces whose latest attempt failed.',
            'subscription_oldest_trace_age_seconds' => 'Age of the oldest retained module trace.',
        ] as $name => $help) {
            $this->family($lines, $name, $help);
        }
        foreach ($rows as $row) {
            $labels = [
                'store_id' => $row['store_id'] ?? 0,
                'subscription_id' => $row['subscription_id'] ?? 0,
            ];
            $this->sample($lines, 'subscription_healthy', $labels, $this->boolean($row['healthy'] ?? false));
            $this->sample($lines, 'subscription_enabled', $labels, $this->boolean($row['enabled'] ?? false));
            foreach ([
                'subscription_trace_total' => 'trace_count',
                'subscription_trace_success_total' => 'trace_success_count',
                'subscription_trace_failure_total' => 'trace_failure_count',
                'subscription_trace_unresolved_failure_total' => 'trace_unresolved_failure_count',
            ] as $metric => $field) {
                $this->sample($lines, $metric, $labels, $this->number($row[$field] ?? 0));
            }
            $this->optionalSample(
                $lines,
                'subscription_oldest_trace_age_seconds',
                $labels,
                $row['oldest_trace_age_seconds'] ?? null
            );
        }
    }

    private function traceCleanup(array &$lines, mixed $cleanup): void
    {
        $cleanup = is_array($cleanup) ? $cleanup : [];
        foreach ([
            'async_event_trace_cleanup_enabled' => 'Whether Async Events trace cleanup is enabled.',
            'async_event_trace_retention_bounded' => 'Whether Async Events trace retention is within the module bound.',
            'async_event_trace_retention_days' => 'Configured Async Events trace retention days.',
            'async_event_oldest_module_trace_age_seconds' => 'Age of the oldest retained module trace.',
        ] as $name => $help) {
            $this->family($lines, $name, $help);
        }
        $this->sample($lines, 'async_event_trace_cleanup_enabled', [], $this->boolean($cleanup['enabled'] ?? false));
        $this->sample($lines, 'async_event_trace_retention_bounded', [], $this->boolean($cleanup['bounded'] ?? false));
        $this->sample($lines, 'async_event_trace_retention_days', [], $this->number($cleanup['retention_days'] ?? 0));
        $this->optionalSample(
            $lines,
            'async_event_oldest_module_trace_age_seconds',
            [],
            $cleanup['oldest_module_trace_age_seconds'] ?? null
        );
    }

    private function events(array &$lines, array $rows, string $kind): void
    {
        $name = $kind . '_events_total';
        $this->family($lines, $name, ucfirst($kind) . ' audit events.', 'counter');
        foreach ($rows as $row) {
            $labels = [
                'store_id' => $row['store_id'] ?? 0,
                'status' => $row['status'] ?? 'UNKNOWN',
            ];
            if ($kind === 'activation') {
                $labels['mode'] = $row['mode'] ?? 'UNKNOWN';
            }
            $this->sample($lines, $name, $labels, $this->number($row['events'] ?? 0));
        }
    }

    private function alerts(array &$lines, array $rows): void
    {
        $this->family($lines, 'alert', 'Current durable operations alert by controlled identity.');
        $aggregated = [];
        foreach ($rows as $row) {
            $labels = [
                'severity' => $row['severity'] ?? 'unknown',
                'code' => $row['code'] ?? 'unknown',
                'store_id' => $row['store_id'] ?? 'none',
                'generation_id' => $row['generation_id'] ?? 'none',
            ];
            $identity = json_encode($labels, JSON_THROW_ON_ERROR);
            if (!isset($aggregated[$identity])) {
                $aggregated[$identity] = ['labels' => $labels, 'count' => 0];
            }
            $aggregated[$identity]['count']++;
        }
        foreach ($aggregated as $alert) {
            $this->sample($lines, 'alert', $alert['labels'], $alert['count']);
        }
    }

    private function queryTelemetry(array &$lines, array $snapshot): void
    {
        foreach ([
            'query_snapshot_schema_version' => 'Query telemetry snapshot schema version.',
            'query_snapshot_generated_timestamp_seconds' => 'Query telemetry snapshot generation time.',
            'query_window_seconds' => 'Query telemetry aggregation window.',
            'query_log_available' => 'Whether the query telemetry log is available.',
            'query_window_complete' => 'Whether the bounded log read covers the full aggregation window.',
            'query_log_bytes_scanned' => 'Query telemetry log bytes scanned for this snapshot.',
            'query_log_records_rejected' => 'Query telemetry records rejected from this snapshot.',
        ] as $name => $help) {
            $this->family($lines, $name, $help);
        }
        $this->sample($lines, 'query_snapshot_schema_version', [], $this->number($snapshot['schema_version'] ?? 0));
        $this->sample(
            $lines,
            'query_snapshot_generated_timestamp_seconds',
            [],
            $this->number($snapshot['generated_timestamp_seconds'] ?? 0)
        );
        $this->sample($lines, 'query_window_seconds', [], $this->number($snapshot['window_seconds'] ?? 0));
        $this->sample($lines, 'query_log_available', [], $this->boolean($snapshot['log_available'] ?? false));
        $this->sample($lines, 'query_window_complete', [], $this->boolean($snapshot['window_complete'] ?? false));
        $this->sample($lines, 'query_log_bytes_scanned', [], $this->number($snapshot['bytes_scanned'] ?? 0));
        $this->sample($lines, 'query_log_records_rejected', [], $this->number($snapshot['records_rejected'] ?? 0));

        $this->family($lines, 'query_window_routes', 'Queries in the current window by controlled route identity.');
        foreach ($this->rows($snapshot, 'routes') as $row) {
            $this->sample($lines, 'query_window_routes', [
                'store_id' => $row['store_id'] ?? 0,
                'request_name' => $row['request_name'] ?? 'unknown',
                'route' => $row['route'] ?? 'unknown',
                'reason' => $row['reason'] ?? 'unknown',
            ], $this->number($row['requests'] ?? 0));
        }

        $this->family($lines, 'query_window_requests', 'Queries in the current window by request identity.');
        $this->family($lines, 'query_window_fallback_ratio', 'Native fallback ratio in the current query window.');
        foreach ($this->rows($snapshot, 'requests') as $row) {
            $labels = [
                'store_id' => $row['store_id'] ?? 0,
                'request_name' => $row['request_name'] ?? 'unknown',
            ];
            $this->sample($lines, 'query_window_requests', $labels, $this->number($row['requests'] ?? 0));
            $this->sample(
                $lines,
                'query_window_fallback_ratio',
                $labels,
                $this->number($row['fallback_ratio'] ?? 0)
            );
        }

        $this->family(
            $lines,
            'query_window_latency_observations',
            'Latency observations in the current query window.'
        );
        $this->family(
            $lines,
            'query_window_p95_duration_milliseconds',
            'P95 end-to-end query duration in the current window.'
        );
        foreach ($this->rows($snapshot, 'latency') as $row) {
            $labels = [
                'store_id' => $row['store_id'] ?? 0,
                'request_name' => $row['request_name'] ?? 'unknown',
                'route' => $row['route'] ?? 'unknown',
            ];
            $this->sample(
                $lines,
                'query_window_latency_observations',
                $labels,
                $this->number($row['observations'] ?? 0)
            );
            $this->sample(
                $lines,
                'query_window_p95_duration_milliseconds',
                $labels,
                $this->number($row['p95_duration_ms'] ?? 0)
            );
        }

        $this->family(
            $lines,
            'query_component_window_observations',
            'Component latency observations in the current query window.'
        );
        $this->family(
            $lines,
            'query_component_window_p95_duration_milliseconds',
            'P95 component duration in the current query window.'
        );
        foreach ($this->rows($snapshot, 'components') as $row) {
            $labels = [
                'store_id' => $row['store_id'] ?? 0,
                'request_name' => $row['request_name'] ?? 'unknown',
                'component' => $row['component'] ?? 'unknown',
                'outcome' => $row['outcome'] ?? 'unknown',
            ];
            $this->sample(
                $lines,
                'query_component_window_observations',
                $labels,
                $this->number($row['observations'] ?? 0)
            );
            $this->sample(
                $lines,
                'query_component_window_p95_duration_milliseconds',
                $labels,
                $this->number($row['p95_duration_ms'] ?? 0)
            );
        }
    }

    private function family(array &$lines, string $name, string $help, string $type = 'gauge'): void
    {
        $metric = self::PREFIX . $name;
        $lines[] = '# HELP ' . $metric . ' ' . $help;
        $lines[] = '# TYPE ' . $metric . ' ' . $type;
    }

    private function optionalSample(array &$lines, string $name, array $labels, mixed $value): void
    {
        if ($value !== null) {
            $this->sample($lines, $name, $labels, $this->number($value));
        }
    }

    private function sample(array &$lines, string $name, array $labels, int|float $value): void
    {
        $labelSet = [];
        foreach ($labels as $label => $labelValue) {
            $labelSet[] = $label . '="' . $this->escapeLabel((string)$labelValue) . '"';
        }
        $lines[] = self::PREFIX . $name
            . ($labelSet === [] ? '' : '{' . implode(',', $labelSet) . '}')
            . ' ' . $this->formatNumber($value);
    }

    private function escapeLabel(string $value): string
    {
        return str_replace(["\\", "\n", '"'], ["\\\\", '\\n', '\\"'], $value);
    }

    private function boolean(mixed $value): int
    {
        return $value === true || $value === 1 || $value === '1' ? 1 : 0;
    }

    private function number(mixed $value): int|float
    {
        if (!is_numeric($value)) {
            return 0;
        }
        $number = $value + 0;

        return is_float($number) && !is_finite($number) ? 0 : $number;
    }

    private function formatNumber(int|float $value): string
    {
        return is_int($value) ? (string)$value : sprintf('%.15g', $value);
    }

    private function rows(array $snapshot, string $key): array
    {
        $rows = $snapshot[$key] ?? [];
        if (!is_array($rows)) {
            return [];
        }
        $rows = array_filter($rows, 'is_array');

        return array_values($rows);
    }
}
