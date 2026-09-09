<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Operations;

use InvalidArgumentException;
use JsonException;
use RuntimeException;
use SplFileObject;

class QueryTelemetryMetricsService
{
    private const REQUEST_NAMES = ['quick_search_container', 'graphql_product_search'];
    private const ROUTES = ['native', 'hybrid', 'native_fallback'];
    private const REASONS = [
        'ineligible',
        'disabled',
        'store_excluded',
        'readiness_closed',
        'unsupported_request',
        'invalid_page',
        'unsupported_sort',
        'circuit_open',
        'known_item',
        'no_candidates',
        'eligible',
        'hybrid_error',
    ];
    private const COMPONENTS = ['native_candidates', 'query_encoder', 'hybrid_opensearch', 'other'];
    private const OUTCOMES = ['success', 'failure'];

    public function summarize(
        string $logPath,
        int $windowSeconds,
        int $maximumBytes,
        int $generatedAt
    ): array {
        if ($windowSeconds < 60 || $windowSeconds > 3600) {
            throw new InvalidArgumentException('Query telemetry window must be between 60 and 3600 seconds.');
        }
        if ($maximumBytes < 65536 || $maximumBytes > 52428800) {
            throw new InvalidArgumentException('Query telemetry byte limit must be between 65536 and 52428800 bytes.');
        }
        if (!is_file($logPath)) {
            return $this->emptySnapshot($generatedAt, $windowSeconds);
        }
        if (!is_readable($logPath)) {
            throw new RuntimeException('Query telemetry log is not readable.');
        }

        $file = new SplFileObject($logPath, 'rb');
        $fileSize = (int)$file->getSize();
        $offset = max(0, $fileSize - $maximumBytes);
        if ($file->fseek($offset) !== 0) {
            throw new RuntimeException('Unable to seek within the query telemetry log.');
        }
        $contents = '';
        while (!$file->eof() && strlen($contents) < $maximumBytes) {
            $remaining = $maximumBytes - strlen($contents);
            $chunk = $file->fread(min(65536, $remaining));
            if ($chunk === false) {
                throw new RuntimeException('Unable to read the query telemetry log.');
            }
            if ($chunk === '') {
                break;
            }
            $contents .= $chunk;
        }
        $bytesScanned = strlen($contents);
        if ($offset > 0) {
            $firstLineEnd = strpos($contents, "\n");
            $contents = $firstLineEnd === false ? '' : substr($contents, $firstLineEnd + 1);
        }

        $cutoff = $generatedAt - $windowSeconds;
        $oldestTimestamp = null;
        $rejected = 0;
        $routes = [];
        $requests = [];
        $latency = [];
        $components = [];
        $lines = preg_split('/\r\n|\n|\r/', $contents) ?: [];
        foreach ($lines as $line) {
            if (!str_contains($line, 'OpenSearch Hybrid query telemetry.')) {
                continue;
            }
            $record = $this->parseRecord($line);
            if ($record === null) {
                $rejected++;
                continue;
            }
            $oldestTimestamp = $oldestTimestamp === null
                ? $record['timestamp']
                : min($oldestTimestamp, $record['timestamp']);
            if ($record['timestamp'] < $cutoff || $record['timestamp'] > $generatedAt + 60) {
                continue;
            }

            $routeKey = implode('|', [
                $record['store_id'],
                $record['request_name'],
                $record['route'],
                $record['reason'],
            ]);
            if (!isset($routes[$routeKey])) {
                $routes[$routeKey] = [
                    'store_id' => $record['store_id'],
                    'request_name' => $record['request_name'],
                    'route' => $record['route'],
                    'reason' => $record['reason'],
                    'requests' => 0,
                ];
            }
            $routes[$routeKey]['requests']++;

            $requestKey = implode('|', [$record['store_id'], $record['request_name']]);
            if (!isset($requests[$requestKey])) {
                $requests[$requestKey] = [
                    'store_id' => $record['store_id'],
                    'request_name' => $record['request_name'],
                    'requests' => 0,
                    'fallbacks' => 0,
                ];
            }
            $requests[$requestKey]['requests']++;
            if ($record['route'] === 'native_fallback') {
                $requests[$requestKey]['fallbacks']++;
            }

            $latencyKey = implode('|', [
                $record['store_id'],
                $record['request_name'],
                $record['route'],
            ]);
            if (!isset($latency[$latencyKey])) {
                $latency[$latencyKey] = [
                    'store_id' => $record['store_id'],
                    'request_name' => $record['request_name'],
                    'route' => $record['route'],
                    'durations' => [],
                ];
            }
            $latency[$latencyKey]['durations'][] = $record['duration_ms'];

            foreach ($record['components'] as $component) {
                $componentKey = implode('|', [
                    $record['store_id'],
                    $record['request_name'],
                    $component['component'],
                    $component['outcome'],
                ]);
                if (!isset($components[$componentKey])) {
                    $components[$componentKey] = [
                        'store_id' => $record['store_id'],
                        'request_name' => $record['request_name'],
                        'component' => $component['component'],
                        'outcome' => $component['outcome'],
                        'durations' => [],
                    ];
                }
                $components[$componentKey]['durations'][] = $component['duration_ms'];
            }
        }

        foreach ($requests as &$request) {
            $request['fallback_ratio'] = $request['requests'] === 0
                ? 0.0
                : $request['fallbacks'] / $request['requests'];
            unset($request['fallbacks']);
        }
        unset($request);
        foreach ($latency as &$row) {
            $row['observations'] = count($row['durations']);
            $row['p95_duration_ms'] = $this->percentile95($row['durations']);
            unset($row['durations']);
        }
        unset($row);
        foreach ($components as &$component) {
            $component['observations'] = count($component['durations']);
            $component['p95_duration_ms'] = $this->percentile95($component['durations']);
            unset($component['durations']);
        }
        unset($component);
        ksort($routes);
        ksort($requests);
        ksort($latency);
        ksort($components);

        return [
            'schema_version' => 1,
            'generated_timestamp_seconds' => $generatedAt,
            'window_seconds' => $windowSeconds,
            'log_available' => true,
            'window_complete' => $offset === 0
                || ($oldestTimestamp !== null && $oldestTimestamp <= $cutoff),
            'bytes_scanned' => $bytesScanned,
            'records_rejected' => $rejected,
            'routes' => array_values($routes),
            'requests' => array_values($requests),
            'latency' => array_values($latency),
            'components' => array_values($components),
        ];
    }

    private function parseRecord(string $line): ?array
    {
        if (!preg_match(
            '/^\[([^\]]+)\].*OpenSearch Hybrid query telemetry\. (\{.*\}) \[\]\s*$/',
            $line,
            $matches
        )) {
            return null;
        }
        $timestamp = strtotime($matches[1]);
        if ($timestamp === false) {
            return null;
        }
        try {
            $context = json_decode($matches[2], true, 512, JSON_THROW_ON_ERROR);
        } catch (JsonException) {
            return null;
        }
        if (!is_array($context)
            || ($context['schema_version'] ?? null) !== 1
            || ($context['event'] ?? null) !== 'query_route'
        ) {
            return null;
        }
        $storeId = filter_var(
            $context['store_id'] ?? null,
            FILTER_VALIDATE_INT,
            ['options' => ['min_range' => 1]]
        );
        $requestName = $context['request_name'] ?? null;
        $route = $context['route'] ?? null;
        $reason = $context['reason'] ?? null;
        $duration = $this->duration($context['duration_ms'] ?? null);
        if ($storeId === false
            || !is_string($requestName)
            || !in_array($requestName, self::REQUEST_NAMES, true)
            || !is_string($route)
            || !in_array($route, self::ROUTES, true)
            || !is_string($reason)
            || !in_array($reason, self::REASONS, true)
            || $duration === null
        ) {
            return null;
        }

        $components = [];
        $rawComponents = $context['components'] ?? [];
        if (is_array($rawComponents)) {
            foreach ($rawComponents as $component => $measurement) {
                if (!is_string($component)
                    || !in_array($component, self::COMPONENTS, true)
                    || !is_array($measurement)
                ) {
                    continue;
                }
                $outcome = $measurement['outcome'] ?? null;
                $componentDuration = $this->duration($measurement['duration_ms'] ?? null);
                if (!is_string($outcome)
                    || !in_array($outcome, self::OUTCOMES, true)
                    || $componentDuration === null
                ) {
                    continue;
                }
                $components[] = [
                    'component' => $component,
                    'outcome' => $outcome,
                    'duration_ms' => $componentDuration,
                ];
            }
        }

        return [
            'timestamp' => $timestamp,
            'store_id' => (int)$storeId,
            'request_name' => $requestName,
            'route' => $route,
            'reason' => $reason,
            'duration_ms' => $duration,
            'components' => $components,
        ];
    }

    private function duration(mixed $value): ?float
    {
        if (!is_numeric($value)) {
            return null;
        }
        $duration = (float)$value;

        return is_finite($duration) && $duration >= 0.0 && $duration <= 60000.0
            ? $duration
            : null;
    }

    private function percentile95(array $durations): float
    {
        sort($durations, SORT_NUMERIC);
        $index = max(0, (int)ceil(count($durations) * 0.95) - 1);

        return (float)$durations[$index];
    }

    private function emptySnapshot(int $generatedAt, int $windowSeconds): array
    {
        return [
            'schema_version' => 1,
            'generated_timestamp_seconds' => $generatedAt,
            'window_seconds' => $windowSeconds,
            'log_available' => false,
            'window_complete' => false,
            'bytes_scanned' => 0,
            'records_rejected' => 0,
            'routes' => [],
            'requests' => [],
            'latency' => [],
            'components' => [],
        ];
    }
}
