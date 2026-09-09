<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Operations;

use Throwable;

class QueryTelemetrySpan
{
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
    private const COMPONENTS = ['native_candidates', 'query_encoder', 'hybrid_opensearch'];

    private int $startedAt;
    private array $components = [];
    private ?string $route = null;
    private ?string $reason = null;
    private ?int $generationId = null;
    private bool $finished = false;

    public function __construct(
        private readonly \Psr\Log\LoggerInterface $logger,
        private readonly int $storeId,
        private readonly string $requestName
    ) {
        $this->startedAt = hrtime(true);
    }

    public function generation(int $generationId): void
    {
        $this->generationId = $generationId;
    }

    public function route(string $route, string $reason, ?int $generationId = null): void
    {
        $this->route = in_array($route, self::ROUTES, true) ? $route : 'native_fallback';
        $this->reason = in_array($reason, self::REASONS, true) ? $reason : 'hybrid_error';
        if ($generationId !== null) {
            $this->generationId = $generationId;
        }
    }

    public function measure(string $component, callable $operation): mixed
    {
        $component = in_array($component, self::COMPONENTS, true) ? $component : 'other';
        $startedAt = hrtime(true);
        try {
            $result = $operation();
            $this->components[$component] = [
                'outcome' => 'success',
                'duration_ms' => $this->elapsedMilliseconds($startedAt),
            ];

            return $result;
        } catch (Throwable $throwable) {
            $this->components[$component] = [
                'outcome' => 'failure',
                'duration_ms' => $this->elapsedMilliseconds($startedAt),
                'error_class' => $throwable::class,
            ];
            throw $throwable;
        }
    }

    public function finish(?Throwable $failure = null): void
    {
        if ($this->finished) {
            return;
        }
        $this->finished = true;
        $context = [
            'schema_version' => 1,
            'event' => 'query_route',
            'store_id' => $this->storeId,
            'request_name' => $this->requestName,
            'route' => $this->route ?? 'native_fallback',
            'reason' => $this->reason ?? 'hybrid_error',
            'generation_id' => $this->generationId,
            'duration_ms' => $this->elapsedMilliseconds($this->startedAt),
            'components' => $this->components,
        ];
        if ($failure !== null) {
            $context['error_class'] = $failure::class;
        }
        $this->emitWithoutAffectingSearch($context);
    }

    private function elapsedMilliseconds(int $startedAt): float
    {
        return round(max(0, hrtime(true) - $startedAt) / 1_000_000, 3);
    }

    private function emitWithoutAffectingSearch(array $context): void
    {
        try {
            $this->logger->info('OpenSearch Hybrid query telemetry.', $context);
        } catch (Throwable) {
            return;
        }
    }
}
