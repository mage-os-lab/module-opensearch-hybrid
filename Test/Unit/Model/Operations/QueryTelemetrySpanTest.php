<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Operations;

class QueryTelemetrySpanTest extends \PHPUnit\Framework\TestCase
{
    public function testEmitsOnePrivacySafeStructuredEventWithComponentTimings(): void
    {
        $logger = $this->createMock(\Psr\Log\LoggerInterface::class);
        $logger->expects($this->once())
            ->method('info')
            ->with(
                'OpenSearch Hybrid query telemetry.',
                $this->callback(static function (array $context): bool {
                    return $context['schema_version'] === 1
                        && $context['event'] === 'query_route'
                        && $context['store_id'] === 2
                        && $context['request_name'] === 'graphql_product_search'
                        && $context['route'] === 'hybrid'
                        && $context['reason'] === 'eligible'
                        && $context['generation_id'] === 3
                        && is_float($context['duration_ms'])
                        && $context['duration_ms'] >= 0.0
                        && $context['components']['query_encoder']['outcome'] === 'success'
                        && is_float($context['components']['query_encoder']['duration_ms'])
                        && !array_key_exists('query', $context)
                        && !str_contains(json_encode($context, JSON_THROW_ON_ERROR), 'compact dining table');
                })
            );
        $span = new \MageOS\OpenSearchHybrid\Model\Operations\QueryTelemetrySpan(
            $logger,
            2,
            'graphql_product_search'
        );

        $result = $span->measure('query_encoder', static fn (): string => 'vector');
        $span->route('hybrid', 'eligible', 3);
        $span->finish();

        self::assertSame('vector', $result);
    }

    public function testRecordsFailureClassWithoutRecordingTheFailureMessage(): void
    {
        $logger = $this->createMock(\Psr\Log\LoggerInterface::class);
        $logger->expects($this->once())
            ->method('info')
            ->with(
                'OpenSearch Hybrid query telemetry.',
                $this->callback(static function (array $context): bool {
                    $encoded = json_encode($context, JSON_THROW_ON_ERROR);

                    return $context['route'] === 'native_fallback'
                        && $context['reason'] === 'hybrid_error'
                        && $context['error_class'] === \RuntimeException::class
                        && $context['components']['query_encoder']['outcome'] === 'failure'
                        && $context['components']['query_encoder']['error_class'] === \RuntimeException::class
                        && !str_contains($encoded, 'sensitive query fragment');
                })
            );
        $span = new \MageOS\OpenSearchHybrid\Model\Operations\QueryTelemetrySpan(
            $logger,
            2,
            'quick_search_container'
        );
        $failure = new \RuntimeException('sensitive query fragment');

        try {
            $span->measure('query_encoder', static function () use ($failure): never {
                throw $failure;
            });
        } catch (\RuntimeException $caught) {
            self::assertSame($failure, $caught);
        }

        $span->route('native_fallback', 'hybrid_error', 3);
        $span->finish($failure);
    }

    public function testPreservesABoundedIneligibilityReason(): void
    {
        $logger = $this->createMock(\Psr\Log\LoggerInterface::class);
        $logger->expects($this->once())
            ->method('info')
            ->with(
                'OpenSearch Hybrid query telemetry.',
                $this->callback(static function (array $context): bool {
                    return $context['route'] === 'native'
                        && $context['reason'] === 'unsupported_sort'
                        && !array_key_exists('query', $context);
                })
            );
        $span = new \MageOS\OpenSearchHybrid\Model\Operations\QueryTelemetrySpan(
            $logger,
            2,
            'quick_search_container'
        );

        $span->route('native', 'unsupported_sort');
        $span->finish();
    }
}
