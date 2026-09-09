<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Operations;

class SimilarityTelemetryTest extends \PHPUnit\Framework\TestCase
{
    public function testEmitsBoundedPrivacySafeRadialMetrics(): void
    {
        $logger = $this->createMock(\Psr\Log\LoggerInterface::class);
        $logger->expects($this->once())
            ->method('info')
            ->with(
                'OpenSearch Hybrid radial similarity telemetry.',
                $this->callback(static function (array $context): bool {
                    return $context['event'] === 'radial_similarity'
                        && $context['store_id'] === 2
                        && $context['generation_id'] === 7
                        && $context['threshold_profile_id'] === 'substitute-v1'
                        && $context['outcome'] === 'fallback'
                        && $context['fallback_reason'] === 'query_failed'
                        && $context['result_count'] === 2
                        && !$context['empty']
                        && $context['duration_ms'] === 12.346
                        && $context['circuit_state'] === 'open'
                        && $context['error_class'] === \RuntimeException::class
                        && !str_contains(json_encode($context, JSON_THROW_ON_ERROR), 'secret response');
                })
            );
        $telemetry = new \MageOS\OpenSearchHybrid\Model\Operations\SimilarityTelemetry($logger);

        $telemetry->emit(
            2,
            7,
            'substitute-v1',
            [
                'items' => [
                    ['product_id' => 9, 'score' => null],
                    ['product_id' => 8, 'score' => null],
                ],
                'fallback_used' => true,
                'fallback_reason' => 'query_failed',
            ],
            12.3456,
            'open',
            new \RuntimeException('secret response')
        );
    }
}
