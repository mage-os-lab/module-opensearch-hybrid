<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Operations;

class SimilarityTelemetry
{
    private const STATES = ['not_checked', 'closed', 'open'];

    public function __construct(
        private readonly \Psr\Log\LoggerInterface $logger
    ) {
    }

    public function emit(
        int $storeId,
        ?int $generationId,
        string $thresholdProfileId,
        array $result,
        float $durationMs,
        string $circuitState,
        ?\Throwable $failure = null
    ): void {
        $context = [
            'schema_version' => 1,
            'event' => 'radial_similarity',
            'store_id' => $storeId,
            'generation_id' => $generationId,
            'threshold_profile_id' => $thresholdProfileId,
            'outcome' => (bool)$result['fallback_used'] ? 'fallback' : 'success',
            'fallback_reason' => $result['fallback_reason'],
            'result_count' => count($result['items']),
            'empty' => $result['items'] === [],
            'duration_ms' => round(max(0.0, $durationMs), 3),
            'circuit_state' => in_array($circuitState, self::STATES, true) ? $circuitState : 'not_checked',
        ];
        if ($failure !== null) {
            $context['error_class'] = $failure::class;
        }
        try {
            $this->logger->info('OpenSearch Hybrid radial similarity telemetry.', $context);
        } catch (\Throwable) {
            return;
        }
    }
}
