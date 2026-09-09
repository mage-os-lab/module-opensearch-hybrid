<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Similarity;

class SimilarityService implements \MageOS\OpenSearchHybrid\Api\SimilarityServiceInterface
{
    private const MAXIMUM_RESULT_COUNT = 20;

    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Config $config,
        private readonly \MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository $generationRepository,
        private readonly \MageOS\OpenSearchHybrid\Model\Similarity\ThresholdProfileRegistry $profileRegistry,
        private readonly \MageOS\OpenSearchHybrid\Model\Similarity\RadialRequestBuilder $requestBuilder,
        private readonly \MageOS\OpenSearchHybrid\Model\Vector\VectorValidator $vectorValidator,
        private readonly \MageOS\OpenSearchHybrid\Model\OpenSearch\HybridTransport $transport,
        private readonly \MageOS\OpenSearchHybrid\Model\Reliability\CircuitBreaker $circuitBreaker,
        private readonly \MageOS\OpenSearchHybrid\Model\Operations\SimilarityTelemetry $telemetry
    ) {
    }

    public function searchByProduct(
        int $storeId,
        int $productId,
        int $limit,
        string $thresholdProfileId,
        array $fallbackProductIds = []
    ): array {
        $this->assertRequest($storeId, $productId, $limit, $thresholdProfileId);
        $startedAt = hrtime(true);
        $fallbackProductIds = $this->normalizeFallback($fallbackProductIds, $productId, $limit);
        if (!$this->config->isSimilarityEnabled($storeId)) {
            $result = $this->fallback(
                $fallbackProductIds,
                'feature_disabled',
                null,
                $thresholdProfileId
            );
            $this->emit($storeId, $thresholdProfileId, $result, $startedAt, 'not_checked');

            return $result;
        }
        $generation = $this->generationRepository->activeForStore($storeId);
        if ($generation === null || !(bool)$generation['result_contract_accepted']) {
            $result = $this->fallback(
                $fallbackProductIds,
                'active_generation_unavailable',
                null,
                $thresholdProfileId
            );
            $this->emit($storeId, $thresholdProfileId, $result, $startedAt, 'not_checked');

            return $result;
        }
        try {
            $profile = $this->profileRegistry->resolve($thresholdProfileId, $generation, $storeId);
        } catch (\Throwable) {
            $result = $this->fallback(
                $fallbackProductIds,
                'threshold_profile_unavailable',
                (int)$generation['generation_id'],
                $thresholdProfileId
            );
            $this->emit($storeId, $thresholdProfileId, $result, $startedAt, 'not_checked');

            return $result;
        }
        $circuitState = $this->circuitState($storeId);
        if ($circuitState !== 'closed') {
            $result = $this->fallback(
                $fallbackProductIds,
                $circuitState === 'open' ? 'circuit_open' : 'circuit_state_unavailable',
                (int)$generation['generation_id'],
                $thresholdProfileId
            );
            $this->emit($storeId, $thresholdProfileId, $result, $startedAt, $circuitState);

            return $result;
        }
        try {
            $seedResponse = $this->transport->search(
                (string)$generation['physical_index'],
                null,
                $this->requestBuilder->buildSeedRequest($generation, $storeId, $productId)
            );
            $seedVector = $this->seedVector($seedResponse, (int)$generation['dimension']);
            $response = $this->transport->search(
                (string)$generation['physical_index'],
                null,
                $this->requestBuilder->buildRadialRequest(
                    $generation,
                    $storeId,
                    $productId,
                    $limit,
                    $seedVector,
                    $profile
                )
            );
            $items = $this->resultItems($response, $productId, $limit, (float)$profile['min_score']);
            try {
                $this->circuitBreaker->success($storeId, 'similarity');
            } catch (\Throwable) {
                $circuitState = 'not_checked';
            }
        } catch (\Throwable $throwable) {
            $circuitCacheFailed = false;
            try {
                $this->circuitBreaker->failure($storeId, 'similarity');
            } catch (\Throwable) {
                $circuitCacheFailed = true;
            }
            $result = $this->fallback(
                $fallbackProductIds,
                'query_failed',
                (int)$generation['generation_id'],
                $thresholdProfileId
            );
            $this->emit(
                $storeId,
                $thresholdProfileId,
                $result,
                $startedAt,
                $circuitCacheFailed ? 'not_checked' : $this->circuitState($storeId),
                $throwable
            );

            return $result;
        }

        $result = [
            'items' => $items,
            'fallback_used' => false,
            'fallback_reason' => null,
            'generation_id' => (int)$generation['generation_id'],
            'threshold_profile_id' => $thresholdProfileId,
        ];
        $this->emit($storeId, $thresholdProfileId, $result, $startedAt, $circuitState);

        return $result;
    }

    private function seedVector(array $response, int $dimension): array
    {
        $hits = $response['hits']['hits'] ?? null;
        if (!is_array($hits) || count($hits) !== 1 || !is_array($hits[0] ?? null)) {
            throw new \RuntimeException('The similarity seed vector is unavailable.');
        }
        $encoded = $hits[0]['fields']['embedding'][0] ?? null;
        if (!is_string($encoded)) {
            throw new \RuntimeException('The similarity seed vector response is invalid.');
        }

        return $this->vectorValidator->fromBase64Float32($encoded, $dimension);
    }

    private function resultItems(
        array $response,
        int $seedProductId,
        int $limit,
        float $minimumScore
    ): array {
        $hits = $response['hits']['hits'] ?? null;
        if (!is_array($hits)) {
            throw new \RuntimeException('The radial similarity response is invalid.');
        }
        $items = [];
        $seen = [];
        foreach ($hits as $hit) {
            $productId = is_array($hit) ? ($hit['fields']['entity_id'][0] ?? null) : null;
            $score = is_array($hit) ? ($hit['_score'] ?? null) : null;
            if ((!is_int($productId) && !ctype_digit((string)$productId))
                || (!is_int($score) && !is_float($score))
                || !is_finite((float)$score)
                || (float)$score < $minimumScore
                || (int)$productId <= 0
                || (int)$productId === $seedProductId
                || isset($seen[(int)$productId])
            ) {
                throw new \RuntimeException('The radial similarity response contains an invalid hit.');
            }
            $seen[(int)$productId] = true;
            $items[] = [
                'product_id' => (int)$productId,
                'score' => (float)$score,
            ];
        }
        usort($items, static function (array $left, array $right): int {
            $scoreOrder = $right['score'] <=> $left['score'];

            return $scoreOrder !== 0 ? $scoreOrder : $left['product_id'] <=> $right['product_id'];
        });

        return array_slice($items, 0, $limit);
    }

    private function assertRequest(
        int $storeId,
        int $productId,
        int $limit,
        string $thresholdProfileId
    ): void {
        if ($storeId <= 0 || $productId <= 0) {
            throw new \InvalidArgumentException('Similarity requires positive store and product IDs.');
        }
        if ($limit <= 0 || $limit > self::MAXIMUM_RESULT_COUNT) {
            throw new \InvalidArgumentException('Similarity result count is outside its safe range.');
        }
        if (preg_match('/^[a-z0-9][a-z0-9._-]{0,63}$/', $thresholdProfileId) !== 1) {
            throw new \InvalidArgumentException('The similarity threshold profile ID is invalid.');
        }
    }

    private function normalizeFallback(array $productIds, int $seedProductId, int $limit): array
    {
        $normalized = [];
        foreach ($productIds as $productId) {
            if (!is_int($productId) || $productId <= 0) {
                throw new \InvalidArgumentException('Fallback product IDs must be positive integers.');
            }
            if ($productId !== $seedProductId) {
                $normalized[$productId] = $productId;
            }
        }

        return array_slice(array_values($normalized), 0, $limit);
    }

    private function fallback(
        array $productIds,
        string $reason,
        ?int $generationId,
        string $thresholdProfileId
    ): array {
        return [
            'items' => array_map(
                static fn (int $productId): array => ['product_id' => $productId, 'score' => null],
                $productIds
            ),
            'fallback_used' => true,
            'fallback_reason' => $reason,
            'generation_id' => $generationId,
            'threshold_profile_id' => $thresholdProfileId,
        ];
    }

    private function emit(
        int $storeId,
        string $thresholdProfileId,
        array $result,
        int $startedAt,
        string $circuitState,
        ?\Throwable $failure = null
    ): void {
        $this->telemetry->emit(
            $storeId,
            $result['generation_id'],
            $thresholdProfileId,
            $result,
            max(0, hrtime(true) - $startedAt) / 1_000_000,
            $circuitState,
            $failure
        );
    }

    private function circuitState(int $storeId): string
    {
        try {
            return $this->circuitBreaker->isOpen($storeId, 'similarity') ? 'open' : 'closed';
        } catch (\Throwable) {
            return 'not_checked';
        }
    }
}
