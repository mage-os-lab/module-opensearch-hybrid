<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Similarity;

class SimilarityServiceTest extends \PHPUnit\Framework\TestCase
{
    public function testDisabledFeatureReturnsDeclaredFallbackWithoutQuerying(): void
    {
        $config = $this->createMock(\MageOS\OpenSearchHybrid\Model\Config::class);
        $config->method('isSimilarityEnabled')->with(2)->willReturn(false);
        $generationRepository = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository::class
        );
        $generationRepository->expects($this->never())->method('activeForStore');
        $transport = $this->createMock(\MageOS\OpenSearchHybrid\Model\OpenSearch\HybridTransport::class);
        $transport->expects($this->never())->method('search');
        $service = $this->service($config, $generationRepository, $transport);

        $result = $service->searchByProduct(2, 41, 2, 'substitute-v1', [41, 9, 8, 7]);

        self::assertSame([
            'items' => [
                ['product_id' => 9, 'score' => null],
                ['product_id' => 8, 'score' => null],
            ],
            'fallback_used' => true,
            'fallback_reason' => 'feature_disabled',
            'generation_id' => null,
            'threshold_profile_id' => 'substitute-v1',
        ], $result);
    }

    public function testProductSeedQueryReturnsDeterministicIdsAndScores(): void
    {
        $config = $this->createStub(\MageOS\OpenSearchHybrid\Model\Config::class);
        $config->method('isSimilarityEnabled')->willReturn(true);
        $generation = $this->generation();
        $generationRepository = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository::class
        );
        $generationRepository->method('activeForStore')->willReturn($generation);
        $profileRegistry = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Similarity\ThresholdProfileRegistry::class
        );
        $profileRegistry->method('resolve')->willReturn(['min_score' => 0.91]);
        $requestBuilder = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Similarity\RadialRequestBuilder::class
        );
        $requestBuilder->method('buildSeedRequest')->willReturn(['seed' => true]);
        $requestBuilder->method('buildRadialRequest')->willReturn(['radial' => true]);
        $vectorValidator = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Vector\VectorValidator::class
        );
        $vectorValidator->method('fromBase64Float32')->willReturn([1.0, 0.0]);
        $transport = $this->createMock(\MageOS\OpenSearchHybrid\Model\OpenSearch\HybridTransport::class);
        $requestNumber = 0;
        $transport->expects($this->exactly(2))
            ->method('search')
            ->willReturnCallback(static function () use (&$requestNumber): array {
                $requestNumber++;
                if ($requestNumber === 1) {
                    return ['hits' => ['hits' => [[
                        'fields' => ['embedding' => ['AACAPwAAAAA=']],
                    ]]]];
                }

                return ['hits' => ['hits' => [
                    ['_score' => 0.92, 'fields' => ['entity_id' => [9]]],
                    ['_score' => 0.95, 'fields' => ['entity_id' => [8]]],
                    ['_score' => 0.95, 'fields' => ['entity_id' => [7]]],
                ]]];
            });
        $service = new \MageOS\OpenSearchHybrid\Model\Similarity\SimilarityService(
            $config,
            $generationRepository,
            $profileRegistry,
            $requestBuilder,
            $vectorValidator,
            $transport,
            $this->createStub(\MageOS\OpenSearchHybrid\Model\Reliability\CircuitBreaker::class),
            $this->createStub(\MageOS\OpenSearchHybrid\Model\Operations\SimilarityTelemetry::class)
        );

        $result = $service->searchByProduct(2, 41, 3, 'substitute-v1');

        self::assertSame([
            ['product_id' => 7, 'score' => 0.95],
            ['product_id' => 8, 'score' => 0.95],
            ['product_id' => 9, 'score' => 0.92],
        ], $result['items']);
        self::assertFalse($result['fallback_used']);
        self::assertSame(7, $result['generation_id']);
    }

    public function testQueryFailureReturnsDeclaredFallbackWithoutLeakingTheError(): void
    {
        $config = $this->createStub(\MageOS\OpenSearchHybrid\Model\Config::class);
        $config->method('isSimilarityEnabled')->willReturn(true);
        $generationRepository = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository::class
        );
        $generationRepository->method('activeForStore')->willReturn($this->generation());
        $profileRegistry = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Similarity\ThresholdProfileRegistry::class
        );
        $profileRegistry->method('resolve')->willReturn(['min_score' => 0.91]);
        $transport = $this->createStub(\MageOS\OpenSearchHybrid\Model\OpenSearch\HybridTransport::class);
        $transport->method('search')->willThrowException(new \RuntimeException('secret response'));
        $circuit = $this->createMock(\MageOS\OpenSearchHybrid\Model\Reliability\CircuitBreaker::class);
        $circuit->expects($this->exactly(2))
            ->method('isOpen')
            ->with(2, 'similarity')
            ->willReturn(false);
        $circuit->expects($this->once())->method('failure')->with(2, 'similarity');
        $service = $this->service(
            $config,
            $generationRepository,
            $transport,
            $profileRegistry,
            $circuit
        );

        $result = $service->searchByProduct(2, 41, 2, 'substitute-v1', [9, 8]);

        self::assertSame('query_failed', $result['fallback_reason']);
        self::assertSame([
            ['product_id' => 9, 'score' => null],
            ['product_id' => 8, 'score' => null],
        ], $result['items']);
        self::assertStringNotContainsString('secret', json_encode($result, JSON_THROW_ON_ERROR));
    }

    public function testOpenSimilarityCircuitReturnsFallbackWithoutQuerying(): void
    {
        $config = $this->createStub(\MageOS\OpenSearchHybrid\Model\Config::class);
        $config->method('isSimilarityEnabled')->willReturn(true);
        $generationRepository = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository::class
        );
        $generationRepository->method('activeForStore')->willReturn($this->generation());
        $profileRegistry = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Similarity\ThresholdProfileRegistry::class
        );
        $profileRegistry->method('resolve')->willReturn(['min_score' => 0.91]);
        $transport = $this->createMock(\MageOS\OpenSearchHybrid\Model\OpenSearch\HybridTransport::class);
        $transport->expects($this->never())->method('search');
        $circuit = $this->createMock(\MageOS\OpenSearchHybrid\Model\Reliability\CircuitBreaker::class);
        $circuit->expects($this->once())
            ->method('isOpen')
            ->with(2, 'similarity')
            ->willReturn(true);
        $service = $this->service($config, $generationRepository, $transport, $profileRegistry, $circuit);

        $result = $service->searchByProduct(2, 41, 2, 'substitute-v1', [9, 8]);

        self::assertSame('circuit_open', $result['fallback_reason']);
        self::assertSame(7, $result['generation_id']);
    }

    private function service(
        \MageOS\OpenSearchHybrid\Model\Config $config,
        \MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository $generationRepository,
        \MageOS\OpenSearchHybrid\Model\OpenSearch\HybridTransport $transport,
        ?\MageOS\OpenSearchHybrid\Model\Similarity\ThresholdProfileRegistry $profileRegistry = null,
        ?\MageOS\OpenSearchHybrid\Model\Reliability\CircuitBreaker $circuitBreaker = null
    ): \MageOS\OpenSearchHybrid\Model\Similarity\SimilarityService {
        return new \MageOS\OpenSearchHybrid\Model\Similarity\SimilarityService(
            $config,
            $generationRepository,
            $profileRegistry ?? $this->createStub(
                \MageOS\OpenSearchHybrid\Model\Similarity\ThresholdProfileRegistry::class
            ),
            $this->createStub(\MageOS\OpenSearchHybrid\Model\Similarity\RadialRequestBuilder::class),
            $this->createStub(\MageOS\OpenSearchHybrid\Model\Vector\VectorValidator::class),
            $transport,
            $circuitBreaker ?? $this->createStub(
                \MageOS\OpenSearchHybrid\Model\Reliability\CircuitBreaker::class
            ),
            $this->createStub(\MageOS\OpenSearchHybrid\Model\Operations\SimilarityTelemetry::class)
        );
    }

    private function generation(): array
    {
        return [
            'generation_id' => 7,
            'physical_index' => 'mageos-hybrid-s2-g7',
            'result_contract_accepted' => 1,
            'dimension' => 2,
        ];
    }
}
