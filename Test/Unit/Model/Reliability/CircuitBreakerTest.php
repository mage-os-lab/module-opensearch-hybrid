<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Reliability;

class CircuitBreakerTest extends \PHPUnit\Framework\TestCase
{
    public function testCatalogAndSimilarityCircuitsUseIndependentCacheKeys(): void
    {
        $removedKeys = [];
        $cache = $this->createMock(\Magento\Framework\App\CacheInterface::class);
        $cache->expects($this->exactly(2))
            ->method('remove')
            ->willReturnCallback(static function (string $key) use (&$removedKeys): bool {
                $removedKeys[] = $key;

                return true;
            });
        $breaker = new \MageOS\OpenSearchHybrid\Model\Reliability\CircuitBreaker(
            $cache,
            $this->createStub(\Magento\Framework\Serialize\Serializer\Json::class),
            $this->createStub(\MageOS\OpenSearchHybrid\Model\Config::class)
        );

        $breaker->success(2);
        $breaker->success(2, 'similarity');

        self::assertSame([
            'mageos_opensearch_hybrid_circuit_2',
            'mageos_opensearch_hybrid_circuit_similarity_2',
        ], $removedKeys);
    }

    public function testRejectsAnUnknownCircuitIdentity(): void
    {
        $breaker = new \MageOS\OpenSearchHybrid\Model\Reliability\CircuitBreaker(
            $this->createStub(\Magento\Framework\App\CacheInterface::class),
            $this->createStub(\Magento\Framework\Serialize\Serializer\Json::class),
            $this->createStub(\MageOS\OpenSearchHybrid\Model\Config::class)
        );
        $this->expectException(\InvalidArgumentException::class);

        $breaker->isOpen(2, 'caller-selected');
    }
}
