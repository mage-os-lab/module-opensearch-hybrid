<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\SearchAdapter;

class AdapterFailureTest extends \PHPUnit\Framework\TestCase
{
    #[\PHPUnit\Framework\Attributes\DataProvider('failedBookkeeping')]
    public function testNativeSearchSurvivesHybridBookkeepingFailure(string $failureMode): void
    {
        $request = $this->createStub(\Magento\Framework\Search\RequestInterface::class);
        $request->method('getName')->willReturn('quick_search_container');
        $response = $this->createStub(\Magento\Framework\Search\Response\QueryResponse::class);
        $native = $this->createMock(\Magento\OpenSearch\SearchAdapter\Adapter::class);
        $native->expects($this->once())->method('query')->willReturn($response);
        $store = $this->createStub(\Magento\Store\Api\Data\StoreInterface::class);
        $store->method('getId')->willReturn(1);
        $storeManager = $this->createStub(\Magento\Store\Model\StoreManagerInterface::class);
        $storeManager->method('getStore')->willReturn($store);
        $eligibility = $this->createStub(\MageOS\OpenSearchHybrid\SearchAdapter\Eligibility::class);
        $eligibility->method('ineligibilityReason')->willReturn(null);
        $executor = $this->createStub(\MageOS\OpenSearchHybrid\SearchAdapter\HybridExecutor::class);
        $executor->method('execute')->willThrowException(new \RuntimeException('encoder unavailable'));
        $circuit = $this->createStub(\MageOS\OpenSearchHybrid\Model\Reliability\CircuitBreaker::class);
        if ($failureMode === 'cache_read') {
            $circuit->method('isOpen')->willThrowException(new \RuntimeException('circuit cache unavailable'));
        } else {
            $circuit->method('isOpen')->willReturn(false);
        }
        if ($failureMode === 'cache_write') {
            $circuit->method('failure')->willThrowException(new \RuntimeException('circuit cache unavailable'));
        }
        $logger = $this->createStub(\Psr\Log\LoggerInterface::class);
        if ($failureMode === 'log_write') {
            $logger->method('warning')->willThrowException(new \RuntimeException('log unavailable'));
        }
        $telemetry = new \MageOS\OpenSearchHybrid\Model\Operations\QueryTelemetry(new \Psr\Log\NullLogger());
        $adapter = new \MageOS\OpenSearchHybrid\SearchAdapter\Adapter(
            $native, $storeManager, $eligibility, $executor, $circuit, $logger, $telemetry,
            new \MageOS\OpenSearchHybrid\SearchAdapter\NativeCandidateRequestFactory(
                new \MageOS\OpenSearchHybrid\SearchAdapter\SortEligibility()
            )
        );
        try {
            $actual = $adapter->query($request);
        } catch (\RuntimeException $error) {
            self::fail('Healthy native search was bypassed: ' . $error->getMessage());
        }
        self::assertSame($response, $actual);
    }
    public static function failedBookkeeping(): array
    {
        return [['cache_read'], ['cache_write'], ['log_write']];
    }
}
