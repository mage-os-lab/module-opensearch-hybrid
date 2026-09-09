<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\SearchAdapter;

class AdapterTest extends \PHPUnit\Framework\TestCase
{
    public function testRecordsIneligibleRequestsAsNativeWithoutCallingHybrid(): void
    {
        $request = $this->createStub(\Magento\Framework\Search\RequestInterface::class);
        $request->method('getName')->willReturn('advanced_search_container');
        $response = $this->createStub(\Magento\Framework\Search\Response\QueryResponse::class);
        $nativeAdapter = $this->createMock(\Magento\OpenSearch\SearchAdapter\Adapter::class);
        $nativeAdapter->expects($this->once())->method('query')->with($request)->willReturn($response);
        $eligibility = $this->createStub(\MageOS\OpenSearchHybrid\SearchAdapter\Eligibility::class);
        $eligibility->method('ineligibilityReason')->willReturn('unsupported_request');
        $hybridExecutor = $this->createMock(\MageOS\OpenSearchHybrid\SearchAdapter\HybridExecutor::class);
        $hybridExecutor->expects($this->never())->method('execute');
        $circuitBreaker = $this->createMock(\MageOS\OpenSearchHybrid\Model\Reliability\CircuitBreaker::class);
        $circuitBreaker->expects($this->never())->method('isOpen');
        $span = $this->createMock(\MageOS\OpenSearchHybrid\Model\Operations\QueryTelemetrySpan::class);
        $span->expects($this->once())->method('route')->with('native', 'unsupported_request');
        $span->expects($this->once())->method('finish')->with();
        $telemetry = $this->createMock(\MageOS\OpenSearchHybrid\Model\Operations\QueryTelemetry::class);
        $telemetry->expects($this->once())
            ->method('start')
            ->with(2, 'advanced_search_container')
            ->willReturn($span);

        $adapter = $this->adapter(
            $nativeAdapter,
            $eligibility,
            $hybridExecutor,
            $circuitBreaker,
            $telemetry
        );

        self::assertSame($response, $adapter->query($request));
    }

    public function testRecordsHybridFailureAndFallsBackToNative(): void
    {
        $request = $this->createStub(\Magento\Framework\Search\RequestInterface::class);
        $request->method('getName')->willReturn('quick_search_container');
        $response = $this->createStub(\Magento\Framework\Search\Response\QueryResponse::class);
        $nativeAdapter = $this->createMock(\Magento\OpenSearch\SearchAdapter\Adapter::class);
        $nativeAdapter->expects($this->once())->method('query')->with($request)->willReturn($response);
        $eligibility = $this->createStub(\MageOS\OpenSearchHybrid\SearchAdapter\Eligibility::class);
        $eligibility->method('ineligibilityReason')->willReturn(null);
        $failure = new \RuntimeException('encoder request failed');
        $hybridExecutor = $this->createMock(\MageOS\OpenSearchHybrid\SearchAdapter\HybridExecutor::class);
        $hybridExecutor->expects($this->once())->method('execute')->willThrowException($failure);
        $circuitBreaker = $this->createMock(\MageOS\OpenSearchHybrid\Model\Reliability\CircuitBreaker::class);
        $circuitBreaker->method('isOpen')->willReturn(false);
        $circuitBreaker->expects($this->once())->method('failure')->with(2);
        $span = $this->createMock(\MageOS\OpenSearchHybrid\Model\Operations\QueryTelemetrySpan::class);
        $span->expects($this->once())->method('route')->with('native_fallback', 'hybrid_error');
        $span->expects($this->once())->method('finish')->with($failure);
        $telemetry = $this->createStub(\MageOS\OpenSearchHybrid\Model\Operations\QueryTelemetry::class);
        $telemetry->method('start')->willReturn($span);

        $adapter = $this->adapter(
            $nativeAdapter,
            $eligibility,
            $hybridExecutor,
            $circuitBreaker,
            $telemetry
        );

        self::assertSame($response, $adapter->query($request));
    }

    public function testRecordsCircuitOpenRequestsAsNative(): void
    {
        $request = $this->createStub(\Magento\Framework\Search\RequestInterface::class);
        $request->method('getName')->willReturn('quick_search_container');
        $response = $this->createStub(\Magento\Framework\Search\Response\QueryResponse::class);
        $nativeAdapter = $this->createMock(\Magento\OpenSearch\SearchAdapter\Adapter::class);
        $nativeAdapter->expects($this->once())->method('query')->with($request)->willReturn($response);
        $eligibility = $this->createStub(\MageOS\OpenSearchHybrid\SearchAdapter\Eligibility::class);
        $eligibility->method('ineligibilityReason')->willReturn(null);
        $hybridExecutor = $this->createMock(\MageOS\OpenSearchHybrid\SearchAdapter\HybridExecutor::class);
        $hybridExecutor->expects($this->never())->method('execute');
        $circuitBreaker = $this->createStub(\MageOS\OpenSearchHybrid\Model\Reliability\CircuitBreaker::class);
        $circuitBreaker->method('isOpen')->willReturn(true);
        $span = $this->createMock(\MageOS\OpenSearchHybrid\Model\Operations\QueryTelemetrySpan::class);
        $span->expects($this->once())->method('route')->with('native', 'circuit_open');
        $span->expects($this->once())->method('finish')->with();
        $telemetry = $this->createStub(\MageOS\OpenSearchHybrid\Model\Operations\QueryTelemetry::class);
        $telemetry->method('start')->willReturn($span);

        $adapter = $this->adapter(
            $nativeAdapter,
            $eligibility,
            $hybridExecutor,
            $circuitBreaker,
            $telemetry
        );

        self::assertSame($response, $adapter->query($request));
    }

    public function testFinishesSuccessfulHybridTelemetryAndClosesTheCircuit(): void
    {
        $request = $this->createStub(\Magento\Framework\Search\RequestInterface::class);
        $request->method('getName')->willReturn('graphql_product_search');
        $response = $this->createStub(\Magento\Framework\Search\Response\QueryResponse::class);
        $nativeAdapter = $this->createMock(\Magento\OpenSearch\SearchAdapter\Adapter::class);
        $nativeAdapter->expects($this->never())->method('query');
        $eligibility = $this->createStub(\MageOS\OpenSearchHybrid\SearchAdapter\Eligibility::class);
        $eligibility->method('ineligibilityReason')->willReturn(null);
        $span = $this->createMock(\MageOS\OpenSearchHybrid\Model\Operations\QueryTelemetrySpan::class);
        $span->expects($this->once())->method('finish')->with();
        $hybridExecutor = $this->createMock(\MageOS\OpenSearchHybrid\SearchAdapter\HybridExecutor::class);
        $hybridExecutor->expects($this->once())
            ->method('execute')
            ->with($request, 2, $span)
            ->willReturn($response);
        $circuitBreaker = $this->createMock(\MageOS\OpenSearchHybrid\Model\Reliability\CircuitBreaker::class);
        $circuitBreaker->method('isOpen')->willReturn(false);
        $circuitBreaker->expects($this->once())->method('success')->with(2);
        $telemetry = $this->createStub(\MageOS\OpenSearchHybrid\Model\Operations\QueryTelemetry::class);
        $telemetry->method('start')->willReturn($span);

        $adapter = $this->adapter(
            $nativeAdapter,
            $eligibility,
            $hybridExecutor,
            $circuitBreaker,
            $telemetry
        );

        self::assertSame($response, $adapter->query($request));
    }

    private function adapter(
        \Magento\OpenSearch\SearchAdapter\Adapter $nativeAdapter,
        \MageOS\OpenSearchHybrid\SearchAdapter\Eligibility $eligibility,
        \MageOS\OpenSearchHybrid\SearchAdapter\HybridExecutor $hybridExecutor,
        \MageOS\OpenSearchHybrid\Model\Reliability\CircuitBreaker $circuitBreaker,
        \MageOS\OpenSearchHybrid\Model\Operations\QueryTelemetry $telemetry
    ): \MageOS\OpenSearchHybrid\SearchAdapter\Adapter {
        $store = $this->createStub(\Magento\Store\Api\Data\StoreInterface::class);
        $store->method('getId')->willReturn(2);
        $storeManager = $this->createStub(\Magento\Store\Model\StoreManagerInterface::class);
        $storeManager->method('getStore')->willReturn($store);

        return new \MageOS\OpenSearchHybrid\SearchAdapter\Adapter(
            $nativeAdapter,
            $storeManager,
            $eligibility,
            $hybridExecutor,
            $circuitBreaker,
            $this->createStub(\Psr\Log\LoggerInterface::class),
            $telemetry,
            new \MageOS\OpenSearchHybrid\SearchAdapter\NativeCandidateRequestFactory(
            new \MageOS\OpenSearchHybrid\SearchAdapter\SortEligibility()
        )
        );
    }
}
