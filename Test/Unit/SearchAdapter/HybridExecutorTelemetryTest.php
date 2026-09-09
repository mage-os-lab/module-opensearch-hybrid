<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\SearchAdapter;

class HybridExecutorTelemetryTest extends \PHPUnit\Framework\TestCase
{
    public function testKnownItemRecordsItsActualNativeRoute(): void
    {
        $generation = ['generation_id' => 3, 'result_contract_accepted' => 1];
        $generationRepository = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository::class
        );
        $generationRepository->method('activeForStore')->willReturn($generation);
        $queryTextExtractor = $this->createStub(
            \MageOS\OpenSearchHybrid\SearchAdapter\QueryTextExtractor::class
        );
        $queryTextExtractor->method('extract')->willReturn('ABC-123');
        $knownItemGuard = $this->createStub(\MageOS\OpenSearchHybrid\SearchAdapter\KnownItemGuard::class);
        $knownItemGuard->method('mustUseLexical')->willReturn(true);
        $request = $this->createStub(\Magento\Framework\Search\Request::class);
        $request->method('getName')->willReturn('quick_search_container');
        $response = $this->createStub(\Magento\Framework\Search\Response\QueryResponse::class);
        $nativeAdapter = $this->createMock(\Magento\OpenSearch\SearchAdapter\Adapter::class);
        $nativeAdapter->expects($this->once())->method('query')->with($request)->willReturn($response);
        $factory = $this->createStub(\MageOS\OpenSearchHybrid\SearchAdapter\NativeCandidateRequestFactory::class);
        $factory->method('createPage')->willReturn($request);
        $span = $this->createMock(\MageOS\OpenSearchHybrid\Model\Operations\QueryTelemetrySpan::class);
        $span->expects($this->once())->method('generation')->with(3);
        $span->expects($this->once())->method('route')->with('native', 'known_item', 3);

        $executor = new \MageOS\OpenSearchHybrid\SearchAdapter\HybridExecutor(
            $nativeAdapter,
            $generationRepository,
            $queryTextExtractor,
            $knownItemGuard,
            $this->createStub(\MageOS\OpenSearchHybrid\Api\QueryEncoderInterface::class),
            $this->createStub(\MageOS\OpenSearchHybrid\SearchAdapter\NativeCandidateScoreExtractor::class),
            $this->createStub(\MageOS\OpenSearchHybrid\SearchAdapter\SortEligibility::class),
            $factory,
            $this->createStub(\MageOS\OpenSearchHybrid\SearchAdapter\HybridRequestBuilder::class),
            $this->createStub(\MageOS\OpenSearchHybrid\SearchAdapter\HybridResultMapper::class),
            $this->createStub(\MageOS\OpenSearchHybrid\Model\OpenSearch\HybridTransport::class)
        );

        self::assertSame($response, $executor->execute($request, 2, $span));
    }
}
