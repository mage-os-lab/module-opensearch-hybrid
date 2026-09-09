<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\SearchAdapter;

class PaginationTest extends \PHPUnit\Framework\TestCase
{
    #[\PHPUnit\Framework\Attributes\DataProvider('pages')]
    public function testEveryProductAppearsOnceAcrossTheHybridBoundary(int $size, array $sort): void
    {
        $nativeIds = range(1, 240);
        $candidateIds = [];
        $aggregations = new \Magento\Framework\Search\Response\Aggregation([]);
        $native = $this->createStub(\Magento\OpenSearch\SearchAdapter\Adapter::class);
        $native->method('query')->willReturnCallback(function ($request) use ($nativeIds, &$candidateIds, $aggregations) {
            $ids = $nativeIds;
            $sort = $request->getSort();
            if (($sort[0]['field'] ?? '') === 'is_out_of_stock') {
                // A highly relevant but unavailable item must remain at the end,
                // including after the hybrid window.
                $ids = array_merge(array_slice($ids, 1), [$ids[0]]);
            }
            if (($sort[0]['field'] ?? '') === 'position') {
                $ids = array_reverse($ids);
            }
            $ids = array_slice($ids, (int)$request->getFrom(), (int)$request->getSize());
            if ($request->getFrom() === 0) {
                $candidateIds = array_slice($ids, 0, 100);
            }
            return new \Magento\Framework\Search\Response\QueryResponse(
                array_map(fn (int $id) => $this->document($id), $ids), $aggregations, 240
            );
        });
        $generation = $this->createStub(\MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository::class);
        $generation->method('activeForStore')->willReturn([
            'generation_id' => 1, 'result_contract_accepted' => 1,
            'physical_index' => 'fixture', 'pipeline_id' => 'fixture',
        ]);
        $queryText = $this->createStub(\MageOS\OpenSearchHybrid\SearchAdapter\QueryTextExtractor::class);
        $queryText->method('extract')->willReturn('chair');
        $guard = $this->createStub(\MageOS\OpenSearchHybrid\SearchAdapter\KnownItemGuard::class);
        $guard->method('mustUseLexical')->willReturn(false);
        $encoder = $this->createStub(\MageOS\OpenSearchHybrid\Api\QueryEncoderInterface::class);
        $encoder->method('encode')->willReturn([1.0]);
        $transport = $this->createStub(\MageOS\OpenSearchHybrid\Model\OpenSearch\HybridTransport::class);
        $transport->method('search')->willReturnCallback(static function () use (&$candidateIds): array {
            return ['hits' => ['hits' => array_map(static fn (int $id): array => [
                '_id' => (string)$id, '_score' => $id / 240,
                'fields' => ['is_salable' => [$id !== 1]],
            ], array_reverse($candidateIds))]];
        });
        $sortEligibility = new \MageOS\OpenSearchHybrid\SearchAdapter\SortEligibility();
        $factory = new \MageOS\OpenSearchHybrid\SearchAdapter\NativeCandidateRequestFactory(
            new \MageOS\OpenSearchHybrid\SearchAdapter\SortEligibility()
        );
        $executor = new \MageOS\OpenSearchHybrid\SearchAdapter\HybridExecutor(
            $native, $generation, $queryText, $guard, $encoder,
            new \MageOS\OpenSearchHybrid\SearchAdapter\NativeCandidateScoreExtractor(),
            $sortEligibility, $factory,
            $this->createStub(\MageOS\OpenSearchHybrid\SearchAdapter\HybridRequestBuilder::class),
            new \MageOS\OpenSearchHybrid\SearchAdapter\HybridResultMapper(), $transport
        );
        $config = $this->createStub(\MageOS\OpenSearchHybrid\Model\Config::class);
        $config->method('isEnabled')->willReturn(true);
        $config->method('isStoreIncluded')->willReturn(true);
        $readiness = $this->createStub(\MageOS\OpenSearchHybrid\Model\Store\ReadinessLatch::class);
        $readiness->method('isOpen')->willReturn(true);
        $eligibility = new \MageOS\OpenSearchHybrid\SearchAdapter\Eligibility($config, $readiness, $sortEligibility);
        $store = $this->createStub(\Magento\Store\Api\Data\StoreInterface::class);
        $store->method('getId')->willReturn(1);
        $stores = $this->createStub(\Magento\Store\Model\StoreManagerInterface::class);
        $stores->method('getStore')->willReturn($store);
        $circuit = $this->createStub(\MageOS\OpenSearchHybrid\Model\Reliability\CircuitBreaker::class);
        $circuit->method('isOpen')->willReturn(false);
        $logger = new \Psr\Log\NullLogger();
        $adapter = new \MageOS\OpenSearchHybrid\SearchAdapter\Adapter(
            $native, $stores, $eligibility, $executor, $circuit, $logger,
            new \MageOS\OpenSearchHybrid\Model\Operations\QueryTelemetry($logger), $factory
        );
        $seen = [];
        for ($from = 0; $from < 240; $from += $size) {
            $request = new \Magento\Framework\Search\Request(
                'quick_search_container', 'catalogsearch_fulltext',
                $this->createStub(\Magento\Framework\Search\Request\QueryInterface::class),
                $from, $size, [], [], $sort
            );
            $response = $adapter->query($request);
            self::assertSame(240, $response->getTotal());
            self::assertSame($aggregations, $response->getAggregations());
            $page = array_map(static fn ($document): int => (int)$document->getId(), iterator_to_array($response));
            self::assertCount(min($size, 240 - $from), $page);
            array_push($seen, ...$page);
        }
        self::assertCount(240, array_unique($seen), 'Duplicate products across page boundaries.');
        $sorted = $seen;
        sort($sorted);
        self::assertSame($nativeIds, $sorted, 'A product was omitted from pagination.');
        if (($sort[0]['field'] ?? '') === 'is_out_of_stock') {
            self::assertSame(1, $seen[239]);
        }
    }

    private function document(int $id): \Magento\Framework\Api\Search\DocumentInterface
    {
        $document = $this->createStub(\Magento\Framework\Api\Search\DocumentInterface::class);
        $document->method('getId')->willReturn($id);
        $document->method('getCustomAttribute')->willReturn(new \Magento\Framework\Api\AttributeValue([
            'attribute_code' => 'score', 'value' => 241.0 - $id,
        ]));
        return $document;
    }

    public static function pages(): array
    {
        $cases = [];
        foreach ([10, 12, 24, 36, 100, 150] as $size) {
            foreach ([
                [['field' => 'relevance', 'direction' => 'DESC']],
                [['field' => 'position', 'direction' => 'ASC']],
                [['field' => 'is_out_of_stock', 'direction' => 'ASC'], ['field' => 'position', 'direction' => 'ASC']],
            ] as $sort) {
                $cases[] = [$size, $sort];
            }
        }
        return $cases;
    }
}
