<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\SearchAdapter;

class HybridRequestBuilderTest extends \PHPUnit\Framework\TestCase
{
    public function testUsesNativeMerchantControlledScoresAsTheLexicalClause(): void
    {
        $contract = $this->createStub(\MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract::class);
        $contract->method('get')->willReturn([
            'ann' => ['k' => 100],
        ]);
        $builder = new \MageOS\OpenSearchHybrid\SearchAdapter\HybridRequestBuilder($contract);
        $request = $builder->build([1.0, 0.0], ['3' => 12.5, '2' => 4.25]);
        $queries = $request['query']['hybrid']['queries'];

        self::assertSame(100, $request['query']['hybrid']['pagination_depth']);
        self::assertSame([
            [
                'constant_score' => [
                    'filter' => ['term' => ['_id' => '3']],
                    'boost' => 12.5,
                ],
            ],
            [
                'constant_score' => [
                    'filter' => ['term' => ['_id' => '2']],
                    'boost' => 4.25,
                ],
            ],
        ], $queries[0]['bool']['should']);
        self::assertSame(1, $queries[0]['bool']['minimum_should_match']);
        self::assertSame(['terms' => ['_id' => ['3', '2']]], $queries[1]['knn']['embedding']['filter']);
        self::assertSame(100, $queries[1]['knn']['embedding']['k']);
        self::assertArrayNotHasKey('fields', $request);
    }

    public function testRejectsInvalidNativeScores(): void
    {
        $contract = $this->createStub(\MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract::class);
        $contract->method('get')->willReturn(['ann' => ['k' => 100]]);
        $builder = new \MageOS\OpenSearchHybrid\SearchAdapter\HybridRequestBuilder($contract);

        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('finite non-negative native score');
        $builder->build([1.0, 0.0], ['3' => INF]);
    }

    public function testRequestsSalabilityMetadataForAvailabilityPartitioning(): void
    {
        $contract = $this->createStub(\MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract::class);
        $contract->method('get')->willReturn(['ann' => ['k' => 100]]);
        $builder = new \MageOS\OpenSearchHybrid\SearchAdapter\HybridRequestBuilder($contract);

        $request = $builder->build([1.0, 0.0], ['3' => 12.5], true);

        self::assertSame(['is_salable'], $request['fields']);
    }
}
