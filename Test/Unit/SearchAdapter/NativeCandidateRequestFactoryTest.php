<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\SearchAdapter;

class NativeCandidateRequestFactoryTest extends \PHPUnit\Framework\TestCase
{
    public function testNormalizesTheCandidateRequestToNativeRelevance(): void
    {
        $query = $this->createStub(\Magento\Framework\Search\Request\QueryInterface::class);
        $dimension = $this->createStub(\Magento\Framework\Search\Request\Dimension::class);
        $bucket = $this->createStub(\Magento\Framework\Search\Request\BucketInterface::class);
        $request = new \Magento\Framework\Search\Request(
            'quick_search_container',
            'catalogsearch_fulltext',
            $query,
            24,
            12,
            [$dimension],
            [$bucket],
            [['field' => 'position', 'direction' => 'ASC']]
        );

        $candidate = (new \MageOS\OpenSearchHybrid\SearchAdapter\NativeCandidateRequestFactory(
            new \MageOS\OpenSearchHybrid\SearchAdapter\SortEligibility()
        ))->create($request);

        self::assertSame('quick_search_container', $candidate->getName());
        self::assertSame('catalogsearch_fulltext', $candidate->getIndex());
        self::assertSame($query, $candidate->getQuery());
        self::assertSame(0, $candidate->getFrom());
        self::assertSame(100, $candidate->getSize());
        self::assertSame([$dimension], $candidate->getDimensions());
        self::assertSame([$bucket], $candidate->getAggregation());
        self::assertSame([
            ['field' => 'relevance', 'direction' => 'DESC'],
            ['field' => 'entity_id', 'direction' => 'ASC'],
        ], $candidate->getSort());
    }
}
