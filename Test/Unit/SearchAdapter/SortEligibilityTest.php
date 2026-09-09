<?php

declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\SearchAdapter;

class SortEligibilityTest extends \PHPUnit\Framework\TestCase
{
    public function testAcceptsThePreparedMagentoRelevanceSortShape(): void
    {
        $eligibility = new \MageOS\OpenSearchHybrid\SearchAdapter\SortEligibility();

        self::assertTrue($eligibility->isRelevanceOnly([['field' => 'relevance', 'direction' => 'DESC']]));
        self::assertTrue($eligibility->isRelevanceOnly(['_score' => 'DESC']));
        self::assertTrue($eligibility->isRelevanceOnly([
            ['field' => 'relevance', 'direction' => 'DESC'],
            ['field' => 'entity_id', 'direction' => 'DESC'],
        ]));
        self::assertTrue($eligibility->isRelevanceOnly([
            'score' => 'DESC',
            'entity_id' => 'ASC',
        ]));
    }

    public function testRejectsNonRelevanceAndUnrecognizedSortShapes(): void
    {
        $eligibility = new \MageOS\OpenSearchHybrid\SearchAdapter\SortEligibility();

        self::assertFalse($eligibility->isRelevanceOnly([['field' => 'price', 'direction' => 'ASC']]));
        self::assertFalse($eligibility->isRelevanceOnly([['field' => 'relevance', 'direction' => 'ASC']]));
        self::assertFalse($eligibility->isRelevanceOnly([['field' => 'entity_id', 'direction' => 'DESC']]));
        self::assertFalse($eligibility->isRelevanceOnly([
            ['field' => 'entity_id', 'direction' => 'DESC'],
            ['field' => 'relevance', 'direction' => 'DESC'],
        ]));
        self::assertFalse($eligibility->isRelevanceOnly([
            ['field' => 'relevance', 'direction' => 'DESC'],
            ['field' => 'price', 'direction' => 'ASC'],
        ]));
        self::assertFalse($eligibility->isRelevanceOnly([['direction' => 'ASC']]));
    }

    public function testPlansMageOsStorefrontDefaultAsAvailabilityPartitionedRelevance(): void
    {
        $eligibility = new \MageOS\OpenSearchHybrid\SearchAdapter\SortEligibility();

        self::assertSame(
            ['out_of_stock_bottom' => true],
            $eligibility->hybridPlan([
                ['field' => 'is_out_of_stock', 'direction' => 'ASC'],
                ['field' => 'position', 'direction' => 'ASC'],
            ])
        );
        self::assertSame(
            ['out_of_stock_bottom' => true],
            $eligibility->hybridPlan([
                ['field' => 'is_out_of_stock', 'direction' => 'ASC'],
                ['field' => 'relevance', 'direction' => 'DESC'],
                ['field' => 'entity_id', 'direction' => 'DESC'],
            ])
        );
        self::assertNull($eligibility->hybridPlan([
            ['field' => 'is_out_of_stock', 'direction' => 'DESC'],
            ['field' => 'position', 'direction' => 'ASC'],
        ]));
        self::assertNull($eligibility->hybridPlan([
            ['field' => 'is_out_of_stock', 'direction' => 'ASC'],
            ['field' => 'price', 'direction' => 'ASC'],
        ]));
    }

    public function testPlansMageOsStorefrontDefaultWithoutAvailabilityPartition(): void
    {
        $eligibility = new \MageOS\OpenSearchHybrid\SearchAdapter\SortEligibility();

        self::assertSame(
            ['out_of_stock_bottom' => false],
            $eligibility->hybridPlan([
                ['field' => 'position', 'direction' => 'ASC'],
            ])
        );
        self::assertSame(
            ['out_of_stock_bottom' => false],
            $eligibility->hybridPlan([
                ['field' => 'position', 'direction' => 'ASC'],
                ['field' => 'entity_id', 'direction' => 'DESC'],
            ])
        );
    }
}
