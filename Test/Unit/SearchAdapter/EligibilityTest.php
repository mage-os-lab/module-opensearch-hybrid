<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\SearchAdapter;

class EligibilityTest extends \PHPUnit\Framework\TestCase
{
    public function testAllowsGraphQlProductSearchWithPreparedRelevanceSort(): void
    {
        $eligibility = $this->eligibility();

        self::assertTrue($eligibility->isEligible($this->request(
            'graphql_product_search',
            [
                ['field' => 'relevance', 'direction' => 'DESC'],
                ['field' => 'entity_id', 'direction' => 'DESC'],
            ]
        ), 2));
    }

    public function testRejectsGraphQlProductSearchWithExplicitPriceSort(): void
    {
        $eligibility = $this->eligibility();

        $request = $this->request(
            'graphql_product_search',
            [
                ['field' => 'price', 'direction' => 'ASC'],
                ['field' => 'entity_id', 'direction' => 'DESC'],
            ]
        );

        self::assertSame('unsupported_sort', $eligibility->ineligibilityReason($request, 2));
        self::assertFalse($eligibility->isEligible($request, 2));
    }

    public function testKeepsStorefrontQuickSearchEligible(): void
    {
        $eligibility = $this->eligibility();

        self::assertTrue($eligibility->isEligible($this->request(
            'quick_search_container',
            [
                ['field' => 'is_out_of_stock', 'direction' => 'ASC'],
                ['field' => 'position', 'direction' => 'ASC'],
            ]
        ), 2));
    }

    public function testRejectsUnknownRequestNames(): void
    {
        $eligibility = $this->eligibility();

        $request = $this->request(
            'catalog_view_container',
            [['field' => 'relevance', 'direction' => 'DESC']]
        );

        self::assertSame('unsupported_request', $eligibility->ineligibilityReason($request, 2));
        self::assertFalse($eligibility->isEligible($request, 2));
    }

    public function testReportsTheFirstFailClosedEligibilityBoundary(): void
    {
        self::assertSame(
            'disabled',
            $this->eligibility(enabled: false)->ineligibilityReason($this->request(), 2)
        );
        self::assertSame(
            'store_excluded',
            $this->eligibility(storeIncluded: false)->ineligibilityReason($this->request(), 2)
        );
        self::assertSame(
            'readiness_closed',
            $this->eligibility(readinessOpen: false)->ineligibilityReason($this->request(), 2)
        );
        self::assertSame(
            'invalid_page',
            $this->eligibility()->ineligibilityReason($this->request(from: 100, size: 10), 2)
        );
    }

    private function eligibility(
        bool $enabled = true,
        bool $storeIncluded = true,
        bool $readinessOpen = true
    ): \MageOS\OpenSearchHybrid\SearchAdapter\Eligibility {
        $config = $this->createStub(\MageOS\OpenSearchHybrid\Model\Config::class);
        $config->method('isEnabled')->willReturn($enabled);
        $config->method('isStoreIncluded')->willReturn($storeIncluded);
        $readinessLatch = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Store\ReadinessLatch::class
        );
        $readinessLatch->method('isOpen')->willReturn($readinessOpen);

        return new \MageOS\OpenSearchHybrid\SearchAdapter\Eligibility(
            $config,
            $readinessLatch,
            new \MageOS\OpenSearchHybrid\SearchAdapter\SortEligibility()
        );
    }

    private function request(
        string $name = 'quick_search_container',
        array $sort = [['field' => 'relevance', 'direction' => 'DESC']],
        int $from = 0,
        int $size = 10
    ): \Magento\Framework\Search\RequestInterface {
        return new \Magento\Framework\Search\Request(
            $name,
            'catalogsearch_fulltext',
            $this->createStub(\Magento\Framework\Search\Request\QueryInterface::class),
            $from,
            $size,
            [],
            [],
            $sort
        );
    }
}
