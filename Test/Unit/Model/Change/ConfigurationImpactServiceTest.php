<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Change;

class ConfigurationImpactServiceTest extends \PHPUnit\Framework\TestCase
{
    public function testIncludedStorePreviewReportsOnlyChangedMembershipAndExactCoverage(): void
    {
        $invalidation = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Change\ScopeInvalidationService::class
        );
        $invalidation->method('storeIdsWithGenerations')->willReturn([1, 2, 3]);
        $resolver = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Change\SearchableProductResolver::class
        );
        $resolver->method('countEligible')->willReturnCallback(
            static fn (int $storeId): int => [1 => 12, 3 => 20][$storeId]
        );
        $service = new \MageOS\OpenSearchHybrid\Model\Change\ConfigurationImpactService(
            $invalidation,
            $resolver
        );

        $preview = $service->preview(
            \MageOS\OpenSearchHybrid\Model\Config::XML_PATH_INCLUDED_STORES,
            'default',
            0,
            '1,2',
            '2,3'
        );

        self::assertSame([1, 3], $preview['affected_store_ids']);
        self::assertSame(32, $preview['estimated_documents']);
        self::assertSame([
            ['store_id' => 1, 'estimated_documents' => 12],
            ['store_id' => 3, 'estimated_documents' => 20],
        ], $preview['stores']);
        self::assertTrue($preview['requires_replacement_generation']);
        self::assertArrayNotHasKey('previous_value', $preview);
        self::assertArrayNotHasKey('proposed_value', $preview);
    }

    public function testWebsiteScopedPreviewUsesOnlyStoresWithGenerations(): void
    {
        $invalidation = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Change\ScopeInvalidationService::class
        );
        $invalidation->method('storeIdsWithGenerations')->willReturn([1, 2, 3]);
        $invalidation->expects($this->once())->method('storeIdsForWebsite')->with(4)->willReturn([2, 3]);
        $resolver = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Change\SearchableProductResolver::class
        );
        $resolver->method('countEligible')->willReturn(7);
        $service = new \MageOS\OpenSearchHybrid\Model\Change\ConfigurationImpactService(
            $invalidation,
            $resolver
        );

        $preview = $service->preview('general/locale/code', 'websites', 4);

        self::assertSame([2, 3], $preview['affected_store_ids']);
        self::assertSame(14, $preview['estimated_documents']);
    }

    public function testRejectsUnsupportedPathsAndScopes(): void
    {
        $service = new \MageOS\OpenSearchHybrid\Model\Change\ConfigurationImpactService(
            $this->createStub(\MageOS\OpenSearchHybrid\Model\Change\ScopeInvalidationService::class),
            $this->createStub(\MageOS\OpenSearchHybrid\Model\Change\SearchableProductResolver::class)
        );

        $this->expectException(\InvalidArgumentException::class);
        $service->preview('catalog/search/opensearch_password', 'default', 0);
    }
}
