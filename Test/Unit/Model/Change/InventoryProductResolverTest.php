<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Change;

class InventoryProductResolverTest extends \PHPUnit\Framework\TestCase
{
    public function testResolvesRecursiveCompositeParentsAndFiltersStoresByReservationStock(): void
    {
        $productIds = $this->createMock(
            \Magento\InventoryCatalogApi\Model\GetProductIdsBySkusInterface::class
        );
        $productIds->expects($this->once())
            ->method('execute')
            ->with(['child', 'parent', 'grandparent'])
            ->willReturn(['child' => 10, 'parent' => 20, 'grandparent' => 30]);
        $parents = $this->createMock(
            \Magento\InventoryCatalogApi\Model\GetParentSkusOfChildrenSkusInterface::class
        );
        $parents->expects($this->exactly(3))
            ->method('execute')
            ->willReturnCallback(static fn (array $skus): array => match ($skus) {
                ['child'] => ['child' => ['parent']],
                ['parent'] => ['parent' => ['grandparent']],
                ['grandparent'] => ['grandparent' => []],
            });
        $storeResolver = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Change\ProductStoreResolver::class
        );
        $storeResolver->expects($this->once())
            ->method('resolveMany')
            ->with([10, 20, 30])
            ->willReturn([1 => [10, 20, 30], 2 => [10, 20]]);
        $stockResolver = $this->createStub(
            \Magento\InventoryCatalog\Model\GetStockIdForByStoreId::class
        );
        $stockResolver->method('execute')->willReturnMap([[1, 5], [2, 8]]);
        $resolver = new \MageOS\OpenSearchHybrid\Model\Change\InventoryProductResolver(
            $productIds,
            $parents,
            $storeResolver,
            $stockResolver
        );

        self::assertSame([1 => [10, 20, 30]], $resolver->resolveByStore([' child ', 'child'], 5));
    }

    public function testRejectsEmptyAndNonScalarSkuInputWithoutQueries(): void
    {
        $productIds = $this->createMock(
            \Magento\InventoryCatalogApi\Model\GetProductIdsBySkusInterface::class
        );
        $productIds->expects($this->never())->method('execute');
        $parents = $this->createMock(
            \Magento\InventoryCatalogApi\Model\GetParentSkusOfChildrenSkusInterface::class
        );
        $parents->expects($this->never())->method('execute');
        $storeResolver = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Change\ProductStoreResolver::class
        );
        $storeResolver->expects($this->once())->method('resolveMany')->with([])->willReturn([]);
        $resolver = new \MageOS\OpenSearchHybrid\Model\Change\InventoryProductResolver(
            $productIds,
            $parents,
            $storeResolver,
            $this->createStub(\Magento\InventoryCatalog\Model\GetStockIdForByStoreId::class)
        );

        self::assertSame([], $resolver->resolveByStore([' ', [], new \stdClass()]));
    }
}
