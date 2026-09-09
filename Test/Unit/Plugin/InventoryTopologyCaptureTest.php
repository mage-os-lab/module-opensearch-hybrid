<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Plugin;

class InventoryTopologyCaptureTest extends \PHPUnit\Framework\TestCase
{
    public function testSalesChannelReplacementComparesTheCompleteMapAfterMutation(): void
    {
        $service = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Change\InventoryTopologyInvalidationService::class
        );
        $service->expects($this->exactly(2))
            ->method('salesChannelSnapshot')
            ->willReturnOnConsecutiveCalls([1 => 1], [1 => 5]);
        $service->expects($this->once())
            ->method('invalidateSalesChannelChanges')
            ->with([1 => 1], [1 => 5]);
        $plugin = new \MageOS\OpenSearchHybrid\Plugin\StockSalesChannelTopologyCapture($service);
        $mutated = false;

        $plugin->aroundExecute(
            $this->createStub(\Magento\InventorySalesApi\Model\ReplaceSalesChannelsForStockInterface::class),
            static function (array $channels, int $stockId) use (&$mutated): void {
                $mutated = $channels === [] && $stockId === 5;
            },
            [],
            5
        );

        self::assertTrue($mutated);
    }

    public function testSourceStatusChangeInvalidatesOnlyAfterAChangedSourceIsSaved(): void
    {
        $service = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Change\InventoryTopologyInvalidationService::class
        );
        $service->expects($this->once())
            ->method('sourceStatusChanged')
            ->with('secondary', false)
            ->willReturn(true);
        $service->expects($this->once())
            ->method('invalidateSource')
            ->with('secondary', 'inventory_source_status_change');
        $plugin = new \MageOS\OpenSearchHybrid\Plugin\SourceStatusTopologyCapture($service);
        $source = $this->createStub(\Magento\InventoryApi\Api\Data\SourceInterface::class);
        $source->method('getSourceCode')->willReturn('secondary');
        $source->method('isEnabled')->willReturn(false);
        $saved = false;

        $plugin->aroundSave(
            $this->createStub(\Magento\InventoryApi\Api\SourceRepositoryInterface::class),
            static function () use (&$saved): void {
                $saved = true;
            },
            $source
        );

        self::assertTrue($saved);
    }

    public function testStockSourceLinkPluginsUseDistinctRebuildReasons(): void
    {
        $service = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Change\InventoryTopologyInvalidationService::class
        );
        $link = $this->createStub(\Magento\InventoryApi\Api\Data\StockSourceLinkInterface::class);
        $calls = [];
        $service->expects($this->exactly(2))
            ->method('invalidateStockSourceLinks')
            ->willReturnCallback(static function (array $links, string $reason) use (&$calls): array {
                $calls[] = [$links, $reason];
                return [];
            });

        $save = new \MageOS\OpenSearchHybrid\Plugin\StockSourceLinksSaveTopologyCapture($service);
        $delete = new \MageOS\OpenSearchHybrid\Plugin\StockSourceLinksDeleteTopologyCapture($service);
        self::assertNull($save->afterExecute(
            $this->createStub(\Magento\InventoryApi\Api\StockSourceLinksSaveInterface::class),
            null,
            [$link]
        ));
        self::assertNull($delete->afterExecute(
            $this->createStub(\Magento\InventoryApi\Api\StockSourceLinksDeleteInterface::class),
            null,
            [$link]
        ));
        self::assertSame([
            [[$link], 'inventory_stock_source_links_save'],
            [[$link], 'inventory_stock_source_links_delete'],
        ], $calls);
    }
}
