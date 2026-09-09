<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Plugin;

class StockSalesChannelTopologyCapture
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Change\InventoryTopologyInvalidationService $invalidationService
    ) {
    }

    public function aroundExecute(
        \Magento\InventorySalesApi\Model\ReplaceSalesChannelsForStockInterface $subject,
        \Closure $proceed,
        array $salesChannels,
        int $stockId
    ): void {
        unset($subject);
        $before = $this->invalidationService->salesChannelSnapshot();
        $proceed($salesChannels, $stockId);
        $this->invalidationService->invalidateSalesChannelChanges(
            $before,
            $this->invalidationService->salesChannelSnapshot()
        );
    }
}
