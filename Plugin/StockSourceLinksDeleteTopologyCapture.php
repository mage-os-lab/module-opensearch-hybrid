<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Plugin;

class StockSourceLinksDeleteTopologyCapture
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Change\InventoryTopologyInvalidationService $invalidationService
    ) {
    }

    public function afterExecute(
        \Magento\InventoryApi\Api\StockSourceLinksDeleteInterface $subject,
        mixed $result,
        array $links
    ): mixed {
        unset($subject);
        $this->invalidationService->invalidateStockSourceLinks(
            $links,
            'inventory_stock_source_links_delete'
        );

        return $result;
    }
}
