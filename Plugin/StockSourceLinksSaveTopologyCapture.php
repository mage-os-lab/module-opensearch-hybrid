<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Plugin;

class StockSourceLinksSaveTopologyCapture
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Change\InventoryTopologyInvalidationService $invalidationService
    ) {
    }

    public function afterExecute(
        \Magento\InventoryApi\Api\StockSourceLinksSaveInterface $subject,
        mixed $result,
        array $links
    ): mixed {
        unset($subject);
        $this->invalidationService->invalidateStockSourceLinks(
            $links,
            'inventory_stock_source_links_save'
        );

        return $result;
    }
}
