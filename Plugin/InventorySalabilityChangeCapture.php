<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Plugin;

class InventorySalabilityChangeCapture
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Change\InventoryProductResolver $productResolver,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\CaptureService $captureService,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\InventoryCaptureContext $captureContext
    ) {
    }

    public function aroundExecute(
        \Magento\InventoryCatalogSearch\Model\UpdateFulltextIndexOnProductSalabilityChange $subject,
        \Closure $proceed,
        array $skus
    ): void {
        if ($this->captureContext->isSourceItemsSave()) {
            $proceed($skus);
            return;
        }
        $capture = $this->captureService->capturePriorityProductsByStore(
            $this->productResolver->resolveByStore($skus),
            'inventory_salability_change'
        );
        $proceed($skus);
        $this->captureService->publishJobs($capture['jobs']);
    }
}
