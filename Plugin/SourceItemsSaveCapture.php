<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Plugin;

class SourceItemsSaveCapture
{
    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\InventoryProductResolver $productResolver,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\CaptureService $captureService,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\InventoryCaptureContext $captureContext
    ) {
    }

    public function aroundExecute(
        \Magento\InventoryApi\Api\SourceItemsSaveInterface $subject,
        \Closure $proceed,
        array $sourceItems
    ): void {
        $skus = [];
        foreach ($sourceItems as $sourceItem) {
            $sku = trim((string)$sourceItem->getSku());
            if ($sku !== '') {
                $skus[$sku] = true;
            }
        }
        if ($skus === []) {
            $proceed($sourceItems);
            return;
        }

        $productsByStore = $this->productResolver->resolveByStore(array_keys($skus));
        if ($productsByStore === []) {
            $proceed($sourceItems);
            return;
        }

        $connection = $this->resourceConnection->getConnection();
        $connection->beginTransaction();
        $this->captureContext->enterSourceItemsSave();
        try {
            $proceed($sourceItems);
            $capture = $this->captureService->capturePriorityProductsByStore(
                $productsByStore,
                'inventory_salability_change'
            );
            $connection->commit();
        } catch (\Throwable $throwable) {
            $connection->rollBack();
            throw $throwable;
        } finally {
            $this->captureContext->leaveSourceItemsSave();
        }
        $this->captureService->publishJobs($capture['jobs']);
    }
}
