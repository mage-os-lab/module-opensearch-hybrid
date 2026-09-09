<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Plugin;

class BulkAttributeChangeCapture
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Change\ProductStoreResolver $storeResolver,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\CaptureService $captureService
    ) {
    }

    public function aroundUpdateAttributes(
        \Magento\Catalog\Model\ResourceModel\Product\Action $subject,
        \Closure $proceed,
        $entityIds,
        $attrData,
        $storeId
    ) {
        $productIds = array_values(array_unique(array_filter(
            array_map('intval', is_array($entityIds) ? $entityIds : []),
            static fn (int $productId): bool => $productId > 0
        )));
        if ($productIds === [] || !is_array($attrData) || $attrData === []) {
            return $proceed($entityIds, $attrData, $storeId);
        }

        $subject->beginTransaction();
        try {
            $result = $proceed($entityIds, $attrData, $storeId);
            $capture = $this->captureService->captureProductsByStore(
                $this->storeResolver->resolveMany($productIds, (int)$storeId),
                'product_attribute_bulk_update'
            );
            if ($capture['jobs'] !== []) {
                $subject->addCommitCallback(function () use ($capture): void {
                    $this->captureService->publishJobs($capture['jobs']);
                });
            }
            $subject->commit();

            return $result;
        } catch (\Throwable $throwable) {
            $subject->rollBack();
            throw $throwable;
        }
    }
}
