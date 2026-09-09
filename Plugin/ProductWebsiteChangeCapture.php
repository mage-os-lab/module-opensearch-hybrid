<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Plugin;

class ProductWebsiteChangeCapture
{
    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\ProductStoreResolver $storeResolver,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\CaptureService $captureService
    ) {
    }

    public function aroundAddProducts(
        \Magento\Catalog\Model\Product\Website $subject,
        \Closure $proceed,
        $websiteIds,
        $productIds
    ) {
        return $this->captureChange(
            $proceed,
            $websiteIds,
            $productIds,
            'product_website_add'
        );
    }

    public function aroundRemoveProducts(
        \Magento\Catalog\Model\Product\Website $subject,
        \Closure $proceed,
        $websiteIds,
        $productIds
    ) {
        return $this->captureChange(
            $proceed,
            $websiteIds,
            $productIds,
            'product_website_remove'
        );
    }

    private function captureChange(
        \Closure $proceed,
        $websiteIds,
        $productIds,
        string $reason
    ) {
        if (!is_array($websiteIds)
            || !is_array($productIds)
            || $websiteIds === []
            || $productIds === []
        ) {
            return $proceed($websiteIds, $productIds);
        }

        $connection = $this->resourceConnection->getConnection();
        $connection->beginTransaction();
        try {
            $result = $proceed($websiteIds, $productIds);
            $capture = $this->captureService->captureProductsByStore(
                $this->storeResolver->resolveWebsiteStores($productIds, $websiteIds),
                $reason
            );
            $connection->commit();
        } catch (\Throwable $throwable) {
            $connection->rollBack();
            throw $throwable;
        }
        $this->captureService->publishJobs($capture['jobs']);

        return $result;
    }
}
