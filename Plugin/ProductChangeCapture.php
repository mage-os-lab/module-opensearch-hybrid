<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Plugin;

class ProductChangeCapture
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Change\ProductStoreResolver $storeResolver,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\CaptureService $captureService
    ) {
    }

    public function aroundSave(
        \Magento\Catalog\Model\ResourceModel\Product $subject,
        \Closure $proceed,
        \Magento\Framework\Model\AbstractModel $product
    ) {
        $hadDataChanges = $product->hasDataChanges() || $product->getId() === null;
        if (!$hadDataChanges) {
            return $proceed($product);
        }
        $productId = (int)($product->getEntityId() ?: $product->getId());
        $previousStoreIds = $productId > 0
            ? $this->storeResolver->resolve($productId, (int)$product->getStoreId())
            : [];
        $subject->beginTransaction();
        try {
            $result = $proceed($product);
            $productId = (int)($product->getEntityId() ?: $product->getId());
            $currentStoreIds = $this->storeResolver->resolve(
                $productId,
                (int)$product->getStoreId()
            );
            $storeIds = array_values(array_unique(array_merge($previousStoreIds, $currentStoreIds)));
            sort($storeIds, SORT_NUMERIC);
            $capture = $this->captureService->captureProduct(
                $productId,
                $storeIds,
                'UPSERT',
                'product_resource_save'
            );
            $this->registerPublication($subject, $capture['jobs']);
            $subject->commit();
        } catch (\Throwable $throwable) {
            $subject->rollBack();
            throw $throwable;
        }

        return $result;
    }

    public function aroundDelete(
        \Magento\Catalog\Model\ResourceModel\Product $subject,
        \Closure $proceed,
        \Magento\Framework\Model\AbstractModel $product
    ) {
        $productId = (int)($product->getEntityId() ?: $product->getId());
        $storeIds = $this->storeResolver->resolve($productId, (int)$product->getStoreId());
        $subject->beginTransaction();
        try {
            $result = $proceed($product);
            $capture = $this->captureService->captureProduct(
                $productId,
                $storeIds,
                'DELETE',
                'product_resource_delete'
            );
            $this->registerPublication($subject, $capture['jobs']);
            $subject->commit();
        } catch (\Throwable $throwable) {
            $subject->rollBack();
            throw $throwable;
        }

        return $result;
    }

    private function registerPublication(
        \Magento\Catalog\Model\ResourceModel\Product $subject,
        array $jobs
    ): void {
        if ($jobs === []) {
            return;
        }
        $subject->addCommitCallback(function () use ($jobs): void {
            $this->captureService->publishJobs($jobs);
        });
    }
}
