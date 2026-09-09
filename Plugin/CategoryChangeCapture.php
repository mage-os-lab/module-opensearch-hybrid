<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Plugin;

class CategoryChangeCapture
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Change\CategoryProductResolver $categoryProductResolver,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\ProductStoreResolver $storeResolver,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\CaptureService $captureService,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\NativeInvalidationScheduler $nativeScheduler
    ) {
    }

    public function aroundSave(
        \Magento\Catalog\Model\ResourceModel\Category $subject,
        \Closure $proceed,
        \Magento\Framework\Model\AbstractModel $category
    ) {
        $nameChanged = $category->dataHasChangedFor('name');
        if (!$nameChanged) {
            return $proceed($category);
        }
        $subject->beginTransaction();
        try {
            $result = $proceed($category);
            if ((int)$category->getId() > 0) {
                $productIds = $this->categoryProductResolver->descendantProductIds((int)$category->getId());
                $productIdsByStore = $this->storeResolver->resolveMany(
                    $productIds,
                    (int)$category->getStoreId()
                );
                $capture = $this->captureService->captureProductsByStore(
                    $productIdsByStore,
                    'category_name_save'
                );
                if ($capture['jobs'] !== []) {
                    $subject->addCommitCallback(function () use ($capture, $productIds): void {
                        $this->captureService->publishJobs($capture['jobs']);
                        $this->nativeScheduler->schedule($productIds);
                    });
                }
            }
            $subject->commit();
        } catch (\Throwable $throwable) {
            $subject->rollBack();
            throw $throwable;
        }

        return $result;
    }
}
