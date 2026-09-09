<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Plugin;

class PriceIndexChangeCapture
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Change\ProductStoreResolver $storeResolver,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\SearchableProductResolver $productResolver,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\CaptureService $captureService,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\ChangeJournalRepository $journalRepository,
        private readonly \MageOS\OpenSearchHybrid\Model\Config $config,
        private readonly \Magento\Store\Model\StoreManagerInterface $storeManager
    ) {
    }

    public function aroundExecute(
        \Magento\Catalog\Model\Indexer\Product\Price $subject,
        \Closure $proceed,
        $ids
    ) {
        $result = $proceed($ids);
        $this->capture(is_array($ids) ? $ids : []);

        return $result;
    }

    public function aroundExecuteList(
        \Magento\Catalog\Model\Indexer\Product\Price $subject,
        \Closure $proceed,
        array $ids
    ) {
        $result = $proceed($ids);
        $this->capture($ids);

        return $result;
    }

    public function aroundExecuteRow(
        \Magento\Catalog\Model\Indexer\Product\Price $subject,
        \Closure $proceed,
        $id
    ) {
        $result = $proceed($id);
        $this->capture([(int)$id]);

        return $result;
    }

    public function aroundExecuteFull(
        \Magento\Catalog\Model\Indexer\Product\Price $subject,
        \Closure $proceed
    ) {
        $result = $proceed();
        $batchSize = $this->config->batchSize();
        foreach ($this->storeManager->getStores(false) as $store) {
            $storeId = (int)$store->getId();
            if ($storeId <= 0 || !$this->config->isStoreIncluded($storeId)) {
                continue;
            }
            $afterProductId = 0;
            do {
                $productIds = $this->productResolver->nextEligibleIds(
                    $storeId,
                    $afterProductId,
                    $batchSize
                );
                if ($productIds === []) {
                    break;
                }
                $this->captureByStore(
                    [$storeId => $productIds],
                    'product_price_full_reindex'
                );
                $afterProductId = max($productIds);
            } while (count($productIds) === $batchSize);
        }

        return $result;
    }

    private function capture(array $productIds): void
    {
        $this->captureByStore(
            $this->storeResolver->resolveMany($productIds),
            'product_price_index'
        );
    }

    private function captureByStore(array $productIdsByStore, string $reason): void
    {
        $capture = $this->captureService->capturePriorityProductsByStore(
            $productIdsByStore,
            $reason
        );
        $this->journalRepository->acknowledgeNativeChanges(array_column($capture['changes'], 'change_id'));
        $this->captureService->publishJobs($capture['jobs']);
    }
}
