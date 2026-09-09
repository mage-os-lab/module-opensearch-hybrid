<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Change;

class InventoryProductResolver
{
    public function __construct(
        private readonly \Magento\InventoryCatalogApi\Model\GetProductIdsBySkusInterface $getProductIdsBySkus,
        private readonly \Magento\InventoryCatalogApi\Model\GetParentSkusOfChildrenSkusInterface $getParentSkus,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\ProductStoreResolver $storeResolver,
        private readonly \Magento\InventoryCatalog\Model\GetStockIdForByStoreId $getStockIdForStore
    ) {
    }

    public function productIds(array $skus): array
    {
        $allSkus = $this->normalizeSkus($skus);
        $frontier = array_values($allSkus);
        while ($frontier !== []) {
            $parentsByChild = $this->getParentSkus->execute($frontier);
            $next = [];
            foreach ($parentsByChild as $parentSkus) {
                foreach ($this->normalizeSkus((array)$parentSkus) as $parentSku) {
                    if (!isset($allSkus[$parentSku])) {
                        $allSkus[$parentSku] = $parentSku;
                        $next[$parentSku] = $parentSku;
                    }
                }
            }
            $frontier = array_values($next);
        }
        if ($allSkus === []) {
            return [];
        }
        $productIds = array_values(array_unique(array_filter(
            array_map('intval', array_values($this->getProductIdsBySkus->execute(array_values($allSkus)))),
            static fn (int $productId): bool => $productId > 0
        )));
        sort($productIds, SORT_NUMERIC);

        return $productIds;
    }

    public function resolveByStore(array $skus, ?int $stockId = null): array
    {
        $resolved = $this->storeResolver->resolveMany($this->productIds($skus));
        if ($stockId === null) {
            return $resolved;
        }
        foreach (array_keys($resolved) as $storeId) {
            if ($this->getStockIdForStore->execute((int)$storeId) !== $stockId) {
                unset($resolved[$storeId]);
            }
        }

        return $resolved;
    }

    private function normalizeSkus(array $skus): array
    {
        $normalized = [];
        foreach ($skus as $sku) {
            if (!is_scalar($sku)) {
                continue;
            }
            $sku = trim((string)$sku);
            if ($sku !== '') {
                $normalized[$sku] = $sku;
            }
        }

        return $normalized;
    }
}
