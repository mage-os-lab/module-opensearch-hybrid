<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Change;

class ProductStoreResolver
{
    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \MageOS\OpenSearchHybrid\Model\Config $config
    ) {
    }

    public function resolve(int $productId, int $contextStoreId = 0): array
    {
        $resolved = $this->resolveMany([$productId], $contextStoreId);
        $storeIds = [];
        foreach ($resolved as $storeId => $productIds) {
            if (in_array($productId, $productIds, true)) {
                $storeIds[] = (int)$storeId;
            }
        }

        return $storeIds;
    }

    public function resolveMany(array $productIds, int $contextStoreId = 0): array
    {
        $productIds = array_values(array_unique(array_filter(
            array_map('intval', $productIds),
            static fn (int $productId): bool => $productId > 0
        )));
        if ($productIds === []) {
            return [];
        }
        $connection = $this->resourceConnection->getConnection();
        $websiteTable = $this->resourceConnection->getTableName('catalog_product_website');
        $storeTable = $this->resourceConnection->getTableName('store');
        $select = $connection->select()
            ->from(['assignment' => $websiteTable], ['product_id'])
            ->joinInner(
                ['store' => $storeTable],
                'store.website_id = assignment.website_id',
                ['store_id']
            )
            ->where('assignment.product_id IN (?)', $productIds)
            ->where('store.store_id != ?', 0);
        if ($contextStoreId > 0) {
            $select->where('store.store_id = ?', $contextStoreId);
        }
        $resolved = [];
        foreach ($connection->fetchAll($select) as $row) {
            $storeId = (int)$row['store_id'];
            if (!$this->config->isStoreIncluded($storeId)) {
                continue;
            }
            $resolved[$storeId][] = (int)$row['product_id'];
        }
        ksort($resolved, SORT_NUMERIC);
        foreach ($resolved as &$storeProductIds) {
            $storeProductIds = array_values(array_unique($storeProductIds));
            sort($storeProductIds, SORT_NUMERIC);
        }
        unset($storeProductIds);

        return $resolved;
    }

    public function resolveWebsiteStores(array $productIds, array $websiteIds): array
    {
        $productIds = array_values(array_unique(array_filter(
            array_map('intval', $productIds),
            static fn (int $productId): bool => $productId > 0
        )));
        $websiteIds = array_values(array_unique(array_filter(
            array_map('intval', $websiteIds),
            static fn (int $websiteId): bool => $websiteId > 0
        )));
        if ($productIds === [] || $websiteIds === []) {
            return [];
        }
        $connection = $this->resourceConnection->getConnection();
        $storeIds = array_map('intval', $connection->fetchCol(
            $connection->select()
                ->from($this->resourceConnection->getTableName('store'), ['store_id'])
                ->where('website_id IN (?)', $websiteIds)
                ->where('store_id != ?', 0)
                ->order('store_id ASC')
        ));
        $resolved = [];
        foreach ($storeIds as $storeId) {
            if ($this->config->isStoreIncluded($storeId)) {
                $resolved[$storeId] = $productIds;
            }
        }

        return $resolved;
    }
}
