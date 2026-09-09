<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Change;

class InventoryTopologyInvalidationService
{
    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\ScopeInvalidationService $invalidationService
    ) {
    }

    public function sourceStatusChanged(string $sourceCode, bool $enabled): bool
    {
        $sourceCode = trim($sourceCode);
        if ($sourceCode === '') {
            return false;
        }
        $connection = $this->resourceConnection->getConnection();
        $current = $connection->fetchOne(
            $connection->select()
                ->from($this->resourceConnection->getTableName('inventory_source'), ['enabled'])
                ->where('source_code = ?', $sourceCode)
                ->limit(1)
        );

        return $current !== false && (bool)$current !== $enabled;
    }

    public function salesChannelSnapshot(): array
    {
        $connection = $this->resourceConnection->getConnection();
        $rows = $connection->fetchAll(
            $connection->select()
                ->from(['channel' => $this->resourceConnection->getTableName(
                    'inventory_stock_sales_channel'
                )], ['stock_id'])
                ->joinInner(
                    ['website' => $this->resourceConnection->getTableName('store_website')],
                    'website.code = channel.code',
                    ['website_id']
                )
                ->where('channel.type = ?', 'website')
                ->order('website.website_id ASC')
        );
        $snapshot = [];
        foreach ($rows as $row) {
            $snapshot[(int)$row['website_id']] = (int)$row['stock_id'];
        }

        return $snapshot;
    }

    public function invalidateStockSourceLinks(array $links, string $reason): array
    {
        $stockIds = [];
        foreach ($links as $link) {
            if (!$link instanceof \Magento\InventoryApi\Api\Data\StockSourceLinkInterface) {
                continue;
            }
            $stockId = (int)$link->getStockId();
            if ($stockId > 0) {
                $stockIds[$stockId] = $stockId;
            }
        }

        return $this->invalidateStockIds(array_values($stockIds), $reason);
    }

    public function invalidateSource(string $sourceCode, string $reason): array
    {
        $sourceCode = trim($sourceCode);
        if ($sourceCode === '') {
            return [];
        }
        $connection = $this->resourceConnection->getConnection();
        $stockIds = array_map('intval', $connection->fetchCol(
            $connection->select()
                ->from(
                    $this->resourceConnection->getTableName('inventory_source_stock_link'),
                    ['stock_id']
                )
                ->where('source_code = ?', $sourceCode)
                ->order('stock_id ASC')
        ));

        return $this->invalidateStockIds($stockIds, $reason);
    }

    public function invalidateSalesChannelChanges(array $before, array $after): array
    {
        $websiteIds = array_values(array_unique(array_merge(array_keys($before), array_keys($after))));
        sort($websiteIds, SORT_NUMERIC);
        $changes = [];
        foreach ($websiteIds as $websiteId) {
            $websiteId = (int)$websiteId;
            if (($before[$websiteId] ?? null) === ($after[$websiteId] ?? null)) {
                continue;
            }
            foreach ($this->invalidationService->invalidateStores(
                $this->invalidationService->storeIdsForWebsite($websiteId),
                'INVENTORY_SALES_CHANNEL',
                $websiteId,
                'inventory_sales_channel_change'
            ) as $change) {
                $changes[] = $change;
            }
        }

        return $changes;
    }

    private function invalidateStockIds(array $stockIds, string $reason): array
    {
        $stockIds = array_values(array_unique(array_filter(
            array_map('intval', $stockIds),
            static fn (int $stockId): bool => $stockId > 0
        )));
        sort($stockIds, SORT_NUMERIC);
        $changes = [];
        foreach ($stockIds as $stockId) {
            foreach ($this->invalidationService->invalidateStores(
                $this->storeIdsForStock($stockId),
                'INVENTORY_STOCK',
                $stockId,
                $reason
            ) as $change) {
                $changes[] = $change;
            }
        }

        return $changes;
    }

    private function storeIdsForStock(int $stockId): array
    {
        $connection = $this->resourceConnection->getConnection();

        return array_map('intval', $connection->fetchCol(
            $connection->select()
                ->from(['channel' => $this->resourceConnection->getTableName(
                    'inventory_stock_sales_channel'
                )], [])
                ->joinInner(
                    ['website' => $this->resourceConnection->getTableName('store_website')],
                    'website.code = channel.code',
                    []
                )
                ->joinInner(
                    ['store' => $this->resourceConnection->getTableName('store')],
                    'store.website_id = website.website_id',
                    ['store_id']
                )
                ->where('channel.type = ?', 'website')
                ->where('channel.stock_id = ?', $stockId)
                ->where('store.store_id > ?', 0)
                ->group('store.store_id')
                ->order('store.store_id ASC')
        ));
    }
}
