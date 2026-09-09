<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Document;

class PriceIndexResolver
{
    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \Magento\Store\Model\StoreManagerInterface $storeManager,
        private readonly \Magento\Catalog\Model\Indexer\Product\Price\PriceTableResolver $priceTableResolver,
        private readonly \Magento\Catalog\Model\Indexer\Product\Price\DimensionModeConfiguration $dimensionMode
    ) {
    }

    public function resolve(int $productId, int $storeId): array
    {
        $websiteId = (int)$this->storeManager->getStore($storeId)->getWebsiteId();
        $connection = $this->resourceConnection->getConnection();
        $customerGroupIds = array_map('intval', $connection->fetchCol(
            $connection->select()
                ->from(
                    $this->resourceConnection->getTableName('customer_group'),
                    ['customer_group_id']
                )
                ->order('customer_group_id ASC')
        ));
        if ($customerGroupIds === []) {
            return [];
        }

        $dimensionNames = array_fill_keys($this->dimensionMode->getDimensionConfiguration(), true);
        $groupIdsByTable = [];
        foreach ($customerGroupIds as $customerGroupId) {
            $dimensions = [];
            if (isset($dimensionNames[\Magento\Store\Model\Indexer\WebsiteDimensionProvider::DIMENSION_NAME])) {
                $dimensions[] = new \Magento\Framework\Indexer\Dimension(
                    \Magento\Store\Model\Indexer\WebsiteDimensionProvider::DIMENSION_NAME,
                    (string)$websiteId
                );
            }
            if (isset($dimensionNames[\Magento\Customer\Model\Indexer\CustomerGroupDimensionProvider::DIMENSION_NAME])) {
                $dimensions[] = new \Magento\Framework\Indexer\Dimension(
                    \Magento\Customer\Model\Indexer\CustomerGroupDimensionProvider::DIMENSION_NAME,
                    (string)$customerGroupId
                );
            }
            $table = $this->priceTableResolver->resolve('catalog_product_index_price', $dimensions);
            $groupIdsByTable[$table][] = $customerGroupId;
        }

        $rows = [];
        foreach ($groupIdsByTable as $table => $groupIds) {
            array_push($rows, ...$connection->fetchAll(
                $connection->select()
                    ->from(
                        $table,
                        [
                            'customer_group_id',
                            'website_id',
                            'tax_class_id',
                            'price',
                            'final_price',
                            'min_price',
                            'max_price',
                            'tier_price',
                        ]
                    )
                    ->where('entity_id = ?', $productId)
                    ->where('website_id = ?', $websiteId)
                    ->where('customer_group_id IN (?)', $groupIds)
            ));
        }
        usort($rows, static fn (array $left, array $right): int =>
            [(int)$left['customer_group_id'], (int)$left['website_id']]
            <=> [(int)$right['customer_group_id'], (int)$right['website_id']]
        );

        return array_map(static fn (array $row): array => [
            'customer_group_id' => (int)$row['customer_group_id'],
            'website_id' => (int)$row['website_id'],
            'tax_class_id' => (int)$row['tax_class_id'],
            'regular_price' => (float)$row['price'],
            'final_price' => (float)$row['final_price'],
            'min_price' => (float)$row['min_price'],
            'max_price' => (float)$row['max_price'],
            'tier_price' => $row['tier_price'] === null ? null : (float)$row['tier_price'],
        ], $rows);
    }
}
