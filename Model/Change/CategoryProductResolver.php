<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Change;

class CategoryProductResolver
{
    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection
    ) {
    }

    public function descendantProductIds(int $categoryId): array
    {
        $connection = $this->resourceConnection->getConnection();
        $categoryTable = $this->resourceConnection->getTableName('catalog_category_entity');
        $categoryProductTable = $this->resourceConnection->getTableName('catalog_category_product');
        $path = $connection->fetchOne(
            $connection->select()
                ->from($categoryTable, ['path'])
                ->where('entity_id = ?', $categoryId)
        );
        if (!is_string($path) || $path === '') {
            return [];
        }
        $select = $connection->select()
            ->distinct()
            ->from(['category' => $categoryTable], [])
            ->joinInner(
                ['relation' => $categoryProductTable],
                'relation.category_id = category.entity_id',
                ['product_id']
            )
            ->where('category.path = ?', $path)
            ->orWhere('category.path LIKE ?', $path . '/%')
            ->order('relation.product_id ASC');

        return array_map('intval', $connection->fetchCol($select));
    }
}
