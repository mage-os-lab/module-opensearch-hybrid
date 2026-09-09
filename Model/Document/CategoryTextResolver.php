<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Document;

class CategoryTextResolver
{
    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \Magento\Catalog\Model\ResourceModel\Category\CollectionFactory $collectionFactory
    ) {
    }

    public function resolve(array $categoryIds, int $storeId): array
    {
        $categoryIds = array_values(array_unique(array_filter(
            array_map('intval', $categoryIds),
            static fn (int $categoryId): bool => $categoryId > 0
        )));
        if ($categoryIds === []) {
            return [];
        }
        $connection = $this->resourceConnection->getConnection();
        $categoryTable = $this->resourceConnection->getTableName('catalog_category_entity');
        $paths = $connection->fetchPairs(
            $connection->select()
                ->from($categoryTable, ['entity_id', 'path'])
                ->where('entity_id IN (?)', $categoryIds)
        );
        $pathIds = [];
        foreach ($paths as $path) {
            foreach (explode('/', (string)$path) as $pathId) {
                if ((int)$pathId > 1) {
                    $pathIds[] = (int)$pathId;
                }
            }
        }
        $pathIds = array_values(array_unique($pathIds));
        if ($pathIds === []) {
            return [];
        }
        $collection = $this->collectionFactory->create();
        $collection->setStoreId($storeId);
        $collection->addAttributeToSelect('name');
        $collection->addFieldToFilter('entity_id', ['in' => $pathIds]);
        $names = [];
        foreach ($collection as $category) {
            $name = trim((string)$category->getName());
            if ($name !== '') {
                $names[(int)$category->getId()] = $name;
            }
        }
        $resolved = [];
        foreach ($categoryIds as $categoryId) {
            if (!isset($paths[$categoryId])) {
                continue;
            }
            $parts = [];
            foreach (explode('/', (string)$paths[$categoryId]) as $pathId) {
                if (isset($names[(int)$pathId])) {
                    $parts[] = $names[(int)$pathId];
                }
            }
            if ($parts !== []) {
                $resolved[] = implode(' / ', $parts);
            }
        }
        $resolved = array_values(array_unique($resolved));
        sort($resolved, SORT_NATURAL | SORT_FLAG_CASE);

        return $resolved;
    }
}
