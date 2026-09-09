<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Change;

class AttributeOptionProductResolver
{
    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\ProductStoreResolver $storeResolver
    ) {
    }

    public function labels(array $optionIds): array
    {
        $optionIds = $this->ids($optionIds);
        if ($optionIds === []) {
            return [];
        }
        $connection = $this->resourceConnection->getConnection();
        $rows = $connection->fetchAll(
            $connection->select()
                ->from(
                    $this->resourceConnection->getTableName('eav_attribute_option_value'),
                    ['option_id', 'store_id', 'value']
                )
                ->where('option_id IN (?)', $optionIds)
        );
        $labels = [];
        foreach ($rows as $row) {
            $labels[(int)$row['option_id']][(int)$row['store_id']] = (string)$row['value'];
        }

        return $labels;
    }

    public function resolve(
        \Magento\Eav\Model\Entity\Attribute\AbstractAttribute $attribute,
        array $optionIds,
        ?array $storeIds
    ): array {
        $optionIds = $this->ids($optionIds);
        if ($optionIds === [] || !$attribute->getId() || !$attribute->getBackendTable()) {
            return [];
        }
        $connection = $this->resourceConnection->getConnection();
        $select = $connection->select()
            ->distinct()
            ->from($attribute->getBackendTable(), ['entity_id'])
            ->where('attribute_id = ?', (int)$attribute->getId());
        if ((string)$attribute->getFrontendInput() === 'multiselect') {
            $conditions = [];
            foreach ($optionIds as $optionId) {
                $conditions[] = $connection->prepareSqlCondition('value', ['finset' => $optionId]);
            }
            $select->where(implode(' OR ', $conditions));
        } else {
            $select->where('value IN (?)', $optionIds);
        }
        $productIds = array_map('intval', $connection->fetchCol($select));
        if ($storeIds === null) {
            return $this->storeResolver->resolveMany($productIds);
        }
        $resolved = [];
        foreach ($this->ids($storeIds) as $storeId) {
            foreach ($this->storeResolver->resolveMany($productIds, $storeId) as $resolvedStoreId => $ids) {
                $resolved[(int)$resolvedStoreId] = array_values(array_unique(array_merge(
                    $resolved[(int)$resolvedStoreId] ?? [],
                    $ids
                )));
            }
        }

        return $resolved;
    }

    private function ids(array $values): array
    {
        $ids = array_values(array_unique(array_filter(
            array_map('intval', $values),
            static fn (int $value): bool => $value > 0
        )));
        sort($ids, SORT_NUMERIC);

        return $ids;
    }
}
