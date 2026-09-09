<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Change;

class SearchableProductResolver
{
    public function __construct(
        private readonly \Magento\Catalog\Model\ResourceModel\Product\CollectionFactory $collectionFactory,
        private readonly \Magento\Catalog\Model\Product\Visibility $visibility
    ) {
    }

    public function eligibleIds(int $storeId, array $productIds): array
    {
        $productIds = array_values(array_unique(array_filter(
            array_map('intval', $productIds),
            static fn (int $productId): bool => $productId > 0
        )));
        if ($storeId <= 0 || $productIds === []) {
            return [];
        }
        $collection = $this->eligibleCollection($storeId);
        $collection->addIdFilter($productIds);
        $eligibleIds = array_map('intval', $collection->getAllIds());
        sort($eligibleIds, SORT_NUMERIC);

        return $eligibleIds;
    }

    public function nextEligibleIds(int $storeId, int $afterProductId, int $batchSize): array
    {
        if ($storeId <= 0 || $afterProductId < 0 || $batchSize <= 0) {
            throw new \InvalidArgumentException('The searchable product cursor is invalid.');
        }
        $collection = $this->eligibleCollection($storeId);
        $collection->addFieldToFilter('entity_id', ['gt' => $afterProductId]);
        $collection->setOrder('entity_id', 'ASC');

        return array_map('intval', $collection->getAllIds($batchSize));
    }

    public function countEligible(int $storeId): int
    {
        if ($storeId <= 0) {
            throw new \InvalidArgumentException('The searchable product store is invalid.');
        }

        return (int)$this->eligibleCollection($storeId)->getSize();
    }

    private function eligibleCollection(
        int $storeId
    ): \Magento\Catalog\Model\ResourceModel\Product\Collection {
        $collection = $this->collectionFactory->create();
        $collection->setStoreId($storeId);
        $collection->addStoreFilter($storeId);
        $collection->addAttributeToFilter(
            'status',
            \Magento\Catalog\Model\Product\Attribute\Source\Status::STATUS_ENABLED
        );
        $collection->addAttributeToFilter('visibility', [
            'in' => $this->visibility->getVisibleInSearchIds(),
        ]);

        return $collection;
    }
}
