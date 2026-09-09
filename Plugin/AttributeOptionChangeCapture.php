<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Plugin;

class AttributeOptionChangeCapture
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Document\SemanticAttributeRegistry $registry,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\AttributeOptionProductResolver $productResolver,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\CaptureService $captureService,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\NativeInvalidationScheduler $nativeScheduler
    ) {
    }

    public function aroundSave(
        \Magento\Eav\Model\ResourceModel\Entity\Attribute $subject,
        \Closure $proceed,
        \Magento\Framework\Model\AbstractModel $attribute
    ) {
        if (!$attribute instanceof \Magento\Eav\Model\Entity\Attribute\AbstractAttribute
            || (string)$attribute->getEntityType()->getEntityTypeCode()
                !== \Magento\Catalog\Model\Product::ENTITY
            || !$this->registry->isMapped((string)$attribute->getAttributeCode())
        ) {
            return $proceed($attribute);
        }
        $option = $attribute->getOption();
        if (!is_array($option) || !isset($option['value']) || !is_array($option['value'])) {
            return $proceed($attribute);
        }
        $optionIds = array_values(array_filter(
            array_keys($option['value']),
            static fn (int|string $optionId): bool => is_numeric($optionId) && (int)$optionId > 0
        ));
        if ($optionIds === []) {
            return $proceed($attribute);
        }
        $existingLabels = $this->productResolver->labels($optionIds);
        $changedStoreIds = $this->changedStoreIds($option, $optionIds, $existingLabels);
        if ($changedStoreIds === []) {
            return $proceed($attribute);
        }
        $productIdsByStore = $this->productResolver->resolve(
            $attribute,
            $optionIds,
            $changedStoreIds === [0] ? null : $changedStoreIds
        );
        $productIds = array_values(array_unique(array_merge(...array_values($productIdsByStore ?: [[]]))));

        $subject->beginTransaction();
        try {
            $result = $proceed($attribute);
            $capture = $this->captureService->captureProductsByStore(
                $productIdsByStore,
                'attribute_option_label_change'
            );
            if ($productIds !== []) {
                $subject->addCommitCallback(function () use ($capture, $productIds): void {
                    $this->captureService->publishJobs($capture['jobs']);
                    $this->nativeScheduler->schedule($productIds);
                });
            }
            $subject->commit();

            return $result;
        } catch (\Throwable $throwable) {
            $subject->rollBack();
            throw $throwable;
        }
    }

    private function changedStoreIds(array $option, array $optionIds, array $existingLabels): array
    {
        $changedStoreIds = [];
        foreach ($optionIds as $optionId) {
            $optionId = (int)$optionId;
            if (!empty($option['delete'][$optionId]) || !empty($option['delete'][(string)$optionId])) {
                return [0];
            }
            $proposed = $option['value'][$optionId] ?? $option['value'][(string)$optionId] ?? [];
            if (!is_array($proposed)) {
                continue;
            }
            $before = $existingLabels[$optionId] ?? [];
            $storeIds = array_values(array_unique(array_merge(array_keys($before), array_keys($proposed))));
            foreach ($storeIds as $storeId) {
                $storeId = (int)$storeId;
                $prior = isset($before[$storeId]) ? (string)$before[$storeId] : '';
                $next = isset($proposed[$storeId]) ? (string)$proposed[$storeId] : '';
                if ($prior !== $next) {
                    if ($storeId === 0) {
                        return [0];
                    }
                    $changedStoreIds[] = $storeId;
                }
            }
        }
        $changedStoreIds = array_values(array_unique($changedStoreIds));
        sort($changedStoreIds, SORT_NUMERIC);

        return $changedStoreIds;
    }
}
