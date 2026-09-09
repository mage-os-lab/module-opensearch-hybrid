<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Plugin;

class StoreDefinitionChangeCapture
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Change\ScopeInvalidationService $invalidationService
    ) {
    }

    public function afterSave(
        \Magento\Framework\Model\ResourceModel\Db\AbstractDb $subject,
        \Magento\Framework\Model\ResourceModel\Db\AbstractDb $result,
        \Magento\Framework\Model\AbstractModel $entity
    ): \Magento\Framework\Model\ResourceModel\Db\AbstractDb {
        unset($subject);
        if ($entity instanceof \Magento\Store\Model\Store) {
            $this->invalidationService->invalidateStores(
                [(int)$entity->getId()],
                'STORE',
                (int)$entity->getId(),
                'store_definition_save'
            );
        } elseif ($entity instanceof \Magento\Store\Model\Website) {
            $this->invalidationService->invalidateStores(
                $this->invalidationService->storeIdsForWebsite((int)$entity->getId()),
                'WEBSITE',
                (int)$entity->getId(),
                'website_definition_save'
            );
        } elseif ($entity instanceof \Magento\Store\Model\Group) {
            $this->invalidationService->invalidateStores(
                $this->invalidationService->storeIdsForGroup((int)$entity->getId()),
                'STORE_GROUP',
                (int)$entity->getId(),
                'store_group_definition_save'
            );
        }

        return $result;
    }
}
