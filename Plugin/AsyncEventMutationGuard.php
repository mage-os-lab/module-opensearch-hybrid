<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Plugin;

class AsyncEventMutationGuard
{
    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection
    ) {
    }

    public function beforeSave(
        \MageOS\AsyncEvents\Api\AsyncEventRepositoryInterface $subject,
        \MageOS\AsyncEvents\Api\Data\AsyncEventInterface $asyncEvent,
        bool $checkResources = true
    ): array {
        if ($asyncEvent->getMetadata() === \MageOS\OpenSearchHybrid\Model\AsyncEvent\SubscriptionReconciler::METADATA
            || $this->isPersistedModuleSubscription($asyncEvent->getSubscriptionId())
        ) {
            throw new \Magento\Framework\Exception\LocalizedException(
                __('OpenSearch Hybrid subscriptions are reconciled by the module and cannot be edited directly.')
            );
        }

        return [$asyncEvent, $checkResources];
    }

    public function beforeDelete(
        \MageOS\AsyncEvents\Model\ResourceModel\AsyncEvent $subject,
        \Magento\Framework\Model\AbstractModel $object
    ): array {
        if ((string)$object->getData('metadata')
            === \MageOS\OpenSearchHybrid\Model\AsyncEvent\SubscriptionReconciler::METADATA
            || $this->isPersistedModuleSubscription((int)$object->getData('subscription_id'))
        ) {
            throw new \Magento\Framework\Exception\LocalizedException(
                __('OpenSearch Hybrid subscriptions are reconciled by the module and cannot be deleted directly.')
            );
        }

        return [$object];
    }

    private function isPersistedModuleSubscription(int $subscriptionId): bool
    {
        if ($subscriptionId < 1) {
            return false;
        }
        $connection = $this->resourceConnection->getConnection();
        $metadata = $connection->fetchOne(
            $connection->select()
                ->from(
                    $this->resourceConnection->getTableName('async_event_subscriber'),
                    ['metadata']
                )
                ->where('subscription_id = ?', $subscriptionId)
        );

        return $metadata === \MageOS\OpenSearchHybrid\Model\AsyncEvent\SubscriptionReconciler::METADATA;
    }
}
