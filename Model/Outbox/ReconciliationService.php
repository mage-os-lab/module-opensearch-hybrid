<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Outbox;

class ReconciliationService
{
    private const TABLE = 'mageos_opensearch_hybrid_outbox';

    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \MageOS\OpenSearchHybrid\Model\AsyncEvent\LifecyclePublisher $publisher,
        private readonly \MageOS\OpenSearchHybrid\Model\AsyncEvent\SubscriptionReconciler $subscriptionReconciler,
        private readonly \MageOS\OpenSearchHybrid\Model\Generation\AutomaticReplacementService $replacementService,
        private readonly \MageOS\OpenSearchHybrid\Model\Generation\BuildService $buildService,
        private readonly \Magento\Store\Model\StoreManagerInterface $storeManager,
        private readonly \Psr\Log\LoggerInterface $logger
    ) {
    }

    public function reconcile(?int $storeId = null): array
    {
        $storeIds = $storeId === null
            ? array_map(
                static fn (\Magento\Store\Api\Data\StoreInterface $store): int => (int)$store->getId(),
                array_values($this->storeManager->getStores())
            )
            : [$storeId];
        $subscriptions = [];
        foreach ($storeIds as $targetStoreId) {
            $subscriptions[$targetStoreId] = $this->subscriptionReconciler->reconcile($targetStoreId);
        }
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName(self::TABLE);
        $where = ['store_id IN (?)' => $storeIds];
        $timedOutClaims = $connection->update(
            $table,
            ['state' => 'TIMED_OUT', 'claim_owner' => null, 'claim_expires_at' => null],
            array_merge($where, ['state = ?' => 'CLAIMED', 'claim_expires_at < UTC_TIMESTAMP()'])
        );
        $timedOutPublications = $connection->update(
            $table,
            ['state' => 'TIMED_OUT'],
            array_merge(
                $where,
                [
                    'state = ?' => 'PUBLISHED',
                    'updated_at < DATE_SUB(UTC_TIMESTAMP(), INTERVAL 5 MINUTE)',
                ]
            )
        );
        $jobIds = $connection->fetchCol(
            $connection->select()->from($table, ['job_id'])
                ->where('store_id IN (?)', $storeIds)
                ->where('state IN (?)', ['PENDING', 'RETRY', 'TIMED_OUT'])
                ->where('next_eligible_at IS NULL OR next_eligible_at <= UTC_TIMESTAMP()')
                ->order('created_at ASC')
                ->limit(100)
        );
        $published = 0;
        $failed = 0;
        foreach ($jobIds as $jobId) {
            try {
                $this->publisher->publish((string)$jobId);
                $published++;
            } catch (\Throwable $throwable) {
                $failed++;
                $this->logger->warning('OpenSearch Hybrid stale job publication failed.', [
                    'job_id' => (string)$jobId,
                    'error_class' => $throwable::class,
                ]);
            }
        }
        $replacements = $this->replacementService->createPending($storeId);
        $failedReplacements = count(array_filter(
            $replacements,
            static fn (array $report): bool => (string)($report['status'] ?? '') === 'FAILED_TO_CREATE'
        ));
        $builds = $this->buildService->resumePendingBuilds($storeId);

        return [
            'schema_version' => 1,
            'store_ids' => $storeIds,
            'subscriptions' => $subscriptions,
            'timed_out_claims' => $timedOutClaims,
            'timed_out_publications' => $timedOutPublications,
            'eligible_jobs' => count($jobIds),
            'published_jobs' => $published,
            'failed_publications' => $failed,
            'replacements' => $replacements,
            'failed_replacements' => $failedReplacements,
            'builds' => $builds,
        ];
    }
}
