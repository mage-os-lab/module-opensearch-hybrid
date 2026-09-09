<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Change;

class CaptureService
{
    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository $generationRepository,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\ChangeJournalRepository $journalRepository,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\SearchableProductResolver $searchableProductResolver,
        private readonly \MageOS\OpenSearchHybrid\Model\Store\ReadinessLatch $readinessLatch,
        private readonly \MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository $outboxRepository,
        private readonly \MageOS\OpenSearchHybrid\Model\AsyncEvent\LifecyclePublisher $publisher,
        private readonly \Psr\Log\LoggerInterface $logger
    ) {
    }

    public function captureProduct(int $productId, array $storeIds, string $operation, string $reason): array
    {
        if (!in_array($operation, ['UPSERT', 'DELETE'], true)) {
            throw new \InvalidArgumentException('The product change operation is invalid.');
        }
        $storeIds = array_values(array_unique(array_filter(
            array_map('intval', $storeIds),
            static fn (int $storeId): bool => $storeId > 0
        )));
        sort($storeIds, SORT_NUMERIC);
        $requests = [];
        foreach ($storeIds as $storeId) {
            $effectiveOperation = $operation;
            if ($operation === 'UPSERT'
                && $this->searchableProductResolver->eligibleIds($storeId, [$productId]) === []
            ) {
                $effectiveOperation = 'DELETE';
            }
            $requests[$storeId][$effectiveOperation] = [$productId];
        }

        return $this->captureRequests($requests, $reason);
    }

    public function captureNativeProducts(int $storeId, array $productIds, string $reason): array
    {
        if ($storeId <= 0) {
            return ['changes' => [], 'jobs' => []];
        }
        $productIds = $this->journalRepository->untrackedNativeProductIds($storeId, $productIds);
        if ($productIds === []) {
            return ['changes' => [], 'jobs' => []];
        }
        $eligibleIds = $this->searchableProductResolver->eligibleIds($storeId, $productIds);
        $requests = [$storeId => []];
        if ($eligibleIds !== []) {
            $requests[$storeId]['UPSERT'] = $eligibleIds;
        }
        $deleteIds = array_values(array_diff($productIds, $eligibleIds));
        if ($deleteIds !== []) {
            $requests[$storeId]['DELETE'] = $deleteIds;
        }

        return $this->captureRequests($requests, $reason);
    }

    public function captureProductsByStore(array $productIdsByStore, string $reason): array
    {
        return $this->captureProductsByStoreMode($productIdsByStore, $reason, true);
    }

    public function capturePriorityProductsByStore(array $productIdsByStore, string $reason): array
    {
        return $this->captureProductsByStoreMode($productIdsByStore, $reason, false);
    }

    private function captureProductsByStoreMode(
        array $productIdsByStore,
        string $reason,
        bool $includeEmbedding
    ): array
    {
        $requests = [];
        foreach ($productIdsByStore as $storeId => $productIds) {
            $storeId = (int)$storeId;
            $productIds = array_values(array_unique(array_filter(
                array_map('intval', $productIds),
                static fn (int $productId): bool => $productId > 0
            )));
            if ($storeId <= 0 || $productIds === []) {
                continue;
            }
            $eligibleIds = $this->searchableProductResolver->eligibleIds($storeId, $productIds);
            if ($eligibleIds !== []) {
                $requests[$storeId][$includeEmbedding ? 'UPSERT' : 'REFRESH'] = $eligibleIds;
            }
            $deleteIds = array_values(array_diff($productIds, $eligibleIds));
            if ($deleteIds !== []) {
                $requests[$storeId]['DELETE'] = $deleteIds;
            }
        }

        return $this->captureRequests($requests, $reason, $includeEmbedding);
    }

    private function captureRequests(
        array $requests,
        string $reason,
        bool $includeEmbedding = true
    ): array
    {
        $targetsByStore = [];
        foreach (array_keys($requests) as $storeId) {
            $targets = $this->generationRepository->writableForStore((int)$storeId);
            if ($targets !== []) {
                $targetsByStore[(int)$storeId] = $targets;
            }
        }
        if ($targetsByStore === []) {
            return ['changes' => [], 'jobs' => []];
        }

        $connection = $this->resourceConnection->getConnection();
        $changes = [];
        $jobs = [];
        $connection->beginTransaction();
        try {
            foreach ($targetsByStore as $storeId => $targets) {
                $lockedTargets = [];
                $generationTable = $this->resourceConnection->getTableName(
                    'mageos_opensearch_hybrid_generation'
                );
                foreach ($targets as $target) {
                    $generation = $connection->fetchRow(
                        $connection->select()
                            ->from($generationTable)
                            ->where('generation_id = ?', (int)$target['generation_id'])
                            ->forUpdate(true)
                    );
                    if (!is_array($generation)
                        || !in_array(
                            (string)$generation['state'],
                            ['ACTIVE', 'BUILDING', 'CATCHING_UP', 'READY'],
                            true
                        )
                    ) {
                        continue;
                    }
                    if ((string)$generation['state'] === 'READY') {
                        $connection->update(
                            $generationTable,
                            ['state' => 'CATCHING_UP', 'validation_report' => null],
                            ['generation_id = ?' => (int)$generation['generation_id'], 'state = ?' => 'READY']
                        );
                        $generation['state'] = 'CATCHING_UP';
                        $generation['validation_report'] = null;
                    }
                    $lockedTargets[] = $generation;
                }
                if ($lockedTargets === []) {
                    continue;
                }
                $targets = $lockedTargets;
                foreach ($requests[$storeId] as $operation => $productIds) {
                    $changeIds = [];
                    foreach ($productIds as $productId) {
                        $changeId = $this->journalRepository->appendProduct(
                            (int)$storeId,
                            (int)$productId,
                            (string)$operation,
                            $reason
                        );
                        $changeIds[] = $changeId;
                        $this->readinessLatch->requireWatermark((int)$storeId, $changeId);
                        $changes[] = [
                            'store_id' => (int)$storeId,
                            'product_id' => (int)$productId,
                            'change_id' => $changeId,
                        ];
                    }
                    $earliestChangeId = min($changeIds);
                    $latestChangeId = max($changeIds);
                    foreach ($targets as $generation) {
                        $generationId = (int)$generation['generation_id'];
                        $priorityEarliestChangeId = $earliestChangeId;
                        foreach ($productIds as $productId) {
                            $supersededBoundary = $this->outboxRepository->supersedeProductJobs(
                                (int)$storeId,
                                $generationId,
                                'PRIORITY',
                                (int)$productId,
                                $latestChangeId
                            );
                            if ($supersededBoundary > 0) {
                                $priorityEarliestChangeId = min(
                                    $priorityEarliestChangeId,
                                    $supersededBoundary
                                );
                            }
                            if ($includeEmbedding || $operation === 'DELETE') {
                                $this->outboxRepository->supersedeProductJobs(
                                    (int)$storeId,
                                    $generationId,
                                    'EMBEDDING',
                                    (int)$productId,
                                    $latestChangeId
                                );
                            }
                        }
                        $priorityJobId = $this->outboxRepository->create(
                            (int)$storeId,
                            $generationId,
                            'PRIORITY',
                            (string)$operation,
                            $productIds,
                            $latestChangeId,
                            $priorityEarliestChangeId,
                            $latestChangeId
                        );
                        $jobs[] = [
                            'job_id' => $priorityJobId,
                            'lane' => 'PRIORITY',
                            'store_id' => (int)$storeId,
                            'generation_id' => $generationId,
                        ];
                        if ($includeEmbedding && $operation === 'UPSERT') {
                            $embeddingJobId = $this->outboxRepository->create(
                                (int)$storeId,
                                $generationId,
                                'EMBEDDING',
                                'UPSERT',
                                $productIds,
                                $latestChangeId,
                                $earliestChangeId,
                                $latestChangeId
                            );
                            $jobs[] = [
                                'job_id' => $embeddingJobId,
                                'lane' => 'EMBEDDING',
                                'store_id' => (int)$storeId,
                                'generation_id' => $generationId,
                            ];
                        }
                    }
                }
            }
            $connection->commit();
        } catch (\Throwable $throwable) {
            $connection->rollBack();
            throw $throwable;
        }

        return ['changes' => $changes, 'jobs' => $jobs];
    }

    public function publishJobs(array $jobs): void
    {
        foreach ($jobs as $job) {
            try {
                $this->publisher->publish((string)$job['job_id']);
            } catch (\Throwable $throwable) {
                $this->logger->warning('OpenSearch Hybrid product-change job remains pending.', [
                    'job_id' => (string)$job['job_id'],
                    'error_class' => $throwable::class,
                ]);
            }
        }
    }
}
