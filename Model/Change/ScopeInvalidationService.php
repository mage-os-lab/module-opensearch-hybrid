<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Change;

class ScopeInvalidationService
{
    private const GENERATION_TABLE = 'mageos_opensearch_hybrid_generation';

    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\ChangeJournalRepository $journalRepository,
        private readonly \MageOS\OpenSearchHybrid\Model\Store\ReadinessLatch $readinessLatch,
        private readonly \MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository $outboxRepository
    ) {
    }

    public function invalidateStores(
        array $storeIds,
        string $entityType,
        int $entityId,
        string $reason
    ): array {
        $storeIds = array_values(array_unique(array_filter(
            array_map('intval', $storeIds),
            static fn (int $storeId): bool => $storeId > 0
        )));
        sort($storeIds, SORT_NUMERIC);
        $generationStores = array_flip($this->storeIdsWithGenerations());
        $storeIds = array_values(array_filter(
            $storeIds,
            static fn (int $storeId): bool => isset($generationStores[$storeId])
        ));
        if ($storeIds === []) {
            return [];
        }
        $connection = $this->resourceConnection->getConnection();
        $generationTable = $this->resourceConnection->getTableName(self::GENERATION_TABLE);
        $changes = [];
        $connection->beginTransaction();
        try {
            foreach ($storeIds as $storeId) {
                $generations = $connection->fetchAll(
                    $connection->select()
                        ->from($generationTable)
                        ->where('store_id = ?', $storeId)
                        ->forUpdate(true)
                );
                $changeId = $this->journalRepository->appendGenerationInvalidation(
                    $storeId,
                    $entityType,
                    $entityId,
                    $reason
                );
                $this->readinessLatch->requireWatermark($storeId, $changeId);
                foreach ($generations as $generation) {
                    if (!in_array(
                        (string)$generation['state'],
                        ['ACTIVE', 'BUILDING', 'CATCHING_UP', 'READY'],
                        true
                    )) {
                        continue;
                    }
                    $generationId = (int)$generation['generation_id'];
                    $connection->update(
                        $generationTable,
                        ['state' => 'FAILED', 'validation_report' => null],
                        ['generation_id = ?' => $generationId]
                    );
                    $this->outboxRepository->supersedeGenerationJobs($generationId);
                }
                $changes[] = ['store_id' => $storeId, 'change_id' => $changeId];
            }
            $connection->commit();
        } catch (\Throwable $throwable) {
            $connection->rollBack();
            throw $throwable;
        }

        return $changes;
    }

    public function storeIdsWithGenerations(): array
    {
        $connection = $this->resourceConnection->getConnection();
        $generationTable = $this->resourceConnection->getTableName(self::GENERATION_TABLE);
        $storeTable = $this->resourceConnection->getTableName('store');
        $storeIds = array_map('intval', $connection->fetchCol(
            $connection->select()
                ->from(['generation' => $generationTable], ['store_id'])
                ->joinInner(['store' => $storeTable], 'store.store_id = generation.store_id', [])
                ->where('generation.store_id > ?', 0)
                ->group('generation.store_id')
                ->order('generation.store_id ASC')
        ));

        return $storeIds;
    }

    public function storeIdsForWebsite(int $websiteId): array
    {
        return $this->storeIdsForColumn('website_id', $websiteId);
    }

    public function storeIdsForGroup(int $groupId): array
    {
        return $this->storeIdsForColumn('group_id', $groupId);
    }

    private function storeIdsForColumn(string $column, int $value): array
    {
        if (!in_array($column, ['website_id', 'group_id'], true) || $value <= 0) {
            return [];
        }
        $connection = $this->resourceConnection->getConnection();
        $generationTable = $this->resourceConnection->getTableName(self::GENERATION_TABLE);
        $storeTable = $this->resourceConnection->getTableName('store');

        return array_map('intval', $connection->fetchCol(
            $connection->select()
                ->from(['generation' => $generationTable], ['store_id'])
                ->joinInner(['store' => $storeTable], 'store.store_id = generation.store_id', [])
                ->where(sprintf('store.%s = ?', $column), $value)
                ->group('generation.store_id')
                ->order('generation.store_id ASC')
        ));
    }
}
