<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Change;

class ChangeJournalRepository
{
    private const TABLE = 'mageos_opensearch_hybrid_change_journal';
    private const OUTBOX_TABLE = 'mageos_opensearch_hybrid_outbox';
    private const COMPLETED = 'COMPLETED';
    private const GENERATION_INVALIDATION_ENTITY_TYPES = [
        'CONFIGURATION',
        'INVENTORY_SALES_CHANNEL',
        'INVENTORY_STOCK',
        'STORE',
        'STORE_GROUP',
        'WEBSITE',
    ];

    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \MageOS\OpenSearchHybrid\Model\Store\ReadinessLatch $readinessLatch
    ) {
    }

    public function appendProduct(int $storeId, int $productId, string $operation, string $reason): int
    {
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName(self::TABLE);
        $connection->insert($table, [
            'store_id' => $storeId,
            'entity_type' => 'PRODUCT',
            'entity_id' => $productId,
            'operation' => $operation,
            'revision' => 0,
            'reason' => substr($reason, 0, 255),
            'native_state' => 'PENDING',
            'hybrid_state' => 'PENDING',
        ]);
        $changeId = (int)$connection->lastInsertId($table);
        $connection->update(
            $table,
            ['revision' => $changeId],
            ['change_id = ?' => $changeId]
        );

        return $changeId;
    }

    public function appendGenerationInvalidation(
        int $storeId,
        string $entityType,
        int $entityId,
        string $reason
    ): int {
        if (!in_array($entityType, self::GENERATION_INVALIDATION_ENTITY_TYPES, true)) {
            throw new \InvalidArgumentException('The generation invalidation entity type is invalid.');
        }
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName(self::TABLE);
        $connection->insert($table, [
            'store_id' => $storeId,
            'entity_type' => $entityType,
            'entity_id' => max(0, $entityId),
            'operation' => 'REBUILD',
            'revision' => 0,
            'reason' => substr($reason, 0, 255),
            'native_state' => self::COMPLETED,
            'hybrid_state' => 'PENDING',
            'native_completed_at' => new \Zend_Db_Expr('UTC_TIMESTAMP()'),
        ]);
        $changeId = (int)$connection->lastInsertId($table);
        $connection->update(
            $table,
            ['revision' => $changeId],
            ['change_id = ?' => $changeId]
        );

        return $changeId;
    }

    public function acknowledgeGenerationInvalidations(int $storeId, int $boundary): void
    {
        if ($boundary <= 0) {
            return;
        }
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName(self::TABLE);
        $connection->update(
            $table,
            [
                'hybrid_state' => self::COMPLETED,
                'hybrid_completed_at' => new \Zend_Db_Expr('UTC_TIMESTAMP()'),
            ],
            [
                'store_id = ?' => $storeId,
                'change_id <= ?' => $boundary,
                'entity_type != ?' => 'PRODUCT',
                'hybrid_state != ?' => self::COMPLETED,
            ]
        );
        $this->readinessLatch->advanceHybrid($storeId, $this->safeWatermark($storeId, 'hybrid_state'));
    }

    public function pendingNativeBoundary(int $storeId, ?array $productIds): int
    {
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName(self::TABLE);
        $select = $connection->select()
            ->from($table, [new \Zend_Db_Expr('COALESCE(MAX(change_id), 0)')])
            ->where('store_id = ?', $storeId)
            ->where('native_state != ?', self::COMPLETED);
        if ($productIds !== null) {
            if ($productIds === []) {
                return 0;
            }
            $select->where('entity_type = ?', 'PRODUCT')->where('entity_id IN (?)', $productIds);
        }

        return (int)$connection->fetchOne($select);
    }

    public function untrackedNativeProductIds(int $storeId, array $productIds): array
    {
        $productIds = array_values(array_unique(array_filter(
            array_map('intval', $productIds),
            static fn (int $productId): bool => $productId > 0
        )));
        if ($productIds === []) {
            return [];
        }
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName(self::TABLE);
        $tracked = array_map('intval', $connection->fetchCol(
            $connection->select()
                ->from($table, ['entity_id'])
                ->where('store_id = ?', $storeId)
                ->where('entity_type = ?', 'PRODUCT')
                ->where('entity_id IN (?)', $productIds)
                ->where('native_state != ?', self::COMPLETED)
                ->group('entity_id')
        ));
        $untracked = array_values(array_diff($productIds, $tracked));
        sort($untracked, SORT_NUMERIC);

        return $untracked;
    }

    public function acknowledgeNative(int $storeId, ?array $productIds, int $boundary): void
    {
        if ($boundary <= 0) {
            return;
        }
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName(self::TABLE);
        $where = [
            'store_id = ?' => $storeId,
            'change_id <= ?' => $boundary,
            'native_state != ?' => self::COMPLETED,
        ];
        if ($productIds !== null) {
            if ($productIds === []) {
                return;
            }
            $where['entity_type = ?'] = 'PRODUCT';
            $where['entity_id IN (?)'] = $productIds;
        }
        $connection->update(
            $table,
            [
                'native_state' => self::COMPLETED,
                'native_completed_at' => new \Zend_Db_Expr('UTC_TIMESTAMP()'),
            ],
            $where
        );
        $this->readinessLatch->advanceNative($storeId, $this->safeWatermark($storeId, 'native_state'));
    }

    public function acknowledgeNativeChanges(array $changeIds): void
    {
        $changeIds = array_values(array_unique(array_filter(
            array_map('intval', $changeIds),
            static fn (int $changeId): bool => $changeId > 0
        )));
        if ($changeIds === []) {
            return;
        }
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName(self::TABLE);
        $storeIds = array_map('intval', $connection->fetchCol(
            $connection->select()
                ->from($table, ['store_id'])
                ->where('change_id IN (?)', $changeIds)
                ->group('store_id')
        ));
        $connection->update(
            $table,
            [
                'native_state' => self::COMPLETED,
                'native_completed_at' => new \Zend_Db_Expr('UTC_TIMESTAMP()'),
            ],
            [
                'change_id IN (?)' => $changeIds,
                'native_state != ?' => self::COMPLETED,
            ]
        );
        foreach ($storeIds as $storeId) {
            $this->readinessLatch->advanceNative($storeId, $this->safeWatermark($storeId, 'native_state'));
        }
    }

    public function acknowledgeHybridJob(array $job): void
    {
        if ((string)($job['lane'] ?? '') !== 'PRIORITY'
            || (string)($job['state'] ?? '') !== self::COMPLETED
        ) {
            return;
        }
        $storeId = (int)$job['store_id'];
        $earliestChangeId = (int)$job['earliest_change_id'];
        $latestChangeId = (int)$job['latest_change_id'];
        if ($earliestChangeId <= 0 || $latestChangeId < $earliestChangeId) {
            return;
        }
        $connection = $this->resourceConnection->getConnection();
        $journalTable = $this->resourceConnection->getTableName(self::TABLE);
        $outboxTable = $this->resourceConnection->getTableName(self::OUTBOX_TABLE);
        $changeIds = array_map('intval', $connection->fetchCol(
            $connection->select()
                ->from($journalTable, ['change_id'])
                ->where('store_id = ?', $storeId)
                ->where('change_id >= ?', $earliestChangeId)
                ->where('change_id <= ?', $latestChangeId)
                ->where('hybrid_state != ?', self::COMPLETED)
        ));
        $completedChangeIds = [];
        foreach ($changeIds as $changeId) {
            $unresolved = (int)$connection->fetchOne(
                $connection->select()
                    ->from($outboxTable, [new \Zend_Db_Expr('COUNT(*)')])
                    ->where('store_id = ?', $storeId)
                    ->where('lane = ?', 'PRIORITY')
                    ->where('earliest_change_id <= ?', $changeId)
                    ->where('latest_change_id >= ?', $changeId)
                    ->where('state NOT IN (?)', [self::COMPLETED, 'SUPERSEDED'])
            );
            if ($unresolved === 0) {
                $completedChangeIds[] = $changeId;
            }
        }
        if ($completedChangeIds !== []) {
            $connection->update(
                $journalTable,
                [
                    'hybrid_state' => self::COMPLETED,
                    'hybrid_completed_at' => new \Zend_Db_Expr('UTC_TIMESTAMP()'),
                ],
                ['change_id IN (?)' => $completedChangeIds]
            );
        }
        $this->readinessLatch->advanceHybrid($storeId, $this->safeWatermark($storeId, 'hybrid_state'));
    }

    private function safeWatermark(int $storeId, string $stateColumn): int
    {
        if (!in_array($stateColumn, ['native_state', 'hybrid_state'], true)) {
            throw new \InvalidArgumentException('The journal state column is invalid.');
        }
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName(self::TABLE);
        $firstPending = $connection->fetchOne(
            $connection->select()
                ->from($table, [new \Zend_Db_Expr('MIN(change_id)')])
                ->where('store_id = ?', $storeId)
                ->where($stateColumn . ' != ?', self::COMPLETED)
        );
        $select = $connection->select()
            ->from($table, [new \Zend_Db_Expr('COALESCE(MAX(change_id), 0)')])
            ->where('store_id = ?', $storeId)
            ->where($stateColumn . ' = ?', self::COMPLETED);
        if ($firstPending !== false && $firstPending !== null) {
            $select->where('change_id < ?', (int)$firstPending);
        }

        return (int)$connection->fetchOne($select);
    }
}
