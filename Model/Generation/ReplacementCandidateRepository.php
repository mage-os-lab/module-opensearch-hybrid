<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Generation;

class ReplacementCandidateRepository
{
    private const GENERATION_TABLE = 'mageos_opensearch_hybrid_generation';
    private const JOURNAL_TABLE = 'mageos_opensearch_hybrid_change_journal';
    private const STORE_STATE_TABLE = 'mageos_opensearch_hybrid_store_state';

    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection
    ) {
    }

    public function storeIds(?int $storeId = null): array
    {
        if ($storeId !== null && $storeId <= 0) {
            throw new \InvalidArgumentException('The replacement store ID must be positive.');
        }
        $connection = $this->resourceConnection->getConnection();
        $select = $connection->select()
            ->from(
                $this->resourceConnection->getTableName(self::STORE_STATE_TABLE),
                ['store_id']
            )
            ->where('activation_mode = ?', 'HYBRID')
            ->where('active_generation_id IS NOT NULL')
            ->order('store_id ASC');
        if ($storeId !== null) {
            $select->where('store_id = ?', $storeId);
        }

        return array_map('intval', $connection->fetchCol($select));
    }

    public function withinStoreLock(int $storeId, callable $operation): mixed
    {
        if ($storeId <= 0) {
            throw new \InvalidArgumentException('The replacement store ID must be positive.');
        }
        $connection = $this->resourceConnection->getConnection();
        $lockName = 'mageos_hybrid_replacement_s' . $storeId;
        $acquired = (int)$connection->fetchOne('SELECT GET_LOCK(?, 0)', [$lockName]);
        if ($acquired !== 1) {
            throw new \RuntimeException('The replacement generation lock is busy.');
        }
        try {
            return $operation();
        } finally {
            $connection->fetchOne('SELECT RELEASE_LOCK(?)', [$lockName]);
        }
    }

    public function assessment(int $storeId): array
    {
        $connection = $this->resourceConnection->getConnection();
        $journalTable = $this->resourceConnection->getTableName(self::JOURNAL_TABLE);
        $generationTable = $this->resourceConnection->getTableName(self::GENERATION_TABLE);
        $storeStateTable = $this->resourceConnection->getTableName(self::STORE_STATE_TABLE);
        $invalidationBoundary = (int)$connection->fetchOne(
            $connection->select()
                ->from($journalTable, [new \Zend_Db_Expr('COALESCE(MAX(change_id), 0)')])
                ->where('store_id = ?', $storeId)
                ->where('entity_type = ?', 'CONFIGURATION')
                ->where('operation = ?', 'REBUILD')
                ->where('hybrid_state != ?', 'COMPLETED')
        );
        if ($invalidationBoundary === 0) {
            return $this->skipped('no_pending_configuration_invalidation');
        }
        $latestGenerationBoundary = (int)$connection->fetchOne(
            $connection->select()
                ->from($generationTable, [new \Zend_Db_Expr('COALESCE(MAX(captured_change_id), 0)')])
                ->where('store_id = ?', $storeId)
        );
        if ($latestGenerationBoundary >= $invalidationBoundary) {
            return $this->skipped('replacement_already_created', $invalidationBoundary);
        }
        $source = $connection->fetchRow(
            $connection->select()
                ->from(['state' => $storeStateTable], [])
                ->joinInner(
                    ['generation' => $generationTable],
                    'generation.generation_id = state.active_generation_id',
                    ['encoder_identity_digest', 'is_fake']
                )
                ->where('state.store_id = ?', $storeId)
                ->where('state.activation_mode = ?', 'HYBRID')
                ->limit(1)
        );
        if (!is_array($source)) {
            return $this->skipped('active_generation_missing', $invalidationBoundary);
        }
        if ((bool)$source['is_fake']) {
            return $this->skipped('active_generation_not_production', $invalidationBoundary);
        }
        $identityDigest = (string)($source['encoder_identity_digest'] ?? '');
        if (!preg_match('/^[0-9a-f]{64}$/D', $identityDigest)) {
            return $this->skipped('active_generation_identity_missing', $invalidationBoundary);
        }

        return [
            'eligible' => true,
            'reason' => 'pending_configuration_invalidation',
            'invalidation_boundary' => $invalidationBoundary,
            'encoder_identity_digest' => $identityDigest,
        ];
    }

    private function skipped(string $reason, int $invalidationBoundary = 0): array
    {
        return [
            'eligible' => false,
            'reason' => $reason,
            'invalidation_boundary' => $invalidationBoundary,
        ];
    }
}
