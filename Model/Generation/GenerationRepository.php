<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Generation;

class GenerationRepository
{
    private const TABLE = 'mageos_opensearch_hybrid_generation';

    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \MageOS\OpenSearchHybrid\Model\Generation\StoreScopeFingerprint $storeScopeFingerprint,
        private readonly \MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract $retrievalContract
    ) {
    }

    public function get(int $generationId): array
    {
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName(self::TABLE);
        $generation = $connection->fetchRow(
            $connection->select()->from($table)->where('generation_id = ?', $generationId)
        );
        if (!is_array($generation)) {
            throw new \Magento\Framework\Exception\NoSuchEntityException(
                __('OpenSearch Hybrid generation %1 does not exist.', $generationId)
            );
        }

        return $generation;
    }

    public function activeForStore(int $storeId): ?array
    {
        $connection = $this->resourceConnection->getConnection();
        $generationTable = $this->resourceConnection->getTableName(self::TABLE);
        $stateTable = $this->resourceConnection->getTableName('mageos_opensearch_hybrid_store_state');
        $select = $connection->select()
            ->from(['state' => $stateTable], [])
            ->joinInner(
                ['generation' => $generationTable],
                'generation.generation_id = state.active_generation_id',
                ['generation.*']
            )
            ->where('state.store_id = ?', $storeId)
            ->where('state.activation_mode = ?', 'HYBRID')
            ->where('state.readiness_latch = ?', 1)
            ->where('generation.state = ?', 'ACTIVE');
        $generation = $connection->fetchRow($select);

        if (!is_array($generation)
            || !hash_equals(
                (string)$generation['scope_digest'],
                $this->storeScopeFingerprint->digest($storeId)
            )
            || !$this->matchesCurrentContract($generation)
        ) {
            return null;
        }

        return $generation;
    }

    public function writableForStore(int $storeId): array
    {
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName(self::TABLE);

        return $connection->fetchAll(
            $connection->select()
                ->from($table)
                ->where('store_id = ?', $storeId)
                ->where('state IN (?)', ['ACTIVE', 'BUILDING', 'CATCHING_UP', 'READY'])
                ->order('generation_id ASC')
        );
    }

    private function matchesCurrentContract(array $generation): bool
    {
        return hash_equals((string)$generation['contract_digest'], $this->retrievalContract->digest())
            && hash_equals(
                (string)$generation['result_contract_digest'],
                $this->retrievalContract->resultContractDigest()
            )
            && hash_equals((string)$generation['mapping_digest'], $this->retrievalContract->mappingDigest())
            && hash_equals((string)$generation['pipeline_digest'], $this->retrievalContract->pipelineDigest());
    }
}
