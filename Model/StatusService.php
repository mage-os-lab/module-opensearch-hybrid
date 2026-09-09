<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model;

class StatusService
{
    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract $retrievalContract,
        private readonly \MageOS\OpenSearchHybrid\Model\Operations\OperationalMetricsService $metricsService,
        private readonly \MageOS\OpenSearchHybrid\Model\OpenSearch\ClusterVersionProvider $clusterVersionProvider,
        private readonly \MageOS\OpenSearchHybrid\Model\OpenSearch\VersionPolicy $versionPolicy,
        private readonly \MageOS\OpenSearchHybrid\Model\OpenSearch\VectorWireQualification $wireQualification,
        private readonly \MageOS\OpenSearchHybrid\Model\Similarity\SimilarityStatusService $similarityStatusService
    ) {
    }

    public function get(?int $storeId = null): array
    {
        $connection = $this->resourceConnection->getConnection();
        $generationTable = $this->resourceConnection->getTableName('mageos_opensearch_hybrid_generation');
        $outboxTable = $this->resourceConnection->getTableName('mageos_opensearch_hybrid_outbox');
        $stateTable = $this->resourceConnection->getTableName('mageos_opensearch_hybrid_store_state');
        $progressTable = $this->resourceConnection->getTableName('mageos_opensearch_hybrid_generation_progress');
        $generationSelect = $connection->select()->from($generationTable)->order('generation_id DESC');
        $outboxSelect = $connection->select()
            ->from($outboxTable, ['state', 'jobs' => new \Zend_Db_Expr('COUNT(*)')])
            ->group('state');
        $stateSelect = $connection->select()->from($stateTable)->order('store_id ASC');
        $progressSelect = $connection->select()->from($progressTable)->order('generation_id DESC');
        if ($storeId !== null) {
            $generationSelect->where('store_id = ?', $storeId);
            $outboxSelect->where('store_id = ?', $storeId);
            $stateSelect->where('store_id = ?', $storeId);
            $progressSelect->where('store_id = ?', $storeId);
        }

        $stores = $connection->fetchAll($stateSelect);
        $currentVersion = null;
        try {
            $currentVersion = $this->clusterVersionProvider->current();
            $openSearch = [
                'supported' => $this->versionPolicy->isSupported($currentVersion),
                'version' => $currentVersion,
                'required' => \MageOS\OpenSearchHybrid\Model\OpenSearch\VersionPolicy::SUPPORTED_RANGE,
            ];
        } catch (\Throwable $throwable) {
            $openSearch = [
                'supported' => false,
                'version' => null,
                'required' => \MageOS\OpenSearchHybrid\Model\OpenSearch\VersionPolicy::SUPPORTED_RANGE,
                'error_class' => $throwable::class,
            ];
        }
        $similarityStoreIds = $storeId === null
            ? array_map(static fn (array $state): int => (int)$state['store_id'], $stores)
            : [$storeId];
        $similarity = [];
        foreach (array_values(array_unique($similarityStoreIds)) as $similarityStoreId) {
            $similarity[] = $this->similarityStatusService->get($similarityStoreId);
        }

        return [
            'schema_version' => 2,
            'contract_digest' => $this->retrievalContract->digest(),
            'result_contract_digest' => $this->retrievalContract->resultContractDigest(),
            'capabilities' => [
                'opensearch' => $openSearch,
                'base64_vector_ingestion' => $this->wireQualification->get($currentVersion),
            ],
            'similarity' => $similarity,
            'stores' => $stores,
            'generations' => $connection->fetchAll($generationSelect),
            'outbox' => $connection->fetchAll($outboxSelect),
            'progress' => $connection->fetchAll($progressSelect),
            'operations' => $this->metricsService->get($storeId),
        ];
    }
}
