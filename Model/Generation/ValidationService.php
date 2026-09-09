<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Generation;

class ValidationService
{
    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository $generationRepository,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\SearchableProductResolver $searchableProductResolver,
        private readonly \MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract $retrievalContract,
        private readonly \MageOS\OpenSearchHybrid\Model\Generation\StoreScopeFingerprint $storeScopeFingerprint,
        private readonly \MageOS\OpenSearchHybrid\Model\OpenSearch\IndexManager $indexManager,
        private readonly \MageOS\OpenSearchHybrid\Model\Generation\BuildRefreshService $buildRefreshService,
        private readonly \Magento\Framework\Serialize\Serializer\Json $json
    ) {
    }

    public function validate(int $generationId, ?string $acceptedResultDigest = null): array
    {
        $generation = $this->snapshotCatchUp($this->generationRepository->get($generationId));
        $this->buildRefreshService->restore($generation);
        $connection = $this->resourceConnection->getConnection();
        $outboxTable = $this->resourceConnection->getTableName('mageos_opensearch_hybrid_outbox');
        $journalTable = $this->resourceConnection->getTableName('mageos_opensearch_hybrid_change_journal');
        $this->indexManager->refreshGeneration($generation);
        $live = $this->indexManager->validateGeneration($generation);
        $indexCount = (int)$live['index_count'];
        $generationTable = $this->resourceConnection->getTableName('mageos_opensearch_hybrid_generation');
        $connection->beginTransaction();
        try {
            $lockedGeneration = $connection->fetchRow(
                $connection->select()
                    ->from($generationTable)
                    ->where('generation_id = ?', $generationId)
                    ->forUpdate(true)
            );
            if (!is_array($lockedGeneration)) {
                throw new \RuntimeException('The generation disappeared during validation.');
            }
            $capturedBoundary = (int)$lockedGeneration['captured_change_id'];
            $unresolved = (int)$connection->fetchOne(
                $connection->select()
                    ->from($outboxTable, [new \Zend_Db_Expr('COUNT(*)')])
                    ->where('generation_id = ?', $generationId)
                    ->where('latest_change_id <= ?', $capturedBoundary)
                    ->where('state NOT IN (?)', ['COMPLETED', 'SUPERSEDED'])
            );
            $unresolvedJournal = (int)$connection->fetchOne(
                $connection->select()
                    ->from($journalTable, [new \Zend_Db_Expr('COUNT(*)')])
                    ->where('store_id = ?', (int)$lockedGeneration['store_id'])
                    ->where('change_id <= ?', $capturedBoundary)
                    ->where(
                        "(native_state != 'COMPLETED' "
                        . "OR (entity_type = 'PRODUCT' AND hybrid_state != 'COMPLETED'))"
                    )
            );
            $latestBoundary = $this->latestJournalBoundary((int)$lockedGeneration['store_id']);
            $currentCoverage = $this->searchableProductResolver->countEligible(
                (int)$lockedGeneration['store_id']
            );
            $checks = [
                'contract_digest' => hash_equals(
                    (string)$lockedGeneration['contract_digest'],
                    $this->retrievalContract->digest()
                ),
                'result_contract_digest' => hash_equals(
                    (string)$lockedGeneration['result_contract_digest'],
                    $this->retrievalContract->resultContractDigest()
                ),
                'mapping_digest' => hash_equals(
                    (string)$lockedGeneration['mapping_digest'],
                    $this->retrievalContract->mappingDigest()
                ),
                'pipeline_digest' => hash_equals(
                    (string)$lockedGeneration['pipeline_digest'],
                    $this->retrievalContract->pipelineDigest()
                ),
                'scope_digest' => hash_equals(
                    (string)$lockedGeneration['scope_digest'],
                    $this->storeScopeFingerprint->digest((int)$lockedGeneration['store_id'])
                ),
                'live_index_exists' => (bool)$live['index_exists'],
                'live_mapping' => (bool)$live['mapping'],
                'live_settings' => (bool)$live['settings'],
                'live_pipeline' => (bool)$live['pipeline'],
                'live_document_identity' => (bool)$live['document_identity'],
                'build_refresh_restored' => $this->buildRefreshService->isRestored($generationId)
                    && (bool)$live['refresh_interval_restored'],
                'coverage_complete' => (int)$lockedGeneration['coverage_complete']
                    === (int)$lockedGeneration['coverage_total'],
                'coverage_failed' => (int)$lockedGeneration['coverage_failed'] === 0,
                'index_count' => $indexCount === (int)$lockedGeneration['coverage_total'],
                'unresolved_work' => $unresolved === 0,
                'journal_complete' => $unresolvedJournal === 0,
                'catch_up_stable' => $latestBoundary === $capturedBoundary
                    && $currentCoverage === (int)$lockedGeneration['coverage_total'],
            ];
            $accepted = (bool)$lockedGeneration['result_contract_accepted']
                || ($acceptedResultDigest !== null
                    && hash_equals($this->retrievalContract->resultContractDigest(), $acceptedResultDigest));
            $ready = !in_array(false, $checks, true);
            $report = [
                'schema_version' => 1,
                'generation_id' => $generationId,
                'checks' => $checks,
                'ready' => $ready,
                'result_contract_accepted' => $accepted,
                'result_contract_digest' => $this->retrievalContract->resultContractDigest(),
            ];
            $connection->update(
                $generationTable,
                [
                    'state' => $ready ? 'READY' : 'CATCHING_UP',
                    'result_contract_accepted' => $accepted ? 1 : 0,
                    'validation_report' => $this->json->serialize($report),
                ],
                ['generation_id = ?' => $generationId]
            );
            $connection->commit();

            return $report;
        } catch (\Throwable $throwable) {
            $connection->rollBack();
            throw $throwable;
        }
    }

    private function snapshotCatchUp(array $generation): array
    {
        if (!in_array((string)$generation['state'], ['BUILDING', 'CATCHING_UP', 'READY'], true)) {
            throw new \RuntimeException('Only building or ready generations can be validated.');
        }
        $connection = $this->resourceConnection->getConnection();
        $generationTable = $this->resourceConnection->getTableName('mageos_opensearch_hybrid_generation');
        $progressTable = $this->resourceConnection->getTableName(
            'mageos_opensearch_hybrid_generation_progress'
        );
        $connection->beginTransaction();
        try {
            $lockedGeneration = $connection->fetchRow(
                $connection->select()
                    ->from($generationTable)
                    ->where('generation_id = ?', (int)$generation['generation_id'])
                    ->forUpdate(true)
            );
            if (!is_array($lockedGeneration)) {
                throw new \RuntimeException('The generation disappeared before catch-up.');
            }
            $seedingState = $connection->fetchOne(
                $connection->select()
                    ->from($progressTable, ['seeding_state'])
                    ->where('generation_id = ?', (int)$generation['generation_id'])
            );
            if ($seedingState !== 'COMPLETED') {
                throw new \RuntimeException('Full-build seeding must complete before validation catch-up.');
            }
            $storeId = (int)$lockedGeneration['store_id'];
            $capturedBoundary = $this->latestJournalBoundary($storeId);
            $coverageTotal = $this->searchableProductResolver->countEligible($storeId);
            $connection->update(
                $generationTable,
                [
                    'state' => 'CATCHING_UP',
                    'captured_change_id' => $capturedBoundary,
                    'coverage_total' => $coverageTotal,
                    'validation_report' => null,
                ],
                ['generation_id = ?' => (int)$generation['generation_id']]
            );
            $connection->update(
                $progressTable,
                ['captured_boundary' => $capturedBoundary],
                ['generation_id = ?' => (int)$generation['generation_id']]
            );
            $connection->commit();
        } catch (\Throwable $throwable) {
            $connection->rollBack();
            throw $throwable;
        }

        return $this->generationRepository->get((int)$generation['generation_id']);
    }

    private function latestJournalBoundary(int $storeId): int
    {
        $connection = $this->resourceConnection->getConnection();
        $journalTable = $this->resourceConnection->getTableName('mageos_opensearch_hybrid_change_journal');

        return (int)$connection->fetchOne(
            $connection->select()
                ->from($journalTable, [new \Zend_Db_Expr('COALESCE(MAX(change_id), 0)')])
                ->where('store_id = ?', $storeId)
        );
    }
}
