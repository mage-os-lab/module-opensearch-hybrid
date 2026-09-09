<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Activation;

class ActivationService
{
    private const GENERATION_TABLE = 'mageos_opensearch_hybrid_generation';
    private const STATE_TABLE = 'mageos_opensearch_hybrid_store_state';
    private const AUDIT_TABLE = 'mageos_opensearch_hybrid_activation';

    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository $generationRepository,
        private readonly \MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract $retrievalContract,
        private readonly \MageOS\OpenSearchHybrid\Model\Generation\StoreScopeFingerprint $storeScopeFingerprint,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\ChangeJournalRepository $journalRepository,
        private readonly \MageOS\OpenSearchHybrid\Model\OpenSearch\IndexManager $indexManager,
        private readonly \Magento\Framework\Serialize\Serializer\Json $json
    ) {
    }

    public function previewActivation(int $storeId, int $generationId): array
    {
        $generation = $this->generationRepository->get($generationId);
        $this->assertActivatable($generation, $storeId, ['READY']);
        $this->assertNoUnresolvedWork($generation);
        $this->assertLiveGeneration($generation);

        return $this->buildPreview('activate', $storeId, $generation);
    }

    public function activateConfirmed(
        int $storeId,
        int $generationId,
        string $confirmationToken,
        string $actor
    ): array {
        return $this->activate($storeId, $generationId, $actor, $confirmationToken);
    }

    public function previewNativeRollback(int $storeId): array
    {
        $state = $this->storeState($storeId);
        if ((string)$state['activation_mode'] === 'NATIVE' && $state['active_generation_id'] === null) {
            throw new \RuntimeException('The store already uses native search.');
        }

        return $this->buildPreview('rollback_native', $storeId, null, $state);
    }

    public function rollbackToNativeConfirmed(
        int $storeId,
        string $confirmationToken,
        string $actor
    ): array {
        return $this->rollbackToNative($storeId, $actor, $confirmationToken);
    }

    public function previewGenerationRollback(int $storeId, int $generationId): array
    {
        $generation = $this->generationRepository->get($generationId);
        $this->assertActivatable($generation, $storeId, ['RETAINED']);
        $this->assertRetainedGenerationIsCurrent($generation);
        $this->assertLiveGeneration($generation);

        return $this->buildPreview('rollback_generation', $storeId, $generation);
    }

    public function rollbackToGenerationConfirmed(
        int $storeId,
        int $generationId,
        string $confirmationToken,
        string $actor
    ): array {
        return $this->rollbackToGeneration($storeId, $generationId, $actor, $confirmationToken);
    }

    public function activate(
        int $storeId,
        int $generationId,
        string $actor = 'cli',
        ?string $confirmationToken = null
    ): array {
        $generation = $this->generationRepository->get($generationId);
        $this->assertActivatable($generation, $storeId, ['READY']);
        $this->assertLiveGeneration($generation);
        $connection = $this->resourceConnection->getConnection();
        $generationTable = $this->resourceConnection->getTableName(self::GENERATION_TABLE);
        $stateTable = $this->resourceConnection->getTableName(self::STATE_TABLE);
        $auditTable = $this->resourceConnection->getTableName(self::AUDIT_TABLE);
        $connection->beginTransaction();
        try {
            $this->lockStoreGenerations($connection, $generationTable, $storeId);
            $lockedGeneration = $connection->fetchRow(
                $connection->select()->from($generationTable)
                    ->where('generation_id = ?', $generationId)
                    ->forUpdate(true)
            );
            if (!is_array($lockedGeneration)) {
                throw new \RuntimeException('The target generation disappeared during activation.');
            }
            $this->assertActivatable($lockedGeneration, $storeId, ['READY']);
            $this->assertNoUnresolvedWork($lockedGeneration);
            $lockedState = $this->lockedStoreState($connection, $stateTable, $storeId);
            if ($confirmationToken !== null) {
                $this->assertConfirmed(
                    $this->buildPreview('activate', $storeId, $lockedGeneration, $lockedState),
                    $confirmationToken
                );
            }
            $priorGenerationId = $lockedState['active_generation_id'] === null
                ? null
                : (int)$lockedState['active_generation_id'];
            $connection->update(
                $generationTable,
                ['state' => 'RETAINED'],
                ['store_id = ?' => $storeId, 'state = ?' => 'ACTIVE']
            );
            $connection->update(
                $generationTable,
                ['state' => 'ACTIVE'],
                ['generation_id = ?' => $generationId, 'state = ?' => 'READY']
            );
            $watermark = (int)$lockedGeneration['captured_change_id'];
            $this->journalRepository->acknowledgeGenerationInvalidations($storeId, $watermark);
            $connection->insertOnDuplicate(
                $stateTable,
                [
                    'store_id' => $storeId,
                    'activation_mode' => 'HYBRID',
                    'active_generation_id' => $generationId,
                    'required_watermark' => $watermark,
                    'native_watermark' => $watermark,
                    'hybrid_watermark' => $watermark,
                    'readiness_latch' => 1,
                    'cache_version' => 1,
                ],
                [
                    'activation_mode',
                    'active_generation_id',
                    'required_watermark',
                    'native_watermark',
                    'hybrid_watermark',
                    'readiness_latch',
                    'cache_version' => new \Zend_Db_Expr('cache_version + 1'),
                ]
            );
            $auditId = $this->insertAudit(
                $connection,
                $auditTable,
                $storeId,
                $generationId,
                $priorGenerationId,
                'HYBRID',
                $lockedGeneration,
                $actor
            );
            $connection->commit();
        } catch (\Throwable $throwable) {
            $connection->rollBack();
            throw $throwable;
        }

        return [
            'schema_version' => 1,
            'status' => 'activated',
            'store_id' => $storeId,
            'generation_id' => $generationId,
            'prior_generation_id' => $priorGenerationId,
            'activation_id' => $auditId,
        ];
    }

    public function rollbackToNative(
        int $storeId,
        string $actor = 'cli',
        ?string $confirmationToken = null
    ): array {
        $connection = $this->resourceConnection->getConnection();
        $generationTable = $this->resourceConnection->getTableName(self::GENERATION_TABLE);
        $stateTable = $this->resourceConnection->getTableName(self::STATE_TABLE);
        $auditTable = $this->resourceConnection->getTableName(self::AUDIT_TABLE);
        $connection->beginTransaction();
        try {
            $this->lockStoreGenerations($connection, $generationTable, $storeId);
            $lockedState = $this->lockedStoreState($connection, $stateTable, $storeId);
            if ($confirmationToken !== null) {
                $this->assertConfirmed(
                    $this->buildPreview('rollback_native', $storeId, null, $lockedState),
                    $confirmationToken
                );
            }
            $priorGenerationId = $lockedState['active_generation_id'] === null
                ? null
                : (int)$lockedState['active_generation_id'];
            $connection->update(
                $generationTable,
                ['state' => 'RETAINED'],
                ['store_id = ?' => $storeId, 'state = ?' => 'ACTIVE']
            );
            $connection->insertOnDuplicate(
                $stateTable,
                [
                    'store_id' => $storeId,
                    'activation_mode' => 'NATIVE',
                    'active_generation_id' => null,
                    'readiness_latch' => 0,
                    'cache_version' => 1,
                ],
                [
                    'activation_mode',
                    'active_generation_id',
                    'readiness_latch',
                    'cache_version' => new \Zend_Db_Expr('cache_version + 1'),
                ]
            );
            $auditId = $this->insertAudit(
                $connection,
                $auditTable,
                $storeId,
                null,
                $priorGenerationId,
                'NATIVE',
                null,
                $actor
            );
            $connection->commit();
        } catch (\Throwable $throwable) {
            $connection->rollBack();
            throw $throwable;
        }

        return [
            'schema_version' => 1,
            'status' => 'rolled_back_to_native',
            'store_id' => $storeId,
            'prior_generation_id' => $priorGenerationId,
            'activation_id' => $auditId,
        ];
    }

    public function rollbackToGeneration(
        int $storeId,
        int $generationId,
        string $actor = 'cli',
        ?string $confirmationToken = null
    ): array {
        $generation = $this->generationRepository->get($generationId);
        $this->assertActivatable($generation, $storeId, ['RETAINED']);
        $this->assertRetainedGenerationIsCurrent($generation);
        $this->assertLiveGeneration($generation);
        $connection = $this->resourceConnection->getConnection();
        $generationTable = $this->resourceConnection->getTableName(self::GENERATION_TABLE);
        $stateTable = $this->resourceConnection->getTableName(self::STATE_TABLE);
        $auditTable = $this->resourceConnection->getTableName(self::AUDIT_TABLE);
        $connection->beginTransaction();
        try {
            $this->lockStoreGenerations($connection, $generationTable, $storeId);
            $lockedGeneration = $connection->fetchRow(
                $connection->select()->from($generationTable)
                    ->where('generation_id = ?', $generationId)
                    ->forUpdate(true)
            );
            if (!is_array($lockedGeneration)) {
                throw new \RuntimeException('The rollback generation disappeared.');
            }
            $this->assertActivatable($lockedGeneration, $storeId, ['RETAINED']);
            $this->assertRetainedGenerationIsCurrent($lockedGeneration);
            $lockedState = $this->lockedStoreState($connection, $stateTable, $storeId);
            if ($confirmationToken !== null) {
                $this->assertConfirmed(
                    $this->buildPreview('rollback_generation', $storeId, $lockedGeneration, $lockedState),
                    $confirmationToken
                );
            }
            $priorGenerationId = $lockedState['active_generation_id'] === null
                ? null
                : (int)$lockedState['active_generation_id'];
            $connection->update(
                $generationTable,
                ['state' => 'RETAINED'],
                [
                    'store_id = ?' => $storeId,
                    'state = ?' => 'ACTIVE',
                    'generation_id != ?' => $generationId,
                ]
            );
            $connection->update(
                $generationTable,
                ['state' => 'ACTIVE'],
                ['generation_id = ?' => $generationId, 'state = ?' => 'RETAINED']
            );
            $watermark = (int)$lockedGeneration['captured_change_id'];
            $this->journalRepository->acknowledgeGenerationInvalidations($storeId, $watermark);
            $connection->insertOnDuplicate(
                $stateTable,
                [
                    'store_id' => $storeId,
                    'activation_mode' => 'HYBRID',
                    'active_generation_id' => $generationId,
                    'required_watermark' => $watermark,
                    'native_watermark' => $watermark,
                    'hybrid_watermark' => $watermark,
                    'readiness_latch' => 1,
                    'cache_version' => 1,
                ],
                [
                    'activation_mode',
                    'active_generation_id',
                    'required_watermark',
                    'native_watermark',
                    'hybrid_watermark',
                    'readiness_latch',
                    'cache_version' => new \Zend_Db_Expr('cache_version + 1'),
                ]
            );
            $auditId = $this->insertAudit(
                $connection,
                $auditTable,
                $storeId,
                $generationId,
                $priorGenerationId,
                'HYBRID',
                $lockedGeneration,
                $actor
            );
            $connection->commit();
        } catch (\Throwable $throwable) {
            $connection->rollBack();
            throw $throwable;
        }

        return [
            'schema_version' => 1,
            'status' => 'rolled_back_to_generation',
            'store_id' => $storeId,
            'generation_id' => $generationId,
            'prior_generation_id' => $priorGenerationId,
            'activation_id' => $auditId,
        ];
    }

    private function assertActivatable(array $generation, int $storeId, array $allowedStates): void
    {
        if ((int)$generation['store_id'] !== $storeId) {
            throw new \InvalidArgumentException('The generation belongs to a different store.');
        }
        if ((bool)$generation['is_fake']) {
            throw new \RuntimeException('Deterministic fake generations can never be activated.');
        }
        if (!in_array((string)$generation['state'], $allowedStates, true)
            || !(bool)$generation['result_contract_accepted']
            || !hash_equals(
                $this->retrievalContract->resultContractDigest(),
                (string)$generation['result_contract_digest']
            )
            || !hash_equals(
                $this->retrievalContract->digest(),
                (string)$generation['contract_digest']
            )
            || !hash_equals(
                $this->retrievalContract->mappingDigest(),
                (string)$generation['mapping_digest']
            )
            || !hash_equals(
                $this->retrievalContract->pipelineDigest(),
                (string)$generation['pipeline_digest']
            )
            || !hash_equals(
                (string)$generation['scope_digest'],
                $this->storeScopeFingerprint->digest($storeId)
            )
        ) {
            throw new \RuntimeException('The generation is not ready for the requested activation.');
        }
        $report = $generation['validation_report'] === null
            ? null
            : $this->json->unserialize((string)$generation['validation_report']);
        if (!is_array($report) || ($report['ready'] ?? false) !== true) {
            throw new \RuntimeException('The generation has no passing validation report.');
        }
    }

    private function assertNoUnresolvedWork(array $generation): void
    {
        $connection = $this->resourceConnection->getConnection();
        $outboxTable = $this->resourceConnection->getTableName('mageos_opensearch_hybrid_outbox');
        $unresolved = (int)$connection->fetchOne(
            $connection->select()->from($outboxTable, [new \Zend_Db_Expr('COUNT(*)')])
                ->where('generation_id = ?', (int)$generation['generation_id'])
                ->where('latest_change_id <= ?', (int)$generation['captured_change_id'])
                ->where('state NOT IN (?)', ['COMPLETED', 'SUPERSEDED'])
        );
        if ($unresolved !== 0
            || (int)$generation['coverage_complete'] !== (int)$generation['coverage_total']
            || (int)$generation['coverage_failed'] !== 0
        ) {
            throw new \RuntimeException('The generation changed after validation and is no longer complete.');
        }
    }

    private function assertRetainedGenerationIsCurrent(array $generation): void
    {
        $connection = $this->resourceConnection->getConnection();
        $journalTable = $this->resourceConnection->getTableName('mageos_opensearch_hybrid_change_journal');
        $latestChangeId = (int)$connection->fetchOne(
            $connection->select()->from($journalTable, [new \Zend_Db_Expr('COALESCE(MAX(change_id), 0)')])
                ->where('store_id = ?', (int)$generation['store_id'])
        );
        if ($latestChangeId > (int)$generation['captured_change_id']) {
            throw new \RuntimeException(
                'The retained generation is behind the store journal and cannot be restored safely.'
            );
        }
        $this->assertNoUnresolvedWork($generation);
    }

    private function assertLiveGeneration(array $generation): void
    {
        $live = $this->indexManager->validateGeneration($generation);
        if (!(bool)$live['index_exists']
            || !(bool)$live['mapping']
            || !(bool)$live['settings']
            || !(bool)$live['pipeline']
            || !(bool)$live['document_identity']
            || !(bool)$live['refresh_interval_restored']
            || (int)$live['index_count'] !== (int)$generation['coverage_total']
        ) {
            throw new \RuntimeException('The live hybrid generation changed after validation.');
        }
    }

    private function lockStoreGenerations(
        \Magento\Framework\DB\Adapter\AdapterInterface $connection,
        string $generationTable,
        int $storeId
    ): void {
        // Match change capture's generation-before-store lock order. Lock the
        // current and target rows together so their preview summaries stay fixed.
        $connection->fetchAll(
            $connection->select()->from($generationTable, ['generation_id'])
                ->where('store_id = ?', $storeId)
                ->order('generation_id ASC')
                ->forUpdate(true)
        );
    }

    private function lockedStoreState(
        \Magento\Framework\DB\Adapter\AdapterInterface $connection,
        string $stateTable,
        int $storeId
    ): array {
        $state = $connection->fetchRow(
            $connection->select()->from($stateTable)->where('store_id = ?', $storeId)->forUpdate(true)
        );

        return is_array($state) ? $state : $this->emptyStoreState();
    }

    private function buildPreview(
        string $operation,
        int $storeId,
        ?array $targetGeneration,
        ?array $knownState = null
    ): array {
        $state = $knownState ?? $this->storeState($storeId);
        $currentGeneration = $this->currentGenerationSummary($state['active_generation_id']);
        $preview = [
            'schema_version' => 1,
            'mode' => 'preview',
            'operation' => $operation,
            'store_id' => $storeId,
            'current_state' => [
                'activation_mode' => (string)$state['activation_mode'],
                'active_generation_id' => $state['active_generation_id'] === null
                    ? null
                    : (int)$state['active_generation_id'],
                'required_watermark' => (int)$state['required_watermark'],
                'native_watermark' => (int)$state['native_watermark'],
                'hybrid_watermark' => (int)$state['hybrid_watermark'],
                'readiness_latch' => (bool)$state['readiness_latch'],
                'cache_version' => (int)$state['cache_version'],
                'updated_at' => $state['updated_at'],
            ],
            'current_generation' => $currentGeneration,
            'target' => [
                'activation_mode' => $operation === 'rollback_native' ? 'NATIVE' : 'HYBRID',
                'generation' => $targetGeneration === null
                    ? null
                    : $this->generationSummary($targetGeneration),
            ],
            'human_confirmation' => $operation === 'rollback_native'
                ? 'native'
                : (string)$targetGeneration['generation_id'],
        ];
        $preview['confirmation_token'] = sprintf(
            '%s-%d-%s',
            str_replace('_', '-', $operation),
            $storeId,
            hash('sha256', $this->json->serialize($preview))
        );

        return $preview;
    }

    private function generationSummary(array $generation): array
    {
        $validationReport = null;
        if (is_string($generation['validation_report']) && $generation['validation_report'] !== '') {
            try {
                $validationReport = $this->json->unserialize($generation['validation_report']);
            } catch (\InvalidArgumentException) {
                $validationReport = [
                    'unreadable' => true,
                    'sha256' => hash('sha256', $generation['validation_report']),
                ];
            }
        }

        return [
            'generation_id' => (int)$generation['generation_id'],
            'state' => (string)$generation['state'],
            'is_fake' => (bool)$generation['is_fake'],
            'coverage_total' => (int)$generation['coverage_total'],
            'coverage_complete' => (int)$generation['coverage_complete'],
            'coverage_failed' => (int)$generation['coverage_failed'],
            'captured_change_id' => (int)$generation['captured_change_id'],
            'result_contract_accepted' => (bool)$generation['result_contract_accepted'],
            'validation_report' => $validationReport,
            'contract_digest' => (string)$generation['contract_digest'],
            'result_contract_digest' => (string)$generation['result_contract_digest'],
            'mapping_digest' => (string)$generation['mapping_digest'],
            'pipeline_digest' => (string)$generation['pipeline_digest'],
            'physical_index' => (string)$generation['physical_index'],
            'model_id' => (string)$generation['model_id'],
            'model_revision' => (string)$generation['model_revision'],
            'encoder_identity_digest' => $generation['encoder_identity_digest'] === null
                ? null
                : (string)$generation['encoder_identity_digest'],
            'updated_at' => (string)$generation['updated_at'],
        ];
    }

    private function currentGenerationSummary(mixed $generationId): ?array
    {
        if ($generationId === null) {
            return null;
        }
        try {
            return $this->generationSummary($this->generationRepository->get((int)$generationId));
        } catch (\Magento\Framework\Exception\NoSuchEntityException) {
            return [
                'generation_id' => (int)$generationId,
                'missing' => true,
            ];
        }
    }

    private function assertConfirmed(array $preview, string $confirmationToken): void
    {
        if (!hash_equals((string)$preview['confirmation_token'], $confirmationToken)) {
            throw new \InvalidArgumentException(
                'The activation confirmation does not match the current exact preview.'
            );
        }
    }

    private function storeState(int $storeId): array
    {
        $connection = $this->resourceConnection->getConnection();
        $state = $connection->fetchRow(
            $connection->select()
                ->from($this->resourceConnection->getTableName(self::STATE_TABLE))
                ->where('store_id = ?', $storeId)
        );
        if (is_array($state)) {
            return $state;
        }

        return $this->emptyStoreState();
    }

    private function emptyStoreState(): array
    {
        return [
            'activation_mode' => 'NATIVE',
            'active_generation_id' => null,
            'required_watermark' => 0,
            'native_watermark' => 0,
            'hybrid_watermark' => 0,
            'readiness_latch' => 0,
            'cache_version' => 0,
            'updated_at' => null,
        ];
    }

    private function insertAudit(
        \Magento\Framework\DB\Adapter\AdapterInterface $connection,
        string $auditTable,
        int $storeId,
        ?int $generationId,
        ?int $priorGenerationId,
        string $mode,
        ?array $generation,
        string $actor
    ): int {
        $connection->insert($auditTable, [
            'store_id' => $storeId,
            'generation_id' => $generationId,
            'prior_generation_id' => $priorGenerationId,
            'mode' => $mode,
            'status' => 'COMPLETED',
            'contract_digest' => $generation['result_contract_digest'] ?? null,
            'physical_index' => $generation['physical_index'] ?? null,
            'high_watermark' => (int)($generation['captured_change_id'] ?? 0),
            'validation_report' => $generation['validation_report'] ?? null,
            'actor' => $this->normalizeActor($actor),
            'completed_at' => new \Zend_Db_Expr('UTC_TIMESTAMP()'),
        ]);

        return (int)$connection->lastInsertId($auditTable);
    }

    private function normalizeActor(string $actor): string
    {
        $actor = trim($actor);
        if ($actor === '' || strlen($actor) > 255) {
            throw new \InvalidArgumentException('The activation audit actor is invalid.');
        }

        return $actor;
    }
}
