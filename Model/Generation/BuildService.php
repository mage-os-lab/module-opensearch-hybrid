<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Generation;

class BuildService
{
    private const GENERATION_TABLE = 'mageos_opensearch_hybrid_generation';
    private const PROGRESS_TABLE = 'mageos_opensearch_hybrid_generation_progress';
    private const STORE_STATE_TABLE = 'mageos_opensearch_hybrid_store_state';

    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract $retrievalContract,
        private readonly \MageOS\OpenSearchHybrid\Model\Config $config,
        private readonly \MageOS\OpenSearchHybrid\Model\Generation\StoreScopeFingerprint $storeScopeFingerprint,
        private readonly \MageOS\OpenSearchHybrid\Model\OpenSearch\IndexManager $indexManager,
        private readonly \MageOS\OpenSearchHybrid\Model\Generation\BuildRefreshService $buildRefreshService,
        private readonly \MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository $generationRepository,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\SearchableProductResolver $searchableProductResolver,
        private readonly \MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository $outboxRepository,
        private readonly \MageOS\OpenSearchHybrid\Model\AsyncEvent\LifecyclePublisher $publisher,
        private readonly \MageOS\OpenSearchHybrid\Model\AsyncEvent\SubscriptionReconciler $subscriptionReconciler,
        private readonly \MageOS\OpenSearchHybrid\Model\Encoder\EncoderIdentityClient $identityClient,
        private readonly \Psr\Log\LoggerInterface $logger,
        private readonly \Magento\Framework\Serialize\Serializer\Json $json
    ) {
    }

    public function buildFake(int $storeId): int
    {
        $fakeRevision = hash('sha256', 'deterministic-fake-v1');

        return $this->createGeneration(
            $storeId,
            [
                'model_id' => 'mageos/deterministic-fake-v1',
                'model_revision' => $fakeRevision,
                'dimension' => 256,
                'recipe_version' => (string)$this->retrievalContract->get()['model']['source_recipe_version'],
                'encoder_identity_digest' => $fakeRevision,
            ],
            'deterministic-fake://local',
            true
        );
    }

    public function planProduction(int $storeId, string $expectedIdentityDigest): array
    {
        $this->assertStoreIncluded($storeId);
        $endpoint = $this->config->encoderEndpoint();
        $identity = $this->identityClient->fetchAt($endpoint, $expectedIdentityDigest);
        $this->indexManager->assertSupportedVersion();
        $eligibleProducts = $this->searchableProductResolver->countEligible($storeId);
        $batchSize = $this->config->batchSize();
        $maxOutstandingBatches = $this->config->maxOutstandingBatches();
        $expectedBatches = $eligibleProducts === 0
            ? 0
            : intdiv($eligibleProducts + $batchSize - 1, $batchSize);
        $initialJobs = min($expectedBatches, $maxOutstandingBatches);
        $initialItems = min($eligibleProducts, $initialJobs * $batchSize);
        $dimension = (int)$identity['dimension'];
        $rawBytesPerVector = $dimension * 4;
        $base64BytesPerVector = 4 * intdiv($rawBytesPerVector + 2, 3);
        $currentGenerations = $this->currentGenerations($storeId);
        $currentActivation = $this->currentActivation($storeId);
        $scopeDigest = $this->storeScopeFingerprint->digest($storeId);

        $preview = [
            'schema_version' => 1,
            'operation' => 'build_production_generation',
            'mode' => 'preview',
            'mutation' => 'none',
            'store_id' => $storeId,
            'encoder' => [
                'endpoint' => $endpoint,
                'identity_digest' => (string)$identity['encoder_identity_digest'],
                'model_id' => (string)$identity['model_id'],
                'model_revision' => (string)$identity['model_revision'],
                'dimension' => $dimension,
                'recipe_version' => (string)$identity['recipe_version'],
                'architecture' => (string)$identity['architecture'],
                'deployment_artifact_digest' => (string)$identity['deployment']['artifact_digest'],
            ],
            'contract' => [
                'contract_digest' => $this->retrievalContract->digest(),
                'result_contract_digest' => $this->retrievalContract->resultContractDigest(),
                'mapping_digest' => $this->retrievalContract->mappingDigest(),
                'pipeline_digest' => $this->retrievalContract->pipelineDigest(),
                'pipeline_id' => $this->retrievalContract->pipelineId(),
            ],
            'catalog' => [
                'eligible_products' => $eligibleProducts,
            ],
            'scope' => [
                'digest' => $scopeDigest,
            ],
            'workload' => [
                'batch_size' => $batchSize,
                'max_outstanding_batches' => $maxOutstandingBatches,
                'expected_total_embedding_batches' => $expectedBatches,
                'initial_outbox_jobs_without_consumer_progress' => $initialJobs,
                'initial_outbox_items_without_consumer_progress' => $initialItems,
                'target_vectors' => $eligibleProducts,
            ],
            'vector_payload' => [
                'raw_float32_bytes_per_vector' => $rawBytesPerVector,
                'raw_float32_bytes' => $eligibleProducts * $rawBytesPerVector,
                'base64_bytes_per_vector' => $base64BytesPerVector,
                'base64_bytes' => $eligibleProducts * $base64BytesPerVector,
                'estimate_scope' => 'Vector values only. Excludes document source, index graph, replicas, and database overhead.',
            ],
            'lifecycle_database_targets' => [
                'generation_rows_created' => 1,
                'progress_rows_created' => 1,
                'embedding_outbox_jobs_expected' => $expectedBatches,
                'outbox_items_expected' => $eligibleProducts,
                'embedding_rows_target' => $eligibleProducts,
            ],
            'opensearch' => [
                'physical_indices_created' => 1,
                'target_documents' => $eligibleProducts,
                'shared_pipeline_put_requests' => 1,
            ],
            'current_generations' => [
                'count' => count($currentGenerations),
                'items' => $currentGenerations,
            ],
            'activation' => [
                'changes' => 0,
                'current_mode' => $currentActivation['mode'],
                'current_generation_id' => $currentActivation['generation_id'],
                'planned_mode' => $currentActivation['mode'],
                'planned_generation_id' => $currentActivation['generation_id'],
                'storefront_routing' => 'unchanged',
                'new_generation_state' => 'BUILDING',
            ],
        ];
        $preview['confirmation_token'] = sprintf(
            'build-production-%d-%s',
            $storeId,
            hash('sha256', $this->json->serialize($preview))
        );

        return $preview;
    }

    public function buildProduction(int $storeId, string $expectedIdentityDigest): int
    {
        $this->assertStoreIncluded($storeId);
        $endpoint = $this->config->encoderEndpoint();
        $identity = $this->identityClient->fetchAt($endpoint, $expectedIdentityDigest);

        return $this->createGeneration($storeId, $identity, $endpoint, false, true);
    }

    public function buildProductionConfirmed(
        int $storeId,
        string $expectedIdentityDigest,
        string $confirmationToken
    ): int {
        $preview = $this->planProduction($storeId, $expectedIdentityDigest);
        if (!hash_equals((string)$preview['confirmation_token'], $confirmationToken)) {
            throw new \InvalidArgumentException(
                'The production build confirmation does not match the current exact preview.'
            );
        }
        $encoder = $preview['encoder'];

        return $this->createGeneration(
            $storeId,
            [
                'model_id' => (string)$encoder['model_id'],
                'model_revision' => (string)$encoder['model_revision'],
                'dimension' => (int)$encoder['dimension'],
                'recipe_version' => (string)$encoder['recipe_version'],
                'encoder_identity_digest' => (string)$encoder['identity_digest'],
            ],
            (string)$encoder['endpoint'],
            false,
            true,
            (string)$preview['scope']['digest']
        );
    }

    private function createGeneration(
        int $storeId,
        array $identity,
        string $endpoint,
        bool $isFake,
        bool $identityVerified = false,
        ?string $scopeDigest = null
    ): int {
        $this->assertStoreIncluded($storeId);
        $this->indexManager->assertSupportedVersion();
        $contract = $this->retrievalContract->get();
        $connection = $this->resourceConnection->getConnection();
        $generationTable = $this->resourceConnection->getTableName(self::GENERATION_TABLE);
        $journalTable = $this->resourceConnection->getTableName('mageos_opensearch_hybrid_change_journal');
        $capturedChangeId = (int)$connection->fetchOne(
            $connection->select()->from($journalTable, [new \Zend_Db_Expr('COALESCE(MAX(change_id), 0)')])
                ->where('store_id = ?', $storeId)
        );
        $coverageTotal = $this->searchableProductResolver->countEligible($storeId);
        $progressTable = $this->resourceConnection->getTableName(self::PROGRESS_TABLE);
        $connection->beginTransaction();
        try {
            $connection->insert($generationTable, [
                'store_id' => $storeId,
                'state' => 'BUILDING',
                'model_id' => (string)$identity['model_id'],
                'model_revision' => (string)$identity['model_revision'],
                'dimension' => (int)$identity['dimension'],
                'recipe_version' => (string)$identity['recipe_version'],
                'contract_version' => (string)$contract['result_contract']['version'],
                'contract_digest' => $this->retrievalContract->digest(),
                'result_contract_digest' => $this->retrievalContract->resultContractDigest(),
                'mapping_digest' => $this->retrievalContract->mappingDigest(),
                'pipeline_digest' => $this->retrievalContract->pipelineDigest(),
                'physical_index' => 'pending',
                'pipeline_id' => $this->retrievalContract->pipelineId(),
                'encoder_endpoint' => $endpoint,
                'encoder_identity_digest' => (string)$identity['encoder_identity_digest'],
                'scope_digest' => $scopeDigest ?? $this->storeScopeFingerprint->digest($storeId),
                'captured_change_id' => $capturedChangeId,
                'coverage_total' => $coverageTotal,
                'is_fake' => $isFake ? 1 : 0,
            ]);
            $generationId = (int)$connection->lastInsertId($generationTable);
            $physicalIndex = $this->retrievalContract->physicalIndexName($storeId, $generationId);
            $connection->update(
                $generationTable,
                ['physical_index' => $physicalIndex],
                ['generation_id = ?' => $generationId]
            );
            $connection->insert($progressTable, [
                'store_id' => $storeId,
                'generation_id' => $generationId,
                'captured_boundary' => $capturedChangeId,
                'processed_boundary' => $capturedChangeId,
                'last_seeded_product_id' => 0,
                'seeded_products' => 0,
                'seeding_state' => 'PENDING',
                'build_refresh_interval' => '-1',
                'normal_refresh_interval' => '1s',
                'refresh_state' => 'NORMAL',
            ]);
            $connection->commit();
        } catch (\Throwable $throwable) {
            $connection->rollBack();
            throw $throwable;
        }
        $generation = $this->generationRepository->get($generationId);
        $this->indexManager->ensureGeneration($generation);
        $this->buildRefreshService->suppress($generation);
        $this->subscriptionReconciler->reconcile($storeId);
        $this->resumeGeneration($generation, $identityVerified);

        return $generationId;
    }

    public function resume(int $generationId): array
    {
        return $this->resumeGeneration($this->generationRepository->get($generationId));
    }

    public function resumeFake(int $generationId): array
    {
        $generation = $this->generationRepository->get($generationId);
        if (!(bool)$generation['is_fake']) {
            throw new \InvalidArgumentException('Only deterministic development generations can use fake resume.');
        }

        return $this->resumeGeneration($generation, true);
    }

    private function resumeGeneration(array $generation, bool $identityVerified = false): array
    {
        $generationId = (int)$generation['generation_id'];
        if (!in_array((string)$generation['state'], ['BUILDING', 'CATCHING_UP'], true)) {
            throw new \RuntimeException('Only building generations can resume full-build seeding.');
        }
        if (!(bool)$generation['is_fake'] && !$identityVerified) {
            $this->identityClient->fetchAt(
                (string)$generation['encoder_endpoint'],
                (string)$generation['encoder_identity_digest']
            );
        }
        $this->resourceConnection->getConnection()->update(
            $this->resourceConnection->getTableName(self::PROGRESS_TABLE),
            ['seeding_state' => 'PENDING'],
            ['generation_id = ?' => $generationId, 'seeding_state = ?' => 'PAUSED']
        );
        $this->indexManager->ensureGeneration($generation);
        $this->buildRefreshService->suppress($generation);
        $seededProducts = 0;
        $seededBatches = 0;
        $seedingState = 'PENDING';
        while (true) {
            $batch = $this->seedNextBatch($generation);
            $seedingState = (string)$batch['seeding_state'];
            if (!isset($batch['job_id'])) {
                break;
            }
            $seededProducts += (int)$batch['product_count'];
            $seededBatches++;
            $this->publish((string)$batch['job_id']);
        }

        return [
            'generation_id' => $generationId,
            'store_id' => (int)$generation['store_id'],
            'seeding_state' => $seedingState,
            'seeded_products' => $seededProducts,
            'seeded_batches' => $seededBatches,
        ];
    }

    public function pauseFake(int $generationId): array
    {
        $generation = $this->generationRepository->get($generationId);
        if (!(bool)$generation['is_fake']) {
            throw new \InvalidArgumentException('Only deterministic development generations can use fake pause.');
        }

        return $this->pauseGeneration($generation);
    }

    public function pause(int $generationId): array
    {
        return $this->pauseGeneration($this->generationRepository->get($generationId));
    }

    private function pauseGeneration(array $generation): array
    {
        $generationId = (int)$generation['generation_id'];
        if (!in_array((string)$generation['state'], ['BUILDING', 'CATCHING_UP'], true)) {
            throw new \RuntimeException('Only building generations can pause full-build seeding.');
        }
        $connection = $this->resourceConnection->getConnection();
        $progressTable = $this->resourceConnection->getTableName(self::PROGRESS_TABLE);
        $affected = $connection->update(
            $progressTable,
            ['seeding_state' => 'PAUSED'],
            ['generation_id = ?' => $generationId, 'seeding_state != ?' => 'COMPLETED']
        );
        $progress = $connection->fetchRow(
            $connection->select()->from($progressTable)->where('generation_id = ?', $generationId)
        );
        if ($affected !== 1
            && (!is_array($progress) || (string)$progress['seeding_state'] !== 'PAUSED')
        ) {
            throw new \RuntimeException('Completed or missing full-build seeding cannot be paused.');
        }

        return [
            'generation_id' => $generationId,
            'store_id' => (int)$generation['store_id'],
            'seeding_state' => 'PAUSED',
            'seeded_products' => (int)$progress['seeded_products'],
        ];
    }

    public function resumePendingBuilds(?int $storeId = null): array
    {
        $connection = $this->resourceConnection->getConnection();
        $generationTable = $this->resourceConnection->getTableName(self::GENERATION_TABLE);
        $progressTable = $this->resourceConnection->getTableName(self::PROGRESS_TABLE);
        $select = $connection->select()
            ->from(['generation' => $generationTable], ['generation_id'])
            ->joinInner(
                ['progress' => $progressTable],
                'progress.generation_id = generation.generation_id',
                []
            )
            ->where('generation.state IN (?)', ['BUILDING', 'CATCHING_UP'])
            ->where('progress.seeding_state = ?', 'PENDING')
            ->order('generation.generation_id ASC')
            ->limit(25);
        if ($storeId !== null) {
            $select->where('generation.store_id = ?', $storeId);
        }
        $reports = [];
        foreach ($connection->fetchCol($select) as $generationId) {
            try {
                $reports[] = $this->resume((int)$generationId);
            } catch (\Throwable $throwable) {
                $this->logger->warning('OpenSearch Hybrid full-build resume failed.', [
                    'generation_id' => (int)$generationId,
                    'error_class' => $throwable::class,
                ]);
                $reports[] = [
                    'generation_id' => (int)$generationId,
                    'seeding_state' => 'FAILED_TO_RESUME',
                    'error_class' => $throwable::class,
                ];
            }
        }

        return $reports;
    }

    private function assertStoreIncluded(int $storeId): void
    {
        if (!$this->config->isStoreIncluded($storeId)) {
            throw new \InvalidArgumentException('The store is not included in OpenSearch Hybrid builds.');
        }
    }

    private function currentGenerations(int $storeId): array
    {
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName(self::GENERATION_TABLE);
        $rows = $connection->fetchAll(
            $connection->select()
                ->from($table, ['generation_id', 'state', 'physical_index', 'is_fake'])
                ->where('store_id = ?', $storeId)
                ->order('generation_id ASC')
        );

        $generations = [];
        foreach ($rows as $row) {
            $generations[] = [
                'generation_id' => (int)$row['generation_id'],
                'state' => (string)$row['state'],
                'physical_index' => (string)$row['physical_index'],
                'is_fake' => (bool)$row['is_fake'],
            ];
        }

        return $generations;
    }

    private function currentActivation(int $storeId): array
    {
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName(self::STORE_STATE_TABLE);
        $row = $connection->fetchRow(
            $connection->select()
                ->from($table, ['activation_mode', 'active_generation_id'])
                ->where('store_id = ?', $storeId)
        );
        if (!is_array($row)) {
            return [
                'mode' => 'NATIVE',
                'generation_id' => null,
            ];
        }

        return [
            'mode' => (string)$row['activation_mode'],
            'generation_id' => $row['active_generation_id'] === null
                ? null
                : (int)$row['active_generation_id'],
        ];
    }

    private function seedNextBatch(array $generation): array
    {
        $connection = $this->resourceConnection->getConnection();
        $progressTable = $this->resourceConnection->getTableName(self::PROGRESS_TABLE);
        $generationId = (int)$generation['generation_id'];
        $storeId = (int)$generation['store_id'];
        $connection->beginTransaction();
        try {
            $progress = $connection->fetchRow(
                $connection->select()
                    ->from($progressTable)
                    ->where('generation_id = ?', $generationId)
                    ->forUpdate(true)
            );
            if (!is_array($progress)) {
                throw new \RuntimeException('The generation has no resumable progress record.');
            }
            if ((string)$progress['seeding_state'] === 'COMPLETED') {
                $connection->commit();
                return ['seeding_state' => 'COMPLETED'];
            }
            if ((string)$progress['seeding_state'] === 'PAUSED') {
                $connection->commit();
                return ['seeding_state' => 'PAUSED'];
            }
            $outstanding = $this->outboxRepository->countOutstandingForGeneration(
                $generationId,
                'EMBEDDING'
            );
            if ($outstanding >= $this->config->maxOutstandingBatches()) {
                $connection->commit();
                return ['seeding_state' => 'PAUSED_CAPACITY'];
            }
            $batchSize = $this->config->batchSize();
            $productIds = $this->searchableProductResolver->nextEligibleIds(
                $storeId,
                (int)$progress['last_seeded_product_id'],
                $batchSize
            );
            if ($productIds === []) {
                $connection->update(
                    $progressTable,
                    ['seeding_state' => 'COMPLETED'],
                    ['progress_id = ?' => (int)$progress['progress_id']]
                );
                $connection->commit();
                return ['seeding_state' => 'COMPLETED'];
            }
            $jobId = $this->outboxRepository->create(
                $storeId,
                $generationId,
                'EMBEDDING',
                'UPSERT',
                $productIds,
                (int)$generation['captured_change_id'],
                0,
                (int)$generation['captured_change_id']
            );
            $connection->update(
                $progressTable,
                [
                    'last_seeded_product_id' => max($productIds),
                    'seeded_products' => new \Zend_Db_Expr(sprintf(
                        'seeded_products + %d',
                        count($productIds)
                    )),
                    'seeding_state' => 'PENDING',
                ],
                ['progress_id = ?' => (int)$progress['progress_id']]
            );
            $connection->commit();

            return [
                'seeding_state' => 'PENDING',
                'job_id' => $jobId,
                'product_count' => count($productIds),
            ];
        } catch (\Throwable $throwable) {
            $connection->rollBack();
            throw $throwable;
        }
    }

    private function publish(string $jobId): void
    {
        try {
            $this->publisher->publish($jobId);
        } catch (\Throwable $throwable) {
            $this->logger->warning('OpenSearch Hybrid job remains pending after publication failed.', [
                'job_id' => $jobId,
                'error_class' => $throwable::class,
            ]);
        }
    }
}
