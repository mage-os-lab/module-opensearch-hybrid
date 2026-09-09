<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Generation;

use Magento\Framework\DB\Adapter\AdapterInterface;
use Magento\Framework\Exception\NoSuchEntityException;
use Zend_Db_Expr;

class GenerationCleanupService
{
    private const GENERATION_TABLE = 'mageos_opensearch_hybrid_generation';
    private const EMBEDDING_TABLE = 'mageos_opensearch_hybrid_embedding';
    private const PROGRESS_TABLE = 'mageos_opensearch_hybrid_generation_progress';
    private const STATE_TABLE = 'mageos_opensearch_hybrid_store_state';
    private const OUTBOX_TABLE = 'mageos_opensearch_hybrid_outbox';
    private const OUTBOX_ITEM_TABLE = 'mageos_opensearch_hybrid_outbox_item';
    private const ACTIVATION_TABLE = 'mageos_opensearch_hybrid_activation';
    private const CLEANUP_TABLE = 'mageos_opensearch_hybrid_cleanup';

    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \MageOS\OpenSearchHybrid\Model\Config $config,
        private readonly \MageOS\OpenSearchHybrid\Model\OpenSearch\IndexManager $indexManager,
        private readonly \Magento\Framework\Serialize\Serializer\Json $json
    ) {
    }

    public function preview(int $storeId, int $generationId): array
    {
        $connection = $this->resourceConnection->getConnection();
        $generation = $this->generation($connection, $generationId);

        return $this->buildPreview($connection, $storeId, $generation, null);
    }

    public function cleanup(int $storeId, int $generationId, string $confirmationToken): array
    {
        $connection = $this->resourceConnection->getConnection();
        $preview = $this->preview($storeId, $generationId);
        $this->assertConfirmedAndEligible($preview, $confirmationToken);
        $cleanupTable = $this->table(self::CLEANUP_TABLE);
        $confirmationDigest = hash('sha256', $confirmationToken);
        $connection->insert($cleanupTable, [
            'store_id' => $storeId,
            'generation_id' => $generationId,
            'status' => 'STARTED',
            'confirmation_digest' => $confirmationDigest,
            'generation_state' => (string)$preview['generation']['state'],
            'physical_index' => (string)$preview['opensearch']['indices'][0]['name'],
            'pipeline_id' => (string)$preview['opensearch']['pipelines'][0]['id'],
            'encoder_identity_digest' => $preview['encoder_artifacts'][0]['identity_digest'],
            'impact_report' => $this->json->serialize($preview),
            'actor' => 'cli',
        ]);
        $auditId = (int)$connection->lastInsertId($cleanupTable);

        $connection->beginTransaction();
        try {
            $generationTable = $this->table(self::GENERATION_TABLE);
            $lockedGeneration = $connection->fetchRow(
                $connection->select()
                    ->from($generationTable)
                    ->where('generation_id = ?', $generationId)
                    ->forUpdate(true)
            );
            if (!is_array($lockedGeneration)) {
                throw new NoSuchEntityException(__('OpenSearch Hybrid generation %1 no longer exists.', $generationId));
            }
            $lockedState = $connection->fetchRow(
                $connection->select()
                    ->from($this->table(self::STATE_TABLE))
                    ->where('store_id = ?', $storeId)
                    ->forUpdate(true)
            );
            $lockedPreview = $this->buildPreview(
                $connection,
                $storeId,
                $lockedGeneration,
                is_array($lockedState) ? $lockedState : []
            );
            $this->assertConfirmedAndEligible($lockedPreview, $confirmationToken);

            $this->indexManager->deleteGenerationIndex($lockedGeneration);
            $deleted = $connection->delete($generationTable, ['generation_id = ?' => $generationId]);
            if ($deleted !== 1) {
                throw new \RuntimeException('The confirmed generation was not deleted exactly once.');
            }
            $connection->update(
                $cleanupTable,
                [
                    'status' => 'COMPLETED',
                    'completed_at' => new Zend_Db_Expr('UTC_TIMESTAMP()'),
                ],
                ['cleanup_id = ?' => $auditId, 'status = ?' => 'STARTED']
            );
            $connection->commit();
        } catch (\Throwable $throwable) {
            $connection->rollBack();
            $connection->update(
                $cleanupTable,
                [
                    'status' => 'FAILED',
                    'last_error_class' => $throwable::class,
                    'last_diagnostic' => 'cleanup failed',
                    'completed_at' => new Zend_Db_Expr('UTC_TIMESTAMP()'),
                ],
                ['cleanup_id = ?' => $auditId]
            );
            throw $throwable;
        }

        return [
            'schema_version' => 1,
            'operation' => 'cleanup_generation',
            'mode' => 'apply',
            'status' => 'completed',
            'store_id' => $storeId,
            'generation_id' => $generationId,
            'cleanup_audit_id' => $auditId,
            'database_rows_deleted' => $preview['database']['delete'],
            'opensearch_index_deleted' => $preview['opensearch']['indices'][0],
            'preserved' => [
                'activation_audit' => $preview['database']['preserve'][0],
                'pipeline' => $preview['opensearch']['pipelines'][0],
                'encoder_artifact' => $preview['encoder_artifacts'][0],
            ],
        ];
    }

    private function buildPreview(
        AdapterInterface $connection,
        int $storeId,
        array $generation,
        ?array $knownState
    ): array {
        if ((int)$generation['store_id'] !== $storeId) {
            throw new \InvalidArgumentException('The generation belongs to a different store.');
        }
        $generationId = (int)$generation['generation_id'];
        $state = $knownState ?? $this->storeState($connection, $storeId);
        $protectionReasons = $this->protectionReasons($generation, $state);
        $outboxTable = $this->table(self::OUTBOX_TABLE);
        $outboxItemTable = $this->table(self::OUTBOX_ITEM_TABLE);
        $activationTable = $this->table(self::ACTIVATION_TABLE);
        $activationSelect = $connection->select()
            ->from($activationTable, [new Zend_Db_Expr('COUNT(*)')])
            ->where('generation_id = ?', $generationId)
            ->orWhere('prior_generation_id = ?', $generationId);
        $impact = [
            'schema_version' => 1,
            'operation' => 'cleanup_generation',
            'mode' => 'preview',
            'eligible' => $protectionReasons === [],
            'protection_reasons' => $protectionReasons,
            'generation' => [
                'generation_id' => $generationId,
                'store_id' => $storeId,
                'state' => (string)$generation['state'],
                'created_at' => (string)$generation['created_at'],
                'updated_at' => (string)$generation['updated_at'],
                'retention_days' => $this->config->retentionDays(),
            ],
            'database' => [
                'delete' => [
                    $this->rowImpact(self::OUTBOX_ITEM_TABLE, $this->outboxItemCount(
                        $connection,
                        $outboxItemTable,
                        $outboxTable,
                        $generationId
                    )),
                    $this->rowImpact(
                        self::OUTBOX_TABLE,
                        $this->generationRowCount($connection, self::OUTBOX_TABLE, $generationId)
                    ),
                    $this->rowImpact(
                        self::EMBEDDING_TABLE,
                        $this->generationRowCount($connection, self::EMBEDDING_TABLE, $generationId)
                    ),
                    $this->rowImpact(
                        self::PROGRESS_TABLE,
                        $this->generationRowCount($connection, self::PROGRESS_TABLE, $generationId)
                    ),
                    $this->rowImpact(self::GENERATION_TABLE, 1),
                ],
                'preserve' => [
                    [
                        'table' => $activationTable,
                        'rows' => (int)$connection->fetchOne($activationSelect),
                        'action' => 'preserve_audit',
                    ],
                    [
                        'table' => $this->table(self::CLEANUP_TABLE),
                        'rows_created_on_apply' => 1,
                        'action' => 'create_audit',
                    ],
                ],
            ],
            'opensearch' => [
                'indices' => [[
                    'name' => (string)$generation['physical_index'],
                    'exists' => $this->indexManager->indexExists((string)$generation['physical_index']),
                    'action' => 'delete',
                ]],
                'pipelines' => [[
                    'id' => (string)$generation['pipeline_id'],
                    'action' => 'preserve_shared_contract_resource',
                ]],
            ],
            'encoder_artifacts' => [[
                'endpoint' => (string)$generation['encoder_endpoint'],
                'identity_digest' => $generation['encoder_identity_digest'] === null
                    ? null
                    : (string)$generation['encoder_identity_digest'],
                'model_id' => (string)$generation['model_id'],
                'model_revision' => (string)$generation['model_revision'],
                'action' => 'retain_external_operator_managed',
            ]],
        ];
        $impact['confirmation_token'] = $impact['eligible']
            ? sprintf('cleanup-%d-%s', $generationId, hash('sha256', $this->json->serialize($impact)))
            : null;

        return $impact;
    }

    private function protectionReasons(array $generation, array $state): array
    {
        $generationId = (int)$generation['generation_id'];
        $generationState = (string)$generation['state'];
        if (($state['active_generation_id'] ?? null) !== null
            && (int)$state['active_generation_id'] === $generationId
        ) {
            return ['active_generation'];
        }
        if ($generationState === 'ACTIVE') {
            return ['active_generation'];
        }
        if (in_array($generationState, ['NEW', 'BUILDING', 'CATCHING_UP', 'READY'], true)) {
            return ['generation_state_not_cleanup_eligible'];
        }
        if ($generationState === 'RETAINED') {
            $retainedUntil = (new \DateTimeImmutable(
                (string)$generation['updated_at'],
                new \DateTimeZone('UTC')
            ))->modify(sprintf('+%d days', $this->config->retentionDays()));
            if ($retainedUntil > new \DateTimeImmutable('now', new \DateTimeZone('UTC'))) {
                return ['rollback_retention_window'];
            }

            return [];
        }
        if ($generationState !== 'FAILED') {
            return ['generation_state_not_cleanup_eligible'];
        }

        return [];
    }

    private function assertConfirmedAndEligible(array $preview, string $confirmationToken): void
    {
        if (!(bool)$preview['eligible']) {
            throw new \RuntimeException(sprintf(
                'Generation cleanup is protected: %s.',
                implode(', ', $preview['protection_reasons'])
            ));
        }
        $expected = $preview['confirmation_token'];
        if (!is_string($expected) || !hash_equals($expected, $confirmationToken)) {
            throw new \InvalidArgumentException('The cleanup confirmation does not match the current exact impact.');
        }
    }

    private function generation(AdapterInterface $connection, int $generationId): array
    {
        $generation = $connection->fetchRow(
            $connection->select()
                ->from($this->table(self::GENERATION_TABLE))
                ->where('generation_id = ?', $generationId)
        );
        if (!is_array($generation)) {
            throw new NoSuchEntityException(__('OpenSearch Hybrid generation %1 does not exist.', $generationId));
        }

        return $generation;
    }

    private function storeState(AdapterInterface $connection, int $storeId): array
    {
        $state = $connection->fetchRow(
            $connection->select()->from($this->table(self::STATE_TABLE))->where('store_id = ?', $storeId)
        );

        return is_array($state) ? $state : [];
    }

    private function generationRowCount(
        AdapterInterface $connection,
        string $table,
        int $generationId
    ): int {
        return (int)$connection->fetchOne(
            $connection->select()
                ->from($this->table($table), [new Zend_Db_Expr('COUNT(*)')])
                ->where('generation_id = ?', $generationId)
        );
    }

    private function outboxItemCount(
        AdapterInterface $connection,
        string $itemTable,
        string $outboxTable,
        int $generationId
    ): int {
        return (int)$connection->fetchOne(
            $connection->select()
                ->from(['item' => $itemTable], [new Zend_Db_Expr('COUNT(*)')])
                ->joinInner(['job' => $outboxTable], 'job.job_id = item.job_id', [])
                ->where('job.generation_id = ?', $generationId)
        );
    }

    private function rowImpact(string $table, int $rows): array
    {
        return [
            'table' => $this->resourceConnection->getTableName($table),
            'rows' => $rows,
            'action' => $table === self::GENERATION_TABLE ? 'delete' : 'delete_cascade',
        ];
    }

    private function table(string $table): string
    {
        return $this->resourceConnection->getTableName($table);
    }
}
