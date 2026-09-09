<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Outbox;

class OutboxRepository
{
    private const TABLE = 'mageos_opensearch_hybrid_outbox';
    private const ITEM_TABLE = 'mageos_opensearch_hybrid_outbox_item';

    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \Magento\Framework\DataObject\IdentityGeneratorInterface $identityGenerator,
        private readonly \MageOS\OpenSearchHybrid\Model\Config $config
    ) {
    }

    public function create(
        int $storeId,
        ?int $generationId,
        string $lane,
        string $operation,
        array $productIds,
        int $revision,
        int $earliestChangeId = 0,
        int $latestChangeId = 0
    ): string {
        if (!in_array($lane, ['EMBEDDING', 'PRIORITY'], true)) {
            throw new \InvalidArgumentException('The outbox lane is invalid.');
        }
        $productIds = array_values(array_unique(array_map('intval', $productIds)));
        if ($productIds === []) {
            throw new \InvalidArgumentException('An outbox job must contain at least one product.');
        }
        $jobId = $this->identityGenerator->generateId();
        $connection = $this->resourceConnection->getConnection();
        $outboxTable = $this->resourceConnection->getTableName(self::TABLE);
        $itemTable = $this->resourceConnection->getTableName(self::ITEM_TABLE);
        $connection->beginTransaction();
        try {
            $connection->insert($outboxTable, [
                'job_id' => $jobId,
                'store_id' => $storeId,
                'generation_id' => $generationId,
                'lane' => $lane,
                'operation' => $operation,
                'state' => 'PENDING',
                'revision' => $revision,
                'earliest_change_id' => $earliestChangeId,
                'latest_change_id' => $latestChangeId,
            ]);
            $rows = [];
            foreach ($productIds as $productId) {
                $rows[] = [
                    'job_id' => $jobId,
                    'product_id' => $productId,
                    'operation' => $operation,
                    'revision' => $revision,
                    'state' => 'PENDING',
                ];
            }
            $connection->insertMultiple($itemTable, $rows);
            $connection->commit();
        } catch (\Throwable $throwable) {
            $connection->rollBack();
            throw $throwable;
        }

        return $jobId;
    }

    public function get(string $jobId): array
    {
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName(self::TABLE);
        $job = $connection->fetchRow(
            $connection->select()->from($table)->where('job_id = ?', $jobId)
        );
        if (!is_array($job)) {
            throw new \Magento\Framework\Exception\NoSuchEntityException(__('Outbox job %1 does not exist.', $jobId));
        }

        return $job;
    }

    public function items(string $jobId): array
    {
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName(self::ITEM_TABLE);

        return $connection->fetchAll(
            $connection->select()->from($table)->where('job_id = ?', $jobId)->order('item_id ASC')
        );
    }

    public function countOutstandingForGeneration(int $generationId, string $lane): int
    {
        if (!in_array($lane, ['EMBEDDING', 'PRIORITY'], true)) {
            throw new \InvalidArgumentException('The outbox lane is invalid.');
        }
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName(self::TABLE);

        return (int)$connection->fetchOne(
            $connection->select()
                ->from($table, [new \Zend_Db_Expr('COUNT(*)')])
                ->where('generation_id = ?', $generationId)
                ->where('lane = ?', $lane)
                ->where('state NOT IN (?)', ['COMPLETED', 'SUPERSEDED'])
        );
    }

    public function markPublished(string $jobId): void
    {
        $this->resourceConnection->getConnection()->update(
            $this->resourceConnection->getTableName(self::TABLE),
            [
                'state' => 'PUBLISHED',
                'publish_count' => new \Zend_Db_Expr('publish_count + 1'),
            ],
            ['job_id = ?' => $jobId, 'state IN (?)' => ['PENDING', 'RETRY', 'TIMED_OUT']]
        );
    }

    public function claim(string $jobId, string $owner, int $claimSeconds = 300): bool
    {
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName(self::TABLE);
        $affected = $connection->update(
            $table,
            [
                'state' => 'CLAIMED',
                'claim_owner' => substr($owner, 0, 128),
                'claim_expires_at' => new \Zend_Db_Expr(sprintf(
                    'DATE_ADD(UTC_TIMESTAMP(), INTERVAL %d SECOND)',
                    max(1, $claimSeconds)
                )),
                'attempt_count' => new \Zend_Db_Expr('attempt_count + 1'),
            ],
            [
                'job_id = ?' => $jobId,
                'state IN (?)' => ['PUBLISHED', 'RETRY', 'TIMED_OUT'],
                '(next_eligible_at IS NULL OR next_eligible_at <= UTC_TIMESTAMP())',
            ]
        );

        return $affected === 1;
    }

    public function complete(string $jobId): void
    {
        $connection = $this->resourceConnection->getConnection();
        $connection->beginTransaction();
        try {
            $connection->update(
                $this->resourceConnection->getTableName(self::ITEM_TABLE),
                ['state' => 'COMPLETED'],
                ['job_id = ?' => $jobId, 'state != ?' => 'SUPERSEDED']
            );
            $connection->update(
                $this->resourceConnection->getTableName(self::TABLE),
                [
                    'state' => 'COMPLETED',
                    'claim_owner' => null,
                    'claim_expires_at' => null,
                    'completed_at' => new \Zend_Db_Expr('UTC_TIMESTAMP()'),
                ],
                ['job_id = ?' => $jobId, 'state = ?' => 'CLAIMED']
            );
            $connection->commit();
        } catch (\Throwable $throwable) {
            $connection->rollBack();
            throw $throwable;
        }
    }

    public function retry(string $jobId, \Throwable $throwable, int $delaySeconds): bool
    {
        $job = $this->get($jobId);
        if ((string)$job['state'] !== 'CLAIMED') {
            return false;
        }
        $attemptCount = (int)$job['attempt_count'];
        $terminal = $attemptCount >= $this->config->jobMaxAttempts();
        $connection = $this->resourceConnection->getConnection();
        $affected = $connection->update(
            $this->resourceConnection->getTableName(self::TABLE),
            [
                'state' => $terminal ? 'DEAD' : 'RETRY',
                'claim_owner' => null,
                'claim_expires_at' => null,
                'next_eligible_at' => $terminal
                    ? null
                    : new \Zend_Db_Expr(sprintf(
                        'DATE_ADD(UTC_TIMESTAMP(), INTERVAL %d SECOND)',
                        max(1, $delaySeconds)
                    )),
                'last_error_class' => substr($throwable::class, 0, 255),
                'last_diagnostic' => substr('processing failed', 0, 1024),
            ],
            [
                'job_id = ?' => $jobId,
                'state = ?' => 'CLAIMED',
                'attempt_count = ?' => $attemptCount,
            ]
        );
        if ($affected !== 1) {
            throw new \RuntimeException('The outbox job changed while failure handling was in progress.');
        }

        return $terminal;
    }

    public function prepareManualRetry(string $jobId): array
    {
        $job = $this->get($jobId);
        if (!in_array((string)$job['state'], ['PUBLISHED', 'RETRY', 'TIMED_OUT', 'DEAD'], true)) {
            throw new \RuntimeException('Only published or failed outbox jobs can be retried manually.');
        }
        $connection = $this->resourceConnection->getConnection();
        $affected = $connection->update(
            $this->resourceConnection->getTableName(self::TABLE),
            [
                'state' => 'PENDING',
                'claim_owner' => null,
                'claim_expires_at' => null,
                'next_eligible_at' => null,
                'last_error_class' => null,
                'last_diagnostic' => null,
                'attempt_count' => 0,
                'replay_count' => new \Zend_Db_Expr('replay_count + 1'),
            ],
            [
                'job_id = ?' => $jobId,
                'state IN (?)' => ['PUBLISHED', 'RETRY', 'TIMED_OUT', 'DEAD'],
            ]
        );
        if ($affected !== 1) {
            throw new \RuntimeException('The outbox job changed while retry was being prepared.');
        }

        return $this->get($jobId);
    }

    public function supersedeProductJobs(
        int $storeId,
        int $generationId,
        string $lane,
        int $productId,
        int $revision
    ): int {
        if (!in_array($lane, ['EMBEDDING', 'PRIORITY'], true)) {
            throw new \InvalidArgumentException('The outbox lane is invalid.');
        }
        $connection = $this->resourceConnection->getConnection();
        $outboxTable = $this->resourceConnection->getTableName(self::TABLE);
        $itemTable = $this->resourceConnection->getTableName(self::ITEM_TABLE);
        $baseSelect = $connection->select()
            ->from(['job' => $outboxTable], [])
            ->joinInner(['item' => $itemTable], 'item.job_id = job.job_id', [])
            ->where('job.store_id = ?', $storeId)
            ->where('job.generation_id = ?', $generationId)
            ->where('job.lane = ?', $lane)
            ->where('job.state IN (?)', [
                'PENDING',
                'PUBLISHED',
                'CLAIMED',
                'RETRY',
                'TIMED_OUT',
                'DEAD',
            ])
            ->where('item.product_id = ?', $productId)
            ->where('item.revision < ?', $revision)
            ->where('item.state NOT IN (?)', ['COMPLETED', 'SUPERSEDED']);
        $boundarySelect = clone $baseSelect;
        $boundarySelect->columns([
            'earliest_change_id' => new \Zend_Db_Expr('MIN(job.earliest_change_id)'),
        ]);
        $earliestChangeId = (int)$connection->fetchOne($boundarySelect);
        $jobSelect = clone $baseSelect;
        $jobSelect->columns(['job_id' => 'job.job_id'])->distinct();
        $jobIds = array_map('strval', $connection->fetchCol($jobSelect));
        if ($jobIds === []) {
            return 0;
        }
        $connection->update(
            $itemTable,
            ['state' => 'SUPERSEDED'],
            [
                'job_id IN (?)' => $jobIds,
                'product_id = ?' => $productId,
                'revision < ?' => $revision,
                'state NOT IN (?)' => ['COMPLETED', 'SUPERSEDED'],
            ]
        );
        foreach ($jobIds as $jobId) {
            $remainingItems = (int)$connection->fetchOne(
                $connection->select()
                    ->from($itemTable, [new \Zend_Db_Expr('COUNT(*)')])
                    ->where('job_id = ?', $jobId)
                    ->where('state NOT IN (?)', ['COMPLETED', 'SUPERSEDED'])
            );
            if ($remainingItems === 0) {
                $connection->update(
                    $outboxTable,
                    [
                        'state' => 'SUPERSEDED',
                        'claim_owner' => null,
                        'claim_expires_at' => null,
                        'next_eligible_at' => null,
                    ],
                    [
                        'job_id = ?' => $jobId,
                        'state NOT IN (?)' => ['COMPLETED', 'SUPERSEDED'],
                    ]
                );
            }
        }

        return $earliestChangeId;
    }

    public function supersedeGenerationJobs(int $generationId): void
    {
        $connection = $this->resourceConnection->getConnection();
        $outboxTable = $this->resourceConnection->getTableName(self::TABLE);
        $itemTable = $this->resourceConnection->getTableName(self::ITEM_TABLE);
        $jobIds = array_map('strval', $connection->fetchCol(
            $connection->select()
                ->from($outboxTable, ['job_id'])
                ->where('generation_id = ?', $generationId)
                ->where('state NOT IN (?)', ['COMPLETED', 'SUPERSEDED'])
        ));
        if ($jobIds === []) {
            return;
        }
        $connection->update(
            $itemTable,
            ['state' => 'SUPERSEDED'],
            [
                'job_id IN (?)' => $jobIds,
                'state NOT IN (?)' => ['COMPLETED', 'SUPERSEDED'],
            ]
        );
        $connection->update(
            $outboxTable,
            [
                'state' => 'SUPERSEDED',
                'claim_owner' => null,
                'claim_expires_at' => null,
                'next_eligible_at' => null,
            ],
            [
                'job_id IN (?)' => $jobIds,
                'state NOT IN (?)' => ['COMPLETED', 'SUPERSEDED'],
            ]
        );
    }
}
