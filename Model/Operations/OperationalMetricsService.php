<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Operations;

use MageOS\OpenSearchHybrid\Model\AsyncEvent\LifecyclePublisher;
use MageOS\OpenSearchHybrid\Model\AsyncEvent\SubscriptionReconciler;
use Magento\Framework\DB\Adapter\AdapterInterface;
use Zend_Db_Expr;

class OperationalMetricsService
{
    private const WORK_AGE_ALERT_SECONDS = 300;
    private const TRACE_RETENTION_MAX_DAYS = 90;
    private const TERMINAL_JOB_STATES = ['COMPLETED', 'SUPERSEDED'];
    private const TRACE_CLEANUP_ENABLED = 'system/async_events/subscriber_log_cleanup_cron';
    private const TRACE_RETENTION_DAYS = 'system/async_events/subscriber_log_cron_delete_period';

    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \Magento\Framework\App\Config\ScopeConfigInterface $scopeConfig,
        private readonly \MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract $retrievalContract
    ) {
    }

    public function get(?int $storeId = null): array
    {
        $connection = $this->resourceConnection->getConnection();
        $alerts = [];
        $readiness = $this->readiness($connection, $storeId, $alerts);
        $activeGenerationIds = [];
        foreach ($readiness as $state) {
            if ($state['activation_mode'] === 'HYBRID' && is_int($state['active_generation_id'])) {
                $activeGenerationIds[$state['store_id']] = $state['active_generation_id'];
            }
        }
        $journalHeads = $this->journalHeads($connection, $storeId);
        $generations = $this->generations(
            $connection,
            $storeId,
            $journalHeads,
            $activeGenerationIds,
            $alerts
        );
        $work = $this->work($connection, $storeId, $alerts);
        $subscriptions = $this->subscriptions($connection, $storeId, $alerts);
        $traceCleanup = $this->traceCleanup($subscriptions, $alerts);

        return [
            'schema_version' => 1,
            'generated_at' => gmdate(\DateTimeInterface::ATOM),
            'scope' => ['store_id' => $storeId],
            'thresholds' => [
                'oldest_unresolved_work_seconds' => self::WORK_AGE_ALERT_SECONDS,
                'maximum_async_event_trace_retention_days' => self::TRACE_RETENTION_MAX_DAYS,
            ],
            'readiness' => $readiness,
            'generations' => $generations,
            'work' => $work,
            'subscriptions' => $subscriptions,
            'async_event_trace_cleanup' => $traceCleanup,
            'activation_events' => $this->eventCounts(
                $connection,
                'mageos_opensearch_hybrid_activation',
                ['store_id', 'mode', 'status'],
                $storeId
            ),
            'cleanup_events' => $this->eventCounts(
                $connection,
                'mageos_opensearch_hybrid_cleanup',
                ['store_id', 'status'],
                $storeId
            ),
            'alerts' => $alerts,
            'external_metrics_required' => [
                'collection and alerting for emitted query route counts and fallback rate',
                'collection and alerting for emitted query encoder, hybrid OpenSearch, and end-to-end latency',
                'encoder errors, timeouts, and circuit-open duration',
                'RabbitMQ queue depth, oldest-message age, redeliveries, and consumer concurrency',
                'document encoding throughput and source-hash mismatch count',
                'reconciliation repair count',
            ],
        ];
    }

    private function readiness(AdapterInterface $connection, ?int $storeId, array &$alerts): array
    {
        $table = $this->table('mageos_opensearch_hybrid_store_state');
        $select = $connection->select()->from($table)->order('store_id ASC');
        if ($storeId !== null) {
            $select->where('store_id = ?', $storeId);
        }
        $metrics = [];
        foreach ($connection->fetchAll($select) as $state) {
            $required = (int)$state['required_watermark'];
            $native = (int)$state['native_watermark'];
            $hybrid = (int)$state['hybrid_watermark'];
            $ready = (string)$state['activation_mode'] === 'HYBRID'
                && $state['active_generation_id'] !== null
                && (bool)$state['readiness_latch']
                && $native >= $required
                && $hybrid >= $required;
            $metric = [
                'store_id' => (int)$state['store_id'],
                'activation_mode' => (string)$state['activation_mode'],
                'active_generation_id' => $state['active_generation_id'] === null
                    ? null
                    : (int)$state['active_generation_id'],
                'ready' => $ready,
                'required_watermark' => $required,
                'native_watermark' => $native,
                'hybrid_watermark' => $hybrid,
                'native_lag' => max(0, $required - $native),
                'hybrid_lag' => max(0, $required - $hybrid),
                'state_age_seconds' => $this->ageSeconds((string)$state['updated_at']),
                'cache_version' => (int)$state['cache_version'],
            ];
            $metrics[] = $metric;
            if ((string)$state['activation_mode'] === 'HYBRID' && !$ready) {
                $alerts[] = $this->alert(
                    'critical',
                    'active_generation_not_ready',
                    (int)$state['store_id'],
                    $metric['active_generation_id'],
                    sprintf('Native lag %d; hybrid lag %d.', $metric['native_lag'], $metric['hybrid_lag'])
                );
            }
        }

        return $metrics;
    }

    private function journalHeads(AdapterInterface $connection, ?int $storeId): array
    {
        $select = $connection->select()
            ->from(
                $this->table('mageos_opensearch_hybrid_change_journal'),
                ['store_id', 'head' => new Zend_Db_Expr('MAX(change_id)')]
            )
            ->group('store_id');
        if ($storeId !== null) {
            $select->where('store_id = ?', $storeId);
        }
        $heads = [];
        foreach ($connection->fetchAll($select) as $row) {
            $heads[(int)$row['store_id']] = (int)$row['head'];
        }

        return $heads;
    }

    private function generations(
        AdapterInterface $connection,
        ?int $storeId,
        array $journalHeads,
        array $activeGenerationIds,
        array &$alerts
    ): array {
        $select = $connection->select()
            ->from(['generation' => $this->table('mageos_opensearch_hybrid_generation')])
            ->joinLeft(
                ['progress' => $this->table('mageos_opensearch_hybrid_generation_progress')],
                'progress.generation_id = generation.generation_id',
                [
                    'seeding_state',
                    'build_refresh_interval',
                    'normal_refresh_interval',
                    'refresh_state',
                ]
            )
            ->order('generation.generation_id DESC');
        if ($storeId !== null) {
            $select->where('generation.store_id = ?', $storeId);
        }
        $metrics = [];
        foreach ($connection->fetchAll($select) as $generation) {
            $total = (int)$generation['coverage_total'];
            $complete = (int)$generation['coverage_complete'];
            $failed = (int)$generation['coverage_failed'];
            $generationStoreId = (int)$generation['store_id'];
            $generationId = (int)$generation['generation_id'];
            $activeForStore = ($activeGenerationIds[$generationStoreId] ?? null) === $generationId;
            $contractCurrent = $this->contractCurrent($generation);
            $metric = [
                'generation_id' => $generationId,
                'store_id' => $generationStoreId,
                'state' => (string)$generation['state'],
                'active_for_store' => $activeForStore,
                'coverage_total' => $total,
                'coverage_complete' => $complete,
                'coverage_failed' => $failed,
                'coverage_percent' => $total === 0 ? 100.0 : round(($complete / $total) * 100, 4),
                'journal_head' => $journalHeads[$generationStoreId] ?? 0,
                'captured_change_id' => (int)$generation['captured_change_id'],
                'journal_lag' => max(
                    0,
                    ($journalHeads[$generationStoreId] ?? 0) - (int)$generation['captured_change_id']
                ),
                'model_id' => (string)$generation['model_id'],
                'model_revision' => (string)$generation['model_revision'],
                'encoder_identity_digest' => $generation['encoder_identity_digest'] === null
                    ? null
                    : (string)$generation['encoder_identity_digest'],
                'contract_digest' => (string)$generation['contract_digest'],
                'result_contract_digest' => (string)$generation['result_contract_digest'],
                'mapping_digest' => (string)$generation['mapping_digest'],
                'pipeline_digest' => (string)$generation['pipeline_digest'],
                'contract_current' => $contractCurrent,
                'physical_index' => (string)$generation['physical_index'],
                'pipeline_id' => (string)$generation['pipeline_id'],
                'seeding_state' => $generation['seeding_state'] === null
                    ? null
                    : (string)$generation['seeding_state'],
                'build_refresh_interval' => $generation['build_refresh_interval'] === null
                    ? null
                    : (string)$generation['build_refresh_interval'],
                'normal_refresh_interval' => $generation['normal_refresh_interval'] === null
                    ? null
                    : (string)$generation['normal_refresh_interval'],
                'refresh_state' => $generation['refresh_state'] === null
                    ? null
                    : (string)$generation['refresh_state'],
            ];
            $metrics[] = $metric;
            if ($activeForStore && ($complete !== $total || $failed !== 0)) {
                $alerts[] = $this->alert(
                    'critical',
                    'active_generation_coverage_gap',
                    $generationStoreId,
                    $generationId,
                    sprintf('Coverage %d of %d with %d failures.', $complete, $total, $failed)
                );
            }
            if ($activeForStore
                && !(bool)$generation['is_fake']
                && $generation['encoder_identity_digest'] === null
            ) {
                $alerts[] = $this->alert(
                    'critical',
                    'active_encoder_identity_missing',
                    $generationStoreId,
                    $generationId,
                    'The production generation has no sealed encoder identity digest.'
                );
            }
            if ($activeForStore && !$contractCurrent) {
                $alerts[] = $this->alert(
                    'critical',
                    'active_generation_contract_drift',
                    $generationStoreId,
                    $generationId,
                    'The active generation does not match the current retrieval contract.'
                );
            }
            if (in_array(
                (string)($generation['refresh_state'] ?? ''),
                ['SUPPRESS_FAILED', 'RESTORE_FAILED'],
                true
            )) {
                $alerts[] = $this->alert(
                    'high',
                    'generation_refresh_state_failed',
                    $generationStoreId,
                    $generationId,
                    sprintf(
                        'Inactive build refresh transition is %s.',
                        (string)$generation['refresh_state']
                    )
                );
            }
            if ((string)$generation['state'] === 'READY'
                && (string)($generation['refresh_state'] ?? '') !== 'RESTORED'
            ) {
                $alerts[] = $this->alert(
                    'critical',
                    'ready_generation_refresh_not_restored',
                    $generationStoreId,
                    $generationId,
                    'The ready generation has not restored its normal refresh interval.'
                );
            }
        }

        return $metrics;
    }

    private function contractCurrent(array $generation): bool
    {
        return hash_equals((string)$generation['contract_digest'], $this->retrievalContract->digest())
            && hash_equals(
                (string)$generation['result_contract_digest'],
                $this->retrievalContract->resultContractDigest()
            )
            && hash_equals((string)$generation['mapping_digest'], $this->retrievalContract->mappingDigest())
            && hash_equals((string)$generation['pipeline_digest'], $this->retrievalContract->pipelineDigest());
    }

    private function work(AdapterInterface $connection, ?int $storeId, array &$alerts): array
    {
        $select = $connection->select()
            ->from($this->table('mageos_opensearch_hybrid_outbox'), [
                'store_id',
                'lane',
                'state',
                'jobs' => new Zend_Db_Expr('COUNT(*)'),
                'attempts' => new Zend_Db_Expr('COALESCE(SUM(attempt_count), 0)'),
                'replays' => new Zend_Db_Expr('COALESCE(SUM(replay_count), 0)'),
                'oldest_created_at' => new Zend_Db_Expr('MIN(created_at)'),
            ])
            ->group(['store_id', 'lane', 'state'])
            ->order(['store_id ASC', 'lane ASC', 'state ASC']);
        if ($storeId !== null) {
            $select->where('store_id = ?', $storeId);
        }
        $metrics = [];
        foreach ($connection->fetchAll($select) as $row) {
            $age = $this->ageSeconds((string)$row['oldest_created_at']);
            $metric = [
                'store_id' => (int)$row['store_id'],
                'lane' => (string)$row['lane'],
                'state' => (string)$row['state'],
                'jobs' => (int)$row['jobs'],
                'attempts' => (int)$row['attempts'],
                'manual_replays' => (int)$row['replays'],
                'oldest_age_seconds' => $age,
            ];
            $metrics[] = $metric;
            if (!in_array((string)$row['state'], self::TERMINAL_JOB_STATES, true)
                && $age !== null
                && $age > self::WORK_AGE_ALERT_SECONDS
            ) {
                $alerts[] = $this->alert(
                    'high',
                    'oldest_unresolved_work_exceeded',
                    (int)$row['store_id'],
                    null,
                    sprintf(
                        '%s %s work is %d seconds old; threshold is %d.',
                        (string)$row['lane'],
                        (string)$row['state'],
                        $age,
                        self::WORK_AGE_ALERT_SECONDS
                    )
                );
            }
            if ((string)$row['state'] === 'DEAD' && (int)$row['jobs'] > 0) {
                $alerts[] = $this->alert(
                    'high',
                    'dead_letter_nonzero',
                    (int)$row['store_id'],
                    null,
                    sprintf('%d %s job(s) require explicit replay.', (int)$row['jobs'], (string)$row['lane'])
                );
            }
        }

        return $metrics;
    }

    private function subscriptions(AdapterInterface $connection, ?int $storeId, array &$alerts): array
    {
        $table = $this->table('async_event_subscriber');
        $select = $connection->select()
            ->from($table)
            ->where('metadata = ?', SubscriptionReconciler::METADATA)
            ->order(['store_id ASC', 'subscription_id ASC']);
        if ($storeId !== null) {
            $select->where('store_id = ?', $storeId);
        }
        $metrics = [];
        foreach ($connection->fetchAll($select) as $subscription) {
            $healthy = (int)$subscription['status'] === 1
                && (string)$subscription['event_name'] === LifecyclePublisher::EVENT_NAME
                && (string)$subscription['recipient_url'] === SubscriptionReconciler::RECIPIENT;
            $trace = $this->trace($connection, (int)$subscription['subscription_id']);
            $metric = [
                'subscription_id' => (int)$subscription['subscription_id'],
                'store_id' => (int)$subscription['store_id'],
                'healthy' => $healthy,
                'enabled' => (int)$subscription['status'] === 1,
                'event_name' => (string)$subscription['event_name'],
                'recipient_url' => (string)$subscription['recipient_url'],
                'trace_count' => $trace['trace_count'],
                'trace_success_count' => $trace['success_count'],
                'trace_failure_count' => $trace['failure_count'],
                'trace_unresolved_failure_count' => $trace['unresolved_failure_count'],
                'oldest_trace_age_seconds' => $trace['oldest_age_seconds'],
                'last_trace_at' => $trace['last_trace_at'],
            ];
            $metrics[] = $metric;
            if (!$healthy) {
                $alerts[] = $this->alert(
                    'high',
                    'subscription_definition_drift',
                    (int)$subscription['store_id'],
                    null,
                    sprintf(
                        'Subscription %d differs from its module-owned definition.',
                        (int)$subscription['subscription_id']
                    )
                );
            }
            if ($trace['unresolved_failure_count'] > 0) {
                $alerts[] = $this->alert(
                    'high',
                    'async_event_handoff_failure',
                    (int)$subscription['store_id'],
                    null,
                    sprintf(
                        'Subscription %d has %d unresolved failed trace(s); '
                            . '%d failed attempt(s) remain in retention.',
                        (int)$subscription['subscription_id'],
                        $trace['unresolved_failure_count'],
                        $trace['failure_count']
                    )
                );
            }
        }
        if ($metrics === []) {
            $alerts[] = $this->alert(
                'high',
                'subscription_missing',
                $storeId,
                null,
                'No module-owned Mage-OS Async Events subscription exists in this scope.'
            );
        }

        return $metrics;
    }

    private function trace(AdapterInterface $connection, int $subscriptionId): array
    {
        $table = $this->table('async_event_subscriber_log');
        $row = $connection->fetchRow(
            $connection->select()
                ->from($table, [
                    'trace_count' => new Zend_Db_Expr('COUNT(*)'),
                    'success_count' => new Zend_Db_Expr('COALESCE(SUM(IF(success = 1, 1, 0)), 0)'),
                    'failure_count' => new Zend_Db_Expr('COALESCE(SUM(IF(success = 0, 1, 0)), 0)'),
                    'oldest_trace_at' => new Zend_Db_Expr('MIN(created)'),
                    'last_trace_at' => new Zend_Db_Expr('MAX(created)'),
                ])
                ->where('subscription_id = ?', $subscriptionId)
        );
        $row = is_array($row) ? $row : [];
        $unresolvedFailureCount = (int)$connection->fetchOne(
            $connection->select()
                ->from(['failed' => $table], [
                    'unresolved_failure_count' => new Zend_Db_Expr('COUNT(*)'),
                ])
                ->joinLeft(
                    ['later' => $table],
                    'later.subscription_id = failed.subscription_id'
                        . ' AND later.uuid = failed.uuid'
                        . ' AND later.log_id > failed.log_id',
                    []
                )
                ->where('failed.subscription_id = ?', $subscriptionId)
                ->where('failed.success = ?', 0)
                ->where('later.log_id IS NULL')
        );

        return [
            'trace_count' => (int)($row['trace_count'] ?? 0),
            'success_count' => (int)($row['success_count'] ?? 0),
            'failure_count' => (int)($row['failure_count'] ?? 0),
            'unresolved_failure_count' => $unresolvedFailureCount,
            'oldest_age_seconds' => $this->ageSeconds($row['oldest_trace_at'] ?? null),
            'last_trace_at' => $row['last_trace_at'] ?? null,
        ];
    }

    private function traceCleanup(array $subscriptions, array &$alerts): array
    {
        $enabled = $this->scopeConfig->isSetFlag(self::TRACE_CLEANUP_ENABLED);
        $retentionDays = (int)$this->scopeConfig->getValue(self::TRACE_RETENTION_DAYS);
        $bounded = $enabled && $retentionDays > 0 && $retentionDays <= self::TRACE_RETENTION_MAX_DAYS;
        if (!$enabled) {
            $alerts[] = $this->alert(
                'medium',
                'async_event_trace_cleanup_disabled',
                null,
                null,
                'Mage-OS Async Events subscriber-log cleanup is disabled.'
            );
        } elseif (!$bounded) {
            $alerts[] = $this->alert(
                'medium',
                'async_event_trace_retention_unbounded',
                null,
                null,
                sprintf(
                    'Subscriber-log retention is %d days; the module maximum is %d.',
                    $retentionDays,
                    self::TRACE_RETENTION_MAX_DAYS
                )
            );
        }
        $oldestAge = null;
        foreach ($subscriptions as $subscription) {
            $age = $subscription['oldest_trace_age_seconds'];
            if (is_int($age) && ($oldestAge === null || $age > $oldestAge)) {
                $oldestAge = $age;
            }
        }

        return [
            'enabled' => $enabled,
            'retention_days' => $retentionDays,
            'bounded' => $bounded,
            'oldest_module_trace_age_seconds' => $oldestAge,
        ];
    }

    private function eventCounts(
        AdapterInterface $connection,
        string $table,
        array $dimensions,
        ?int $storeId
    ): array {
        $columns = $dimensions;
        $columns['events'] = new Zend_Db_Expr('COUNT(*)');
        $select = $connection->select()
            ->from($this->table($table), $columns)
            ->group($dimensions)
            ->order(array_map(static fn (string $dimension): string => $dimension . ' ASC', $dimensions));
        if ($storeId !== null) {
            $select->where('store_id = ?', $storeId);
        }
        $metrics = [];
        foreach ($connection->fetchAll($select) as $row) {
            foreach ($dimensions as $dimension) {
                if (str_ends_with($dimension, '_id')) {
                    $row[$dimension] = (int)$row[$dimension];
                } else {
                    $row[$dimension] = (string)$row[$dimension];
                }
            }
            $row['events'] = (int)$row['events'];
            $metrics[] = $row;
        }

        return $metrics;
    }

    private function alert(
        string $severity,
        string $code,
        ?int $storeId,
        ?int $generationId,
        string $detail
    ): array {
        return [
            'severity' => $severity,
            'code' => $code,
            'store_id' => $storeId,
            'generation_id' => $generationId,
            'detail' => $detail,
        ];
    }

    private function ageSeconds(mixed $date): ?int
    {
        if (!is_string($date) || $date === '') {
            return null;
        }
        $timestamp = strtotime($date . ' UTC');

        return $timestamp === false ? null : max(0, time() - $timestamp);
    }

    private function table(string $table): string
    {
        return $this->resourceConnection->getTableName($table);
    }
}
