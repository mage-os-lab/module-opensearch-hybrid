<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\AsyncEvent;

class SubscriptionReconciler
{
    public const METADATA = 'mageos_opensearch_hybrid';
    public const RECIPIENT = 'mageos-opensearch-hybrid://handoff';

    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \Magento\Framework\Serialize\Serializer\Json $json
    ) {
    }

    public function reconcile(int $storeId): int
    {
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName('async_event_subscriber');
        $connection->beginTransaction();
        try {
            $rows = $this->rows($connection, $table, $storeId, true);
            $subscriptionId = $this->apply($connection, $table, $storeId, $rows);
            $connection->commit();

            return $subscriptionId;
        } catch (\Throwable $exception) {
            $connection->rollBack();
            throw $exception;
        }
    }

    public function preview(int $storeId): array
    {
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName('async_event_subscriber');
        $preview = $this->buildPreview($storeId, $this->rows($connection, $table, $storeId));
        $preview['confirmation_token'] = sprintf(
            'events-install-%d-%s',
            $storeId,
            hash('sha256', $this->json->serialize($preview))
        );

        return $preview;
    }

    public function reconcileConfirmed(int $storeId, string $confirmationToken): array
    {
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName('async_event_subscriber');
        $connection->beginTransaction();
        try {
            $rows = $this->rows($connection, $table, $storeId, true);
            $preview = $this->buildPreview($storeId, $rows);
            $expectedToken = sprintf(
                'events-install-%d-%s',
                $storeId,
                hash('sha256', $this->json->serialize($preview))
            );
            if (!hash_equals($expectedToken, $confirmationToken)) {
                throw new \InvalidArgumentException(
                    'The subscription reconciliation confirmation does not match the current exact preview.'
                );
            }
            $subscriptionId = $this->apply($connection, $table, $storeId, $rows);
            $connection->commit();

            return [
                'schema_version' => 1,
                'operation' => 'reconcile_async_event_subscription',
                'mode' => 'applied',
                'store_id' => $storeId,
                'subscription_id' => $subscriptionId,
                'change' => $preview['change'],
            ];
        } catch (\Throwable $exception) {
            $connection->rollBack();
            throw $exception;
        }
    }

    private function rows(
        \Magento\Framework\DB\Adapter\AdapterInterface $connection,
        string $table,
        int $storeId,
        bool $forUpdate = false
    ): array {
        $select = $connection->select()
            ->from($table)
            ->where('metadata = ?', self::METADATA)
            ->where('store_id = ?', $storeId)
            ->order('subscription_id ASC');
        if ($forUpdate) {
            $select->forUpdate(true);
        }

        return $connection->fetchAll($select);
    }

    private function buildPreview(int $storeId, array $rows): array
    {
        return [
            'schema_version' => 1,
            'operation' => 'reconcile_async_event_subscription',
            'mode' => 'preview',
            'mutation' => 'none',
            'store_id' => $storeId,
            'desired_subscription' => [
                'event_name' => LifecyclePublisher::EVENT_NAME,
                'recipient_url' => self::RECIPIENT,
                'status' => 1,
                'metadata' => self::METADATA,
                'store_id' => $storeId,
                'verification_token_on_create_sha256' => hash('sha256', ''),
                'existing_verification_token' => 'preserved',
            ],
            'current_subscriptions' => array_map(
                fn (array $row): array => $this->summarize($row),
                $rows
            ),
            'change' => $this->assess($storeId, $rows),
        ];
    }

    private function assess(int $storeId, array $rows): array
    {
        if ($rows === []) {
            return [
                'canonical_action' => 'create',
                'canonical_subscription_id' => null,
                'canonical_fields' => [
                    'event_name',
                    'recipient_url',
                    'verification_token',
                    'status',
                    'subscribed_at',
                    'metadata',
                    'store_id',
                ],
                'duplicate_subscription_ids' => [],
            ];
        }
        $canonical = $rows[0];
        $desired = $this->desiredMutableFields($storeId);
        $changedFields = [];
        foreach ($desired as $field => $value) {
            $current = in_array($field, ['status', 'store_id'], true)
                ? (int)($canonical[$field] ?? 0)
                : (string)($canonical[$field] ?? '');
            if ($current !== $value) {
                $changedFields[] = $field;
            }
        }
        $duplicateIds = [];
        foreach (array_slice($rows, 1) as $duplicate) {
            if ((int)($duplicate['status'] ?? 0) !== 0) {
                $duplicateIds[] = (int)$duplicate['subscription_id'];
            }
        }

        return [
            'canonical_action' => $changedFields === [] ? 'none' : 'repair',
            'canonical_subscription_id' => (int)$canonical['subscription_id'],
            'canonical_fields' => $changedFields,
            'duplicate_subscription_ids' => $duplicateIds,
        ];
    }

    private function summarize(array $row): array
    {
        return [
            'subscription_id' => (int)$row['subscription_id'],
            'event_name' => (string)($row['event_name'] ?? ''),
            'recipient_url' => (string)($row['recipient_url'] ?? ''),
            'verification_token_sha256' => hash('sha256', (string)($row['verification_token'] ?? '')),
            'status' => (int)($row['status'] ?? 0),
            'subscribed_at' => isset($row['subscribed_at']) ? (string)$row['subscribed_at'] : null,
            'metadata' => (string)($row['metadata'] ?? ''),
            'store_id' => (int)($row['store_id'] ?? 0),
        ];
    }

    private function apply(
        \Magento\Framework\DB\Adapter\AdapterInterface $connection,
        string $table,
        int $storeId,
        array $rows
    ): int {
        if ($rows === []) {
            $connection->insert($table, [
                'event_name' => LifecyclePublisher::EVENT_NAME,
                'recipient_url' => self::RECIPIENT,
                'verification_token' => '',
                'status' => 1,
                'subscribed_at' => new \Zend_Db_Expr('UTC_TIMESTAMP()'),
                'metadata' => self::METADATA,
                'store_id' => $storeId,
            ]);
            return (int)$connection->lastInsertId($table);
        }
        $canonicalId = (int)$rows[0]['subscription_id'];
        $change = $this->assess($storeId, $rows);
        if ($change['canonical_fields'] !== []) {
            $desired = $this->desiredMutableFields($storeId);
            $updates = array_intersect_key($desired, array_flip($change['canonical_fields']));
            $connection->update($table, $updates, ['subscription_id = ?' => $canonicalId]);
        }
        $duplicateIds = $change['duplicate_subscription_ids'];
        if ($duplicateIds !== []) {
            $connection->update($table, ['status' => 0], ['subscription_id IN (?)' => $duplicateIds]);
        }

        return $canonicalId;
    }

    private function desiredMutableFields(int $storeId): array
    {
        return [
            'event_name' => LifecyclePublisher::EVENT_NAME,
            'recipient_url' => self::RECIPIENT,
            'status' => 1,
            'metadata' => self::METADATA,
            'store_id' => $storeId,
        ];
    }
}
