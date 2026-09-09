<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\AsyncEvent;

class SubscriptionReconcilerTest extends \PHPUnit\Framework\TestCase
{
    public function testPreviewReportsExactCreateWithoutWriting(): void
    {
        $select = $this->select();
        $connection = $this->createMock(\Magento\Framework\DB\Adapter\AdapterInterface::class);
        $connection->expects($this->once())->method('select')->willReturn($select);
        $connection->expects($this->once())->method('fetchAll')->with($select)->willReturn([]);
        $connection->expects($this->never())->method('insert');
        $connection->expects($this->never())->method('update');
        $json = $this->createMock(\Magento\Framework\Serialize\Serializer\Json::class);
        $json->expects($this->once())
            ->method('serialize')
            ->with(self::callback(static fn (array $preview): bool =>
                ($preview['change']['canonical_action'] ?? null) === 'create'
                && ($preview['change']['duplicate_subscription_ids'] ?? null) === []
                && !array_key_exists('confirmation_token', $preview)
            ))
            ->willReturn('canonical-subscription-create-preview');

        $preview = $this->reconciler($connection, $json)->preview(3);

        self::assertSame('preview', $preview['mode']);
        self::assertSame('create', $preview['change']['canonical_action']);
        self::assertSame([], $preview['current_subscriptions']);
        self::assertSame(
            'events-install-3-' . hash('sha256', 'canonical-subscription-create-preview'),
            $preview['confirmation_token']
        );
    }

    public function testPreviewRecognizesOwnedEventNameDriftAndDoesNotExposeVerificationToken(): void
    {
        $whereClauses = [];
        $select = $this->createStub(\Magento\Framework\DB\Select::class);
        $select->method('from')->willReturnSelf();
        $select->method('where')->willReturnCallback(
            static function (string $condition) use ($select, &$whereClauses): \Magento\Framework\DB\Select {
                $whereClauses[] = $condition;
                return $select;
            }
        );
        $select->method('order')->willReturnSelf();
        $connection = $this->createMock(\Magento\Framework\DB\Adapter\AdapterInterface::class);
        $connection->expects($this->once())->method('select')->willReturn($select);
        $connection->expects($this->once())->method('fetchAll')->with($select)->willReturn([[
            'subscription_id' => '8',
            'event_name' => 'drifted.event',
            'recipient_url' => 'https://drifted.example.test',
            'verification_token' => 'secret-token',
            'status' => '0',
            'metadata' => \MageOS\OpenSearchHybrid\Model\AsyncEvent\SubscriptionReconciler::METADATA,
            'store_id' => '3',
        ]]);
        $json = $this->createStub(\Magento\Framework\Serialize\Serializer\Json::class);
        $json->method('serialize')->willReturn('canonical-subscription-repair-preview');

        $preview = $this->reconciler($connection, $json)->preview(3);

        self::assertSame(['metadata = ?', 'store_id = ?'], $whereClauses);
        self::assertSame('repair', $preview['change']['canonical_action']);
        self::assertSame(['event_name', 'recipient_url', 'status'], $preview['change']['canonical_fields']);
        self::assertSame(hash('sha256', 'secret-token'), $preview['current_subscriptions'][0]['verification_token_sha256']);
        self::assertArrayNotHasKey('verification_token', $preview['current_subscriptions'][0]);
    }

    public function testConfirmedReconciliationRejectsAStalePreviewWithoutWriting(): void
    {
        $select = $this->select();
        $connection = $this->createMock(\Magento\Framework\DB\Adapter\AdapterInterface::class);
        $connection->expects($this->once())->method('beginTransaction');
        $connection->expects($this->once())->method('select')->willReturn($select);
        $connection->expects($this->once())->method('fetchAll')->with($select)->willReturn([]);
        $connection->expects($this->never())->method('insert');
        $connection->expects($this->never())->method('update');
        $connection->expects($this->never())->method('commit');
        $connection->expects($this->once())->method('rollBack');
        $json = $this->createStub(\Magento\Framework\Serialize\Serializer\Json::class);
        $json->method('serialize')->willReturn('current-subscription-preview');

        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('confirmation does not match');

        $this->reconciler($connection, $json)->reconcileConfirmed(3, 'stale-token');
    }

    public function testConfirmedReconciliationRepairsTheLockedPreviewState(): void
    {
        $select = $this->createMock(\Magento\Framework\DB\Select::class);
        $select->method('from')->willReturnSelf();
        $select->method('where')->willReturnSelf();
        $select->method('order')->willReturnSelf();
        $select->expects($this->once())->method('forUpdate')->with(true)->willReturnSelf();
        $connection = $this->createMock(\Magento\Framework\DB\Adapter\AdapterInterface::class);
        $connection->expects($this->once())->method('beginTransaction');
        $connection->expects($this->once())->method('select')->willReturn($select);
        $connection->expects($this->once())->method('fetchAll')->with($select)->willReturn([
            [
                'subscription_id' => '8',
                'event_name' => 'drifted.event',
                'recipient_url' => \MageOS\OpenSearchHybrid\Model\AsyncEvent\SubscriptionReconciler::RECIPIENT,
                'verification_token' => '',
                'status' => '1',
                'metadata' => \MageOS\OpenSearchHybrid\Model\AsyncEvent\SubscriptionReconciler::METADATA,
                'store_id' => '3',
            ],
            [
                'subscription_id' => '9',
                'event_name' => \MageOS\OpenSearchHybrid\Model\AsyncEvent\LifecyclePublisher::EVENT_NAME,
                'recipient_url' => \MageOS\OpenSearchHybrid\Model\AsyncEvent\SubscriptionReconciler::RECIPIENT,
                'verification_token' => '',
                'status' => '1',
                'metadata' => \MageOS\OpenSearchHybrid\Model\AsyncEvent\SubscriptionReconciler::METADATA,
                'store_id' => '3',
            ],
        ]);
        $connection->expects($this->exactly(2))
            ->method('update')
            ->willReturnCallback(static function (string $table, array $data, array $where): int {
                self::assertSame('async_event_subscriber', $table);
                if ($where === ['subscription_id = ?' => 8]) {
                    self::assertSame([
                        'event_name' => \MageOS\OpenSearchHybrid\Model\AsyncEvent\LifecyclePublisher::EVENT_NAME,
                    ], $data);
                    return 1;
                }
                self::assertSame(['subscription_id IN (?)' => [9]], $where);
                self::assertSame(['status' => 0], $data);
                return 1;
            });
        $connection->expects($this->once())->method('commit');
        $connection->expects($this->never())->method('rollBack');
        $json = $this->createStub(\Magento\Framework\Serialize\Serializer\Json::class);
        $json->method('serialize')->willReturn('locked-subscription-preview');

        $report = $this->reconciler($connection, $json)->reconcileConfirmed(
            3,
            'events-install-3-' . hash('sha256', 'locked-subscription-preview')
        );

        self::assertSame('applied', $report['mode']);
        self::assertSame(8, $report['subscription_id']);
        self::assertSame(['event_name'], $report['change']['canonical_fields']);
        self::assertSame([9], $report['change']['duplicate_subscription_ids']);
        self::assertArrayNotHasKey('confirmation_token', $report);
    }

    private function select(): \Magento\Framework\DB\Select
    {
        $select = $this->createStub(\Magento\Framework\DB\Select::class);
        $select->method('from')->willReturnSelf();
        $select->method('where')->willReturnSelf();
        $select->method('order')->willReturnSelf();

        return $select;
    }

    private function reconciler(
        \Magento\Framework\DB\Adapter\AdapterInterface $connection,
        \Magento\Framework\Serialize\Serializer\Json $json
    ): \MageOS\OpenSearchHybrid\Model\AsyncEvent\SubscriptionReconciler {
        $resourceConnection = $this->createStub(\Magento\Framework\App\ResourceConnection::class);
        $resourceConnection->method('getConnection')->willReturn($connection);
        $resourceConnection->method('getTableName')->willReturnArgument(0);

        return new \MageOS\OpenSearchHybrid\Model\AsyncEvent\SubscriptionReconciler(
            $resourceConnection,
            $json
        );
    }
}
