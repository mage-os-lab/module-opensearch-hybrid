<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Plugin;

class AsyncEventMutationGuardTest extends \PHPUnit\Framework\TestCase
{
    public function testSaveCannotEvadeOwnershipGuardByClearingMetadata(): void
    {
        $guard = $this->guardWithPersistedMetadata(
            8,
            \MageOS\OpenSearchHybrid\Model\AsyncEvent\SubscriptionReconciler::METADATA
        );
        $subject = $this->createStub(\MageOS\AsyncEvents\Api\AsyncEventRepositoryInterface::class);
        $event = $this->createStub(\MageOS\AsyncEvents\Api\Data\AsyncEventInterface::class);
        $event->method('getSubscriptionId')->willReturn(8);
        $event->method('getMetadata')->willReturn('cleared-by-caller');

        $this->expectException(\Magento\Framework\Exception\LocalizedException::class);
        $this->expectExceptionMessage('cannot be edited directly');

        $guard->beforeSave($subject, $event);
    }

    public function testDeleteCannotEvadeOwnershipGuardByClearingMetadata(): void
    {
        $guard = $this->guardWithPersistedMetadata(
            8,
            \MageOS\OpenSearchHybrid\Model\AsyncEvent\SubscriptionReconciler::METADATA
        );
        $subject = $this->createStub(\MageOS\AsyncEvents\Model\ResourceModel\AsyncEvent::class);
        $event = $this->createStub(\Magento\Framework\Model\AbstractModel::class);
        $event->method('getData')->willReturnMap([
            ['metadata', null, 'cleared-by-caller'],
            ['subscription_id', null, 8],
        ]);

        $this->expectException(\Magento\Framework\Exception\LocalizedException::class);
        $this->expectExceptionMessage('cannot be deleted directly');

        $guard->beforeDelete($subject, $event);
    }

    private function guardWithPersistedMetadata(
        int $subscriptionId,
        string $metadata
    ): \MageOS\OpenSearchHybrid\Plugin\AsyncEventMutationGuard {
        $select = $this->createMock(\Magento\Framework\DB\Select::class);
        $select->method('from')->willReturnSelf();
        $select->expects($this->once())
            ->method('where')
            ->with('subscription_id = ?', $subscriptionId)
            ->willReturnSelf();
        $connection = $this->createMock(\Magento\Framework\DB\Adapter\AdapterInterface::class);
        $connection->expects($this->once())->method('select')->willReturn($select);
        $connection->expects($this->once())
            ->method('fetchOne')
            ->with($select)
            ->willReturn($metadata);
        $resourceConnection = $this->createStub(\Magento\Framework\App\ResourceConnection::class);
        $resourceConnection->method('getConnection')->willReturn($connection);
        $resourceConnection->method('getTableName')->willReturnArgument(0);

        return new \MageOS\OpenSearchHybrid\Plugin\AsyncEventMutationGuard($resourceConnection);
    }
}
