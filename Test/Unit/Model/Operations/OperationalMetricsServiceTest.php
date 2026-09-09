<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Operations;

class OperationalMetricsServiceTest extends \PHPUnit\Framework\TestCase
{
    public function testRecoveredTraceFailuresRemainVisibleWithoutBeingUnresolved(): void
    {
        $summarySelect = $this->select();
        $terminalSelect = $this->select();
        $connection = $this->createMock(\Magento\Framework\DB\Adapter\AdapterInterface::class);
        $connection->expects($this->exactly(2))
            ->method('select')
            ->willReturnOnConsecutiveCalls($summarySelect, $terminalSelect);
        $connection->expects($this->once())
            ->method('fetchRow')
            ->with($summarySelect)
            ->willReturn([
                'trace_count' => '3',
                'success_count' => '2',
                'failure_count' => '1',
                'oldest_trace_at' => null,
                'last_trace_at' => null,
            ]);
        $connection->expects($this->once())
            ->method('fetchOne')
            ->with($terminalSelect)
            ->willReturn('0');

        $result = $this->trace($this->service(), $connection, 7);

        self::assertSame(3, $result['trace_count']);
        self::assertSame(2, $result['success_count']);
        self::assertSame(1, $result['failure_count']);
        self::assertSame(0, $result['unresolved_failure_count']);
    }

    private function select(): \Magento\Framework\DB\Select
    {
        $select = $this->createStub(\Magento\Framework\DB\Select::class);
        $select->method('from')->willReturnSelf();
        $select->method('joinLeft')->willReturnSelf();
        $select->method('where')->willReturnSelf();

        return $select;
    }

    private function service(): \MageOS\OpenSearchHybrid\Model\Operations\OperationalMetricsService
    {
        $resourceConnection = $this->createStub(\Magento\Framework\App\ResourceConnection::class);
        $resourceConnection->method('getTableName')->willReturnArgument(0);

        return new \MageOS\OpenSearchHybrid\Model\Operations\OperationalMetricsService(
            $resourceConnection,
            $this->createStub(\Magento\Framework\App\Config\ScopeConfigInterface::class),
            $this->createStub(\MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract::class)
        );
    }

    private function trace(
        \MageOS\OpenSearchHybrid\Model\Operations\OperationalMetricsService $service,
        \Magento\Framework\DB\Adapter\AdapterInterface $connection,
        int $subscriptionId
    ): array {
        $method = new \ReflectionMethod(
            \MageOS\OpenSearchHybrid\Model\Operations\OperationalMetricsService::class,
            'trace'
        );

        return $method->invoke($service, $connection, $subscriptionId);
    }
}
