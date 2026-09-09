<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Plugin;

class ReservationSalabilityChangeCaptureTest extends \PHPUnit\Framework\TestCase
{
    public function testCapturesTheReservationSkuSetForOnlyTheAffectedStockStores(): void
    {
        $resolver = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Change\InventoryProductResolver::class
        );
        $resolver->expects($this->once())
            ->method('resolveByStore')
            ->with(['child'], 5)
            ->willReturn([1 => [10, 20]]);
        $captureService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Change\CaptureService::class
        );
        $captureService->expects($this->once())
            ->method('capturePriorityProductsByStore')
            ->with([1 => [10, 20]], 'inventory_reservation_change')
            ->willReturn(['changes' => [['change_id' => 9]], 'jobs' => [['job_id' => 'job-9']]]);
        $captureService->expects($this->once())
            ->method('publishJobs')
            ->with([['job_id' => 'job-9']]);
        $plugin = new \MageOS\OpenSearchHybrid\Plugin\ReservationSalabilityChangeCapture(
            $resolver,
            $captureService
        );
        $reservation = new \Magento\InventoryIndexer\Model\Queue\ReservationData(['child'], 5);

        self::assertSame(
            ['child' => false],
            $plugin->afterExecute(
                $this->createStub(\Magento\InventoryIndexer\Model\Queue\UpdateIndexSalabilityStatus::class),
                ['child' => false],
                $reservation
            )
        );
    }
}
