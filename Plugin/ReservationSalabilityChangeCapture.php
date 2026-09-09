<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Plugin;

class ReservationSalabilityChangeCapture
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Change\InventoryProductResolver $productResolver,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\CaptureService $captureService
    ) {
    }

    public function afterExecute(
        \Magento\InventoryIndexer\Model\Queue\UpdateIndexSalabilityStatus $subject,
        array $result,
        \Magento\InventoryIndexer\Model\Queue\ReservationData $reservationData
    ): array {
        unset($subject);
        $capture = $this->captureService->capturePriorityProductsByStore(
            $this->productResolver->resolveByStore(
                $reservationData->getSkus(),
                $reservationData->getStock()
            ),
            'inventory_reservation_change'
        );
        $this->captureService->publishJobs($capture['jobs']);

        return $result;
    }
}
