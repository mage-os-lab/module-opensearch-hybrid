<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Plugin;

class SourceStatusTopologyCapture
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Change\InventoryTopologyInvalidationService $invalidationService
    ) {
    }

    public function aroundSave(
        \Magento\InventoryApi\Api\SourceRepositoryInterface $subject,
        \Closure $proceed,
        \Magento\InventoryApi\Api\Data\SourceInterface $source
    ) {
        unset($subject);
        $sourceCode = (string)$source->getSourceCode();
        $changed = $this->invalidationService->sourceStatusChanged(
            $sourceCode,
            (bool)$source->isEnabled()
        );
        $result = $proceed($source);
        if ($changed) {
            $this->invalidationService->invalidateSource(
                $sourceCode,
                'inventory_source_status_change'
            );
        }

        return $result;
    }
}
