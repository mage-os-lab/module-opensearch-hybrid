<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Change;

class InventoryCaptureContext
{
    private int $sourceItemsSaveDepth = 0;

    public function enterSourceItemsSave(): void
    {
        $this->sourceItemsSaveDepth++;
    }

    public function leaveSourceItemsSave(): void
    {
        $this->sourceItemsSaveDepth = max(0, $this->sourceItemsSaveDepth - 1);
    }

    public function isSourceItemsSave(): bool
    {
        return $this->sourceItemsSaveDepth > 0;
    }
}
