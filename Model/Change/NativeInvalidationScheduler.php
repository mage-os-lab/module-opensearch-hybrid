<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Change;

class NativeInvalidationScheduler
{
    public function __construct(
        private readonly \Magento\CatalogSearch\Model\Indexer\Fulltext\Processor $processor
    ) {
    }

    public function schedule(array $productIds): void
    {
        $productIds = array_values(array_unique(array_filter(
            array_map('intval', $productIds),
            static fn (int $productId): bool => $productId > 0
        )));
        if ($productIds === []) {
            return;
        }
        if ($this->processor->isIndexerScheduled()) {
            $this->processor->getIndexer()->getView()->getChangelog()->addList($productIds);
            return;
        }
        $this->processor->reindexList($productIds);
    }
}
