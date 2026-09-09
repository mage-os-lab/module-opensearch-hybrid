<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Plugin;

class NativeIndexerAcknowledgement
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Change\ChangeJournalRepository $journalRepository,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\CaptureService $captureService,
        private readonly \Magento\CatalogSearch\Model\ResourceModel\Fulltext $fulltextResource
    ) {
    }

    public function aroundExecuteByDimensions(
        \Magento\CatalogSearch\Model\Indexer\Fulltext $subject,
        \Closure $proceed,
        array $dimensions,
        ?\Traversable $entityIds = null
    ) {
        $dimensionName = \Magento\Store\Model\StoreDimensionProvider::DIMENSION_NAME;
        if (!isset($dimensions[$dimensionName])) {
            return $proceed($dimensions, $entityIds);
        }
        $storeId = (int)$dimensions[$dimensionName]->getValue();
        $productIds = null;
        $forwardIds = $entityIds;
        if ($entityIds !== null) {
            $productIds = array_values(array_unique(array_map(
                'intval',
                iterator_to_array($entityIds, false)
            )));
            $productIds = array_values(array_unique(array_merge(
                $productIds,
                array_map('intval', $this->fulltextResource->getRelationsByChild($productIds))
            )));
            $forwardIds = new \ArrayIterator($productIds);
            $capture = $this->captureService->captureNativeProducts(
                $storeId,
                $productIds,
                'native_partial_reindex'
            );
            $this->captureService->publishJobs($capture['jobs']);
        }
        $boundary = $this->journalRepository->pendingNativeBoundary($storeId, $productIds);
        $result = $proceed($dimensions, $forwardIds);
        $this->journalRepository->acknowledgeNative($storeId, $productIds, $boundary);

        return $result;
    }
}
