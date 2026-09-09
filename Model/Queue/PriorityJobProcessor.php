<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Queue;

class PriorityJobProcessor
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository $outboxRepository,
        private readonly \MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository $generationRepository,
        private readonly \MageOS\OpenSearchHybrid\Api\EmbeddingStoreInterface $embeddingStore,
        private readonly \MageOS\OpenSearchHybrid\Model\Document\ProductDocumentFactory $documentFactory,
        private readonly \MageOS\OpenSearchHybrid\Model\OpenSearch\IndexManager $indexManager
    ) {
    }

    public function process(string $jobId): void
    {
        $job = $this->outboxRepository->get($jobId);
        if ((string)$job['lane'] !== 'PRIORITY') {
            throw new \InvalidArgumentException('The priority consumer received the wrong job lane.');
        }
        if ($job['generation_id'] === null) {
            throw new \InvalidArgumentException('The priority job has no target generation.');
        }
        $storeId = (int)$job['store_id'];
        $generationId = (int)$job['generation_id'];
        $generation = $this->generationRepository->get($generationId);
        $this->indexManager->ensureGeneration($generation);
        foreach ($this->outboxRepository->items($jobId) as $item) {
            if ((string)$item['state'] === 'SUPERSEDED') {
                continue;
            }
            $productId = (int)$item['product_id'];
            if ((string)$item['operation'] === 'DELETE') {
                $this->embeddingStore->delete(
                    $storeId,
                    $productId,
                    $generationId,
                    (int)$item['revision']
                );
                $this->indexManager->deleteDocument(
                    $generation,
                    $productId,
                    ((int)$item['revision'] * 2) + 2
                );
                continue;
            }
            $document = $this->documentFactory->create($productId, $storeId);
            $vector = (string)$item['operation'] === 'REFRESH'
                ? $this->embeddingStore->getVectorForSource(
                    $storeId,
                    $productId,
                    $generationId,
                    (string)$document['source_hash']
                )
                : null;
            $this->indexManager->indexDocument(
                $generation,
                $document,
                $vector,
                ((int)$item['revision'] * 2) + 1
            );
        }
    }
}
