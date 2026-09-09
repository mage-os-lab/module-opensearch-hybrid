<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Queue;

class EmbeddingJobProcessor
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository $outboxRepository,
        private readonly \MageOS\OpenSearchHybrid\Model\Document\ProductDocumentFactory $documentFactory,
        private readonly \MageOS\OpenSearchHybrid\Model\Vector\DeterministicVector $deterministicVector,
        private readonly \MageOS\OpenSearchHybrid\Api\DocumentEncoderInterface $documentEncoder,
        private readonly \MageOS\OpenSearchHybrid\Api\EmbeddingStoreInterface $embeddingStore,
        private readonly \MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository $generationRepository,
        private readonly \MageOS\OpenSearchHybrid\Model\OpenSearch\IndexManager $indexManager
    ) {
    }

    public function process(string $jobId): void
    {
        $job = $this->outboxRepository->get($jobId);
        if ((string)$job['lane'] !== 'EMBEDDING' || $job['generation_id'] === null) {
            throw new \InvalidArgumentException('The embedding consumer received the wrong job lane.');
        }
        $storeId = (int)$job['store_id'];
        $generationId = (int)$job['generation_id'];
        $generation = $this->generationRepository->get($generationId);
        $this->indexManager->ensureGeneration($generation);
        $items = array_values(array_filter(
            $this->outboxRepository->items($jobId),
            static fn (array $item): bool => (string)$item['state'] !== 'SUPERSEDED'
        ));
        if ($items === []) {
            return;
        }
        $documents = array_map(
            fn (array $item): array => $this->documentFactory->create((int)$item['product_id'], $storeId),
            $items
        );
        $vectors = (bool)$generation['is_fake']
            ? array_map(
                fn (array $document): array => $this->deterministicVector->encode(
                    (string)$document['rendered']
                ),
                $documents
            )
            : $this->documentEncoder->encode(
                array_map(static fn (array $document): string => (string)$document['rendered'], $documents),
                $generation
            );
        $indexRecords = [];
        foreach ($items as $index => $item) {
            $document = $documents[$index];
            $vector = $vectors[$index];
            $accepted = $this->embeddingStore->save(
                $storeId,
                (int)$item['product_id'],
                $generationId,
                (string)$document['source_hash'],
                $vector,
                (int)$item['revision']
            );
            if ($accepted) {
                $indexRecords[] = [
                    'document' => $document,
                    'vector' => $vector,
                    'revision' => ((int)$item['revision'] * 2) + 2,
                ];
            }
        }
        $this->indexManager->indexDocuments($generation, $indexRecords);
    }
}
