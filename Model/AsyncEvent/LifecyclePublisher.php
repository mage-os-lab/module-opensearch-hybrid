<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\AsyncEvent;

class LifecyclePublisher
{
    public const EVENT_NAME = 'mageos.opensearch_hybrid.embedding.batch.v1';

    public function __construct(
        private readonly \MageOS\AsyncEvents\Api\AsyncEventPublisherInterface $asyncEventPublisher,
        private readonly \MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository $outboxRepository
    ) {
    }

    public function publish(string $jobId): void
    {
        $job = $this->outboxRepository->get($jobId);
        $this->asyncEventPublisher->publish(
            self::EVENT_NAME,
            ['jobId' => $jobId],
            (int)$job['store_id']
        );
        $this->outboxRepository->markPublished($jobId);
    }
}
