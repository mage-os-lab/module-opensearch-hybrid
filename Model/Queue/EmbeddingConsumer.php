<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Queue;

class EmbeddingConsumer
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository $outboxRepository,
        private readonly \MageOS\OpenSearchHybrid\Model\Queue\EmbeddingJobProcessor $processor
    ) {
    }

    public function process(string $jobId): void
    {
        $owner = sprintf('embedding:%s:%d', gethostname() ?: 'unknown', getmypid() ?: 0);
        if (!$this->outboxRepository->claim($jobId, $owner)) {
            return;
        }
        try {
            $this->processor->process($jobId);
            $this->outboxRepository->complete($jobId);
        } catch (\Throwable $throwable) {
            if ($this->outboxRepository->retry($jobId, $throwable, 5)) {
                throw $throwable;
            }
        }
    }
}
