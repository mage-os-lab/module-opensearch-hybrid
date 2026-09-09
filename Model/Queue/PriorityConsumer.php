<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Queue;

class PriorityConsumer
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository $outboxRepository,
        private readonly \MageOS\OpenSearchHybrid\Model\Queue\PriorityJobProcessor $processor,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\ChangeJournalRepository $journalRepository
    ) {
    }

    public function process(string $jobId): void
    {
        $owner = sprintf('priority:%s:%d', gethostname() ?: 'unknown', getmypid() ?: 0);
        if (!$this->outboxRepository->claim($jobId, $owner, 30)) {
            $job = $this->outboxRepository->get($jobId);
            if ((string)$job['state'] === 'COMPLETED') {
                $this->journalRepository->acknowledgeHybridJob($job);
            }
            return;
        }
        try {
            $this->processor->process($jobId);
            $this->outboxRepository->complete($jobId);
            $this->journalRepository->acknowledgeHybridJob($this->outboxRepository->get($jobId));
        } catch (\Throwable $throwable) {
            if ($this->outboxRepository->retry($jobId, $throwable, 2)) {
                throw $throwable;
            }
        }
    }
}
