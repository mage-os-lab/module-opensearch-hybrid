<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Cron;

class ReconcileStaleJobs
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Outbox\ReconciliationService $reconciliationService,
        private readonly \Psr\Log\LoggerInterface $logger
    ) {
    }

    public function execute(): void
    {
        try {
            $this->reconciliationService->reconcile();
        } catch (\Throwable $throwable) {
            $this->logger->error('OpenSearch Hybrid reconciliation failed.', [
                'error_class' => $throwable::class,
            ]);
        }
    }
}
