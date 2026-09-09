<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\AsyncEvent;

use MageOS\OpenSearchHybrid\Api\Data\JobPayloadInterface as JobPayloadDataInterface;

class JobPayload implements \MageOS\OpenSearchHybrid\Api\JobPayloadInterface
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository $outboxRepository,
        private readonly \MageOS\OpenSearchHybrid\Model\AsyncEvent\JobPayloadDataFactory $jobPayloadDataFactory
    ) {
    }

    public function get(string $jobId): JobPayloadDataInterface
    {
        $job = $this->outboxRepository->get($jobId);

        return $this->jobPayloadDataFactory->create(['data' => [
            JobPayloadDataInterface::JOB_ID => (string)$job['job_id'],
            JobPayloadDataInterface::STORE_ID => (int)$job['store_id'],
            JobPayloadDataInterface::GENERATION_ID => $job['generation_id'] === null
                ? null
                : (int)$job['generation_id'],
            JobPayloadDataInterface::LANE => (string)$job['lane'],
            JobPayloadDataInterface::OPERATION => (string)$job['operation'],
            JobPayloadDataInterface::REVISION => (int)$job['revision'],
        ]]);
    }
}
