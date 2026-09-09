<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Api\Data;

interface JobPayloadInterface
{
    public const JOB_ID = 'job_id';
    public const STORE_ID = 'store_id';
    public const GENERATION_ID = 'generation_id';
    public const LANE = 'lane';
    public const OPERATION = 'operation';
    public const REVISION = 'revision';

    /**
     * @return string
     */
    public function getJobId(): string;

    /**
     * @return int
     */
    public function getStoreId(): int;

    /**
     * @return int|null
     */
    public function getGenerationId(): ?int;

    /**
     * @return string
     */
    public function getLane(): string;

    /**
     * @return string
     */
    public function getOperation(): string;

    /**
     * @return int
     */
    public function getRevision(): int;
}
