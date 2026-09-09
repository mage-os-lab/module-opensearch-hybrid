<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Api;

use MageOS\OpenSearchHybrid\Api\Data\JobPayloadInterface as JobPayloadDataInterface;

interface JobPayloadInterface
{
    /**
     * Return the bounded identity payload resolved by Mage-OS Async Events.
     *
     * @return \MageOS\OpenSearchHybrid\Api\Data\JobPayloadInterface
     */
    public function get(string $jobId): JobPayloadDataInterface;
}
