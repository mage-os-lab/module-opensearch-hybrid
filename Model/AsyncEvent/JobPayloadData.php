<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\AsyncEvent;

use MageOS\OpenSearchHybrid\Api\Data\JobPayloadInterface;
use Magento\Framework\Api\AbstractSimpleObject;

class JobPayloadData extends AbstractSimpleObject implements JobPayloadInterface
{
    public function getJobId(): string
    {
        return (string)$this->_get(self::JOB_ID);
    }

    public function getStoreId(): int
    {
        return (int)$this->_get(self::STORE_ID);
    }

    public function getGenerationId(): ?int
    {
        $generationId = $this->_get(self::GENERATION_ID);

        return $generationId === null ? null : (int)$generationId;
    }

    public function getLane(): string
    {
        return (string)$this->_get(self::LANE);
    }

    public function getOperation(): string
    {
        return (string)$this->_get(self::OPERATION);
    }

    public function getRevision(): int
    {
        return (int)$this->_get(self::REVISION);
    }
}
