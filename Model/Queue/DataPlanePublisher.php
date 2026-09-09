<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Queue;

class DataPlanePublisher
{
    public function __construct(
        private readonly \Magento\Framework\MessageQueue\PublisherInterface $publisher
    ) {
    }

    public function publish(string $jobId, string $lane): void
    {
        $topic = $lane === 'PRIORITY' ? Topics::PRIORITY : Topics::EMBEDDING;
        $this->publisher->publish($topic, $jobId);
    }
}
