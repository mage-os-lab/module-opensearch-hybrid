<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Queue;

class EmbeddingConsumerTest extends \PHPUnit\Framework\TestCase
{
    public function testRetryableFailureIsPersistedWithoutDeadLetteringTheDelivery(): void
    {
        $failure = new \RuntimeException('retryable fixture failure');
        $repository = $this->createMock(\MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository::class);
        $repository->expects($this->once())->method('claim')->willReturn(true);
        $repository->expects($this->once())
            ->method('retry')
            ->with('job-1', $failure, 5)
            ->willReturn(false);
        $processor = $this->createMock(\MageOS\OpenSearchHybrid\Model\Queue\EmbeddingJobProcessor::class);
        $processor->expects($this->once())->method('process')->with('job-1')->willThrowException($failure);

        $consumer = new \MageOS\OpenSearchHybrid\Model\Queue\EmbeddingConsumer($repository, $processor);
        $consumer->process('job-1');
    }

    public function testTerminalFailureEscapesSoRabbitMqDeadLettersTheDelivery(): void
    {
        $failure = new \RuntimeException('terminal fixture failure');
        $repository = $this->createMock(\MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository::class);
        $repository->expects($this->once())->method('claim')->willReturn(true);
        $repository->expects($this->once())->method('retry')->willReturn(true);
        $processor = $this->createMock(\MageOS\OpenSearchHybrid\Model\Queue\EmbeddingJobProcessor::class);
        $processor->expects($this->once())->method('process')->willThrowException($failure);

        $this->expectExceptionObject($failure);
        (new \MageOS\OpenSearchHybrid\Model\Queue\EmbeddingConsumer($repository, $processor))->process('job-2');
    }
}
