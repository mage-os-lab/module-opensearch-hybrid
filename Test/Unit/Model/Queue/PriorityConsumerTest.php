<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Queue;

class PriorityConsumerTest extends \PHPUnit\Framework\TestCase
{
    public function testRetryableFailureIsPersistedWithoutAdvancingTheJournal(): void
    {
        $failure = new \RuntimeException('retryable fixture failure');
        $repository = $this->createMock(\MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository::class);
        $repository->expects($this->once())->method('claim')->willReturn(true);
        $repository->expects($this->once())
            ->method('retry')
            ->with('job-1', $failure, 2)
            ->willReturn(false);
        $processor = $this->createMock(\MageOS\OpenSearchHybrid\Model\Queue\PriorityJobProcessor::class);
        $processor->expects($this->once())->method('process')->with('job-1')->willThrowException($failure);
        $journal = $this->createMock(\MageOS\OpenSearchHybrid\Model\Change\ChangeJournalRepository::class);
        $journal->expects($this->never())->method('acknowledgeHybridJob');

        $consumer = new \MageOS\OpenSearchHybrid\Model\Queue\PriorityConsumer(
            $repository,
            $processor,
            $journal
        );
        $consumer->process('job-1');
    }

    public function testTerminalFailureEscapesSoRabbitMqDeadLettersTheDelivery(): void
    {
        $failure = new \RuntimeException('terminal fixture failure');
        $repository = $this->createMock(\MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository::class);
        $repository->expects($this->once())->method('claim')->willReturn(true);
        $repository->expects($this->once())->method('retry')->willReturn(true);
        $processor = $this->createMock(\MageOS\OpenSearchHybrid\Model\Queue\PriorityJobProcessor::class);
        $processor->expects($this->once())->method('process')->willThrowException($failure);
        $journal = $this->createStub(\MageOS\OpenSearchHybrid\Model\Change\ChangeJournalRepository::class);

        $this->expectExceptionObject($failure);
        $consumer = new \MageOS\OpenSearchHybrid\Model\Queue\PriorityConsumer(
            $repository,
            $processor,
            $journal
        );
        $consumer->process('job-2');
    }
}
