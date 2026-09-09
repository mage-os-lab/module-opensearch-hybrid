<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\AsyncEvent;

class HandoffNotifierTest extends \PHPUnit\Framework\TestCase
{
    public function testSuccessReturnsTheConcreteV4ResultWithoutRunningInference(): void
    {
        $publisher = $this->createMock(\MageOS\OpenSearchHybrid\Model\Queue\DataPlanePublisher::class);
        $publisher->expects($this->once())
            ->method('publish')
            ->with('job-1', 'EMBEDDING');
        $notifier = new \MageOS\OpenSearchHybrid\Model\AsyncEvent\HandoffNotifier(
            $publisher,
            new \Magento\Framework\Serialize\Serializer\Json()
        );
        $asyncEvent = $this->createStub(\MageOS\AsyncEvents\Api\Data\AsyncEventInterface::class);
        $asyncEvent->method('getSubscriptionId')->willReturn(42);
        $event = new \CloudEvents\V1\CloudEventImmutable(
            'event-1',
            'mageos://fixture',
            'mageos.opensearch_hybrid.embedding.batch.v1',
            ['job_id' => 'job-1', 'lane' => 'EMBEDDING']
        );

        $result = $notifier->notify($asyncEvent, $event);

        self::assertInstanceOf(\MageOS\AsyncEvents\Helper\NotifierResult::class, $result);
        self::assertTrue($result->getSuccess());
        self::assertFalse($result->getIsRetryable());
        self::assertSame(42, $result->getSubscriptionId());
        self::assertSame(['job_id' => 'job-1', 'lane' => 'EMBEDDING'], $result->getAsyncEventData());
    }

    public function testEveryHandoffFailureReturnsARetryableConcreteResult(): void
    {
        $publisher = $this->createMock(\MageOS\OpenSearchHybrid\Model\Queue\DataPlanePublisher::class);
        $publisher->expects($this->once())
            ->method('publish')
            ->willThrowException(new \RuntimeException('fixture failure'));
        $notifier = new \MageOS\OpenSearchHybrid\Model\AsyncEvent\HandoffNotifier(
            $publisher,
            new \Magento\Framework\Serialize\Serializer\Json()
        );
        $asyncEvent = $this->createStub(\MageOS\AsyncEvents\Api\Data\AsyncEventInterface::class);
        $asyncEvent->method('getSubscriptionId')->willReturn(43);
        $event = new \CloudEvents\V1\CloudEventImmutable(
            'event-2',
            'mageos://fixture',
            'mageos.opensearch_hybrid.embedding.batch.v1',
            ['job_id' => 'job-2', 'lane' => 'PRIORITY']
        );

        $result = $notifier->notify($asyncEvent, $event);

        self::assertInstanceOf(\MageOS\AsyncEvents\Helper\NotifierResult::class, $result);
        self::assertFalse($result->getSuccess());
        self::assertTrue($result->getIsRetryable());
        self::assertSame(5, $result->getRetryAfter());
        self::assertStringNotContainsString('fixture failure', $result->getResponseData());
    }
}
