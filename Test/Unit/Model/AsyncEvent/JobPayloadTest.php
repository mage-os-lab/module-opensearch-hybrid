<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\AsyncEvent;

class JobPayloadTest extends \PHPUnit\Framework\TestCase
{
    public function testApiGettersDeclareWebApiReturnAnnotations(): void
    {
        $interface = new \ReflectionClass(
            \MageOS\OpenSearchHybrid\Api\Data\JobPayloadInterface::class
        );
        $expected = [
            'getJobId' => 'string',
            'getStoreId' => 'int',
            'getGenerationId' => 'int|null',
            'getLane' => 'string',
            'getOperation' => 'string',
            'getRevision' => 'int',
        ];

        foreach ($expected as $method => $returnType) {
            self::assertStringContainsString(
                '@return ' . $returnType,
                (string)$interface->getMethod($method)->getDocComment()
            );
        }
    }

    public function testReturnsATypedPayloadThatPreservesAsyncEventFieldNames(): void
    {
        $row = [
            'job_id' => 'job-1',
            'store_id' => 2,
            'generation_id' => 7,
            'lane' => 'EMBEDDING',
            'operation' => 'UPSERT',
            'revision' => 11,
        ];
        $repository = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository::class
        );
        $repository->expects($this->once())->method('get')->with('job-1')->willReturn($row);
        $factory = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\AsyncEvent\JobPayloadDataFactory::class
        );
        $data = new \MageOS\OpenSearchHybrid\Model\AsyncEvent\JobPayloadData($row);
        $factory->expects($this->once())
            ->method('create')
            ->with(['data' => $row])
            ->willReturn($data);
        $payload = new \MageOS\OpenSearchHybrid\Model\AsyncEvent\JobPayload($repository, $factory);

        $result = $payload->get('job-1');

        self::assertInstanceOf(
            \MageOS\OpenSearchHybrid\Api\Data\JobPayloadInterface::class,
            $result
        );
        self::assertSame('job-1', $result->getJobId());
        self::assertSame(2, $result->getStoreId());
        self::assertSame(7, $result->getGenerationId());
        self::assertSame('EMBEDDING', $result->getLane());
        self::assertSame('UPSERT', $result->getOperation());
        self::assertSame(11, $result->getRevision());
    }
}
