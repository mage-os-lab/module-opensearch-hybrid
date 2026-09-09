<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Encoder;

class HttpDocumentEncoderTest extends \PHPUnit\Framework\TestCase
{
    public function testSplitsAnOutboxBatchByTheConfiguredRuntimeBatchSize(): void
    {
        $config = $this->createStub(\MageOS\OpenSearchHybrid\Model\Config::class);
        $config->method('validateEncoderEndpoint')->willReturn('http://10.0.0.20:8080');
        $config->method('encoderToken')->willReturn('');
        $config->method('encoderConnectTimeoutMs')->willReturn(100);
        $config->method('documentTimeoutSeconds')->willReturn(30);
        $config->method('batchSize')->willReturn(2);
        $calls = [];
        $client = $this->createStub(\MageOS\OpenSearchHybrid\Model\Http\CurlJsonClient::class);
        $client->method('post')->willReturnCallback(
            function (string $url, array $payload) use (&$calls): array {
                $calls[] = $payload['documents'];

                return $this->response(
                    $payload['request_id'],
                    true,
                    array_map(
                        static fn (string $document): array => array_fill(0, 256, 0.0625),
                        $payload['documents']
                    )
                );
            }
        );
        $encoder = new \MageOS\OpenSearchHybrid\Model\Encoder\HttpDocumentEncoder(
            $config,
            $client,
            new \MageOS\OpenSearchHybrid\Model\Vector\VectorValidator()
        );

        $vectors = $encoder->encode(['one', 'two', 'three'], $this->generation());

        self::assertSame([['one', 'two'], ['three']], $calls);
        self::assertCount(3, $vectors);
    }

    public function testBatchesDocumentsAgainstTheGenerationBoundIdentity(): void
    {
        $config = $this->createStub(\MageOS\OpenSearchHybrid\Model\Config::class);
        $config->method('validateEncoderEndpoint')->willReturn('http://10.0.0.20:8080');
        $config->method('encoderToken')->willReturn('fixture-token');
        $config->method('encoderConnectTimeoutMs')->willReturn(100);
        $config->method('documentTimeoutSeconds')->willReturn(30);
        $config->method('batchSize')->willReturn(16);
        $client = $this->createMock(\MageOS\OpenSearchHybrid\Model\Http\CurlJsonClient::class);
        $client->expects($this->once())
            ->method('post')
            ->with(
                'http://10.0.0.20:8080/v1/embed/documents',
                $this->callback(static fn (array $payload): bool => $payload['documents'] === ['one', 'two']
                    && $payload['encoder_identity_digest'] === str_repeat('b', 64)),
                100,
                30000,
                ['Authorization' => 'Bearer fixture-token']
            )
            ->willReturnCallback(function (string $url, array $payload): array {
                return $this->response($payload['request_id'], true, [
                    array_fill(0, 256, 0.0625),
                    array_fill(0, 256, 0.0625),
                ]);
            });
        $encoder = new \MageOS\OpenSearchHybrid\Model\Encoder\HttpDocumentEncoder(
            $config,
            $client,
            new \MageOS\OpenSearchHybrid\Model\Vector\VectorValidator()
        );

        $vectors = $encoder->encode(['one', 'two'], $this->generation());

        self::assertCount(2, $vectors);
        self::assertCount(256, $vectors[0]);
    }

    public function testRejectsDevelopmentOnlyOrWrongSizedResponses(): void
    {
        $config = $this->createStub(\MageOS\OpenSearchHybrid\Model\Config::class);
        $config->method('validateEncoderEndpoint')->willReturn('http://10.0.0.20:8080');
        $config->method('encoderToken')->willReturn('');
        $config->method('encoderConnectTimeoutMs')->willReturn(100);
        $config->method('documentTimeoutSeconds')->willReturn(30);
        $config->method('batchSize')->willReturn(16);
        $client = $this->createStub(\MageOS\OpenSearchHybrid\Model\Http\CurlJsonClient::class);
        $client->method('post')->willReturnCallback(function (string $url, array $payload): array {
            return $this->response($payload['request_id'], false, [array_fill(0, 256, 0.0625)]);
        });
        $encoder = new \MageOS\OpenSearchHybrid\Model\Encoder\HttpDocumentEncoder(
            $config,
            $client,
            new \MageOS\OpenSearchHybrid\Model\Vector\VectorValidator()
        );

        $this->expectExceptionMessage('identity or batch shape');
        $encoder->encode(['one', 'two'], $this->generation());
    }

    public function testRejectsBatchesLargerThanTheEncoderContractBeforeSending(): void
    {
        $config = $this->createStub(\MageOS\OpenSearchHybrid\Model\Config::class);
        $client = $this->createMock(\MageOS\OpenSearchHybrid\Model\Http\CurlJsonClient::class);
        $client->expects($this->never())->method('post');
        $encoder = new \MageOS\OpenSearchHybrid\Model\Encoder\HttpDocumentEncoder(
            $config,
            $client,
            new \MageOS\OpenSearchHybrid\Model\Vector\VectorValidator()
        );

        $this->expectExceptionMessage('at most 64 documents');
        $encoder->encode(array_fill(0, 65, 'document'), $this->generation());
    }

    private function generation(): array
    {
        return [
            'generation_id' => 7,
            'encoder_endpoint' => 'http://10.0.0.20:8080',
            'model_id' => 'Snowflake/snowflake-arctic-embed-m-v2.0',
            'model_revision' => str_repeat('a', 64),
            'dimension' => 256,
            'recipe_version' => 'mageos-v1',
            'encoder_identity_digest' => str_repeat('b', 64),
        ];
    }

    private function response(string $requestId, bool $productionEligible, array $vectors): array
    {
        return [
            'request_id' => $requestId,
            'model_id' => 'Snowflake/snowflake-arctic-embed-m-v2.0',
            'model_revision' => str_repeat('a', 64),
            'dimension' => 256,
            'recipe_version' => 'mageos-v1',
            'encoder_identity_digest' => str_repeat('b', 64),
            'production_eligible' => $productionEligible,
            'vectors' => $vectors,
        ];
    }
}
