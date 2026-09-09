<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Encoder;

class HttpQueryEncoderTest extends \PHPUnit\Framework\TestCase
{
    public function testUsesAndEchoValidatesTheGenerationBoundIdentity(): void
    {
        $generation = $this->generation();
        $config = $this->createMock(\MageOS\OpenSearchHybrid\Model\Config::class);
        $config->expects($this->once())
            ->method('validateEncoderEndpoint')
            ->with('http://10.0.0.20:8080')
            ->willReturn('http://10.0.0.20:8080');
        $config->method('encoderToken')->willReturn('');
        $config->method('encoderConnectTimeoutMs')->willReturn(100);
        $config->method('encoderQueryTimeoutMs')->willReturn(350);
        $client = $this->createMock(\MageOS\OpenSearchHybrid\Model\Http\CurlJsonClient::class);
        $client->expects($this->once())
            ->method('post')
            ->with(
                'http://10.0.0.20:8080/v1/embed/query',
                $this->callback(static fn (array $payload): bool =>
                    $payload['recipe_version'] === 'mageos-v1'
                    && $payload['encoder_identity_digest'] === str_repeat('b', 64)),
                100,
                350,
                []
            )
            ->willReturn($this->response(true));
        $encoder = new \MageOS\OpenSearchHybrid\Model\Encoder\HttpQueryEncoder(
            $config,
            $client,
            new \MageOS\OpenSearchHybrid\Model\Vector\VectorValidator()
        );

        $vector = $encoder->encode('red trail shoe', $generation);

        self::assertCount(256, $vector);
    }

    public function testRejectsAnEncoderThatIsNotProductionEligible(): void
    {
        $config = $this->createStub(\MageOS\OpenSearchHybrid\Model\Config::class);
        $config->method('validateEncoderEndpoint')->willReturn('http://10.0.0.20:8080');
        $config->method('encoderToken')->willReturn('');
        $config->method('encoderConnectTimeoutMs')->willReturn(100);
        $config->method('encoderQueryTimeoutMs')->willReturn(350);
        $client = $this->createStub(\MageOS\OpenSearchHybrid\Model\Http\CurlJsonClient::class);
        $client->method('post')->willReturn($this->response(false));
        $encoder = new \MageOS\OpenSearchHybrid\Model\Encoder\HttpQueryEncoder(
            $config,
            $client,
            new \MageOS\OpenSearchHybrid\Model\Vector\VectorValidator()
        );

        $this->expectException(\RuntimeException::class);
        $this->expectExceptionMessage('identity does not match');

        $encoder->encode('red trail shoe', $this->generation());
    }

    private function generation(): array
    {
        return [
            'generation_id' => 9,
            'model_id' => 'fixture/model',
            'model_revision' => str_repeat('a', 64),
            'dimension' => 256,
            'recipe_version' => 'mageos-v1',
            'encoder_endpoint' => 'http://10.0.0.20:8080',
            'encoder_identity_digest' => str_repeat('b', 64),
        ];
    }

    private function response(bool $productionEligible): array
    {
        return [
            'request_id' => hash('sha256', "9\0red trail shoe"),
            'model_id' => 'fixture/model',
            'model_revision' => str_repeat('a', 64),
            'dimension' => 256,
            'recipe_version' => 'mageos-v1',
            'encoder_identity_digest' => str_repeat('b', 64),
            'production_eligible' => $productionEligible,
            'vectors' => [array_fill(0, 256, 0.0625)],
        ];
    }
}
