<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Encoder;

class EncoderIdentityClientTest extends \PHPUnit\Framework\TestCase
{
    public function testFetchesAuthenticatedIdentityAndAppliesTheOperatorPin(): void
    {
        $identity = ['encoder_identity_digest' => str_repeat('a', 64)];
        $config = $this->createStub(\MageOS\OpenSearchHybrid\Model\Config::class);
        $config->method('encoderEndpoint')->willReturn('https://encoder.internal');
        $config->method('validateEncoderEndpoint')->willReturn('https://encoder.internal');
        $config->method('encoderToken')->willReturn('secret');
        $config->method('encoderConnectTimeoutMs')->willReturn(100);
        $config->method('encoderQueryTimeoutMs')->willReturn(350);
        $client = $this->createMock(\MageOS\OpenSearchHybrid\Model\Http\CurlJsonClient::class);
        $client->expects($this->once())->method('get')->with(
            'https://encoder.internal/v1/identity',
            100,
            350,
            ['Authorization' => 'Bearer secret']
        )->willReturn($identity);
        $verifier = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Encoder\EncoderIdentityVerifier::class
        );
        $verifier->expects($this->once())
            ->method('verify')
            ->with($identity, str_repeat('a', 64))
            ->willReturn($identity);
        $subject = new \MageOS\OpenSearchHybrid\Model\Encoder\EncoderIdentityClient(
            $config,
            $client,
            $verifier
        );

        self::assertSame($identity, $subject->fetch(str_repeat('a', 64)));
    }
}
