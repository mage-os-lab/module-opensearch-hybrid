<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Security;

class EndpointValidatorTest extends \PHPUnit\Framework\TestCase
{
    public function testAllowsPrivateAddressAndExplicitDnsHost(): void
    {
        $validator = new \MageOS\OpenSearchHybrid\Model\Security\EndpointValidator();

        self::assertSame('http://10.1.2.3:8080', $validator->validate('http://10.1.2.3:8080/', []));
        self::assertSame('http://172.16.1.2', $validator->validate('http://172.16.1.2', []));
        self::assertSame('http://192.168.1.2', $validator->validate('http://192.168.1.2', []));
        self::assertSame('http://[fd00::1]:8080', $validator->validate('http://[fd00::1]:8080', []));
        self::assertSame(
            'https://encoder.internal',
            $validator->validate('https://encoder.internal', ['encoder.internal'])
        );
    }

    public function testAllowsLoopbackOnlyWhenTheExactHostIsExplicitlyAllowed(): void
    {
        $validator = new \MageOS\OpenSearchHybrid\Model\Security\EndpointValidator();

        self::assertSame(
            'http://127.0.0.1:8080',
            $validator->validate('http://127.0.0.1:8080', ['127.0.0.1'])
        );
        self::assertSame(
            'http://[::1]:8080',
            $validator->validate('http://[::1]:8080', ['::1'])
        );
        self::assertSame(
            'http://localhost:8080',
            $validator->validate('http://localhost:8080', ['localhost'])
        );
    }

    #[\PHPUnit\Framework\Attributes\DataProvider('forbiddenEndpoints')]
    public function testRejectsUnlistedLoopbackAndAlwaysRejectsLinkLocal(string $endpoint): void
    {
        $validator = new \MageOS\OpenSearchHybrid\Model\Security\EndpointValidator();
        $this->expectException(\InvalidArgumentException::class);

        $validator->validate($endpoint, ['169.254.169.254']);
    }

    public static function forbiddenEndpoints(): array
    {
        return [
            ['http://127.0.0.1:8080'],
            ['http://169.254.1.1'],
            ['http://169.254.169.254/latest/meta-data'],
            ['http://[::1]:8080'],
            ['http://0.0.0.0:8080'],
            ['http://[fe80::1]:8080'],
            ['http://localhost:8080'],
        ];
    }

    #[\PHPUnit\Framework\Attributes\DataProvider('forbiddenUrlComponents')]
    public function testRejectsComponentsOnOtherwiseAllowedHosts(string $endpoint, array $hosts): void
    {
        $validator = new \MageOS\OpenSearchHybrid\Model\Security\EndpointValidator();
        $this->expectException(\InvalidArgumentException::class);
        $validator->validate($endpoint, $hosts);
    }

    public static function forbiddenUrlComponents(): array
    {
        $cases = [];
        foreach (['10.1.2.3', 'encoder.internal'] as $host) {
            foreach (['user@' . $host, 'user:placeholder@' . $host, $host . '?q=1', $host . '#part'] as $url) {
                $cases[] = ['http://' . $url, [$host]];
            }
        }
        return $cases;
    }

    public function testRejectsUserInfoPathsAndUnexpectedPublicHosts(): void
    {
        $validator = new \MageOS\OpenSearchHybrid\Model\Security\EndpointValidator();

        foreach (
            [
                'https://user:pass@encoder.test',
                'https://encoder.test/api',
                'https://8.8.8.8',
                'https://100.64.0.1',
                'https://198.51.100.1',
            ] as $endpoint
        ) {
            try {
                $validator->validate($endpoint, []);
                self::fail('Expected endpoint rejection for ' . $endpoint);
            } catch (\InvalidArgumentException) {
            }
        }
        self::addToAssertionCount(5);
    }
}
