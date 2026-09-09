<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\OpenSearch;

class VersionPolicyTest extends \PHPUnit\Framework\TestCase
{
    #[\PHPUnit\Framework\Attributes\DataProvider('supportedVersionProvider')]
    public function testSupportedVersions(string $version): void
    {
        $policy = new \MageOS\OpenSearchHybrid\Model\OpenSearch\VersionPolicy();

        self::assertTrue($policy->isSupported($version));
        $policy->assertSupported($version);
    }

    public static function supportedVersionProvider(): array
    {
        return [
            ['3.8.0'],
            ['3.8.1'],
            ['3.8.99'],
        ];
    }

    #[\PHPUnit\Framework\Attributes\DataProvider('unsupportedVersionProvider')]
    public function testUnsupportedVersions(string $version): void
    {
        $policy = new \MageOS\OpenSearchHybrid\Model\OpenSearch\VersionPolicy();

        self::assertFalse($policy->isSupported($version));
        $this->expectException(\RuntimeException::class);
        $this->expectExceptionMessage('OpenSearch Hybrid requires >=3.8.0 <3.9.0.');
        $policy->assertSupported($version);
    }

    public static function unsupportedVersionProvider(): array
    {
        return [
            ['2.19.6'],
            ['3.1.0'],
            ['3.7.99'],
            ['3.9.0'],
            ['3.99.0'],
            ['4.0.0'],
        ];
    }
}
