<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Vector;

class DeterministicVectorTest extends \PHPUnit\Framework\TestCase
{
    public function testProducesDeterministicNormalizedDevelopmentVector(): void
    {
        $encoder = new \MageOS\OpenSearchHybrid\Model\Vector\DeterministicVector();
        $first = $encoder->encode('one document');
        $second = $encoder->encode('one document');

        self::assertCount(256, $first);
        self::assertSame($first, $second);
        $norm = sqrt(array_sum(array_map(static fn (float $value): float => $value * $value, $first)));
        self::assertEqualsWithDelta(1.0, $norm, 0.000001);
    }
}
