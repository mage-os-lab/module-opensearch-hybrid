<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Vector;

class VectorValidatorTest extends \PHPUnit\Framework\TestCase
{
    public function testCreatesCanonicalLittleEndianFloat32Binary(): void
    {
        $validator = new \MageOS\OpenSearchHybrid\Model\Vector\VectorValidator();
        $binary = $validator->toFloat32Binary([1.0, 0.0], 2);

        self::assertSame(8, strlen($binary));
        self::assertSame([1.0, 0.0], array_values(unpack('g2', $binary)));
    }

    public function testCreatesOpenSearch38Base64VectorPayload(): void
    {
        $validator = new \MageOS\OpenSearchHybrid\Model\Vector\VectorValidator();

        self::assertSame('AACAPwAAAAA=', $validator->toBase64Float32([1.0, 0.0], 2));
    }

    public function testRejectsWrongDimension(): void
    {
        $validator = new \MageOS\OpenSearchHybrid\Model\Vector\VectorValidator();
        $this->expectException(\InvalidArgumentException::class);

        $validator->validate([1.0], 2);
    }

    public function testRejectsNonFiniteValues(): void
    {
        $validator = new \MageOS\OpenSearchHybrid\Model\Vector\VectorValidator();
        $this->expectException(\InvalidArgumentException::class);

        $validator->validate([INF, 0.0], 2);
    }

    public function testDecodesCanonicalBase64Float32Vector(): void
    {
        $validator = new \MageOS\OpenSearchHybrid\Model\Vector\VectorValidator();

        self::assertSame([1.0, 0.0], $validator->fromBase64Float32('AACAPwAAAAA=', 2));
    }

    public function testRejectsStoredVectorWithWrongByteLength(): void
    {
        $validator = new \MageOS\OpenSearchHybrid\Model\Vector\VectorValidator();
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('stored vector bytes are invalid');

        $validator->fromBase64Float32('AAAAAA==', 2);
    }
}
