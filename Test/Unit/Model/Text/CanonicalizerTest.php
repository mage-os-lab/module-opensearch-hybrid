<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Text;

class CanonicalizerTest extends \PHPUnit\Framework\TestCase
{
    public function testNormalizesNfkcWhitespaceAndCaseWhilePreservingPunctuation(): void
    {
        $canonicalizer = new \MageOS\OpenSearchHybrid\Model\Text\Canonicalizer();

        self::assertSame('abc sku-1/2', $canonicalizer->normalize("  ＡＢＣ\u{2003}SKU-1/2  "));
    }

    public function testEquivalentInputsCollideByDesign(): void
    {
        $canonicalizer = new \MageOS\OpenSearchHybrid\Model\Text\Canonicalizer();

        self::assertSame(
            $canonicalizer->normalize('Trail   Shoe'),
            $canonicalizer->normalize("trail\tshoe")
        );
    }
}
