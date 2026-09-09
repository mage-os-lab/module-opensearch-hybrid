<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\SearchAdapter;

class NativeCandidateScoreExtractorTest extends \PHPUnit\Framework\TestCase
{
    public function testExtractsNativeDocumentScoresWithoutChangingOrderOrMagnitude(): void
    {
        $extractor = new \MageOS\OpenSearchHybrid\SearchAdapter\NativeCandidateScoreExtractor();

        self::assertSame(
            ['19' => 18.75, '7' => 2.5],
            $extractor->extract([
                $this->document('19', 18.75),
                $this->document('7', 2.5),
            ])
        );
    }

    public function testRejectsCandidateWithoutNativeScore(): void
    {
        $extractor = new \MageOS\OpenSearchHybrid\SearchAdapter\NativeCandidateScoreExtractor();
        $document = new \Magento\Framework\Api\Search\Document([
            \Magento\Framework\Api\Search\DocumentInterface::ID => '19',
            \Magento\Framework\Api\CustomAttributesDataInterface::CUSTOM_ATTRIBUTES => [],
        ]);

        $this->expectException(\RuntimeException::class);
        $this->expectExceptionMessage('native score');
        $extractor->extract([$document]);
    }

    private function document(string $id, float $score): \Magento\Framework\Api\Search\Document
    {
        $attribute = new \Magento\Framework\Api\AttributeValue([
            \Magento\Framework\Api\AttributeInterface::ATTRIBUTE_CODE => '_score',
            \Magento\Framework\Api\AttributeInterface::VALUE => $score,
        ]);

        return new \Magento\Framework\Api\Search\Document([
            \Magento\Framework\Api\Search\DocumentInterface::ID => $id,
            \Magento\Framework\Api\CustomAttributesDataInterface::CUSTOM_ATTRIBUTES => ['score' => $attribute],
        ]);
    }
}
