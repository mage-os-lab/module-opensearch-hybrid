<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\SearchAdapter;

class HybridResultMapperTest extends \PHPUnit\Framework\TestCase
{
    public function testReturnsHybridOrderAndReplacesNativeScores(): void
    {
        $mapper = new \MageOS\OpenSearchHybrid\SearchAdapter\HybridResultMapper();
        $documents = [$this->document('1', 20.0), $this->document('2', 5.0)];

        $ordered = $mapper->map($documents, [
            ['_id' => '2', '_score' => 0.9],
            ['_id' => '1', '_score' => 0.7],
        ]);

        self::assertSame(['2', '1'], array_map(
            static fn (\Magento\Framework\Api\Search\DocumentInterface $document): string =>
                (string)$document->getId(),
            $ordered
        ));
        self::assertSame(0.9, $ordered[0]->getCustomAttribute('score')->getValue());
        self::assertSame(0.7, $ordered[1]->getCustomAttribute('score')->getValue());
    }

    public function testRejectsAResultUniverseMismatch(): void
    {
        $mapper = new \MageOS\OpenSearchHybrid\SearchAdapter\HybridResultMapper();

        $this->expectException(\RuntimeException::class);
        $this->expectExceptionMessage('result universe');
        $mapper->map([$this->document('1', 20.0)], [['_id' => '2', '_score' => 1.0]]);
    }

    public function testKeepsSalableResultsAheadOfOutOfStockResults(): void
    {
        $mapper = new \MageOS\OpenSearchHybrid\SearchAdapter\HybridResultMapper();
        $documents = [$this->document('1', 20.0), $this->document('2', 5.0)];

        $ordered = $mapper->map($documents, [
            ['_id' => '1', '_score' => 0.9, 'fields' => ['is_salable' => [false]]],
            ['_id' => '2', '_score' => 0.7, 'fields' => ['is_salable' => [true]]],
        ], true);

        self::assertSame(['2', '1'], array_map(
            static fn (\Magento\Framework\Api\Search\DocumentInterface $document): string =>
                (string)$document->getId(),
            $ordered
        ));
        self::assertSame(0.7, $ordered[0]->getCustomAttribute('score')->getValue());
        self::assertSame(0.9, $ordered[1]->getCustomAttribute('score')->getValue());
    }

    public function testRejectsMissingSalabilityMetadataWhenAvailabilityOrderingIsRequired(): void
    {
        $mapper = new \MageOS\OpenSearchHybrid\SearchAdapter\HybridResultMapper();

        $this->expectException(\RuntimeException::class);
        $this->expectExceptionMessage('salability metadata');
        $mapper->map(
            [$this->document('1', 20.0)],
            [['_id' => '1', '_score' => 0.9]],
            true
        );
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
