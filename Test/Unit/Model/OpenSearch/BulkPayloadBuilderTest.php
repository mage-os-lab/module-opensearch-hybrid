<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\OpenSearch;

class BulkPayloadBuilderTest extends \PHPUnit\Framework\TestCase
{
    public function testBuildsOpenSearch38BulkPayloadWithBase64Vector(): void
    {
        $body = $this->builder()->build($this->generation(), [[
            'document' => $this->document(),
            'vector' => [1.0, 0.0],
            'revision' => 7,
        ]]);

        self::assertSame([
            'index' => [
                '_index' => 'mageos-hybrid-s1-g4',
                '_id' => '42',
                'version' => 7,
                'version_type' => 'external_gte',
            ],
        ], $body[0]);
        self::assertTrue($body[1]['embedding_eligible']);
        self::assertSame('AACAPwAAAAA=', $body[1]['embedding']);
        self::assertSame(7, $body[1]['revision']);
    }

    public function testOmitsVectorForIneligibleDocument(): void
    {
        $body = $this->builder()->build($this->generation(), [[
            'document' => $this->document(),
            'vector' => null,
            'revision' => 0,
        ]]);

        self::assertSame(1, $body[0]['index']['version']);
        self::assertFalse($body[1]['embedding_eligible']);
        self::assertArrayNotHasKey('embedding', $body[1]);
    }

    public function testRejectsNonArrayVector(): void
    {
        $this->expectException(\InvalidArgumentException::class);

        $this->builder()->build($this->generation(), [[
            'document' => $this->document(),
            'vector' => 'not-a-vector',
            'revision' => 7,
        ]]);
    }

    private function builder(): \MageOS\OpenSearchHybrid\Model\OpenSearch\BulkPayloadBuilder
    {
        return new \MageOS\OpenSearchHybrid\Model\OpenSearch\BulkPayloadBuilder(
            new \MageOS\OpenSearchHybrid\Model\Vector\VectorValidator()
        );
    }

    private function generation(): array
    {
        return [
            'physical_index' => 'mageos-hybrid-s1-g4',
            'generation_id' => 4,
            'model_revision' => 'model-revision',
            'dimension' => 2,
        ];
    }

    private function document(): array
    {
        return [
            'entity_id' => 42,
            'store_id' => 1,
            'sku' => 'SKU-42',
            'sku_normalized' => 'sku-42',
            'title' => 'Product',
            'title_normalized' => 'product',
            'description' => 'Description',
            'product_class' => 'simple',
            'category' => 'Category',
            'brand' => 'Brand',
            'features' => 'blue',
            'visibility' => 4,
            'status' => 1,
            'price' => 12.5,
            'customer_group_prices' => [],
            'stock_id' => 1,
            'is_salable' => true,
            'source_hash' => str_repeat('a', 64),
        ];
    }
}
