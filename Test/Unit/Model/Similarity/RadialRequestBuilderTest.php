<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Similarity;

class RadialRequestBuilderTest extends \PHPUnit\Framework\TestCase
{
    public function testBuildsBoundedFilteredMinScoreRequestWithoutTopK(): void
    {
        $builder = new \MageOS\OpenSearchHybrid\Model\Similarity\RadialRequestBuilder(
            new \MageOS\OpenSearchHybrid\Model\Vector\VectorValidator()
        );
        $request = $builder->buildRadialRequest(
            $this->generation(),
            2,
            41,
            8,
            [1.0, 0.0],
            ['min_score' => 0.91]
        );
        $radial = $request['query']['knn']['embedding'];
        $filter = $radial['filter']['bool'];

        self::assertSame(8, $request['size']);
        self::assertFalse($request['_source']);
        self::assertSame([['_score' => 'desc'], ['entity_id' => 'asc']], $request['sort']);
        self::assertSame(0.91, $radial['min_score']);
        self::assertArrayNotHasKey('k', $radial);
        self::assertContains(['term' => ['store_id' => 2]], $filter['filter']);
        self::assertContains(['term' => ['generation_id' => 7]], $filter['filter']);
        self::assertContains(['term' => ['model_revision' => 'revision-1']], $filter['filter']);
        self::assertContains(['term' => ['embedding_eligible' => true]], $filter['filter']);
        self::assertContains(['term' => ['status' => 1]], $filter['filter']);
        self::assertContains(['term' => ['is_salable' => true]], $filter['filter']);
        self::assertSame([['term' => ['entity_id' => 41]]], $filter['must_not']);
    }

    public function testSeedRequestUsesBinaryDocValuesAndExactGenerationIdentity(): void
    {
        $builder = new \MageOS\OpenSearchHybrid\Model\Similarity\RadialRequestBuilder(
            new \MageOS\OpenSearchHybrid\Model\Vector\VectorValidator()
        );

        $request = $builder->buildSeedRequest($this->generation(), 2, 41);

        self::assertSame([['field' => 'embedding', 'format' => 'binary']], $request['docvalue_fields']);
        self::assertContains(
            ['term' => ['entity_id' => 41]],
            $request['query']['bool']['filter']
        );
        self::assertContains(
            ['term' => ['generation_id' => 7]],
            $request['query']['bool']['filter']
        );
    }

    private function generation(): array
    {
        return [
            'generation_id' => 7,
            'model_revision' => 'revision-1',
            'dimension' => 2,
        ];
    }
}
