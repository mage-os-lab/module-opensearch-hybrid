<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Similarity;

class RadialRequestBuilder
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Vector\VectorValidator $vectorValidator
    ) {
    }

    public function buildSeedRequest(array $generation, int $storeId, int $productId): array
    {
        return [
            'size' => 1,
            'track_total_hits' => false,
            '_source' => false,
            'fields' => ['entity_id'],
            'docvalue_fields' => [
                ['field' => 'embedding', 'format' => 'binary'],
            ],
            'query' => [
                'bool' => [
                    'filter' => array_merge(
                        $this->identityFilters($generation, $storeId),
                        [
                            ['term' => ['entity_id' => $productId]],
                            ['term' => ['embedding_eligible' => true]],
                        ]
                    ),
                ],
            ],
        ];
    }

    public function buildRadialRequest(
        array $generation,
        int $storeId,
        int $productId,
        int $limit,
        array $vector,
        array $thresholdProfile
    ): array {
        $this->vectorValidator->validate($vector, (int)$generation['dimension']);

        return [
            'size' => $limit,
            'track_total_hits' => false,
            '_source' => false,
            'fields' => ['entity_id'],
            'sort' => [
                ['_score' => 'desc'],
                ['entity_id' => 'asc'],
            ],
            'query' => [
                'knn' => [
                    'embedding' => [
                        'vector' => $vector,
                        'min_score' => (float)$thresholdProfile['min_score'],
                        'filter' => [
                            'bool' => [
                                'filter' => array_merge(
                                    $this->identityFilters($generation, $storeId),
                                    [
                                        ['term' => ['embedding_eligible' => true]],
                                        ['term' => ['status' => 1]],
                                        ['terms' => ['visibility' => [
                                            \Magento\Catalog\Model\Product\Visibility::VISIBILITY_IN_SEARCH,
                                            \Magento\Catalog\Model\Product\Visibility::VISIBILITY_BOTH,
                                        ]]],
                                        ['term' => ['is_salable' => true]],
                                    ]
                                ),
                                'must_not' => [
                                    ['term' => ['entity_id' => $productId]],
                                ],
                            ],
                        ],
                    ],
                ],
            ],
        ];
    }

    private function identityFilters(array $generation, int $storeId): array
    {
        return [
            ['term' => ['store_id' => $storeId]],
            ['term' => ['generation_id' => (int)$generation['generation_id']]],
            ['term' => ['model_revision' => (string)$generation['model_revision']]],
        ];
    }
}
