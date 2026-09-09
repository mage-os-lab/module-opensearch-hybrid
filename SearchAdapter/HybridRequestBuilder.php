<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\SearchAdapter;

class HybridRequestBuilder
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract $retrievalContract
    ) {
    }

    public function build(
        array $vector,
        array $nativeCandidateScores,
        bool $outOfStockBottom = false
    ): array
    {
        if ($nativeCandidateScores === []) {
            throw new \InvalidArgumentException('A hybrid request requires native candidate IDs.');
        }
        $candidateIds = [];
        $scoreClauses = [];
        foreach ($nativeCandidateScores as $candidateId => $nativeScore) {
            $candidateId = (string)$candidateId;
            if ($candidateId === ''
                || (!is_int($nativeScore) && !is_float($nativeScore))
                || !is_finite((float)$nativeScore)
                || $nativeScore < 0
            ) {
                throw new \InvalidArgumentException(
                    'Each hybrid candidate requires an ID and a finite non-negative native score.'
                );
            }
            $candidateIds[] = $candidateId;
            $scoreClauses[] = [
                'constant_score' => [
                    'filter' => ['term' => ['_id' => $candidateId]],
                    'boost' => (float)$nativeScore,
                ],
            ];
        }
        $contract = $this->retrievalContract->get();
        $ann = $contract['ann'];
        $candidateFilter = ['terms' => ['_id' => $candidateIds]];
        $lexicalClause = [
            'bool' => [
                'should' => $scoreClauses,
                'minimum_should_match' => 1,
            ],
        ];
        $denseClause = [
            'knn' => [
                'embedding' => [
                    'vector' => $vector,
                    'k' => $ann['k'],
                    'filter' => $candidateFilter,
                ],
            ],
        ];

        $request = [
            'size' => 100,
            'track_total_hits' => false,
            '_source' => false,
            'query' => [
                'hybrid' => [
                    'pagination_depth' => 100,
                    'queries' => [$lexicalClause, $denseClause],
                ],
            ],
        ];
        if ($outOfStockBottom) {
            $request['fields'] = ['is_salable'];
        }

        return $request;
    }
}
