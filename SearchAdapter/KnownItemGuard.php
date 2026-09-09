<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\SearchAdapter;

class KnownItemGuard
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Text\Canonicalizer $canonicalizer,
        private readonly \MageOS\OpenSearchHybrid\Model\OpenSearch\HybridTransport $transport
    ) {
    }

    public function mustUseLexical(string $query, array $generation): bool
    {
        $normalized = $this->canonicalizer->normalize($query);
        if ($normalized === '') {
            return true;
        }
        $response = $this->transport->search(
            (string)$generation['physical_index'],
            null,
            [
                'size' => 2,
                'track_total_hits' => true,
                '_source' => false,
                'query' => [
                    'bool' => [
                        'should' => [
                            ['term' => ['sku_normalized' => $normalized]],
                            ['term' => ['title_normalized' => $normalized]],
                        ],
                        'minimum_should_match' => 1,
                    ],
                ],
            ]
        );
        $total = $response['hits']['total']['value'] ?? 0;

        return (int)$total > 0;
    }
}
