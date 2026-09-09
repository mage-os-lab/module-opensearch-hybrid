<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\SearchAdapter;

class HybridResultMapper
{
    public function map(
        array $candidateDocuments,
        array $hits,
        bool $outOfStockBottom = false
    ): array
    {
        if (count($hits) !== count($candidateDocuments)) {
            throw new \RuntimeException(
                'The hybrid result universe differs from the native candidate universe.'
            );
        }
        $documentsById = [];
        foreach ($candidateDocuments as $document) {
            if (!$document instanceof \Magento\Framework\Api\Search\DocumentInterface) {
                throw new \RuntimeException('The native candidate response contains an invalid document.');
            }
            $id = (string)$document->getId();
            if ($id === '' || isset($documentsById[$id])) {
                throw new \RuntimeException(
                    'The hybrid result universe differs from the native candidate universe.'
                );
            }
            $documentsById[$id] = $document;
        }
        $ordered = [];
        $salable = [];
        $outOfStock = [];
        $seen = [];
        foreach ($hits as $hit) {
            $id = is_array($hit) && isset($hit['_id']) ? (string)$hit['_id'] : '';
            $score = is_array($hit) ? ($hit['_score'] ?? null) : null;
            if ($id === ''
                || !isset($documentsById[$id])
                || isset($seen[$id])
                || (!is_int($score) && !is_float($score))
                || !is_finite((float)$score)
                || $score < 0
            ) {
                throw new \RuntimeException(
                    'The hybrid result universe differs from the native candidate universe.'
                );
            }
            $scoreAttribute = $documentsById[$id]->getCustomAttribute('score');
            if (!$scoreAttribute instanceof \Magento\Framework\Api\AttributeInterface) {
                throw new \RuntimeException('The native candidate response is missing its score attribute.');
            }
            $scoreAttribute->setValue((float)$score);
            if ($outOfStockBottom) {
                $salability = $hit['fields']['is_salable'] ?? null;
                if (!is_array($salability) || count($salability) !== 1 || !is_bool($salability[0])) {
                    throw new \RuntimeException('The hybrid result is missing salability metadata.');
                }
                if ($salability[0]) {
                    $salable[] = $documentsById[$id];
                } else {
                    $outOfStock[] = $documentsById[$id];
                }
            } else {
                $ordered[] = $documentsById[$id];
            }
            $seen[$id] = true;
        }

        return $outOfStockBottom ? array_merge($salable, $outOfStock) : $ordered;
    }
}
