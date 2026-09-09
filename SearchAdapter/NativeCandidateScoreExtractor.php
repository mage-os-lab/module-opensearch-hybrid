<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\SearchAdapter;

class NativeCandidateScoreExtractor
{
    public function extract(array $documents): array
    {
        $scores = [];
        foreach ($documents as $document) {
            if (!$document instanceof \Magento\Framework\Api\Search\DocumentInterface) {
                throw new \RuntimeException('The native candidate response contains an invalid document.');
            }
            $id = (string)$document->getId();
            $scoreAttribute = $document->getCustomAttribute('score');
            $score = $scoreAttribute?->getValue();
            if ($id === ''
                || array_key_exists($id, $scores)
                || (!is_int($score) && !is_float($score))
                || !is_finite((float)$score)
                || $score < 0
            ) {
                throw new \RuntimeException(
                    'Each native candidate must have a unique ID and a finite non-negative native score.'
                );
            }
            $scores[$id] = (float)$score;
        }

        return $scores;
    }
}
