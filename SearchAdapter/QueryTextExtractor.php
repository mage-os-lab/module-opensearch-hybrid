<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\SearchAdapter;

class QueryTextExtractor
{
    public function extract(\Magento\Framework\Search\RequestInterface $request): string
    {
        $values = [];
        $this->collect($request->getQuery(), $values);
        $values = array_values(array_unique(array_filter($values, static fn (string $value): bool => $value !== '')));
        if ($values === []) {
            throw new \InvalidArgumentException('The quick-search request has no search term.');
        }

        return $values[0];
    }

    private function collect(\Magento\Framework\Search\Request\QueryInterface $query, array &$values): void
    {
        if ($query->getType() === \Magento\Framework\Search\Request\QueryInterface::TYPE_MATCH
            && $query instanceof \Magento\Framework\Search\Request\Query\MatchQuery
        ) {
            $values[] = trim((string)$query->getValue());
            return;
        }
        if ($query instanceof \Magento\Framework\Search\Request\Query\BoolExpression) {
            foreach (array_merge($query->getMust(), $query->getShould(), $query->getMustNot()) as $child) {
                $this->collect($child, $values);
            }
            return;
        }
        if ($query instanceof \Magento\Framework\Search\Request\Query\Filter
            && $query->getReferenceType() === \Magento\Framework\Search\Request\Query\Filter::REFERENCE_QUERY
        ) {
            $this->collect($query->getReference(), $values);
        }
    }
}
