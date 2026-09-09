<?php

declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\SearchAdapter;

class SortEligibility
{
    public function isRelevanceOnly(mixed $sort): bool
    {
        if (!is_array($sort) || $sort === []) {
            return true;
        }
        $entries = $this->entries($sort);
        return $entries !== null && $this->isPreparedRelevance($entries);
    }

    public function hybridPlan(mixed $sort): ?array
    {
        if ($this->isRelevanceOnly($sort)) {
            return ['out_of_stock_bottom' => false];
        }
        $entries = $this->entries($sort);
        if ($entries === null) {
            return null;
        }
        if ($this->isStorefrontDefaultPosition($entries)) {
            return ['out_of_stock_bottom' => false];
        }
        if (count($entries) < 2) {
            return null;
        }
        $availability = array_shift($entries);
        if ($availability['field'] !== 'is_out_of_stock' || $availability['direction'] !== 'ASC') {
            return null;
        }
        if ($this->isPreparedRelevance($entries) || $this->isStorefrontDefaultPosition($entries)) {
            return ['out_of_stock_bottom' => true];
        }

        return null;
    }

    public function nativeSort(mixed $sort): array
    {
        $plan = $this->hybridPlan($sort);
        if ($plan === null) {
            throw new \InvalidArgumentException('The search sort is not hybrid compatible.');
        }
        $normalized = $plan['out_of_stock_bottom']
            ? [['field' => 'is_out_of_stock', 'direction' => 'ASC']]
            : [];
        $normalized[] = ['field' => 'relevance', 'direction' => 'DESC'];
        $tieDirection = 'ASC';
        foreach ($this->entries(is_array($sort) ? $sort : []) ?? [] as $entry) {
            if ($entry['field'] === 'entity_id') {
                $tieDirection = $entry['direction'];
            }
        }
        $normalized[] = ['field' => 'entity_id', 'direction' => $tieDirection];

        return $normalized;
    }

    private function entries(array $sort): ?array
    {
        $entries = [];
        foreach ($sort as $key => $entry) {
            if ($entry instanceof \Magento\Framework\Api\SortOrder) {
                $field = $entry->getField();
                $direction = $entry->getDirection();
            } elseif (is_array($entry) && isset($entry['field'])) {
                $field = (string)$entry['field'];
                $direction = $entry['direction'] ?? null;
            } elseif (is_string($key)) {
                $field = $key;
                $direction = $entry;
            } else {
                return null;
            }
            if (!is_string($direction)) {
                return null;
            }
            $entries[] = [
                'field' => (string)$field,
                'direction' => strtoupper($direction),
            ];
        }

        return $entries;
    }

    private function isPreparedRelevance(array $entries): bool
    {
        if ($entries === []) {
            return false;
        }
        $hasRelevance = false;
        $hasEntityTieBreaker = false;
        foreach ($entries as $entry) {
            $field = $entry['field'];
            $direction = $entry['direction'];
            if (in_array($field, ['_score', 'score', 'relevance'], true)) {
                if (
                    $hasRelevance
                    || $hasEntityTieBreaker
                    || $direction !== \Magento\Framework\Api\SortOrder::SORT_DESC
                ) {
                    return false;
                }
                $hasRelevance = true;
                continue;
            }
            if (
                $field !== 'entity_id'
                || !$hasRelevance
                || $hasEntityTieBreaker
                || !in_array($direction, [
                    \Magento\Framework\Api\SortOrder::SORT_ASC,
                    \Magento\Framework\Api\SortOrder::SORT_DESC,
                ], true)
            ) {
                return false;
            }
            $hasEntityTieBreaker = true;
        }

        return $hasRelevance;
    }

    private function isStorefrontDefaultPosition(array $entries): bool
    {
        if (count($entries) < 1 || count($entries) > 2) {
            return false;
        }
        if ($entries[0] !== ['field' => 'position', 'direction' => 'ASC']) {
            return false;
        }

        return !isset($entries[1])
            || ($entries[1]['field'] === 'entity_id'
                && in_array($entries[1]['direction'], ['ASC', 'DESC'], true));
    }
}
