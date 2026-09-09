<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\SearchAdapter;

class Eligibility
{
    private const HYBRID_REQUEST_NAMES = [
        'quick_search_container',
        'graphql_product_search',
    ];

    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Config $config,
        private readonly \MageOS\OpenSearchHybrid\Model\Store\ReadinessLatch $readinessLatch,
        private readonly \MageOS\OpenSearchHybrid\SearchAdapter\SortEligibility $sortEligibility
    ) {
    }

    public function isEligible(\Magento\Framework\Search\RequestInterface $request, int $storeId): bool
    {
        return $this->ineligibilityReason($request, $storeId) === null;
    }

    public function ineligibilityReason(
        \Magento\Framework\Search\RequestInterface $request,
        int $storeId
    ): ?string {
        if (!$this->config->isEnabled($storeId)) {
            return 'disabled';
        }
        if (!$this->config->isStoreIncluded($storeId)) {
            return 'store_excluded';
        }
        if (!$this->readinessLatch->isOpen($storeId)) {
            return 'readiness_closed';
        }
        if (!in_array($request->getName(), self::HYBRID_REQUEST_NAMES, true)) {
            return 'unsupported_request';
        }
        if (method_exists($request, 'getSort')
            && $this->sortEligibility->hybridPlan($request->getSort()) === null
        ) {
            return 'unsupported_sort';
        }
        $from = max(0, (int)($request->getFrom() ?? 0));
        $size = max(0, (int)($request->getSize() ?? 0));
        if ($size === 0 || $size > 100 || $from >= 100) {
            return 'invalid_page';
        }

        return null;
    }
}
