<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\SearchAdapter;

class NativeCandidateRequestFactory
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\SearchAdapter\SortEligibility $sortEligibility
    ) {
    }

    public function create(
        \Magento\Framework\Search\RequestInterface $request
    ): \Magento\Framework\Search\Request {
        $end = max(0, (int)$request->getFrom()) + max(0, (int)$request->getSize());

        return $this->build($request, 0, max(100, min(199, $end)));
    }

    public function createPage(
        \Magento\Framework\Search\RequestInterface $request
    ): \Magento\Framework\Search\Request {
        return $this->build($request, (int)$request->getFrom(), (int)$request->getSize());
    }

    private function build(
        \Magento\Framework\Search\RequestInterface $request,
        int $from,
        int $size
    ): \Magento\Framework\Search\Request {
        return new \Magento\Framework\Search\Request(
            $request->getName(),
            $request->getIndex(),
            $request->getQuery(),
            $from,
            $size,
            $request->getDimensions(),
            $request->getAggregation(),
            $this->sortEligibility->nativeSort(method_exists($request, 'getSort') ? $request->getSort() : [])
        );
    }
}
