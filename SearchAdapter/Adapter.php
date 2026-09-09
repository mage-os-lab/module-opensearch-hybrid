<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\SearchAdapter;

class Adapter implements \Magento\Framework\Search\AdapterInterface
{
    public function __construct(
        private readonly \Magento\OpenSearch\SearchAdapter\Adapter $nativeAdapter,
        private readonly \Magento\Store\Model\StoreManagerInterface $storeManager,
        private readonly \MageOS\OpenSearchHybrid\SearchAdapter\Eligibility $eligibility,
        private readonly \MageOS\OpenSearchHybrid\SearchAdapter\HybridExecutor $hybridExecutor,
        private readonly \MageOS\OpenSearchHybrid\Model\Reliability\CircuitBreaker $circuitBreaker,
        private readonly \Psr\Log\LoggerInterface $logger,
        private readonly \MageOS\OpenSearchHybrid\Model\Operations\QueryTelemetry $queryTelemetry,
        private readonly \MageOS\OpenSearchHybrid\SearchAdapter\NativeCandidateRequestFactory $candidateRequestFactory
    ) {
    }

    public function query(
        \Magento\Framework\Search\RequestInterface $request
    ): \Magento\Framework\Search\Response\QueryResponse {
        $storeId = (int)$this->storeManager->getStore()->getId();
        $span = $this->queryTelemetry->start($storeId, (string)$request->getName());
        try {
            $ineligibilityReason = $this->eligibility->ineligibilityReason($request, $storeId);
            if ($ineligibilityReason === null && $this->circuitBreaker->isOpen($storeId)) {
                $ineligibilityReason = 'circuit_open';
            }
        } catch (\Throwable $throwable) {
            $this->recordFailure($storeId, $throwable);
            $span->route('native_fallback', 'hybrid_error');

            return $this->native($request, $span, $throwable);
        }
        if ($ineligibilityReason !== null) {
            if ($ineligibilityReason === 'invalid_page'
                && (int)$request->getFrom() >= 100
                && (int)$request->getSize() > 0
                && (int)$request->getSize() <= 100
            ) {
                $request = $this->candidateRequestFactory->createPage($request);
            }
            $span->route('native', $ineligibilityReason);

            return $this->native($request, $span);
        }
        try {
            $response = $this->hybridExecutor->execute($request, $storeId, $span);
        } catch (\Throwable $throwable) {
            $this->recordFailure($storeId, $throwable);
            $span->route('native_fallback', 'hybrid_error');

            return $this->native($request, $span, $throwable);
        }
        try {
            $this->circuitBreaker->success($storeId);
        } catch (\Throwable) {
            // Optional circuit bookkeeping must not discard a successful search.
        }
        $span->finish();

        return $response;
    }

    private function recordFailure(int $storeId, \Throwable $throwable): void
    {
        try {
            $this->circuitBreaker->failure($storeId);
        } catch (\Throwable) {
            // A cache failure must not prevent native delegation.
        }
        try {
            $this->logger->warning('OpenSearch Hybrid routed the request to native search.', [
                'store_id' => $storeId,
                'error_class' => $throwable::class,
            ]);
        } catch (\Throwable) {
            // Logging is independent of search availability.
        }
    }

    private function native(
        \Magento\Framework\Search\RequestInterface $request,
        \MageOS\OpenSearchHybrid\Model\Operations\QueryTelemetrySpan $span,
        ?\Throwable $hybridFailure = null
    ): \Magento\Framework\Search\Response\QueryResponse {
        try {
            $response = $this->nativeAdapter->query($request);
        } catch (\Throwable $throwable) {
            $span->finish($throwable);
            throw $throwable;
        }
        $span->finish($hybridFailure);

        return $response;
    }
}
