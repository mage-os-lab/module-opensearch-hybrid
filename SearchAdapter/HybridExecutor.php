<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\SearchAdapter;

class HybridExecutor
{
    public function __construct(
        private readonly \Magento\OpenSearch\SearchAdapter\Adapter $nativeAdapter,
        private readonly \MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository $generationRepository,
        private readonly \MageOS\OpenSearchHybrid\SearchAdapter\QueryTextExtractor $queryTextExtractor,
        private readonly \MageOS\OpenSearchHybrid\SearchAdapter\KnownItemGuard $knownItemGuard,
        private readonly \MageOS\OpenSearchHybrid\Api\QueryEncoderInterface $queryEncoder,
        private readonly \MageOS\OpenSearchHybrid\SearchAdapter\NativeCandidateScoreExtractor $scoreExtractor,
        private readonly \MageOS\OpenSearchHybrid\SearchAdapter\SortEligibility $sortEligibility,
        private readonly \MageOS\OpenSearchHybrid\SearchAdapter\NativeCandidateRequestFactory $candidateRequestFactory,
        private readonly \MageOS\OpenSearchHybrid\SearchAdapter\HybridRequestBuilder $requestBuilder,
        private readonly \MageOS\OpenSearchHybrid\SearchAdapter\HybridResultMapper $resultMapper,
        private readonly \MageOS\OpenSearchHybrid\Model\OpenSearch\HybridTransport $transport
    ) {
    }

    public function execute(
        \Magento\Framework\Search\RequestInterface $request,
        int $storeId,
        \MageOS\OpenSearchHybrid\Model\Operations\QueryTelemetrySpan $span
    ): \Magento\Framework\Search\Response\QueryResponse {
        $generation = $this->generationRepository->activeForStore($storeId);
        if ($generation === null || !(bool)$generation['result_contract_accepted']) {
            throw new \RuntimeException('No accepted active generation is available.');
        }
        $generationId = (int)$generation['generation_id'];
        $span->generation($generationId);
        $queryText = $this->queryTextExtractor->extract($request);
        if ($this->knownItemGuard->mustUseLexical($queryText, $generation)) {
            $span->route('native', 'known_item', $generationId);
            return $this->nativeAdapter->query($this->candidateRequestFactory->createPage($request));
        }
        $sortPlan = $this->sortEligibility->hybridPlan(
            method_exists($request, 'getSort') ? $request->getSort() : []
        );
        if ($sortPlan === null) {
            throw new \RuntimeException('The search sort is not hybrid compatible.');
        }
        $outOfStockBottom = (bool)$sortPlan['out_of_stock_bottom'];
        $candidateRequest = $this->candidateRequestFactory->create($request);
        $nativeCandidates = $span->measure(
            'native_candidates',
            fn (): \Magento\Framework\Search\Response\QueryResponse =>
                $this->nativeAdapter->query($candidateRequest)
        );
        $nativeDocuments = iterator_to_array($nativeCandidates->getIterator(), false);
        $candidateDocuments = array_slice($nativeDocuments, 0, 100);
        $nativeTail = array_slice($nativeDocuments, 100);
        if ($candidateDocuments === []) {
            $span->route('native', 'no_candidates', $generationId);
            return $nativeCandidates;
        }
        $nativeCandidateScores = $this->scoreExtractor->extract($candidateDocuments);
        $vector = $span->measure(
            'query_encoder',
            fn (): array => $this->queryEncoder->encode($queryText, $generation)
        );
        $hybridRequest = $this->requestBuilder->build(
            $vector,
            $nativeCandidateScores,
            $outOfStockBottom
        );
        $response = $span->measure(
            'hybrid_opensearch',
            fn (): array => $this->transport->search(
                (string)$generation['physical_index'],
                (string)$generation['pipeline_id'],
                $hybridRequest
            )
        );
        $hits = $response['hits']['hits'] ?? null;
        if (!is_array($hits)) {
            throw new \RuntimeException('The hybrid result universe differs from the native candidate universe.');
        }
        $ordered = $this->resultMapper->map($candidateDocuments, $hits, $outOfStockBottom);
        $from = max(0, (int)($request->getFrom() ?? 0));
        $size = max(0, (int)($request->getSize() ?? 0));
        $page = array_slice(array_merge($ordered, $nativeTail), $from, $size);
        $span->route('hybrid', 'eligible', $generationId);

        return new \Magento\Framework\Search\Response\QueryResponse(
            $page,
            $nativeCandidates->getAggregations(),
            $nativeCandidates->getTotal()
        );
    }
}
