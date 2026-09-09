<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Change;

class ConfigurationImpactService
{
    private const GENERATION_AFFECTING_PATHS = [
        \MageOS\OpenSearchHybrid\Model\Config::XML_PATH_INCLUDED_STORES,
        \MageOS\OpenSearchHybrid\Model\Config::XML_PATH_ENCODER_ENDPOINT,
        'general/locale/code',
        'currency/options/base',
        'catalog/price/scope',
    ];

    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Change\ScopeInvalidationService $invalidationService,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\SearchableProductResolver $productResolver
    ) {
    }

    public function paths(): array
    {
        return self::GENERATION_AFFECTING_PATHS;
    }

    public function isGenerationAffecting(string $path): bool
    {
        return in_array($path, self::GENERATION_AFFECTING_PATHS, true);
    }

    public function preview(
        string $path,
        string $scope,
        int $scopeId,
        ?string $previousValue = null,
        ?string $proposedValue = null
    ): array {
        $storeIds = $this->affectedStoreIds($path, $scope, $scopeId, $previousValue, $proposedValue);
        $stores = [];
        $estimatedDocuments = 0;
        foreach ($storeIds as $storeId) {
            $documentCount = $this->productResolver->countEligible($storeId);
            $estimatedDocuments += $documentCount;
            $stores[] = [
                'store_id' => $storeId,
                'estimated_documents' => $documentCount,
            ];
        }

        return [
            'path' => $path,
            'scope' => $scope,
            'scope_id' => $scopeId,
            'affected_store_ids' => $storeIds,
            'estimated_documents' => $estimatedDocuments,
            'stores' => $stores,
            'requires_replacement_generation' => $storeIds !== [],
            'activation_required' => $storeIds !== [],
        ];
    }

    public function affectedStoreIds(
        string $path,
        string $scope,
        int $scopeId,
        ?string $previousValue = null,
        ?string $proposedValue = null
    ): array {
        if (!$this->isGenerationAffecting($path)) {
            throw new \InvalidArgumentException('The configuration path is not generation-affecting.');
        }
        if (!in_array($scope, ['default', 'websites', 'stores'], true)
            || ($scope === 'default' && $scopeId !== 0)
            || ($scope !== 'default' && $scopeId <= 0)
        ) {
            throw new \InvalidArgumentException('The configuration scope is invalid.');
        }
        $generationStores = $this->invalidationService->storeIdsWithGenerations();
        if ($path === \MageOS\OpenSearchHybrid\Model\Config::XML_PATH_INCLUDED_STORES) {
            if ($scope !== 'default') {
                throw new \InvalidArgumentException('Included stores are configured only at default scope.');
            }
            if ($previousValue === null || $proposedValue === null) {
                return $generationStores;
            }

            return array_values(array_filter(
                $generationStores,
                fn (int $storeId): bool => $this->isIncluded($previousValue, $storeId)
                    !== $this->isIncluded($proposedValue, $storeId)
            ));
        }
        if ($scope === 'websites') {
            return $this->invalidationService->storeIdsForWebsite($scopeId);
        }
        if ($scope === 'stores') {
            return in_array($scopeId, $generationStores, true) ? [$scopeId] : [];
        }

        return $generationStores;
    }

    private function isIncluded(string $configured, int $storeId): bool
    {
        if (trim($configured) === '') {
            return true;
        }
        $storeIds = array_map('intval', array_filter(array_map('trim', explode(',', $configured))));

        return in_array($storeId, $storeIds, true);
    }
}
