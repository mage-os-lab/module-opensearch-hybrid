<?php

declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Change;

class ConfigurationSaveConfirmation
{
    private const FIELD_PATHS = [
        'catalog' => [
            'price' => ['scope' => 'catalog/price/scope'],
        ],
        'currency' => [
            'options' => ['base' => 'currency/options/base'],
        ],
        'general' => [
            'locale' => ['code' => 'general/locale/code'],
        ],
        'mageos_opensearch_hybrid' => [
            'encoder' => ['endpoint' => \MageOS\OpenSearchHybrid\Model\Config::XML_PATH_ENCODER_ENDPOINT],
            'general' => [
                'included_stores' => \MageOS\OpenSearchHybrid\Model\Config::XML_PATH_INCLUDED_STORES,
            ],
        ],
    ];

    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \Magento\Framework\App\Config\ScopeConfigInterface $scopeConfig,
        private readonly \Magento\Store\Model\StoreManagerInterface $storeManager,
        private readonly \Magento\Framework\Serialize\Serializer\Json $json,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\ConfigurationImpactService $impactService
    ) {
    }

    public function isSectionSupported(string $section): bool
    {
        return isset(self::FIELD_PATHS[$section]);
    }

    public function preview(\Magento\Config\Model\Config $config): array
    {
        $section = trim((string)$config->getSection());
        if (!$this->isSectionSupported($section)) {
            return $this->emptyPreview($section);
        }
        [$scope, $scopeId] = $this->scope($config);
        $groups = $config->getGroups();
        if (!is_array($groups)) {
            $groups = [];
        }
        $changes = [];
        $stores = [];
        foreach (self::FIELD_PATHS[$section] as $group => $fields) {
            foreach ($fields as $field => $path) {
                $fieldData = $groups[$group]['fields'][$field] ?? null;
                if (!is_array($fieldData)) {
                    continue;
                }
                $change = $this->change($path, $scope, $scopeId, $fieldData);
                if ($change === null) {
                    continue;
                }
                $impact = $this->impactService->preview(
                    $path,
                    $scope,
                    $scopeId,
                    $change['previous_effective'],
                    $change['proposed_effective']
                );
                $affectedStoreIds = array_map('intval', $impact['affected_store_ids']);
                $changes[] = [
                    'path' => $path,
                    'scope' => $scope,
                    'scope_id' => $scopeId,
                    'mode' => $change['mode'],
                    'previous_digest' => $change['previous_digest'],
                    'proposed_digest' => $change['proposed_digest'],
                    'affected_store_ids' => $affectedStoreIds,
                ];
                foreach ($impact['stores'] as $store) {
                    $stores[(int)$store['store_id']] = [
                        'store_id' => (int)$store['store_id'],
                        'estimated_documents' => (int)$store['estimated_documents'],
                    ];
                }
            }
        }
        ksort($stores, SORT_NUMERIC);
        usort($changes, static fn (array $left, array $right): int => $left['path'] <=> $right['path']);
        $payload = [
            'schema_version' => 1,
            'section' => $section,
            'scope' => $scope,
            'scope_id' => $scopeId,
            'changes' => $changes,
            'affected_store_ids' => array_keys($stores),
            'estimated_documents' => array_sum(array_column($stores, 'estimated_documents')),
            'stores' => array_values($stores),
        ];
        $payload['requires_confirmation'] = $payload['affected_store_ids'] !== [];
        $payload['human_confirmation'] = $payload['requires_confirmation'] ? 'apply' : null;
        $payload['confirmation_token'] = $payload['requires_confirmation']
            ? 'config-save-' . hash('sha256', $this->json->serialize($payload))
            : null;

        return $payload;
    }

    public function assertConfirmed(
        \Magento\Config\Model\Config $config,
        mixed $confirmationToken,
        mixed $humanConfirmation
    ): void {
        $preview = $this->preview($config);
        if (!(bool)$preview['requires_confirmation']) {
            return;
        }
        if (
            !is_string($humanConfirmation)
            || !hash_equals((string)$preview['human_confirmation'], trim($humanConfirmation))
        ) {
            throw new \Magento\Framework\Exception\LocalizedException(
                __('Type apply after reviewing the current OpenSearch Hybrid generation impact.')
            );
        }
        if (
            !is_string($confirmationToken)
            || !hash_equals((string)$preview['confirmation_token'], trim($confirmationToken))
        ) {
            throw new \Magento\Framework\Exception\LocalizedException(
                __('Refresh the OpenSearch Hybrid impact preview before saving this configuration.')
            );
        }
    }

    private function change(string $path, string $scope, int $scopeId, array $fieldData): ?array
    {
        $connection = $this->resourceConnection->getConnection();
        $row = $connection->fetchRow(
            $connection->select()
                ->from($this->resourceConnection->getTableName('core_config_data'), ['value'])
                ->where('scope = ?', $scope)
                ->where('scope_id = ?', $scopeId)
                ->where('path = ?', $path)
                ->limit(1)
        );
        $hasRawValue = is_array($row);
        $rawValue = $hasRawValue ? (string)$row['value'] : null;
        $inherit = !empty($fieldData['inherit']);
        $proposedRawValue = $inherit ? null : $this->normalizeValue($fieldData['value'] ?? '');
        if (
            ($inherit && !$hasRawValue)
            || (!$inherit && $hasRawValue && hash_equals((string)$rawValue, $proposedRawValue))
        ) {
            return null;
        }
        $previousEffective = $this->effectiveValue($path, $scope, $scopeId);
        $proposedEffective = $inherit
            ? $this->parentEffectiveValue($path, $scope, $scopeId)
            : $proposedRawValue;

        return [
            'mode' => $inherit ? 'inherit' : 'value',
            'previous_effective' => $previousEffective,
            'proposed_effective' => $proposedEffective,
            'previous_digest' => hash('sha256', $this->json->serialize([
                'exists' => $hasRawValue,
                'value' => $rawValue,
            ])),
            'proposed_digest' => hash('sha256', $this->json->serialize([
                'exists' => !$inherit,
                'value' => $proposedRawValue,
            ])),
        ];
    }

    private function scope(\Magento\Config\Model\Config $config): array
    {
        $storeCode = trim((string)$config->getStore());
        if ($storeCode !== '') {
            return ['stores', (int)$this->storeManager->getStore($storeCode)->getId()];
        }
        $websiteCode = trim((string)$config->getWebsite());
        if ($websiteCode !== '') {
            return ['websites', (int)$this->storeManager->getWebsite($websiteCode)->getId()];
        }

        return ['default', 0];
    }

    private function effectiveValue(string $path, string $scope, int $scopeId): string
    {
        $scopeType = match ($scope) {
            'websites' => \Magento\Store\Model\ScopeInterface::SCOPE_WEBSITE,
            'stores' => \Magento\Store\Model\ScopeInterface::SCOPE_STORE,
            default => \Magento\Framework\App\Config\ScopeConfigInterface::SCOPE_TYPE_DEFAULT,
        };

        return (string)$this->scopeConfig->getValue($path, $scopeType, $scopeId);
    }

    private function parentEffectiveValue(string $path, string $scope, int $scopeId): string
    {
        if ($scope === 'stores') {
            $websiteId = (int)$this->storeManager->getStore($scopeId)->getWebsiteId();
            return (string)$this->scopeConfig->getValue(
                $path,
                \Magento\Store\Model\ScopeInterface::SCOPE_WEBSITE,
                $websiteId
            );
        }

        return (string)$this->scopeConfig->getValue(
            $path,
            \Magento\Framework\App\Config\ScopeConfigInterface::SCOPE_TYPE_DEFAULT,
            0
        );
    }

    private function normalizeValue(mixed $value): string
    {
        if (is_array($value)) {
            return implode(',', array_map('strval', $value));
        }

        return (string)$value;
    }

    private function emptyPreview(string $section): array
    {
        return [
            'schema_version' => 1,
            'section' => $section,
            'scope' => 'default',
            'scope_id' => 0,
            'changes' => [],
            'affected_store_ids' => [],
            'estimated_documents' => 0,
            'stores' => [],
            'requires_confirmation' => false,
            'human_confirmation' => null,
            'confirmation_token' => null,
        ];
    }
}
