<?php

declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Plugin;

class ConfigurationChangeCapture
{
    private \WeakMap $saveSnapshots;

    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Change\ScopeInvalidationService $invalidationService,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\ConfigurationImpactService $impactService,
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \Magento\Framework\App\Config\ScopeConfigInterface $scopeConfig
    ) {
        $this->saveSnapshots = new \WeakMap();
    }

    public function beforeSave(
        \Magento\Config\Model\ResourceModel\Config\Data $subject,
        \Magento\Framework\Model\AbstractModel $value
    ): void {
        unset($subject);
        $path = (string)$value->getPath();
        if (!$this->impactService->isGenerationAffecting($path)) {
            return;
        }
        $scope = (string)$value->getScope();
        $scopeId = (int)$value->getScopeId();
        $row = $this->resourceConnection->getConnection()->fetchRow(
            $this->resourceConnection->getConnection()->select()
                ->from($this->resourceConnection->getTableName('core_config_data'), ['value'])
                ->where('scope = ?', $scope)
                ->where('scope_id = ?', $scopeId)
                ->where('path = ?', $path)
                ->limit(1)
        );
        $this->saveSnapshots[$value] = [
            'effective_value' => is_array($row)
                ? (string)$row['value']
                : (string)$this->scopeConfig->getValue($path, $this->scopeType($scope), $scopeId),
        ];
    }

    public function afterSave(
        \Magento\Config\Model\ResourceModel\Config\Data $subject,
        \Magento\Config\Model\ResourceModel\Config\Data $result,
        \Magento\Framework\Model\AbstractModel $value
    ): \Magento\Config\Model\ResourceModel\Config\Data {
        unset($subject);
        $path = (string)$value->getPath();
        if (!$this->impactService->isGenerationAffecting($path)) {
            return $result;
        }
        $snapshot = $this->saveSnapshots[$value] ?? null;
        unset($this->saveSnapshots[$value]);
        if (!is_array($snapshot)) {
            return $result;
        }
        $previousValue = (string)$snapshot['effective_value'];
        $proposedValue = (string)$value->getValue();
        if (hash_equals($previousValue, $proposedValue)) {
            return $result;
        }
        $storeIds = $this->impactService->affectedStoreIds(
            $path,
            (string)$value->getScope(),
            (int)$value->getScopeId(),
            $previousValue,
            $proposedValue
        );
        $this->invalidationService->invalidateStores(
            $storeIds,
            'CONFIGURATION',
            (int)sprintf('%u', crc32($path)),
            'configuration_save:' . $path
        );

        return $result;
    }

    public function afterDelete(
        \Magento\Config\Model\ResourceModel\Config\Data $subject,
        \Magento\Config\Model\ResourceModel\Config\Data $result,
        \Magento\Framework\Model\AbstractModel $value
    ): \Magento\Config\Model\ResourceModel\Config\Data {
        unset($subject);
        $path = (string)$value->getPath();
        if (!$this->impactService->isGenerationAffecting($path)) {
            return $result;
        }
        $this->invalidationService->invalidateStores(
            $this->impactService->affectedStoreIds(
                $path,
                (string)$value->getScope(),
                (int)$value->getScopeId()
            ),
            'CONFIGURATION',
            (int)sprintf('%u', crc32($path)),
            'configuration_delete:' . $path
        );

        return $result;
    }

    private function scopeType(string $scope): string
    {
        return match ($scope) {
            'websites' => \Magento\Store\Model\ScopeInterface::SCOPE_WEBSITE,
            'stores' => \Magento\Store\Model\ScopeInterface::SCOPE_STORE,
            default => \Magento\Framework\App\Config\ScopeConfigInterface::SCOPE_TYPE_DEFAULT,
        };
    }
}
