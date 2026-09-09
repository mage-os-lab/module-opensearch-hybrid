<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Generation;

class StoreScopeFingerprint
{
    private const CONFIG_PATHS = [
        'general/locale/code',
        'currency/options/base',
        'catalog/price/scope',
    ];

    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \Magento\Framework\App\Config\ScopeConfigInterface $scopeConfig,
        private readonly \MageOS\OpenSearchHybrid\Model\Config $config,
        private readonly \Magento\Framework\Serialize\Serializer\Json $json
    ) {
    }

    public function digest(int $storeId): string
    {
        $connection = $this->resourceConnection->getConnection();
        $storeTable = $this->resourceConnection->getTableName('store');
        $websiteTable = $this->resourceConnection->getTableName('store_website');
        $groupTable = $this->resourceConnection->getTableName('store_group');
        $scope = $connection->fetchRow(
            $connection->select()
                ->from(['store' => $storeTable], [
                    'store_id',
                    'code',
                    'website_id',
                    'group_id',
                    'is_active',
                ])
                ->joinLeft(
                    ['website' => $websiteTable],
                    'website.website_id = store.website_id',
                    [
                        'website_code' => 'code',
                        'website_default_group_id' => 'default_group_id',
                    ]
                )
                ->joinLeft(
                    ['store_group' => $groupTable],
                    'store_group.group_id = store.group_id',
                    [
                        'group_website_id' => 'website_id',
                        'root_category_id',
                        'default_store_id',
                    ]
                )
                ->where('store.store_id = ?', $storeId)
        );
        if (!is_array($scope)) {
            throw new \Magento\Framework\Exception\NoSuchEntityException(
                __('Store %1 does not exist.', $storeId)
            );
        }
        $configuration = [];
        foreach (self::CONFIG_PATHS as $path) {
            $configuration[$path] = (string)$this->scopeConfig->getValue(
                $path,
                \Magento\Store\Model\ScopeInterface::SCOPE_STORE,
                $storeId
            );
        }
        $configuration[\MageOS\OpenSearchHybrid\Model\Config::XML_PATH_INCLUDED_STORES]
            = $this->config->isStoreIncluded($storeId);
        $configuration[\MageOS\OpenSearchHybrid\Model\Config::XML_PATH_ENCODER_ENDPOINT]
            = trim((string)$this->scopeConfig->getValue(
                \MageOS\OpenSearchHybrid\Model\Config::XML_PATH_ENCODER_ENDPOINT
            ));
        ksort($configuration);

        return hash('sha256', $this->json->serialize([
            'schema_version' => 2,
            'store' => $scope,
            'configuration' => $configuration,
            'inventory' => $this->inventoryScope((string)$scope['website_code']),
        ]));
    }

    private function inventoryScope(string $websiteCode): array
    {
        $connection = $this->resourceConnection->getConnection();
        $stockId = $connection->fetchOne(
            $connection->select()
                ->from(
                    $this->resourceConnection->getTableName('inventory_stock_sales_channel'),
                    ['stock_id']
                )
                ->where('type = ?', 'website')
                ->where('code = ?', $websiteCode)
                ->limit(1)
        );
        if ($stockId === false) {
            return ['stock_id' => null, 'sources' => []];
        }
        $stockId = (int)$stockId;
        $sources = $connection->fetchAll(
            $connection->select()
                ->from(['link' => $this->resourceConnection->getTableName(
                    'inventory_source_stock_link'
                )], ['source_code', 'priority'])
                ->joinLeft(
                    ['source' => $this->resourceConnection->getTableName('inventory_source')],
                    'source.source_code = link.source_code',
                    ['enabled']
                )
                ->where('link.stock_id = ?', $stockId)
                ->order('link.source_code ASC')
        );

        return [
            'stock_id' => $stockId,
            'sources' => array_map(static fn (array $source): array => [
                'source_code' => (string)$source['source_code'],
                'priority' => (int)$source['priority'],
                'enabled' => (bool)$source['enabled'],
            ], $sources),
        ];
    }
}
