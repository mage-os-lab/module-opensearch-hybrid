<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Store;

class ReadinessLatch
{
    private const TABLE = 'mageos_opensearch_hybrid_store_state';

    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection
    ) {
    }

    public function isOpen(int $storeId): bool
    {
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName(self::TABLE);
        $state = $connection->fetchRow(
            $connection->select()->from($table)->where('store_id = ?', $storeId)
        );
        if (!is_array($state) || (string)$state['activation_mode'] !== 'HYBRID') {
            return false;
        }
        $required = (int)$state['required_watermark'];

        return (bool)$state['readiness_latch']
            && (int)$state['native_watermark'] >= $required
            && (int)$state['hybrid_watermark'] >= $required
            && $state['active_generation_id'] !== null;
    }

    public function requireWatermark(int $storeId, int $watermark): void
    {
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName(self::TABLE);
        $watermark = max(0, $watermark);
        $requiredWatermark = sprintf('GREATEST(required_watermark, %d)', $watermark);
        $connection->insertOnDuplicate(
            $table,
            [
                'store_id' => $storeId,
                'required_watermark' => $watermark,
                'readiness_latch' => 0,
                'cache_version' => 1,
            ],
            [
                'required_watermark' => new \Zend_Db_Expr($requiredWatermark),
                'readiness_latch' => $this->readinessExpression(
                    'native_watermark',
                    'hybrid_watermark',
                    $requiredWatermark
                ),
                'cache_version' => new \Zend_Db_Expr('cache_version + 1'),
            ]
        );
    }

    public function advanceNative(int $storeId, int $watermark): void
    {
        $this->advance($storeId, 'native_watermark', $watermark);
    }

    public function advanceHybrid(int $storeId, int $watermark): void
    {
        $this->advance($storeId, 'hybrid_watermark', $watermark);
    }

    private function advance(int $storeId, string $column, int $watermark): void
    {
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName(self::TABLE);
        $watermark = max(0, $watermark);
        $advancedWatermark = sprintf('GREATEST(%s, %d)', $column, $watermark);
        $nativeWatermark = $column === 'native_watermark' ? $advancedWatermark : 'native_watermark';
        $hybridWatermark = $column === 'hybrid_watermark' ? $advancedWatermark : 'hybrid_watermark';
        $connection->update(
            $table,
            [
                $column => new \Zend_Db_Expr($advancedWatermark),
                'readiness_latch' => $this->readinessExpression(
                    $nativeWatermark,
                    $hybridWatermark,
                    'required_watermark'
                ),
                'cache_version' => new \Zend_Db_Expr('cache_version + 1'),
            ],
            ['store_id = ?' => $storeId]
        );
    }

    private function readinessExpression(
        string $nativeWatermark,
        string $hybridWatermark,
        string $requiredWatermark
    ): \Zend_Db_Expr {
        return new \Zend_Db_Expr(sprintf(
            "IF(activation_mode = 'HYBRID' AND active_generation_id IS NOT NULL "
            . 'AND %s >= %s AND %s >= %s, 1, 0)',
            $nativeWatermark,
            $requiredWatermark,
            $hybridWatermark,
            $requiredWatermark
        ));
    }
}
