<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Embedding;

class EmbeddingStore implements \MageOS\OpenSearchHybrid\Api\EmbeddingStoreInterface
{
    private const TABLE = 'mageos_opensearch_hybrid_embedding';

    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository $generationRepository,
        private readonly \MageOS\OpenSearchHybrid\Model\Vector\VectorValidator $vectorValidator
    ) {
    }

    public function save(
        int $storeId,
        int $productId,
        int $generationId,
        string $sourceHash,
        array $vector,
        int $revision
    ): bool {
        $generation = $this->generationRepository->get($generationId);
        if ((int)$generation['store_id'] !== $storeId) {
            throw new \InvalidArgumentException('The embedding store scope differs from its generation.');
        }
        $dimension = (int)$generation['dimension'];
        $binary = $this->vectorValidator->toFloat32Binary($vector, $dimension);
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName(self::TABLE);
        $connection->beginTransaction();
        try {
            $this->lockGeneration($connection, $generationId);
            $existing = $connection->fetchRow(
                $connection->select()
                    ->from($table, ['embedding_id', 'revision', 'source_hash', 'status'])
                    ->where('store_id = ?', $storeId)
                    ->where('product_id = ?', $productId)
                    ->where('generation_id = ?', $generationId)
                    ->forUpdate(true)
            );
            if (is_array($existing) && (int)$existing['revision'] > $revision) {
                $connection->commit();
                return false;
            }
            if (is_array($existing)
                && (int)$existing['revision'] === $revision
                && (string)$existing['status'] === 'DELETED'
            ) {
                $connection->commit();
                return false;
            }
            if (is_array($existing) && (int)$existing['revision'] === $revision
                && !hash_equals((string)$existing['source_hash'], $sourceHash)
            ) {
                throw new \RuntimeException('Equal embedding revisions have different source hashes.');
            }
            $row = [
                'store_id' => $storeId,
                'product_id' => $productId,
                'generation_id' => $generationId,
                'model_id' => (string)$generation['model_id'],
                'model_revision' => (string)$generation['model_revision'],
                'dimension' => $dimension,
                'recipe_version' => (string)$generation['recipe_version'],
                'source_hash' => $sourceHash,
                'vector_hash' => hash('sha256', $binary),
                'vector' => $binary,
                'status' => 'COMPLETE',
                'revision' => $revision,
                'last_error_class' => null,
                'last_diagnostic' => null,
                'completed_at' => new \Zend_Db_Expr('UTC_TIMESTAMP()'),
            ];
            if (is_array($existing)) {
                $connection->update($table, $row, ['embedding_id = ?' => (int)$existing['embedding_id']]);
                $wasComplete = (string)$existing['status'] === 'COMPLETE';
            } else {
                $connection->insert($table, $row);
                $wasComplete = false;
            }
            if (!$wasComplete) {
                $connection->update(
                    $this->resourceConnection->getTableName('mageos_opensearch_hybrid_generation'),
                    ['coverage_complete' => new \Zend_Db_Expr('coverage_complete + 1')],
                    ['generation_id = ?' => $generationId]
                );
            }
            $connection->commit();
            return true;
        } catch (\Throwable $throwable) {
            $connection->rollBack();
            throw $throwable;
        }
    }

    public function delete(int $storeId, int $productId, int $generationId, int $revision): bool
    {
        $generation = $this->generationRepository->get($generationId);
        if ((int)$generation['store_id'] !== $storeId) {
            throw new \InvalidArgumentException('The embedding tombstone scope differs from its generation.');
        }
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName(self::TABLE);
        $connection->beginTransaction();
        try {
            $this->lockGeneration($connection, $generationId);
            $existing = $connection->fetchRow(
                $connection->select()
                    ->from($table, ['embedding_id', 'revision', 'status'])
                    ->where('store_id = ?', $storeId)
                    ->where('product_id = ?', $productId)
                    ->where('generation_id = ?', $generationId)
                    ->forUpdate(true)
            );
            if (is_array($existing) && (int)$existing['revision'] > $revision) {
                $connection->commit();
                return false;
            }
            if (is_array($existing)
                && (int)$existing['revision'] === $revision
                && (string)$existing['status'] === 'DELETED'
            ) {
                $connection->commit();
                return true;
            }
            $row = [
                'store_id' => $storeId,
                'product_id' => $productId,
                'generation_id' => $generationId,
                'model_id' => (string)$generation['model_id'],
                'model_revision' => (string)$generation['model_revision'],
                'dimension' => (int)$generation['dimension'],
                'recipe_version' => (string)$generation['recipe_version'],
                'source_hash' => hash('sha256', 'DELETED'),
                'vector_hash' => null,
                'vector' => null,
                'status' => 'DELETED',
                'revision' => $revision,
                'last_error_class' => null,
                'last_diagnostic' => null,
                'completed_at' => new \Zend_Db_Expr('UTC_TIMESTAMP()'),
            ];
            $wasComplete = is_array($existing) && (string)$existing['status'] === 'COMPLETE';
            if (is_array($existing)) {
                $connection->update($table, $row, ['embedding_id = ?' => (int)$existing['embedding_id']]);
            } else {
                $connection->insert($table, $row);
            }
            if ($wasComplete) {
                $connection->update(
                    $this->resourceConnection->getTableName('mageos_opensearch_hybrid_generation'),
                    [
                        'coverage_complete' => new \Zend_Db_Expr(
                            'IF(coverage_complete > 0, coverage_complete - 1, 0)'
                        ),
                    ],
                    ['generation_id = ?' => $generationId]
                );
            }
            $connection->commit();
            return true;
        } catch (\Throwable $throwable) {
            $connection->rollBack();
            throw $throwable;
        }
    }

    private function lockGeneration(
        \Magento\Framework\DB\Adapter\AdapterInterface $connection,
        int $generationId
    ): void {
        $table = $this->resourceConnection->getTableName('mageos_opensearch_hybrid_generation');
        $lockedGenerationId = $connection->fetchOne(
            $connection->select()
                ->from($table, ['generation_id'])
                ->where('generation_id = ?', $generationId)
                ->forUpdate(true)
        );
        if ((int)$lockedGenerationId !== $generationId) {
            throw new \RuntimeException('The embedding generation disappeared during persistence.');
        }
    }

    public function getVectorForSource(
        int $storeId,
        int $productId,
        int $generationId,
        string $sourceHash
    ): ?array
    {
        $connection = $this->resourceConnection->getConnection();
        $table = $this->resourceConnection->getTableName(self::TABLE);
        $row = $connection->fetchRow(
            $connection->select()
                ->from($table, ['dimension', 'vector'])
                ->where('store_id = ?', $storeId)
                ->where('product_id = ?', $productId)
                ->where('generation_id = ?', $generationId)
                ->where('source_hash = ?', $sourceHash)
                ->where('status = ?', 'COMPLETE')
        );
        if (!is_array($row) || !is_string($row['vector'])) {
            return null;
        }
        $dimension = (int)$row['dimension'];
        $values = unpack('g' . $dimension, $row['vector']);
        if (!is_array($values) || count($values) !== $dimension) {
            throw new \RuntimeException('The stored embedding binary is invalid.');
        }

        return array_values(array_map('floatval', $values));
    }
}
