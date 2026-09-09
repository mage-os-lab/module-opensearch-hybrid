<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Embedding;

class EmbeddingStoreTest extends \PHPUnit\Framework\TestCase
{
    public function testSaveLocksGenerationBeforeEmbeddingWrite(): void
    {
        $events = [];
        $select = $this->select();
        $connection = $this->connection($select, $events, false);
        $connection->expects($this->once())
            ->method('insert')
            ->willReturnCallback(static function () use (&$events): int {
                $events[] = 'embedding_write';
                return 1;
            });
        $connection->expects($this->once())
            ->method('update')
            ->willReturnCallback(static function () use (&$events): int {
                $events[] = 'coverage_write';
                return 1;
            });

        $saved = $this->store($connection)->save(
            1,
            42,
            5,
            str_repeat('a', 64),
            array_fill(0, 256, 0.0),
            0
        );

        self::assertTrue($saved);
        self::assertSame(
            ['generation_lock', 'embedding_lock', 'embedding_write', 'coverage_write'],
            $events
        );
    }

    public function testDeleteLocksGenerationBeforeEmbeddingWrite(): void
    {
        $events = [];
        $select = $this->select();
        $connection = $this->connection($select, $events, [
            'embedding_id' => '9',
            'revision' => '0',
            'status' => 'COMPLETE',
        ]);
        $connection->expects($this->exactly(2))
            ->method('update')
            ->willReturnCallback(static function (string $table) use (&$events): int {
                $events[] = $table === 'mageos_opensearch_hybrid_embedding'
                    ? 'embedding_write'
                    : 'coverage_write';
                return 1;
            });

        $deleted = $this->store($connection)->delete(1, 42, 5, 1);

        self::assertTrue($deleted);
        self::assertSame(
            ['generation_lock', 'embedding_lock', 'embedding_write', 'coverage_write'],
            $events
        );
    }

    private function connection(
        \Magento\Framework\DB\Select $select,
        array &$events,
        array|false $existing
    ): \Magento\Framework\DB\Adapter\AdapterInterface {
        $connection = $this->createMock(\Magento\Framework\DB\Adapter\AdapterInterface::class);
        $connection->expects($this->exactly(2))->method('select')->willReturn($select);
        $connection->expects($this->once())
            ->method('fetchOne')
            ->with($select)
            ->willReturnCallback(static function () use (&$events): string {
                $events[] = 'generation_lock';
                return '5';
            });
        $connection->expects($this->once())
            ->method('fetchRow')
            ->with($select)
            ->willReturnCallback(static function () use (&$events, $existing): array|false {
                $events[] = 'embedding_lock';
                return $existing;
            });

        return $connection;
    }

    private function select(): \Magento\Framework\DB\Select
    {
        $select = $this->createStub(\Magento\Framework\DB\Select::class);
        $select->method('from')->willReturnSelf();
        $select->method('where')->willReturnSelf();
        $select->method('forUpdate')->willReturnSelf();

        return $select;
    }

    private function store(
        \Magento\Framework\DB\Adapter\AdapterInterface $connection
    ): \MageOS\OpenSearchHybrid\Model\Embedding\EmbeddingStore {
        $resourceConnection = $this->createStub(\Magento\Framework\App\ResourceConnection::class);
        $resourceConnection->method('getConnection')->willReturn($connection);
        $resourceConnection->method('getTableName')->willReturnArgument(0);
        $generationRepository = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository::class
        );
        $generationRepository->method('get')->willReturn([
            'generation_id' => '5',
            'store_id' => '1',
            'model_id' => 'model',
            'model_revision' => 'revision',
            'dimension' => '256',
            'recipe_version' => 'recipe',
        ]);
        $vectorValidator = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Vector\VectorValidator::class
        );
        $vectorValidator->method('toFloat32Binary')->willReturn(str_repeat("\0", 1024));

        return new \MageOS\OpenSearchHybrid\Model\Embedding\EmbeddingStore(
            $resourceConnection,
            $generationRepository,
            $vectorValidator
        );
    }
}
