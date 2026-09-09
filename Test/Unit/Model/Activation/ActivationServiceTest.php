<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Activation;

class ActivationServiceTest extends \PHPUnit\Framework\TestCase
{
    #[\PHPUnit\Framework\Attributes\DataProvider('operations')]
    public function testConfirmationIsBoundToLockedState(string $operation, bool $changed, bool $firstActivation = false): void
    {
        $state = [
            'activation_mode' => 'HYBRID', 'active_generation_id' => 10,
            'required_watermark' => 0, 'native_watermark' => 0, 'hybrid_watermark' => 0,
            'readiness_latch' => 1, 'cache_version' => 1, 'updated_at' => '2026-09-09 12:00:00',
        ];
        if ($firstActivation) {
            $state = false;
        }
        $generation = [
            'generation_id' => 20, 'store_id' => 1,
            'state' => $operation === 'activate' ? 'READY' : 'RETAINED', 'is_fake' => 0,
            'coverage_total' => 1, 'coverage_complete' => 1, 'coverage_failed' => 0,
            'captured_change_id' => 0, 'result_contract_accepted' => 1,
            'validation_report' => '{"ready":true}', 'contract_digest' => 'c',
            'result_contract_digest' => 'r', 'mapping_digest' => 'm', 'pipeline_digest' => 'p',
            'physical_index' => 'fixture_generation_20', 'model_id' => 'fixture',
            'model_revision' => 'fixture', 'encoder_identity_digest' => 'fixture', 'scope_digest' => 's',
            'updated_at' => '2026-09-09 12:00:00',
        ];
        $table = '';
        $select = $this->createStub(\Magento\Framework\DB\Select::class);
        $select->method('from')->willReturnCallback(static function ($from) use (&$table, $select) {
            $table = $from;
            return $select;
        });
        foreach (['where', 'forUpdate', 'order'] as $method) {
            $select->method($method)->willReturnSelf();
        }
        $connection = $changed
            ? $this->createMock(\Magento\Framework\DB\Adapter\Pdo\Mysql::class)
            : $this->createStub(\Magento\Framework\DB\Adapter\Pdo\Mysql::class);
        $connection->method('select')->willReturn($select);
        $connection->method('fetchRow')->willReturnCallback(
            static function () use (&$table, &$state, $generation): array|false {
                return str_ends_with($table, '_generation') ? $generation : $state;
            }
        );
        $connection->method('fetchAll')->willReturn([$generation]);
        $connection->method('fetchOne')->willReturn(0);
        $connection->method('beginTransaction')->willReturnCallback(static function () use (&$state, $changed) {
            if ($changed) {
                $state['active_generation_id'] = 11;
                $state['cache_version'] = 2;
            }
        });
        $connection->method('lastInsertId')->willReturn('1');
        if ($changed) {
            $connection->expects($this->never())->method('update');
            $connection->expects($this->never())->method('insert');
            $connection->expects($this->never())->method('insertOnDuplicate');
            $connection->expects($this->once())->method('rollBack');
        }
        $resource = $this->createStub(\Magento\Framework\App\ResourceConnection::class);
        $resource->method('getConnection')->willReturn($connection);
        $resource->method('getTableName')->willReturnArgument(0);
        $generations = $this->createStub(\MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository::class);
        $generations->method('get')->willReturnCallback(static fn (int $id): array => array_replace(
            $generation, ['generation_id' => $id, 'state' => $id === 20 ? $generation['state'] : 'ACTIVE']
        ));
        $contract = $this->createStub(\MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract::class);
        foreach (['digest' => 'c', 'resultContractDigest' => 'r', 'mappingDigest' => 'm', 'pipelineDigest' => 'p'] as $method => $value) {
            $contract->method($method)->willReturn($value);
        }
        $scope = $this->createStub(\MageOS\OpenSearchHybrid\Model\Generation\StoreScopeFingerprint::class);
        $scope->method('digest')->willReturn('s');
        $index = $this->createStub(\MageOS\OpenSearchHybrid\Model\OpenSearch\IndexManager::class);
        $index->method('validateGeneration')->willReturn([
            'index_exists' => true, 'mapping' => true, 'settings' => true, 'pipeline' => true,
            'document_identity' => true, 'refresh_interval_restored' => true, 'index_count' => 1,
        ]);
        $service = new \MageOS\OpenSearchHybrid\Model\Activation\ActivationService(
            $resource, $generations, $contract, $scope,
            $this->createStub(\MageOS\OpenSearchHybrid\Model\Change\ChangeJournalRepository::class),
            $index, new \Magento\Framework\Serialize\Serializer\Json()
        );
        $preview = match ($operation) {
            'activate' => $service->previewActivation(1, 20),
            'native' => $service->previewNativeRollback(1),
            'retained' => $service->previewGenerationRollback(1, 20),
        };
        if ($changed) {
            $this->expectException(\InvalidArgumentException::class);
            $this->expectExceptionMessage('confirmation does not match');
        }
        $token = $preview['confirmation_token'];
        $result = match ($operation) {
            'activate' => $service->activateConfirmed(1, 20, $token, 'test'),
            'native' => $service->rollbackToNativeConfirmed(1, $token, 'test'),
            'retained' => $service->rollbackToGenerationConfirmed(1, 20, $token, 'test'),
        };
        self::assertSame($firstActivation ? null : 10, $result['prior_generation_id']);
    }

    public static function operations(): array
    {
        return [
            ['activate', true], ['native', true], ['retained', true],
            ['activate', false], ['native', false], ['retained', false],
            ['activate', false, true],
        ];
    }
}
