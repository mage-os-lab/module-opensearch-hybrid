<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Generation;

class BuildServiceTest extends \PHPUnit\Framework\TestCase
{
    public function testProductionPlanReportsCompleteAndInitiallyAdmittedWorkWithoutWriting(): void
    {
        $expectedIdentity = str_repeat('a', 64);
        $generationSelect = $this->createStub(\Magento\Framework\DB\Select::class);
        $generationSelect->method('from')->willReturnSelf();
        $generationSelect->method('where')->willReturnSelf();
        $generationSelect->method('order')->willReturnSelf();
        $stateSelect = $this->createStub(\Magento\Framework\DB\Select::class);
        $stateSelect->method('from')->willReturnSelf();
        $stateSelect->method('where')->willReturnSelf();
        $connection = $this->createMock(\Magento\Framework\DB\Adapter\AdapterInterface::class);
        $connection->expects($this->exactly(2))
            ->method('select')
            ->willReturnOnConsecutiveCalls($generationSelect, $stateSelect);
        $connection->expects($this->once())->method('fetchAll')->with($generationSelect)->willReturn([
            [
                'generation_id' => '3',
                'state' => 'RETAINED',
                'physical_index' => 'mageos-opensearch-hybrid-s1-g3-contract',
                'is_fake' => '0',
            ],
            [
                'generation_id' => '4',
                'state' => 'ACTIVE',
                'physical_index' => 'mageos-opensearch-hybrid-s1-g4-contract',
                'is_fake' => '0',
            ],
        ]);
        $connection->expects($this->once())->method('fetchRow')->with($stateSelect)->willReturn([
            'activation_mode' => 'HYBRID',
            'active_generation_id' => '4',
        ]);
        $connection->expects($this->never())->method('insert');
        $connection->expects($this->never())->method('insertMultiple');
        $connection->expects($this->never())->method('update');
        $connection->expects($this->never())->method('delete');

        $resourceConnection = $this->createStub(\Magento\Framework\App\ResourceConnection::class);
        $resourceConnection->method('getConnection')->willReturn($connection);
        $resourceConnection->method('getTableName')->willReturnArgument(0);
        $config = $this->createMock(\MageOS\OpenSearchHybrid\Model\Config::class);
        $config->expects($this->once())->method('isStoreIncluded')->with(1)->willReturn(true);
        $config->expects($this->once())->method('encoderEndpoint')->willReturn('http://encoder:8080');
        $config->expects($this->once())->method('batchSize')->willReturn(16);
        $config->expects($this->once())->method('maxOutstandingBatches')->willReturn(8);
        $identityClient = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Encoder\EncoderIdentityClient::class
        );
        $identityClient->expects($this->once())
            ->method('fetchAt')
            ->with('http://encoder:8080', $expectedIdentity)
            ->willReturn([
                'model_id' => 'Snowflake/snowflake-arctic-embed-m-v2.0',
                'model_revision' => 'revision',
                'dimension' => 256,
                'recipe_version' => 'mageos-v1',
                'encoder_identity_digest' => $expectedIdentity,
                'architecture' => 'linux/amd64',
                'deployment' => ['artifact_digest' => 'sha256:' . str_repeat('b', 64)],
            ]);
        $indexManager = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\OpenSearch\IndexManager::class
        );
        $indexManager->expects($this->once())->method('assertSupportedVersion');
        $resolver = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Change\SearchableProductResolver::class
        );
        $resolver->expects($this->once())->method('countEligible')->with(1)->willReturn(50_000);
        $contract = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract::class
        );
        $contract->method('digest')->willReturn(str_repeat('c', 64));
        $contract->method('resultContractDigest')->willReturn(str_repeat('d', 64));
        $contract->method('mappingDigest')->willReturn(str_repeat('e', 64));
        $contract->method('pipelineDigest')->willReturn(str_repeat('f', 64));
        $contract->method('pipelineId')->willReturn('mageos-opensearch-hybrid-pipeline');
        $json = $this->createMock(\Magento\Framework\Serialize\Serializer\Json::class);
        $json->expects($this->once())
            ->method('serialize')
            ->with(self::callback(static fn (array $preview): bool =>
                ($preview['operation'] ?? null) === 'build_production_generation'
                && !array_key_exists('confirmation_token', $preview)
            ))
            ->willReturn('canonical-production-build-preview');
        $scopeFingerprint = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Generation\StoreScopeFingerprint::class
        );
        $scopeFingerprint->expects($this->once())
            ->method('digest')
            ->with(1)
            ->willReturn(str_repeat('1', 64));

        $service = new \MageOS\OpenSearchHybrid\Model\Generation\BuildService(
            $resourceConnection,
            $contract,
            $config,
            $scopeFingerprint,
            $indexManager,
            $this->createStub(\MageOS\OpenSearchHybrid\Model\Generation\BuildRefreshService::class),
            $this->createStub(\MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository::class),
            $resolver,
            $this->createStub(\MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository::class),
            $this->createStub(\MageOS\OpenSearchHybrid\Model\AsyncEvent\LifecyclePublisher::class),
            $this->createStub(
                \MageOS\OpenSearchHybrid\Model\AsyncEvent\SubscriptionReconciler::class
            ),
            $identityClient,
            $this->createStub(\Psr\Log\LoggerInterface::class),
            $json
        );

        $plan = $service->planProduction(1, $expectedIdentity);

        self::assertSame('preview', $plan['mode']);
        self::assertSame('none', $plan['mutation']);
        self::assertSame(50_000, $plan['catalog']['eligible_products']);
        self::assertSame(3_125, $plan['workload']['expected_total_embedding_batches']);
        self::assertSame(8, $plan['workload']['initial_outbox_jobs_without_consumer_progress']);
        self::assertSame(128, $plan['workload']['initial_outbox_items_without_consumer_progress']);
        self::assertSame(51_200_000, $plan['vector_payload']['raw_float32_bytes']);
        self::assertSame(68_400_000, $plan['vector_payload']['base64_bytes']);
        self::assertSame(2, $plan['current_generations']['count']);
        self::assertSame('RETAINED', $plan['current_generations']['items'][0]['state']);
        self::assertSame(0, $plan['activation']['changes']);
        self::assertSame('HYBRID', $plan['activation']['current_mode']);
        self::assertSame(4, $plan['activation']['current_generation_id']);
        self::assertSame('HYBRID', $plan['activation']['planned_mode']);
        self::assertSame(4, $plan['activation']['planned_generation_id']);
        self::assertSame(str_repeat('1', 64), $plan['scope']['digest']);
        self::assertSame(
            'build-production-1-' . hash('sha256', 'canonical-production-build-preview'),
            $plan['confirmation_token']
        );
    }
}
