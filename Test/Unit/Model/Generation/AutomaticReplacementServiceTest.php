<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Generation;

class AutomaticReplacementServiceTest extends \PHPUnit\Framework\TestCase
{
    public function testCreatesProductionReplacementWithActiveGenerationIdentity(): void
    {
        $digest = str_repeat('a', 64);
        $repository = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Generation\ReplacementCandidateRepository::class
        );
        $repository->method('storeIds')->with(null)->willReturn([2]);
        $repository->expects($this->once())
            ->method('withinStoreLock')
            ->with(2, $this->isCallable())
            ->willReturnCallback(static fn (int $storeId, callable $operation): array => $operation());
        $repository->method('assessment')->with(2)->willReturn([
            'eligible' => true,
            'reason' => 'pending_configuration_invalidation',
            'invalidation_boundary' => 41,
            'encoder_identity_digest' => $digest,
        ]);
        $config = $this->createMock(\MageOS\OpenSearchHybrid\Model\Config::class);
        $config->expects($this->once())->method('isEnabled')->with(2)->willReturn(true);
        $config->expects($this->once())->method('isStoreIncluded')->with(2)->willReturn(true);
        $buildService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Generation\BuildService::class
        );
        $buildService->expects($this->once())->method('buildProduction')->with(2, $digest)->willReturn(9);
        $service = new \MageOS\OpenSearchHybrid\Model\Generation\AutomaticReplacementService(
            $repository,
            $buildService,
            $config,
            $this->createStub(\Psr\Log\LoggerInterface::class)
        );

        self::assertSame([[
            'store_id' => 2,
            'status' => 'CREATED',
            'reason' => 'pending_configuration_invalidation',
            'invalidation_boundary' => 41,
            'generation_id' => 9,
        ]], $service->createPending());
    }

    public function testDisabledStoreRemainsFailClosedWithoutCreatingWork(): void
    {
        $repository = $this->eligibleRepository();
        $config = $this->createMock(\MageOS\OpenSearchHybrid\Model\Config::class);
        $config->expects($this->once())->method('isEnabled')->with(2)->willReturn(false);
        $config->expects($this->never())->method('isStoreIncluded');
        $buildService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Generation\BuildService::class
        );
        $buildService->expects($this->never())->method('buildProduction');
        $service = new \MageOS\OpenSearchHybrid\Model\Generation\AutomaticReplacementService(
            $repository,
            $buildService,
            $config,
            $this->createStub(\Psr\Log\LoggerInterface::class)
        );

        self::assertSame('module_disabled', $service->createPending()[0]['reason']);
    }

    public function testAlreadyCoveredInvalidationDoesNotReachConfigurationOrBuilder(): void
    {
        $repository = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Generation\ReplacementCandidateRepository::class
        );
        $repository->method('storeIds')->willReturn([2]);
        $repository->method('withinStoreLock')
            ->willReturnCallback(static fn (int $storeId, callable $operation): array => $operation());
        $repository->method('assessment')->willReturn([
            'eligible' => false,
            'reason' => 'replacement_already_created',
            'invalidation_boundary' => 41,
        ]);
        $config = $this->createMock(\MageOS\OpenSearchHybrid\Model\Config::class);
        $config->expects($this->never())->method('isEnabled');
        $buildService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Generation\BuildService::class
        );
        $buildService->expects($this->never())->method('buildProduction');
        $service = new \MageOS\OpenSearchHybrid\Model\Generation\AutomaticReplacementService(
            $repository,
            $buildService,
            $config,
            $this->createStub(\Psr\Log\LoggerInterface::class)
        );

        self::assertSame('replacement_already_created', $service->createPending()[0]['reason']);
    }

    public function testBuildFailureEmitsOnlyBoundedErrorClass(): void
    {
        $repository = $this->eligibleRepository();
        $config = $this->createStub(\MageOS\OpenSearchHybrid\Model\Config::class);
        $config->method('isEnabled')->willReturn(true);
        $config->method('isStoreIncluded')->willReturn(true);
        $buildService = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Generation\BuildService::class
        );
        $buildService->method('buildProduction')->willThrowException(
            new \RuntimeException('secret endpoint response')
        );
        $logger = $this->createMock(\Psr\Log\LoggerInterface::class);
        $logger->expects($this->once())
            ->method('warning')
            ->with(
                'OpenSearch Hybrid replacement generation creation failed.',
                [
                    'store_id' => 2,
                    'invalidation_boundary' => 41,
                    'error_class' => \RuntimeException::class,
                ]
            );
        $service = new \MageOS\OpenSearchHybrid\Model\Generation\AutomaticReplacementService(
            $repository,
            $buildService,
            $config,
            $logger
        );

        $report = $service->createPending()[0];
        self::assertSame('FAILED_TO_CREATE', $report['status']);
        self::assertSame(\RuntimeException::class, $report['error_class']);
        self::assertStringNotContainsString('secret', json_encode($report, JSON_THROW_ON_ERROR));
    }

    private function eligibleRepository(): \MageOS\OpenSearchHybrid\Model\Generation\ReplacementCandidateRepository
    {
        $repository = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Generation\ReplacementCandidateRepository::class
        );
        $repository->method('storeIds')->willReturn([2]);
        $repository->method('withinStoreLock')
            ->willReturnCallback(static fn (int $storeId, callable $operation): array => $operation());
        $repository->method('assessment')->willReturn([
            'eligible' => true,
            'reason' => 'pending_configuration_invalidation',
            'invalidation_boundary' => 41,
            'encoder_identity_digest' => str_repeat('a', 64),
        ]);

        return $repository;
    }
}
