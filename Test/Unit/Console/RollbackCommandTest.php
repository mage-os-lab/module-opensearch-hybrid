<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Console;

class RollbackCommandTest extends \PHPUnit\Framework\TestCase
{
    public function testDryRunPrintsTheStateBoundNativeRollbackPreview(): void
    {
        $preview = [
            'schema_version' => 1,
            'operation' => 'rollback_native',
            'mode' => 'preview',
            'store_id' => 3,
            'confirmation_token' => 'rollback-native-3-' . str_repeat('a', 64),
        ];
        $activationService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Activation\ActivationService::class
        );
        $activationService->expects($this->once())
            ->method('previewNativeRollback')
            ->with(3)
            ->willReturn($preview);
        $activationService->expects($this->never())->method('rollbackToNativeConfirmed');
        $tester = $this->commandTester($activationService);

        $result = $tester->execute([
            '--store' => '3',
            '--dry-run' => true,
        ]);

        self::assertSame(0, $result);
        self::assertSame($preview, json_decode(trim($tester->getDisplay()), true, 512, JSON_THROW_ON_ERROR));
    }

    public function testApplyNativeRollbackRequiresTheCurrentPreviewToken(): void
    {
        $token = 'rollback-native-3-' . str_repeat('b', 64);
        $report = [
            'schema_version' => 1,
            'status' => 'rolled_back_to_native',
            'store_id' => 3,
        ];
        $activationService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Activation\ActivationService::class
        );
        $activationService->expects($this->never())->method('previewNativeRollback');
        $activationService->expects($this->once())
            ->method('rollbackToNativeConfirmed')
            ->with(3, $token, 'cli')
            ->willReturn($report);
        $tester = $this->commandTester($activationService);

        $result = $tester->execute([
            '--store' => '3',
            '--confirm' => $token,
        ]);

        self::assertSame(0, $result);
        self::assertSame($report, json_decode(trim($tester->getDisplay()), true, 512, JSON_THROW_ON_ERROR));
    }

    public function testDryRunPrintsTheStateBoundGenerationRollbackPreview(): void
    {
        $preview = [
            'schema_version' => 1,
            'operation' => 'rollback_generation',
            'mode' => 'preview',
            'store_id' => 3,
            'confirmation_token' => 'rollback-generation-3-8-' . str_repeat('c', 64),
        ];
        $activationService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Activation\ActivationService::class
        );
        $activationService->expects($this->once())
            ->method('previewGenerationRollback')
            ->with(3, 8)
            ->willReturn($preview);
        $activationService->expects($this->never())->method('rollbackToGenerationConfirmed');
        $tester = $this->commandTester($activationService);

        $result = $tester->execute([
            '--store' => '3',
            '--generation' => '8',
            '--dry-run' => true,
        ]);

        self::assertSame(0, $result);
        self::assertSame($preview, json_decode(trim($tester->getDisplay()), true, 512, JSON_THROW_ON_ERROR));
    }

    public function testApplyGenerationRollbackRequiresTheCurrentPreviewToken(): void
    {
        $token = 'rollback-generation-3-8-' . str_repeat('d', 64);
        $report = [
            'schema_version' => 1,
            'status' => 'rolled_back_to_generation',
            'store_id' => 3,
            'generation_id' => 8,
        ];
        $activationService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Activation\ActivationService::class
        );
        $activationService->expects($this->never())->method('previewGenerationRollback');
        $activationService->expects($this->once())
            ->method('rollbackToGenerationConfirmed')
            ->with(3, 8, $token, 'cli')
            ->willReturn($report);
        $tester = $this->commandTester($activationService);

        $result = $tester->execute([
            '--store' => '3',
            '--generation' => '8',
            '--confirm' => $token,
        ]);

        self::assertSame(0, $result);
        self::assertSame($report, json_decode(trim($tester->getDisplay()), true, 512, JSON_THROW_ON_ERROR));
    }

    public function testCommandRejectsImplicitRollback(): void
    {
        $activationService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Activation\ActivationService::class
        );
        $activationService->expects($this->never())->method('previewNativeRollback');
        $activationService->expects($this->never())->method('rollbackToNativeConfirmed');
        $activationService->expects($this->never())->method('previewGenerationRollback');
        $activationService->expects($this->never())->method('rollbackToGenerationConfirmed');
        $tester = $this->commandTester($activationService);

        $result = $tester->execute(['--store' => '3']);

        self::assertSame(2, $result);
        self::assertStringContainsString('--dry-run', $tester->getDisplay());
        self::assertStringContainsString('--confirm', $tester->getDisplay());
    }

    private function commandTester(
        \MageOS\OpenSearchHybrid\Model\Activation\ActivationService $activationService
    ): \Symfony\Component\Console\Tester\CommandTester {
        return new \Symfony\Component\Console\Tester\CommandTester(
            new \MageOS\OpenSearchHybrid\Console\RollbackCommand(
                $activationService,
                new \Magento\Framework\Serialize\Serializer\Json()
            )
        );
    }
}
