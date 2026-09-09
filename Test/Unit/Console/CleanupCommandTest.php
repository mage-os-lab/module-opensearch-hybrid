<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Console;

class CleanupCommandTest extends \PHPUnit\Framework\TestCase
{
    public function testDryRunPrintsTheExactCleanupPreview(): void
    {
        $preview = [
            'schema_version' => 1,
            'operation' => 'cleanup_generation',
            'mode' => 'preview',
            'eligible' => true,
            'confirmation_token' => 'cleanup-8-' . str_repeat('a', 64),
        ];
        $cleanupService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Generation\GenerationCleanupService::class
        );
        $cleanupService->expects($this->once())->method('preview')->with(3, 8)->willReturn($preview);
        $cleanupService->expects($this->never())->method('cleanup');
        $tester = new \Symfony\Component\Console\Tester\CommandTester(
            new \MageOS\OpenSearchHybrid\Console\CleanupCommand(
                $cleanupService,
                new \Magento\Framework\Serialize\Serializer\Json()
            )
        );

        $result = $tester->execute([
            '--store' => '3',
            '--generation' => '8',
            '--dry-run' => true,
        ]);

        self::assertSame(0, $result);
        self::assertSame($preview, json_decode(trim($tester->getDisplay()), true, 512, JSON_THROW_ON_ERROR));
    }

    public function testApplyRequiresThePreviewConfirmationToken(): void
    {
        $token = 'cleanup-8-' . str_repeat('b', 64);
        $report = [
            'schema_version' => 1,
            'operation' => 'cleanup_generation',
            'mode' => 'apply',
            'status' => 'completed',
            'generation_id' => 8,
        ];
        $cleanupService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Generation\GenerationCleanupService::class
        );
        $cleanupService->expects($this->never())->method('preview');
        $cleanupService->expects($this->once())->method('cleanup')->with(3, 8, $token)->willReturn($report);
        $tester = new \Symfony\Component\Console\Tester\CommandTester(
            new \MageOS\OpenSearchHybrid\Console\CleanupCommand(
                $cleanupService,
                new \Magento\Framework\Serialize\Serializer\Json()
            )
        );

        $result = $tester->execute([
            '--store' => '3',
            '--generation' => '8',
            '--confirm' => $token,
        ]);

        self::assertSame(0, $result);
        self::assertSame($report, json_decode(trim($tester->getDisplay()), true, 512, JSON_THROW_ON_ERROR));
    }

    public function testCommandRejectsImplicitMutation(): void
    {
        $cleanupService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Generation\GenerationCleanupService::class
        );
        $cleanupService->expects($this->never())->method('preview');
        $cleanupService->expects($this->never())->method('cleanup');
        $tester = new \Symfony\Component\Console\Tester\CommandTester(
            new \MageOS\OpenSearchHybrid\Console\CleanupCommand(
                $cleanupService,
                new \Magento\Framework\Serialize\Serializer\Json()
            )
        );

        $result = $tester->execute([
            '--store' => '3',
            '--generation' => '8',
        ]);

        self::assertSame(2, $result);
        self::assertStringContainsString('--dry-run', $tester->getDisplay());
        self::assertStringContainsString('--confirm', $tester->getDisplay());
    }
}
