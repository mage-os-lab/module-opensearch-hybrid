<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Console;

class ActivateCommandTest extends \PHPUnit\Framework\TestCase
{
    public function testDryRunPrintsTheStateBoundActivationPreview(): void
    {
        $preview = [
            'schema_version' => 1,
            'operation' => 'activate',
            'mode' => 'preview',
            'store_id' => 3,
            'confirmation_token' => 'activate-3-8-' . str_repeat('a', 64),
        ];
        $activationService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Activation\ActivationService::class
        );
        $activationService->expects($this->once())
            ->method('previewActivation')
            ->with(3, 8)
            ->willReturn($preview);
        $activationService->expects($this->never())->method('activateConfirmed');
        $tester = new \Symfony\Component\Console\Tester\CommandTester(
            new \MageOS\OpenSearchHybrid\Console\ActivateCommand(
                $activationService,
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

    public function testApplyRequiresTheCurrentPreviewToken(): void
    {
        $token = 'activate-3-8-' . str_repeat('b', 64);
        $report = [
            'schema_version' => 1,
            'status' => 'activated',
            'store_id' => 3,
            'generation_id' => 8,
        ];
        $activationService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Activation\ActivationService::class
        );
        $activationService->expects($this->never())->method('previewActivation');
        $activationService->expects($this->once())
            ->method('activateConfirmed')
            ->with(3, 8, $token, 'cli')
            ->willReturn($report);
        $tester = new \Symfony\Component\Console\Tester\CommandTester(
            new \MageOS\OpenSearchHybrid\Console\ActivateCommand(
                $activationService,
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

    public function testCommandRejectsImplicitActivation(): void
    {
        $activationService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Activation\ActivationService::class
        );
        $activationService->expects($this->never())->method('previewActivation');
        $activationService->expects($this->never())->method('activateConfirmed');
        $tester = new \Symfony\Component\Console\Tester\CommandTester(
            new \MageOS\OpenSearchHybrid\Console\ActivateCommand(
                $activationService,
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
