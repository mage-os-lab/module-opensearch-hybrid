<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Console;

class BuildCommandTest extends \PHPUnit\Framework\TestCase
{
    public function testDryRunPrintsTheReadOnlyProductionImpact(): void
    {
        $plan = [
            'schema_version' => 1,
            'operation' => 'build_production_generation',
            'mode' => 'preview',
            'store_id' => 3,
        ];
        $buildService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Generation\BuildService::class
        );
        $buildService->expects($this->once())
            ->method('planProduction')
            ->with(3, str_repeat('a', 64))
            ->willReturn($plan);
        $buildService->expects($this->never())->method('buildProduction');
        $buildService->expects($this->never())->method('buildProductionConfirmed');
        $json = $this->createMock(\Magento\Framework\Serialize\Serializer\Json::class);
        $json->expects($this->once())->method('serialize')->with($plan)->willReturn('{"mode":"preview"}');
        $tester = new \Symfony\Component\Console\Tester\CommandTester(
            new \MageOS\OpenSearchHybrid\Console\BuildCommand($buildService, $json)
        );

        $result = $tester->execute([
            '--store' => '3',
            '--encoder-identity' => str_repeat('a', 64),
            '--dry-run' => true,
        ]);

        self::assertSame(0, $result);
        self::assertSame('{"mode":"preview"}' . PHP_EOL, $tester->getDisplay());
    }

    public function testDryRunRefusesFakeBuilds(): void
    {
        $buildService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Generation\BuildService::class
        );
        $buildService->expects($this->never())->method('planProduction');
        $buildService->expects($this->never())->method('buildFake');
        $json = $this->createStub(\Magento\Framework\Serialize\Serializer\Json::class);
        $tester = new \Symfony\Component\Console\Tester\CommandTester(
            new \MageOS\OpenSearchHybrid\Console\BuildCommand($buildService, $json)
        );

        $result = $tester->execute([
            '--store' => '3',
            '--fake' => true,
            '--dry-run' => true,
        ]);

        self::assertSame(2, $result);
        self::assertStringContainsString('Production preview', $tester->getDisplay());
    }

    public function testProductionBuildRequiresAndPassesTheCurrentPreviewConfirmation(): void
    {
        $token = 'build-production-3-' . str_repeat('b', 64);
        $buildService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Generation\BuildService::class
        );
        $buildService->expects($this->once())
            ->method('buildProductionConfirmed')
            ->with(3, str_repeat('a', 64), $token)
            ->willReturn(8);
        $buildService->expects($this->never())->method('buildProduction');
        $json = $this->createStub(\Magento\Framework\Serialize\Serializer\Json::class);
        $tester = new \Symfony\Component\Console\Tester\CommandTester(
            new \MageOS\OpenSearchHybrid\Console\BuildCommand($buildService, $json)
        );

        $result = $tester->execute([
            '--store' => '3',
            '--encoder-identity' => str_repeat('a', 64),
            '--confirm' => $token,
        ]);

        self::assertSame(0, $result);
        self::assertStringContainsString('production generation 8', $tester->getDisplay());
        self::assertStringContainsString(str_repeat('a', 64), $tester->getDisplay());
    }

    public function testProductionBuildRefusesAnUnpinnedEndpointIdentity(): void
    {
        $buildService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Generation\BuildService::class
        );
        $buildService->expects($this->never())->method('buildProduction');
        $json = $this->createStub(\Magento\Framework\Serialize\Serializer\Json::class);
        $tester = new \Symfony\Component\Console\Tester\CommandTester(
            new \MageOS\OpenSearchHybrid\Console\BuildCommand($buildService, $json)
        );

        $result = $tester->execute(['--store' => '3']);

        self::assertSame(2, $result);
        self::assertStringContainsString('--encoder-identity', $tester->getDisplay());
    }

    public function testProductionBuildRefusesImplicitApplyWithoutPreviewConfirmation(): void
    {
        $buildService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Generation\BuildService::class
        );
        $buildService->expects($this->never())->method('planProduction');
        $buildService->expects($this->never())->method('buildProduction');
        $buildService->expects($this->never())->method('buildProductionConfirmed');
        $json = $this->createStub(\Magento\Framework\Serialize\Serializer\Json::class);
        $tester = new \Symfony\Component\Console\Tester\CommandTester(
            new \MageOS\OpenSearchHybrid\Console\BuildCommand($buildService, $json)
        );

        $result = $tester->execute([
            '--store' => '3',
            '--encoder-identity' => str_repeat('a', 64),
        ]);

        self::assertSame(2, $result);
        self::assertStringContainsString('--confirm', $tester->getDisplay());
    }
}
