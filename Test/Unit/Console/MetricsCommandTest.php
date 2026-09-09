<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Console;

class MetricsCommandTest extends \PHPUnit\Framework\TestCase
{
    public function testCommandEmitsTheMachineReadableOperationsSnapshot(): void
    {
        $snapshot = [
            'schema_version' => 1,
            'generated_at' => '2026-08-25T00:00:00+00:00',
            'scope' => ['store_id' => 3],
            'alerts' => [],
        ];
        $metricsService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Operations\OperationalMetricsService::class
        );
        $metricsService->expects($this->once())->method('get')->with(3)->willReturn($snapshot);
        $tester = new \Symfony\Component\Console\Tester\CommandTester(
            new \MageOS\OpenSearchHybrid\Console\MetricsCommand(
                $metricsService,
                new \Magento\Framework\Serialize\Serializer\Json(),
                $this->createStub(\MageOS\OpenSearchHybrid\Model\Operations\OpenMetricsFormatter::class),
                $this->createStub(\MageOS\OpenSearchHybrid\Model\Operations\QueryTelemetryMetricsService::class),
                $this->createStub(\Magento\Framework\App\Filesystem\DirectoryList::class)
            )
        );

        $result = $tester->execute(['--store' => '3']);

        self::assertSame(0, $result);
        self::assertSame($snapshot, json_decode(trim($tester->getDisplay()), true, 512, JSON_THROW_ON_ERROR));
    }

    public function testCommandRejectsAnInvalidStoreScope(): void
    {
        $metricsService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Operations\OperationalMetricsService::class
        );
        $metricsService->expects($this->never())->method('get');
        $tester = new \Symfony\Component\Console\Tester\CommandTester(
            new \MageOS\OpenSearchHybrid\Console\MetricsCommand(
                $metricsService,
                new \Magento\Framework\Serialize\Serializer\Json(),
                $this->createStub(\MageOS\OpenSearchHybrid\Model\Operations\OpenMetricsFormatter::class),
                $this->createStub(\MageOS\OpenSearchHybrid\Model\Operations\QueryTelemetryMetricsService::class),
                $this->createStub(\Magento\Framework\App\Filesystem\DirectoryList::class)
            )
        );

        $result = $tester->execute(['--store' => 'all']);

        self::assertSame(2, $result);
        self::assertStringContainsString('positive integer', $tester->getDisplay());
    }

    public function testCommandEmitsOpenMetricsWhenRequested(): void
    {
        $snapshot = [
            'schema_version' => 1,
            'generated_at' => '2026-08-27T00:00:00+00:00',
            'scope' => ['store_id' => 3],
            'alerts' => [],
        ];
        $metricsService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Operations\OperationalMetricsService::class
        );
        $metricsService->expects($this->once())->method('get')->with(3)->willReturn($snapshot);
        $formatter = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Operations\OpenMetricsFormatter::class
        );
        $formatter->expects($this->once())->method('format')->with($snapshot)->willReturn("# EOF\n");
        $tester = new \Symfony\Component\Console\Tester\CommandTester(
            new \MageOS\OpenSearchHybrid\Console\MetricsCommand(
                $metricsService,
                new \Magento\Framework\Serialize\Serializer\Json(),
                $formatter,
                $this->createStub(\MageOS\OpenSearchHybrid\Model\Operations\QueryTelemetryMetricsService::class),
                $this->createStub(\Magento\Framework\App\Filesystem\DirectoryList::class)
            )
        );

        $result = $tester->execute(['--store' => '3', '--format' => 'openmetrics']);

        self::assertSame(0, $result);
        self::assertSame("# EOF\n", $tester->getDisplay());
    }

    public function testCommandRejectsAnUnknownFormat(): void
    {
        $metricsService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Operations\OperationalMetricsService::class
        );
        $metricsService->expects($this->never())->method('get');
        $tester = new \Symfony\Component\Console\Tester\CommandTester(
            new \MageOS\OpenSearchHybrid\Console\MetricsCommand(
                $metricsService,
                new \Magento\Framework\Serialize\Serializer\Json(),
                $this->createStub(\MageOS\OpenSearchHybrid\Model\Operations\OpenMetricsFormatter::class),
                $this->createStub(\MageOS\OpenSearchHybrid\Model\Operations\QueryTelemetryMetricsService::class),
                $this->createStub(\Magento\Framework\App\Filesystem\DirectoryList::class)
            )
        );

        $result = $tester->execute(['--format' => 'prometheus']);

        self::assertSame(2, $result);
        self::assertStringContainsString('json or openmetrics', $tester->getDisplay());
    }

    public function testCommandIncludesBoundedQueryMetricsWhenRequested(): void
    {
        $snapshot = [
            'schema_version' => 1,
            'generated_at' => '2026-08-31T00:00:00+00:00',
            'alerts' => [],
        ];
        $querySnapshot = [
            'schema_version' => 1,
            'generated_timestamp_seconds' => 1788134400,
            'window_seconds' => 300,
            'routes' => [],
        ];
        $metricsService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Operations\OperationalMetricsService::class
        );
        $metricsService->expects($this->once())->method('get')->with(null)->willReturn($snapshot);
        $queryMetricsService = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Operations\QueryTelemetryMetricsService::class
        );
        $queryMetricsService->expects($this->once())
            ->method('summarize')
            ->with(
                '/tmp/magento-log/opensearch-hybrid-query.log',
                300,
                5242880,
                $this->isType('int')
            )
            ->willReturn($querySnapshot);
        $directoryList = $this->createMock(\Magento\Framework\App\Filesystem\DirectoryList::class);
        $directoryList->expects($this->once())
            ->method('getPath')
            ->with(\Magento\Framework\App\Filesystem\DirectoryList::LOG)
            ->willReturn('/tmp/magento-log');
        $formatter = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Operations\OpenMetricsFormatter::class
        );
        $formatter->expects($this->once())
            ->method('format')
            ->with($snapshot, $querySnapshot)
            ->willReturn("query_metrics 1\n# EOF\n");
        $tester = new \Symfony\Component\Console\Tester\CommandTester(
            new \MageOS\OpenSearchHybrid\Console\MetricsCommand(
                $metricsService,
                new \Magento\Framework\Serialize\Serializer\Json(),
                $formatter,
                $queryMetricsService,
                $directoryList
            )
        );

        $result = $tester->execute([
            '--format' => 'openmetrics',
            '--query-window' => '300',
            '--query-max-bytes' => '5242880',
        ]);

        self::assertSame(0, $result);
        self::assertSame("query_metrics 1\n# EOF\n", $tester->getDisplay());
    }
}
