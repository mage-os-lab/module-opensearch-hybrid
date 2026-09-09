<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Console;

class ReconcileCommandTest extends \PHPUnit\Framework\TestCase
{
    public function testCommandFailsWhenAutomaticReplacementCreationFails(): void
    {
        $report = [
            'schema_version' => 1,
            'failed_publications' => 0,
            'failed_replacements' => 1,
            'replacements' => [[
                'store_id' => 2,
                'status' => 'FAILED_TO_CREATE',
                'error_class' => \RuntimeException::class,
            ]],
        ];
        $service = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Outbox\ReconciliationService::class
        );
        $service->expects($this->once())->method('reconcile')->with(2)->willReturn($report);
        $tester = new \Symfony\Component\Console\Tester\CommandTester(
            new \MageOS\OpenSearchHybrid\Console\ReconcileCommand(
                $service,
                new \Magento\Framework\Serialize\Serializer\Json()
            )
        );

        $result = $tester->execute(['--store' => '2']);

        self::assertSame(1, $result);
        self::assertSame($report, json_decode(
            trim($tester->getDisplay()),
            true,
            512,
            JSON_THROW_ON_ERROR
        ));
    }
}
