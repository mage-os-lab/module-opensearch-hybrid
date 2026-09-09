<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Console;

class EventsInstallCommandTest extends \PHPUnit\Framework\TestCase
{
    public function testDryRunPrintsTheStateBoundSubscriptionPreview(): void
    {
        $preview = [
            'schema_version' => 1,
            'operation' => 'reconcile_async_event_subscription',
            'mode' => 'preview',
            'store_id' => 3,
            'confirmation_token' => 'events-install-3-' . str_repeat('a', 64),
        ];
        $reconciler = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\AsyncEvent\SubscriptionReconciler::class
        );
        $reconciler->expects($this->once())->method('preview')->with(3)->willReturn($preview);
        $reconciler->expects($this->never())->method('reconcile');
        $reconciler->expects($this->never())->method('reconcileConfirmed');
        $tester = new \Symfony\Component\Console\Tester\CommandTester(
            new \MageOS\OpenSearchHybrid\Console\EventsInstallCommand(
                $reconciler,
                new \Magento\Framework\Serialize\Serializer\Json(),
                $this->storeManager()
            )
        );

        $result = $tester->execute([
            '--store' => '3',
            '--dry-run' => true,
        ]);

        self::assertSame(0, $result);
        self::assertSame($preview, json_decode(trim($tester->getDisplay()), true, 512, JSON_THROW_ON_ERROR));
    }

    public function testApplyRequiresTheCurrentPreviewToken(): void
    {
        $token = 'events-install-3-' . str_repeat('b', 64);
        $report = [
            'schema_version' => 1,
            'operation' => 'reconcile_async_event_subscription',
            'mode' => 'applied',
            'store_id' => 3,
            'subscription_id' => 8,
        ];
        $reconciler = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\AsyncEvent\SubscriptionReconciler::class
        );
        $reconciler->expects($this->never())->method('preview');
        $reconciler->expects($this->never())->method('reconcile');
        $reconciler->expects($this->once())
            ->method('reconcileConfirmed')
            ->with(3, $token)
            ->willReturn($report);
        $tester = new \Symfony\Component\Console\Tester\CommandTester(
            new \MageOS\OpenSearchHybrid\Console\EventsInstallCommand(
                $reconciler,
                new \Magento\Framework\Serialize\Serializer\Json(),
                $this->storeManager()
            )
        );

        $result = $tester->execute([
            '--store' => '3',
            '--confirm' => $token,
        ]);

        self::assertSame(0, $result);
        self::assertSame($report, json_decode(trim($tester->getDisplay()), true, 512, JSON_THROW_ON_ERROR));
    }

    public function testCommandRejectsImplicitSubscriptionMutation(): void
    {
        $reconciler = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\AsyncEvent\SubscriptionReconciler::class
        );
        $reconciler->expects($this->never())->method('preview');
        $reconciler->expects($this->never())->method('reconcile');
        $reconciler->expects($this->never())->method('reconcileConfirmed');
        $tester = new \Symfony\Component\Console\Tester\CommandTester(
            new \MageOS\OpenSearchHybrid\Console\EventsInstallCommand(
                $reconciler,
                new \Magento\Framework\Serialize\Serializer\Json(),
                $this->storeManager()
            )
        );

        $result = $tester->execute(['--store' => '3']);

        self::assertSame(2, $result);
        self::assertStringContainsString('--dry-run', $tester->getDisplay());
        self::assertStringContainsString('--confirm', $tester->getDisplay());
    }

    public function testCommandRejectsANonexistentStoreBeforePreview(): void
    {
        $reconciler = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\AsyncEvent\SubscriptionReconciler::class
        );
        $reconciler->expects($this->never())->method('preview');
        $reconciler->expects($this->never())->method('reconcileConfirmed');
        $storeManager = $this->createMock(\Magento\Store\Model\StoreManagerInterface::class);
        $storeManager->expects($this->once())
            ->method('getStore')
            ->with(99)
            ->willThrowException(new \Magento\Framework\Exception\NoSuchEntityException());
        $tester = new \Symfony\Component\Console\Tester\CommandTester(
            new \MageOS\OpenSearchHybrid\Console\EventsInstallCommand(
                $reconciler,
                new \Magento\Framework\Serialize\Serializer\Json(),
                $storeManager
            )
        );

        $result = $tester->execute([
            '--store' => '99',
            '--dry-run' => true,
        ]);

        self::assertSame(2, $result);
        self::assertStringContainsString('does not exist', $tester->getDisplay());
    }

    private function storeManager(): \Magento\Store\Model\StoreManagerInterface
    {
        $store = $this->createStub(\Magento\Store\Api\Data\StoreInterface::class);
        $storeManager = $this->createStub(\Magento\Store\Model\StoreManagerInterface::class);
        $storeManager->method('getStore')->willReturn($store);

        return $storeManager;
    }
}
