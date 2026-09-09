<?php

declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Plugin;

class ConfigurationSaveConfirmationTest extends \PHPUnit\Framework\TestCase
{
    public function testAdminSystemSaveRequiresTheCurrentRequestConfirmation(): void
    {
        $request = $this->createStub(\Magento\Framework\App\Request\Http::class);
        $request->method('getFullActionName')->willReturn('adminhtml_system_config_save');
        $request->method('getParam')->willReturnMap([
            ['mageos_opensearch_hybrid_confirmation_token', null, 'config-save-token'],
            ['mageos_opensearch_hybrid_human_confirmation', null, 'apply'],
        ]);
        $confirmation = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Change\ConfigurationSaveConfirmation::class
        );
        $config = $this->createStub(\Magento\Config\Model\Config::class);
        $confirmation->expects($this->once())
            ->method('assertConfirmed')
            ->with($config, 'config-save-token', 'apply');
        $plugin = new \MageOS\OpenSearchHybrid\Plugin\ConfigurationSaveConfirmation(
            $request,
            $confirmation
        );

        $plugin->beforeSave($config);
    }

    public function testNonAdminSaveDoesNotRequireAnInteractiveConfirmation(): void
    {
        $request = $this->createStub(\Magento\Framework\App\Request\Http::class);
        $request->method('getFullActionName')->willReturn('crontab_default_jobs');
        $confirmation = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Change\ConfigurationSaveConfirmation::class
        );
        $confirmation->expects($this->never())->method('assertConfirmed');
        $plugin = new \MageOS\OpenSearchHybrid\Plugin\ConfigurationSaveConfirmation(
            $request,
            $confirmation
        );

        $plugin->beforeSave($this->createStub(\Magento\Config\Model\Config::class));
    }
}
