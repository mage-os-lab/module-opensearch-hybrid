<?php

declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Plugin;

class ConfigurationSaveConfirmation
{
    public function __construct(
        private readonly \Magento\Framework\App\Request\Http $request,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\ConfigurationSaveConfirmation $confirmation
    ) {
    }

    public function beforeSave(\Magento\Config\Model\Config $subject): void
    {
        if ($this->request->getFullActionName() !== 'adminhtml_system_config_save') {
            return;
        }
        $this->confirmation->assertConfirmed(
            $subject,
            $this->request->getParam('mageos_opensearch_hybrid_confirmation_token'),
            $this->request->getParam('mageos_opensearch_hybrid_human_confirmation')
        );
    }
}
