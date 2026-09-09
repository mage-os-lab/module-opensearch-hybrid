<?php

declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Block\Adminhtml;

class ConfigConfirmation extends \Magento\Backend\Block\Template
{
    public function __construct(
        \Magento\Backend\Block\Template\Context $context,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\ConfigurationSaveConfirmation $confirmation,
        array $data = []
    ) {
        parent::__construct($context, $data);
    }

    public function shouldRender(): bool
    {
        return $this->confirmation->isSectionSupported(
            trim((string)$this->getRequest()->getParam('section'))
        );
    }

    public function getPreviewUrl(): string
    {
        return $this->getUrl('mageos_opensearch_hybrid/config/preview');
    }
}
