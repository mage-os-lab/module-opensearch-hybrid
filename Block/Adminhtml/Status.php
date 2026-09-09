<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Block\Adminhtml;

class Status extends \Magento\Backend\Block\Template
{
    private const PERSIST_KEY = 'mageos_opensearch_hybrid_impact_preview';
    private const ACTIVATION_PERSIST_KEY = 'mageos_opensearch_hybrid_activation_preview';
    private ?array $status = null;
    private ?array $preview = null;
    private ?array $activationPreview = null;

    public function __construct(
        \Magento\Backend\Block\Template\Context $context,
        private readonly \MageOS\OpenSearchHybrid\Model\StatusService $statusService,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\ConfigurationImpactService $impactService,
        private readonly \Magento\Framework\App\Request\DataPersistorInterface $dataPersistor,
        private readonly \Magento\Framework\Serialize\Serializer\Json $json,
        array $data = []
    ) {
        parent::__construct($context, $data);
    }

    public function getStatus(): array
    {
        if ($this->status === null) {
            $this->status = $this->statusService->get();
        }

        return $this->status;
    }

    public function getImpactPreview(): ?array
    {
        if ($this->preview === null) {
            $preview = $this->dataPersistor->get(self::PERSIST_KEY);
            $this->dataPersistor->clear(self::PERSIST_KEY);
            $this->preview = is_array($preview) ? $preview : [];
        }

        return $this->preview === [] ? null : $this->preview;
    }

    public function getImpactPaths(): array
    {
        return array_combine($this->impactService->paths(), $this->impactService->paths());
    }

    public function getPreviewUrl(): string
    {
        return $this->getUrl('mageos_opensearch_hybrid/status/preview');
    }

    public function canManageActivation(): bool
    {
        return $this->_authorization->isAllowed('MageOS_OpenSearchHybrid::activate');
    }

    public function getActivationPreview(): ?array
    {
        if ($this->activationPreview === null) {
            $preview = $this->dataPersistor->get(self::ACTIVATION_PERSIST_KEY);
            $this->dataPersistor->clear(self::ACTIVATION_PERSIST_KEY);
            $this->activationPreview = is_array($preview) ? $preview : [];
        }

        return $this->activationPreview === [] ? null : $this->activationPreview;
    }

    public function getActivationPreviewUrl(): string
    {
        return $this->getUrl('mageos_opensearch_hybrid/activation/preview');
    }

    public function getActivationApplyUrl(): string
    {
        return $this->getUrl('mageos_opensearch_hybrid/activation/apply');
    }

    public function serializePreviewValue(mixed $value): string
    {
        return $this->json->serialize($value);
    }
}
