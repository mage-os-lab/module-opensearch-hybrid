<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Controller\Adminhtml\Status;

class Preview extends \Magento\Backend\App\Action implements \Magento\Framework\App\Action\HttpPostActionInterface
{
    public const ADMIN_RESOURCE = 'MageOS_OpenSearchHybrid::status';
    private const PERSIST_KEY = 'mageos_opensearch_hybrid_impact_preview';

    public function __construct(
        \Magento\Backend\App\Action\Context $context,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\ConfigurationImpactService $impactService,
        private readonly \Magento\Framework\App\Config\ScopeConfigInterface $scopeConfig,
        private readonly \Magento\Framework\App\Request\DataPersistorInterface $dataPersistor
    ) {
        parent::__construct($context);
    }

    public function execute(): \Magento\Framework\Controller\ResultInterface
    {
        try {
            $path = trim((string)$this->getRequest()->getParam('path'));
            $scope = trim((string)$this->getRequest()->getParam('scope', 'default'));
            $scopeId = filter_var(
                $this->getRequest()->getParam('scope_id', 0),
                FILTER_VALIDATE_INT,
                ['options' => ['min_range' => 0]]
            );
            if ($scopeId === false) {
                throw new \InvalidArgumentException('Scope ID must be a non-negative integer.');
            }
            $scopeType = match ($scope) {
                'websites' => \Magento\Store\Model\ScopeInterface::SCOPE_WEBSITE,
                'stores' => \Magento\Store\Model\ScopeInterface::SCOPE_STORE,
                default => \Magento\Framework\App\Config\ScopeConfigInterface::SCOPE_TYPE_DEFAULT,
            };
            $previousValue = (string)$this->scopeConfig->getValue($path, $scopeType, (int)$scopeId);
            $proposedValue = $path === \MageOS\OpenSearchHybrid\Model\Config::XML_PATH_INCLUDED_STORES
                ? (string)$this->getRequest()->getParam('proposed_included_stores', '')
                : null;
            $preview = $this->impactService->preview(
                $path,
                $scope,
                (int)$scopeId,
                $previousValue,
                $proposedValue
            );
            $this->dataPersistor->set(self::PERSIST_KEY, $preview);
            $this->messageManager->addSuccessMessage(__('Configuration impact preview refreshed.'));
        } catch (\InvalidArgumentException $exception) {
            $this->dataPersistor->clear(self::PERSIST_KEY);
            $this->messageManager->addErrorMessage(__($exception->getMessage()));
        }

        return $this->resultRedirectFactory->create()->setPath('*/*/index');
    }
}
