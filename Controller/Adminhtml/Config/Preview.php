<?php

declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Controller\Adminhtml\Config;

class Preview extends \Magento\Backend\App\Action implements \Magento\Framework\App\Action\HttpPostActionInterface
{
    public const ADMIN_RESOURCE = 'MageOS_OpenSearchHybrid::config';

    public function __construct(
        \Magento\Backend\App\Action\Context $context,
        private readonly \Magento\Config\Model\Config\Factory $configFactory,
        private readonly \MageOS\OpenSearchHybrid\Model\Change\ConfigurationSaveConfirmation $confirmation,
        private readonly \Magento\Framework\Controller\Result\JsonFactory $jsonFactory,
        private readonly \Psr\Log\LoggerInterface $logger
    ) {
        parent::__construct($context);
    }

    public function execute(): \Magento\Framework\Controller\ResultInterface
    {
        $result = $this->jsonFactory->create();
        try {
            $section = trim((string)$this->getRequest()->getParam('section'));
            if (!$this->confirmation->isSectionSupported($section)) {
                throw new \InvalidArgumentException('The configuration section has no generation-affecting fields.');
            }
            $config = $this->configFactory->create(['data' => [
                'section' => $section,
                'website' => trim((string)$this->getRequest()->getParam('website')),
                'store' => trim((string)$this->getRequest()->getParam('store')),
                'groups' => $this->getRequest()->getParam('groups', []),
            ]]);
            return $result->setData([
                'success' => true,
                'preview' => $this->confirmation->preview($config),
            ]);
        } catch (\InvalidArgumentException $exception) {
            return $result->setHttpResponseCode(400)->setData([
                'success' => false,
                'message' => $exception->getMessage(),
            ]);
        } catch (\Throwable $throwable) {
            $this->logger->critical('OpenSearch Hybrid Admin configuration preview failed.', [
                'error_class' => $throwable::class,
            ]);
            return $result->setHttpResponseCode(500)->setData([
                'success' => false,
                'message' => (string)__('The configuration impact preview failed.'),
            ]);
        }
    }
}
