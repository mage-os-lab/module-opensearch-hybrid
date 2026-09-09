<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Controller\Adminhtml\Activation;

use Magento\Framework\Exception\LocalizedException;

class Preview extends \Magento\Backend\App\Action implements \Magento\Framework\App\Action\HttpPostActionInterface
{
    public const ADMIN_RESOURCE = 'MageOS_OpenSearchHybrid::activate';
    public const PERSIST_KEY = 'mageos_opensearch_hybrid_activation_preview';

    public function __construct(
        \Magento\Backend\App\Action\Context $context,
        private readonly \MageOS\OpenSearchHybrid\Model\Activation\ActivationService $activationService,
        private readonly \MageOS\OpenSearchHybrid\Model\Activation\AdminRequestValidator $requestValidator,
        private readonly \Magento\Framework\App\Request\DataPersistorInterface $dataPersistor,
        private readonly \Psr\Log\LoggerInterface $logger
    ) {
        parent::__construct($context);
    }

    public function execute(): \Magento\Framework\Controller\ResultInterface
    {
        try {
            $input = $this->requestValidator->preview(
                $this->getRequest()->getParam('operation'),
                $this->getRequest()->getParam('store'),
                $this->getRequest()->getParam('generation')
            );
            $preview = match ($input['operation']) {
                'activate' => $this->activationService->previewActivation(
                    $input['store_id'],
                    $input['generation_id']
                ),
                'rollback_native' => $this->activationService->previewNativeRollback($input['store_id']),
                'rollback_generation' => $this->activationService->previewGenerationRollback(
                    $input['store_id'],
                    $input['generation_id']
                ),
            };
            $this->dataPersistor->set(self::PERSIST_KEY, $preview);
            $this->messageManager->addSuccessMessage(
                __('Review the exact activation impact and type its confirmation value.')
            );
        } catch (LocalizedException|\InvalidArgumentException $exception) {
            $this->dataPersistor->clear(self::PERSIST_KEY);
            $this->messageManager->addErrorMessage($exception->getMessage());
        } catch (\Throwable $throwable) {
            $this->dataPersistor->clear(self::PERSIST_KEY);
            $this->logger->critical('OpenSearch Hybrid Admin activation preview failed.', [
                'error_class' => $throwable::class,
            ]);
            $this->messageManager->addErrorMessage(__('The activation preview failed unexpectedly.'));
        }

        return $this->resultRedirectFactory->create()->setPath('mageos_opensearch_hybrid/status/index');
    }
}
