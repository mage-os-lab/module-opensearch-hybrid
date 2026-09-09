<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Controller\Adminhtml\Activation;

use Magento\Framework\Exception\LocalizedException;

class Apply extends \Magento\Backend\App\Action implements \Magento\Framework\App\Action\HttpPostActionInterface
{
    public const ADMIN_RESOURCE = 'MageOS_OpenSearchHybrid::activate';

    public function __construct(
        \Magento\Backend\App\Action\Context $context,
        private readonly \MageOS\OpenSearchHybrid\Model\Activation\ActivationService $activationService,
        private readonly \MageOS\OpenSearchHybrid\Model\Activation\AdminRequestValidator $requestValidator,
        private readonly \Magento\Backend\Model\Auth\Session $authSession,
        private readonly \Psr\Log\LoggerInterface $logger
    ) {
        parent::__construct($context);
    }

    public function execute(): \Magento\Framework\Controller\ResultInterface
    {
        try {
            $input = $this->requestValidator->apply(
                $this->getRequest()->getParam('operation'),
                $this->getRequest()->getParam('store'),
                $this->getRequest()->getParam('generation'),
                $this->getRequest()->getParam('human_confirmation'),
                $this->getRequest()->getParam('confirmation_token')
            );
            $actor = $this->actor();
            $report = match ($input['operation']) {
                'activate' => $this->activationService->activateConfirmed(
                    $input['store_id'],
                    $input['generation_id'],
                    $input['confirmation_token'],
                    $actor
                ),
                'rollback_native' => $this->activationService->rollbackToNativeConfirmed(
                    $input['store_id'],
                    $input['confirmation_token'],
                    $actor
                ),
                'rollback_generation' => $this->activationService->rollbackToGenerationConfirmed(
                    $input['store_id'],
                    $input['generation_id'],
                    $input['confirmation_token'],
                    $actor
                ),
            };
            $this->messageManager->addSuccessMessage(
                __('OpenSearch Hybrid operation completed under activation audit %1.', $report['activation_id'])
            );
        } catch (LocalizedException|\InvalidArgumentException $exception) {
            $this->messageManager->addErrorMessage($exception->getMessage());
        } catch (\Throwable $throwable) {
            $this->logger->critical('OpenSearch Hybrid Admin activation mutation failed.', [
                'error_class' => $throwable::class,
            ]);
            $this->messageManager->addErrorMessage(__('The activation operation failed unexpectedly.'));
        }

        return $this->resultRedirectFactory->create()->setPath('mageos_opensearch_hybrid/status/index');
    }

    private function actor(): string
    {
        $user = $this->authSession->getUser();
        $userId = $user === null ? 0 : (int)$user->getId();
        if ($userId <= 0) {
            throw new \RuntimeException('The authenticated Admin user identity is unavailable.');
        }

        return 'admin_user_id:' . $userId;
    }
}
