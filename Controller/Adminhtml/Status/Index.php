<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Controller\Adminhtml\Status;

class Index extends \Magento\Backend\App\Action implements \Magento\Framework\App\Action\HttpGetActionInterface
{
    public const ADMIN_RESOURCE = 'MageOS_OpenSearchHybrid::status';

    public function __construct(
        \Magento\Backend\App\Action\Context $context,
        private readonly \Magento\Framework\View\Result\PageFactory $resultPageFactory
    ) {
        parent::__construct($context);
    }

    public function execute(): \Magento\Framework\Controller\ResultInterface
    {
        $page = $this->resultPageFactory->create();
        $page->setActiveMenu(self::ADMIN_RESOURCE);
        $page->getConfig()->getTitle()->prepend(__('OpenSearch Hybrid'));

        return $page;
    }
}
