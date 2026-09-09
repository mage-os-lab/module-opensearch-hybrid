<?php

declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Plugin;

class ConfigEditConfirmationBlock
{
    public function afterGetChildHtml(
        \Magento\Config\Block\System\Config\Edit $subject,
        string $result,
        string $alias = ''
    ): string {
        if ($alias !== 'form') {
            return $result;
        }
        $block = $subject->getLayout()->createBlock(
            \MageOS\OpenSearchHybrid\Block\Adminhtml\ConfigConfirmation::class,
            'mageos.opensearch_hybrid.config_confirmation'
        );
        if (
            !$block instanceof \MageOS\OpenSearchHybrid\Block\Adminhtml\ConfigConfirmation
            || !$block->shouldRender()
        ) {
            return $result;
        }

        return $result . $block->setTemplate(
            'MageOS_OpenSearchHybrid::config/confirmation.phtml'
        )->toHtml();
    }
}
