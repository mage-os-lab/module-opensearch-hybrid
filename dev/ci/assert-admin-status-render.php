<?php
declare(strict_types=1);

use MageOS\OpenSearchHybrid\Block\Adminhtml\Status;
use MageOS\OpenSearchHybrid\Model\Change\ConfigurationImpactService;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\Request\DataPersistorInterface;
use Magento\Framework\App\State;
use Magento\Framework\View\LayoutFactory;

if ($argc !== 2) {
    fwrite(STDERR, "Usage: php assert-admin-status-render.php /path/to/mageos\n");
    exit(2);
}

$mageOsRoot = rtrim((string)$argv[1], DIRECTORY_SEPARATOR);
$bootstrapFile = $mageOsRoot . '/app/bootstrap.php';
if (!is_file($bootstrapFile)) {
    fwrite(STDERR, "Mage-OS bootstrap not found: {$bootstrapFile}\n");
    exit(2);
}
require $bootstrapFile;

$bootstrap = Bootstrap::create(BP, $_SERVER);
$objectManager = $bootstrap->getObjectManager();
$state = $objectManager->get(State::class);
$html = $state->emulateAreaCode('adminhtml', static function () use ($objectManager): string {
    $impact = $objectManager->get(ConfigurationImpactService::class)->preview(
        'general/locale/code',
        'default',
        0
    );
    $objectManager->get(DataPersistorInterface::class)->set(
        'mageos_opensearch_hybrid_impact_preview',
        $impact
    );
    $layout = $objectManager->get(LayoutFactory::class)->create();
    $layout->createBlock(
        \Magento\Backend\Block\Admin\Formkey::class,
        'formkey'
    )->setTemplate('Magento_Backend::admin/formkey.phtml');
    /** @var Status $block */
    $block = $layout->createBlock(Status::class);
    $block->setTemplate('MageOS_OpenSearchHybrid::status.phtml');

    return $block->toHtml();
});

foreach ([
    'Configuration impact preview',
    'Activation and rollback',
    'Store readiness',
    'Generations and coverage',
    'Work by state',
    'name="form_key"',
] as $expected) {
    if (!str_contains($html, $expected)) {
        throw new RuntimeException('Admin status rendering missed: ' . $expected);
    }
}

fwrite(STDOUT, sprintf(
    "Admin status rendered %d bytes with an exact impact preview.\n",
    strlen($html)
));
