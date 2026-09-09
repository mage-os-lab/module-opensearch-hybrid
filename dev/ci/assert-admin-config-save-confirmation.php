<?php

declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Change\ConfigurationSaveConfirmation;
use MageOS\OpenSearchHybrid\Model\Config;
use Magento\Config\Model\Config\Factory as ConfigFactory;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\Config\ReinitableConfigInterface;
use Magento\Framework\App\Request\Http;
use Magento\Framework\App\ResourceConnection;
use Magento\Framework\App\State;
use Magento\Framework\View\LayoutFactory;

$fixtureRoot = $argv[1] ?? '';
if ($fixtureRoot === '' || !is_file($fixtureRoot . '/app/bootstrap.php')) {
    fwrite(STDERR, "Provide the installed Mage-OS fixture root.\n");
    exit(2);
}

require $fixtureRoot . '/app/bootstrap.php';
$bootstrap = Bootstrap::create(BP, $_SERVER);
$objectManager = $bootstrap->getObjectManager();
try {
    $objectManager->get(State::class)->setAreaCode('adminhtml');
} catch (\Magento\Framework\Exception\LocalizedException) {
}

/** @var Http $request */
$request = $objectManager->get(Http::class);
$request->setRouteName('adminhtml');
$request->setControllerName('system_config');
$request->setActionName('edit');
$request->setParam('section', 'mageos_opensearch_hybrid');
/** @var LayoutFactory $layoutFactory */
$layoutFactory = $objectManager->get(LayoutFactory::class);
$layout = $layoutFactory->create();
$layout->createBlock(
    \Magento\Backend\Block\Admin\Formkey::class,
    'formkey'
)->setTemplate('Magento_Backend::admin/formkey.phtml');
$layout->createBlock(
    \Magento\Theme\Block\Html\Title::class,
    'page.title'
);
$layout->createBlock(
    \Magento\Framework\View\Element\Template::class,
    'page.actions.toolbar'
);
$edit = $layout->createBlock(\Magento\Config\Block\System\Config\Edit::class);
$html = $edit->toHtml();
if (
    !str_contains($html, 'Preview generation impact')
    || !str_contains($html, 'mageos_opensearch_hybrid_confirmation_token')
    || !str_contains($html, 'MageOS_OpenSearchHybrid/js/config-confirmation')
) {
    throw new RuntimeException('The system configuration form omitted its generation-impact controls.');
}

/** @var ResourceConnection $resourceConnection */
$resourceConnection = $objectManager->get(ResourceConnection::class);
$connection = $resourceConnection->getConnection();
$configTable = $resourceConnection->getTableName('core_config_data');
$journalTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_change_journal');
$existing = $connection->fetchRow(
    $connection->select()
        ->from($configTable)
        ->where('scope = ?', 'default')
        ->where('scope_id = ?', 0)
        ->where('path = ?', Config::XML_PATH_INCLUDED_STORES)
        ->limit(1)
);
if (is_array($existing)) {
    throw new RuntimeException('The Admin confirmation fixture requires no included-store override.');
}
$beforeBoundary = (int)$connection->fetchOne(
    $connection->select()
        ->from($journalTable, [new Zend_Db_Expr('COALESCE(MAX(change_id), 0)')])
);

/** @var ConfigFactory $configFactory */
$configFactory = $objectManager->get(ConfigFactory::class);
$createConfig = static function (array $fieldData) use ($configFactory): \Magento\Config\Model\Config {
    return $configFactory->create(['data' => [
        'section' => 'mageos_opensearch_hybrid',
        'website' => '',
        'store' => '',
        'groups' => [
            'general' => [
                'fields' => [
                    'included_stores' => $fieldData,
                ],
            ],
        ],
    ]]);
};
/** @var ConfigurationSaveConfirmation $confirmation */
$confirmation = $objectManager->get(ConfigurationSaveConfirmation::class);
$proposed = $createConfig(['value' => '2']);
$preview = $confirmation->preview($proposed);
if (
    !(bool)$preview['requires_confirmation']
    || $preview['affected_store_ids'] !== [1]
    || (int)$preview['estimated_documents'] < 1
    || count($preview['changes']) !== 1
    || (string)$preview['changes'][0]['path'] !== Config::XML_PATH_INCLUDED_STORES
    || array_key_exists('previous_effective', $preview['changes'][0])
    || array_key_exists('proposed_effective', $preview['changes'][0])
    || preg_match('/^config-save-[0-9a-f]{64}$/D', (string)$preview['confirmation_token']) !== 1
) {
    throw new RuntimeException('The Admin configuration preview is not exact or value-safe.');
}

$request->setActionName('save');
$request->setParam('mageos_opensearch_hybrid_confirmation_token', null);
$request->setParam('mageos_opensearch_hybrid_human_confirmation', null);
try {
    $proposed->save();
    throw new RuntimeException('An unconfirmed generation-affecting Admin save was accepted.');
} catch (\Magento\Framework\Exception\LocalizedException $exception) {
    if (!str_contains($exception->getMessage(), 'Type apply')) {
        throw $exception;
    }
}
$unchanged = $connection->fetchOne(
    $connection->select()
        ->from($configTable, ['value'])
        ->where('scope = ?', 'default')
        ->where('scope_id = ?', 0)
        ->where('path = ?', Config::XML_PATH_INCLUDED_STORES)
        ->limit(1)
);
$unchangedJournal = (int)$connection->fetchOne(
    $connection->select()
        ->from($journalTable, [new Zend_Db_Expr('COUNT(*)')])
        ->where('change_id > ?', $beforeBoundary)
);
if ($unchanged !== false || $unchangedJournal !== 0) {
    throw new RuntimeException('A rejected Admin configuration save changed durable state.');
}

$request->setParam(
    'mageos_opensearch_hybrid_confirmation_token',
    (string)$preview['confirmation_token']
);
$request->setParam('mageos_opensearch_hybrid_human_confirmation', 'apply');
$createConfig(['value' => '2'])->save();
/** @var ReinitableConfigInterface $reinitableConfig */
$reinitableConfig = $objectManager->get(ReinitableConfigInterface::class);
$reinitableConfig->reinit();
$saved = $connection->fetchOne(
    $connection->select()
        ->from($configTable, ['value'])
        ->where('scope = ?', 'default')
        ->where('scope_id = ?', 0)
        ->where('path = ?', Config::XML_PATH_INCLUDED_STORES)
        ->limit(1)
);
if ($saved !== '2') {
    throw new RuntimeException('The confirmed Admin configuration value was not saved.');
}

$restore = $createConfig(['value' => '']);
try {
    $restore->save();
    throw new RuntimeException('A stale Admin configuration confirmation was accepted.');
} catch (\Magento\Framework\Exception\LocalizedException $exception) {
    if (!str_contains($exception->getMessage(), 'Refresh')) {
        throw $exception;
    }
}
$restorePreview = $confirmation->preview($createConfig(['value' => '']));
if (
    !(bool)$restorePreview['requires_confirmation']
    || (string)$restorePreview['changes'][0]['mode'] !== 'value'
    || hash_equals(
        (string)$preview['confirmation_token'],
        (string)$restorePreview['confirmation_token']
    )
) {
    throw new RuntimeException('The restored Admin configuration preview was not state-bound.');
}
$request->setParam(
    'mageos_opensearch_hybrid_confirmation_token',
    (string)$restorePreview['confirmation_token']
);
$createConfig(['value' => ''])->save();
$reinitableConfig->reinit();
$restored = $connection->fetchOne(
    $connection->select()
        ->from($configTable, ['value'])
        ->where('scope = ?', 'default')
        ->where('scope_id = ?', 0)
        ->where('path = ?', Config::XML_PATH_INCLUDED_STORES)
        ->limit(1)
);
$changes = $connection->fetchAll(
    $connection->select()
        ->from($journalTable)
        ->where('change_id > ?', $beforeBoundary)
        ->where('entity_type = ?', 'CONFIGURATION')
        ->where('reason IN (?)', [
            'configuration_save:' . Config::XML_PATH_INCLUDED_STORES,
            'configuration_delete:' . Config::XML_PATH_INCLUDED_STORES,
        ])
        ->order('change_id ASC')
);
if (
    $restored !== null
    || count($changes) !== 2
    || (string)$changes[0]['operation'] !== 'REBUILD'
    || (string)$changes[1]['operation'] !== 'REBUILD'
) {
    throw new RuntimeException(sprintf(
        'Confirmed Admin save and restore state mismatch: value=%s changes=%d operations=%s reasons=%s.',
        var_export($restored, true),
        count($changes),
        implode(',', array_map(static fn (array $change): string => (string)$change['operation'], $changes)),
        implode(',', array_map(static fn (array $change): string => (string)$change['reason'], $changes))
    ));
}

printf(
    "Admin configuration preview bound %d store and %d documents; stale confirmation was rejected.\n",
    count($preview['affected_store_ids']),
    (int)$preview['estimated_documents']
);
