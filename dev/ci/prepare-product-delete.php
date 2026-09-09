<?php

declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository;
use MageOS\OpenSearchHybrid\Model\Queue\DataPlanePublisher;
use Magento\Catalog\Api\ProductRepositoryInterface;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\ResourceConnection;
use Magento\Framework\App\State;
use Magento\Framework\Registry;

if ($argc !== 3) {
    fwrite(STDERR, "Usage: php prepare-product-delete.php /path/to/mageos /path/to/manifest.json\n");
    exit(2);
}

$mageOsRoot = rtrim((string)$argv[1], DIRECTORY_SEPARATOR);
$manifestPath = (string)$argv[2];
$bootstrapFile = $mageOsRoot . '/app/bootstrap.php';
if (!is_file($bootstrapFile) || !is_file($manifestPath)) {
    fwrite(STDERR, "Mage-OS bootstrap or queue-isolation manifest is missing.\n");
    exit(2);
}

require $bootstrapFile;

/** @var array<string, int|string> $manifest */
$manifest = json_decode((string)file_get_contents($manifestPath), true, 512, JSON_THROW_ON_ERROR);
$bootstrap = Bootstrap::create(BP, $_SERVER);
$objectManager = $bootstrap->getObjectManager();
try {
    $objectManager->get(State::class)->setAreaCode('global');
} catch (\Magento\Framework\Exception\LocalizedException) {
    // Another bootstrap participant already set the area code.
}
/** @var ProductRepositoryInterface $productRepository */
$productRepository = $objectManager->get(ProductRepositoryInterface::class);
/** @var ResourceConnection $resourceConnection */
$resourceConnection = $objectManager->get(ResourceConnection::class);
/** @var OutboxRepository $outboxRepository */
$outboxRepository = $objectManager->get(OutboxRepository::class);
/** @var DataPlanePublisher $publisher */
$publisher = $objectManager->get(DataPlanePublisher::class);
/** @var Registry $registry */
$registry = $objectManager->get(Registry::class);

$alreadySecure = (bool)$registry->registry('isSecureArea');
if (!$alreadySecure) {
    $registry->register('isSecureArea', true);
}
try {
    $productRepository->deleteById((string)$manifest['product_sku']);
} finally {
    if (!$alreadySecure) {
        $registry->unregister('isSecureArea');
    }
}

$connection = $resourceConnection->getConnection();
$journalTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_change_journal');
$outboxTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_outbox');
$journal = $connection->fetchRow(
    $connection->select()
        ->from($journalTable)
        ->where('store_id = ?', (int)$manifest['store_id'])
        ->where('entity_type = ?', 'PRODUCT')
        ->where('entity_id = ?', (int)$manifest['product_id'])
        ->where('operation = ?', 'DELETE')
        ->order('change_id DESC')
        ->limit(1)
);
if (!is_array($journal) || (string)$journal['native_state'] !== 'PENDING') {
    throw new RuntimeException('Scheduled product deletion did not create a pending native journal record.');
}
$deleteChangeId = (int)$journal['change_id'];
$deleteJobs = $connection->fetchAll(
    $connection->select()
        ->from($outboxTable)
        ->where('store_id = ?', (int)$manifest['store_id'])
        ->where('generation_id = ?', (int)$manifest['generation_id'])
        ->where('earliest_change_id = ?', $deleteChangeId)
        ->where('latest_change_id = ?', $deleteChangeId)
);
if (count($deleteJobs) !== 1 || (string)$deleteJobs[0]['lane'] !== 'PRIORITY'
    || (string)$deleteJobs[0]['operation'] !== 'DELETE'
) {
    throw new RuntimeException('Product deletion did not create one priority tombstone job.');
}
$deleteJobId = (string)$deleteJobs[0]['job_id'];
if ((string)$outboxRepository->get($deleteJobId)['state'] !== 'PUBLISHED') {
    throw new RuntimeException('Product deletion priority job was not published after commit.');
}
$publisher->publish($deleteJobId, 'PRIORITY');

$manifest['delete_change_id'] = $deleteChangeId;
$manifest['delete_job_id'] = $deleteJobId;
$encodedManifest = json_encode($manifest, JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR);
if (file_put_contents($manifestPath, $encodedManifest . "\n") === false) {
    throw new RuntimeException("Could not update queue-isolation manifest: {$manifestPath}");
}

fwrite(STDOUT, "Captured a scheduled product deletion and published its priority tombstone job.\n");
