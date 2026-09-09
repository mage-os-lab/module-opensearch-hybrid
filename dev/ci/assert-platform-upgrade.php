<?php

declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Generation\BuildService;
use Magento\Catalog\Api\ProductRepositoryInterface;
use Magento\Catalog\Model\ProductFactory;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\ResourceConnection;
use Magento\Framework\App\State;

if ($argc !== 4 || !in_array($argv[2], ['prepare', 'verify'], true)) {
    fwrite(
        STDERR,
        "Usage: php assert-platform-upgrade.php /path/to/mageos prepare|verify /path/to/manifest.json\n"
    );
    exit(2);
}

$mageOsRoot = rtrim((string)$argv[1], DIRECTORY_SEPARATOR);
$phase = (string)$argv[2];
$manifestPath = (string)$argv[3];
$bootstrapFile = $mageOsRoot . '/app/bootstrap.php';
if (!is_file($bootstrapFile)) {
    fwrite(STDERR, "Mage-OS bootstrap not found: {$bootstrapFile}\n");
    exit(2);
}

require $bootstrapFile;

$bootstrap = Bootstrap::create(BP, $_SERVER);
$objectManager = $bootstrap->getObjectManager();
try {
    $objectManager->get(State::class)->setAreaCode('global');
} catch (\Magento\Framework\Exception\LocalizedException) {
    // Another bootstrap participant already set the area code.
}

/** @var ProductRepositoryInterface $productRepository */
$productRepository = $objectManager->get(ProductRepositoryInterface::class);
/** @var ProductFactory $productFactory */
$productFactory = $objectManager->get(ProductFactory::class);
/** @var ResourceConnection $resourceConnection */
$resourceConnection = $objectManager->get(ResourceConnection::class);
$connection = $resourceConnection->getConnection();
$catalogTable = $resourceConnection->getTableName('catalog_product_entity');

if ($phase === 'prepare') {
    if (is_file($manifestPath)) {
        throw new RuntimeException("Refusing to overwrite upgrade fixture manifest: {$manifestPath}");
    }

    $product = $productFactory->create();
    $product->setTypeId('simple');
    $product->setAttributeSetId($product->getDefaultAttributeSetId());
    $product->setSku('mageos-hybrid-platform-upgrade');
    $product->setName('Mage-OS Hybrid Upgrade Fixture Prepare');
    $product->setDescription('upgrade_fixture_prepare');
    $product->setPrice(39.99);
    $product->setWeight(1);
    $product->setStatus(1);
    $product->setVisibility(4);
    $product->setWebsiteIds([1]);
    $product = $productRepository->save($product);
    $productId = (int)$product->getId();

    $catalogRow = $connection->fetchRow(
        $connection->select()->from($catalogTable)->where('entity_id = ?', $productId)
    );
    if (!is_array($catalogRow)) {
        throw new RuntimeException('Mage-OS 3.3 did not persist the catalog upgrade fixture.');
    }

    $manifest = [
        'schema_version' => 1,
        'platform_version' => '3.3.0',
        'store_id' => 1,
        'product_id' => $productId,
        'product_sku' => (string)$product->getSku(),
        'product_name' => (string)$product->getName(),
        'catalog_created_at' => (string)$catalogRow['created_at'],
    ];
    $encoded = json_encode($manifest, JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR);
    if (file_put_contents($manifestPath, $encoded . "\n") === false) {
        throw new RuntimeException("Could not write upgrade fixture manifest: {$manifestPath}");
    }

    printf("Prepared catalog product %d on Mage-OS 3.3.0.\n", $productId);
    exit(0);
}

if (!is_file($manifestPath)) {
    throw new RuntimeException("Upgrade fixture manifest not found: {$manifestPath}");
}
$manifest = json_decode((string)file_get_contents($manifestPath), true, flags: JSON_THROW_ON_ERROR);
if (($manifest['schema_version'] ?? null) !== 1 || ($manifest['platform_version'] ?? null) !== '3.3.0') {
    throw new RuntimeException('Upgrade fixture manifest has an unsupported contract.');
}

$productId = (int)($manifest['product_id'] ?? 0);
$product = $productRepository->getById($productId, false, 0, true);
if ((string)$product->getSku() !== (string)$manifest['product_sku']
    || (string)$product->getName() !== (string)$manifest['product_name']
) {
    throw new RuntimeException('The Mage-OS 3.4 upgrade did not preserve the fixture product.');
}
$catalogCreatedAt = $connection->fetchOne(
    $connection->select()->from($catalogTable, ['created_at'])->where('entity_id = ?', $productId)
);
if ((string)$catalogCreatedAt !== (string)$manifest['catalog_created_at']) {
    throw new RuntimeException('The Mage-OS 3.4 upgrade did not preserve catalog identity.');
}

/** @var BuildService $buildService */
$buildService = $objectManager->get(BuildService::class);
$generationId = $buildService->buildFake((int)$manifest['store_id']);
$generationTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_generation');
$progressTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_generation_progress');
$journalTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_change_journal');
$outboxTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_outbox');
$generation = $connection->fetchRow(
    $connection->select()->from($generationTable)->where('generation_id = ?', $generationId)
);
$progress = $connection->fetchRow(
    $connection->select()->from($progressTable)->where('generation_id = ?', $generationId)
);
if (!is_array($generation) || !is_array($progress)) {
    throw new RuntimeException('The module could not create lifecycle state on the upgraded store.');
}

$product->setName('Mage-OS Hybrid Upgrade Fixture Verify');
$product->setDescription('upgrade_fixture_verify');
$productRepository->save($product);

$journal = $connection->fetchRow(
    $connection->select()
        ->from($journalTable)
        ->where('store_id = ?', (int)$manifest['store_id'])
        ->where('entity_type = ?', 'PRODUCT')
        ->where('entity_id = ?', $productId)
        ->where('operation = ?', 'UPSERT')
        ->order('change_id DESC')
        ->limit(1)
);
if (!is_array($journal)
    || (int)$journal['revision'] <= 0
    || (string)$journal['native_state'] !== 'PENDING'
    || (string)$journal['hybrid_state'] !== 'PENDING'
) {
    throw new RuntimeException('Mage-OS 3.4 did not capture a post-upgrade product revision.');
}

$changeId = (int)$journal['change_id'];
$jobs = $connection->fetchAll(
    $connection->select()
        ->from($outboxTable)
        ->where('store_id = ?', (int)$manifest['store_id'])
        ->where('generation_id = ?', $generationId)
        ->where('latest_change_id = ?', $changeId)
);
$lanes = array_values(array_unique(array_map(
    static fn (array $job): string => (string)$job['lane'],
    $jobs
)));
sort($lanes, SORT_STRING);
if ($lanes !== ['EMBEDDING', 'PRIORITY']) {
    throw new RuntimeException('Post-upgrade capture did not create both durable processing lanes.');
}

$reloaded = $productRepository->getById($productId, false, 0, true);
if ((string)$reloaded->getDescription() !== 'upgrade_fixture_verify') {
    throw new RuntimeException('The post-upgrade product mutation was not durable.');
}

printf(
    "Verified product %d, generation %d, and post-upgrade change %d on Mage-OS 3.4.0.\n",
    $productId,
    $generationId,
    $changeId
);
