<?php
declare(strict_types=1);

use Magento\Catalog\Api\ProductRepositoryInterface;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\ResourceConnection;
use Magento\Framework\App\State;

$fixtureRoot = $argv[1] ?? '';
if ($fixtureRoot === '' || !is_file($fixtureRoot . '/app/bootstrap.php')) {
    fwrite(STDERR, "Provide the installed Mage-OS fixture root.\n");
    exit(2);
}

require $fixtureRoot . '/app/bootstrap.php';
$bootstrap = Bootstrap::create(BP, $_SERVER);
$objectManager = $bootstrap->getObjectManager();
try {
    $objectManager->get(State::class)->setAreaCode('global');
} catch (\Magento\Framework\Exception\LocalizedException) {
    // Another bootstrap participant already set the area code.
}

/** @var ResourceConnection $resourceConnection */
$resourceConnection = $objectManager->get(ResourceConnection::class);
$connection = $resourceConnection->getConnection();
$journalTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_change_journal');
$productTable = $resourceConnection->getTableName('catalog_product_entity');
$productId = (int)$connection->fetchOne(
    $connection->select()->from($productTable, ['entity_id'])->order('entity_id ASC')->limit(1)
);
if ($productId <= 0) {
    throw new RuntimeException('The atomic rollback fixture requires a catalog product.');
}

/** @var ProductRepositoryInterface $productRepository */
$productRepository = $objectManager->get(ProductRepositoryInterface::class);
$product = $productRepository->getById($productId, false, 0, true);
$originalName = (string)$product->getName();
$journalCount = (int)$connection->fetchOne(
    $connection->select()->from($journalTable, [new Zend_Db_Expr('COUNT(*)')])
);
$triggerName = 'mageos_hybrid_fixture_reject_journal';
$connection->query(sprintf('DROP TRIGGER IF EXISTS `%s`', $triggerName));
$connection->query(sprintf(
    "CREATE TRIGGER `%s` BEFORE INSERT ON `%s` FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'fixture journal rejection'",
    $triggerName,
    $journalTable
));

$failed = false;
try {
    $product->setName($originalName . ' should rollback');
    $productRepository->save($product);
} catch (Throwable) {
    $failed = true;
} finally {
    $connection->query(sprintf('DROP TRIGGER IF EXISTS `%s`', $triggerName));
}
if (!$failed) {
    throw new RuntimeException('The injected journal failure did not fail the catalog save.');
}

$reloaded = $productRepository->getById($productId, false, 0, true);
$reloadedJournalCount = (int)$connection->fetchOne(
    $connection->select()->from($journalTable, [new Zend_Db_Expr('COUNT(*)')])
);
if ((string)$reloaded->getName() !== $originalName || $reloadedJournalCount !== $journalCount) {
    throw new RuntimeException('Catalog data committed without its required hybrid journal record.');
}

printf("Product %d and its journal remained unchanged after injected capture failure.\n", $productId);
