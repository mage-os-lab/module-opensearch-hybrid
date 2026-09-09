<?php
declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Generation\ValidationService;
use MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository;
use MageOS\OpenSearchHybrid\Model\Queue\PriorityConsumer;
use Magento\Catalog\Api\Data\TierPriceInterface;
use Magento\Catalog\Api\Data\TierPriceInterfaceFactory;
use Magento\Catalog\Api\ProductRepositoryInterface;
use Magento\Catalog\Api\TierPriceStorageInterface;
use Magento\Elasticsearch\SearchAdapter\ConnectionManager;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\ResourceConnection;
use Magento\Framework\App\State;
use Magento\Framework\Indexer\IndexerRegistry;
use Magento\InventoryApi\Api\Data\SourceItemInterface;
use Magento\InventoryApi\Api\Data\SourceItemInterfaceFactory;
use Magento\InventoryApi\Api\SourceItemsSaveInterface;

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

$storeId = 1;
/** @var ResourceConnection $resourceConnection */
$resourceConnection = $objectManager->get(ResourceConnection::class);
$connection = $resourceConnection->getConnection();
$generationTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_generation');
$progressTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_generation_progress');
$journalTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_change_journal');
$outboxTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_outbox');
$embeddingTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_embedding');
$generation = $connection->fetchRow(
    $connection->select()
        ->from(['generation' => $generationTable])
        ->joinInner(
            ['progress' => $progressTable],
            'progress.generation_id = generation.generation_id',
            []
        )
        ->where('generation.store_id = ?', $storeId)
        ->where('generation.state = ?', 'READY')
        ->where('progress.seeding_state = ?', 'COMPLETED')
        ->order('generation.generation_id DESC')
        ->limit(1)
);
if (!is_array($generation)) {
    throw new RuntimeException('The price fixture requires a ready generation.');
}
$generationId = (int)$generation['generation_id'];

/** @var ProductRepositoryInterface $productRepository */
$productRepository = $objectManager->get(ProductRepositoryInterface::class);
$product = $productRepository->get('mageos-hybrid-resume-02', false, $storeId, true);
$productId = (int)$product->getId();
$sku = (string)$product->getSku();

/** @var OutboxRepository $outboxRepository */
$outboxRepository = $objectManager->get(OutboxRepository::class);
/** @var PriorityConsumer $priorityConsumer */
$priorityConsumer = $objectManager->get(PriorityConsumer::class);
/** @var ValidationService $validationService */
$validationService = $objectManager->get(ValidationService::class);
/** @var IndexerRegistry $indexerRegistry */
$indexerRegistry = $objectManager->get(IndexerRegistry::class);
$processPriorityChange = static function (int $changeId, string $context) use (
    $connection,
    $outboxTable,
    $outboxRepository,
    $priorityConsumer
): void {
    $priorityJobs = array_map('strval', $connection->fetchCol(
        $connection->select()
            ->from($outboxTable, ['job_id'])
            ->where('latest_change_id = ?', $changeId)
            ->where('lane = ?', 'PRIORITY')
            ->where('state NOT IN (?)', ['COMPLETED', 'SUPERSEDED'])
            ->order('generation_id ASC')
    ));
    if ($priorityJobs === []) {
        throw new RuntimeException($context . ' has no priority work.');
    }
    foreach ($priorityJobs as $jobId) {
        if ((string)$outboxRepository->get($jobId)['state'] === 'PENDING') {
            $outboxRepository->markPublished($jobId);
        }
        $priorityConsumer->process($jobId);
    }
};

$inventoryBoundary = (int)$connection->fetchOne(
    $connection->select()
        ->from($journalTable, [new Zend_Db_Expr('COALESCE(MAX(change_id), 0)')])
);
/** @var SourceItemInterfaceFactory $sourceItemFactory */
$sourceItemFactory = $objectManager->get(SourceItemInterfaceFactory::class);
/** @var SourceItemInterface $sourceItem */
$sourceItem = $sourceItemFactory->create();
$sourceItem->setSourceCode('default');
$sourceItem->setSku($sku);
$sourceItem->setQuantity(100.0);
$sourceItem->setStatus(SourceItemInterface::STATUS_IN_STOCK);
/** @var SourceItemsSaveInterface $sourceItemsSave */
$sourceItemsSave = $objectManager->get(SourceItemsSaveInterface::class);
$sourceItemsSave->execute([$sourceItem]);
$inventoryChanges = $connection->fetchAll(
    $connection->select()
        ->from($journalTable)
        ->where('change_id > ?', $inventoryBoundary)
        ->where('entity_id = ?', $productId)
        ->where('reason = ?', 'inventory_salability_change')
        ->order('change_id ASC')
);
if (count($inventoryChanges) !== 1) {
    throw new RuntimeException('The price fixture could not establish one salable product change.');
}
$indexerRegistry->get('catalogsearch_fulltext')->reindexList([$productId]);
$processPriorityChange((int)$inventoryChanges[0]['change_id'], 'Salable product setup');
if (!(bool)$validationService->validate($generationId)['ready']) {
    throw new RuntimeException('The price fixture salable product setup did not return the generation to READY.');
}

$embeddingBefore = $connection->fetchRow(
    $connection->select()
        ->from($embeddingTable)
        ->where('store_id = ?', $storeId)
        ->where('product_id = ?', $productId)
        ->where('generation_id = ?', $generationId)
);
if (!is_array($embeddingBefore) || (string)$embeddingBefore['status'] !== 'COMPLETE') {
    throw new RuntimeException('The price fixture product has no completed embedding to preserve.');
}

$fullPriceBoundary = (int)$connection->fetchOne(
    $connection->select()
        ->from($journalTable, [new Zend_Db_Expr('COALESCE(MAX(change_id), 0)')])
);
$indexerRegistry->get('catalog_product_price')->reindexAll();
$fullPriceJobIds = array_map('strval', $connection->fetchCol(
    $connection->select()
        ->from($outboxTable, ['job_id'])
        ->where('latest_change_id > ?', $fullPriceBoundary)
        ->where('lane = ?', 'PRIORITY')
        ->where('state NOT IN (?)', ['COMPLETED', 'SUPERSEDED'])
        ->order('latest_change_id ASC')
));
if ($fullPriceJobIds === []) {
    throw new RuntimeException('The baseline full price reindex created no priority work.');
}
foreach ($fullPriceJobIds as $jobId) {
    if ((string)$outboxRepository->get($jobId)['state'] === 'PENDING') {
        $outboxRepository->markPublished($jobId);
    }
    $priorityConsumer->process($jobId);
}
if (!(bool)$validationService->validate($generationId)['ready']) {
    throw new RuntimeException('The baseline full price reindex did not return the generation to READY.');
}
$beforeBoundary = (int)$connection->fetchOne(
    $connection->select()
        ->from($journalTable, [new Zend_Db_Expr('COALESCE(MAX(change_id), 0)')])
);

/** @var TierPriceInterfaceFactory $tierPriceFactory */
$tierPriceFactory = $objectManager->get(TierPriceInterfaceFactory::class);
/** @var TierPriceInterface $tierPrice */
$tierPrice = $tierPriceFactory->create();
$tierPrice->setSku($sku);
$tierPrice->setPrice(6.5);
$tierPrice->setPriceType(TierPriceInterface::PRICE_TYPE_FIXED);
$tierPrice->setWebsiteId(0);
$tierPrice->setCustomerGroup('all groups');
$tierPrice->setQuantity(1.0);
/** @var TierPriceStorageInterface $tierPriceStorage */
$tierPriceStorage = $objectManager->get(TierPriceStorageInterface::class);
if ($tierPriceStorage->update([$tierPrice]) !== []) {
    throw new RuntimeException('The tier price fixture update failed validation.');
}
$indexerRegistry->get('catalog_product_price')->getView()->update();

$changes = $connection->fetchAll(
    $connection->select()
        ->from($journalTable)
        ->where('change_id > ?', $beforeBoundary)
        ->where('entity_id = ?', $productId)
        ->where('reason = ?', 'product_price_index')
        ->order('change_id ASC')
);
if (count($changes) !== 1
    || (string)$changes[0]['operation'] !== 'REFRESH'
    || (string)$changes[0]['native_state'] !== 'COMPLETED'
) {
    throw new RuntimeException('The tier price update did not capture one native-complete REFRESH change.');
}
$changeId = (int)$changes[0]['change_id'];
$embeddingJobs = (int)$connection->fetchOne(
    $connection->select()
        ->from($outboxTable, [new Zend_Db_Expr('COUNT(*)')])
        ->where('latest_change_id = ?', $changeId)
        ->where('lane = ?', 'EMBEDDING')
);
if ($embeddingJobs !== 0) {
    throw new RuntimeException('A price-only change created encoder work.');
}

$processPriorityChange($changeId, 'Tier price update');
if (!(bool)$validationService->validate($generationId)['ready']) {
    throw new RuntimeException('Price catch-up did not return the generation to READY.');
}

/** @var ConnectionManager $connectionManager */
$connectionManager = $objectManager->get(ConnectionManager::class);
$searchConnection = $connectionManager->getConnection();
if (!$searchConnection instanceof \Magento\OpenSearch\Model\SearchClient) {
    throw new RuntimeException('The price fixture has no OpenSearch client.');
}
$document = $searchConnection->getOpenSearchClient()->get([
    'index' => (string)$generation['physical_index'],
    'id' => (string)$productId,
]);
$priceRows = $document['_source']['customer_group_prices'] ?? null;
if (!is_array($priceRows) || count($priceRows) < 4) {
    throw new RuntimeException('The hybrid document is missing customer-group price rows.');
}
foreach ($priceRows as $priceRow) {
    if (!array_key_exists('tier_price', $priceRow) || (float)$priceRow['tier_price'] !== 6.5) {
        throw new RuntimeException('The hybrid document has the wrong indexed tier price.');
    }
}

$embeddingAfter = $connection->fetchRow(
    $connection->select()
        ->from($embeddingTable)
        ->where('store_id = ?', $storeId)
        ->where('product_id = ?', $productId)
        ->where('generation_id = ?', $generationId)
);
if (!is_array($embeddingAfter)
    || (string)$embeddingAfter['status'] !== 'COMPLETE'
    || (int)$embeddingAfter['revision'] !== (int)$embeddingBefore['revision']
    || !hash_equals((string)$embeddingBefore['source_hash'], (string)$embeddingAfter['source_hash'])
    || !hash_equals((string)$embeddingBefore['vector_hash'], (string)$embeddingAfter['vector_hash'])
) {
    throw new RuntimeException('Price refresh changed or tombstoned the semantic embedding.');
}

printf(
    "Tier price REFRESH change %d indexed %d customer-group rows and preserved embedding revision %d.\n",
    $changeId,
    count($priceRows),
    (int)$embeddingAfter['revision']
);
