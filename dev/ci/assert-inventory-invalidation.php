<?php
declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Generation\ValidationService;
use MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository;
use MageOS\OpenSearchHybrid\Model\Queue\PriorityConsumer;
use Magento\Catalog\Api\ProductRepositoryInterface;
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
    throw new RuntimeException('The inventory fixture requires a ready generation.');
}
$generationId = (int)$generation['generation_id'];

/** @var ProductRepositoryInterface $productRepository */
$productRepository = $objectManager->get(ProductRepositoryInterface::class);
$product = $productRepository->get('mageos-hybrid-resume-02', false, $storeId, true);
$productId = (int)$product->getId();
$sku = (string)$product->getSku();
$embeddingBefore = $connection->fetchRow(
    $connection->select()
        ->from($embeddingTable)
        ->where('store_id = ?', $storeId)
        ->where('product_id = ?', $productId)
        ->where('generation_id = ?', $generationId)
);
if (!is_array($embeddingBefore) || (string)$embeddingBefore['status'] !== 'COMPLETE') {
    throw new RuntimeException('The inventory fixture product has no completed embedding to preserve.');
}

/** @var SourceItemInterfaceFactory $sourceItemFactory */
$sourceItemFactory = $objectManager->get(SourceItemInterfaceFactory::class);
/** @var SourceItemsSaveInterface $sourceItemsSave */
$sourceItemsSave = $objectManager->get(SourceItemsSaveInterface::class);
/** @var OutboxRepository $outboxRepository */
$outboxRepository = $objectManager->get(OutboxRepository::class);
/** @var PriorityConsumer $priorityConsumer */
$priorityConsumer = $objectManager->get(PriorityConsumer::class);
/** @var IndexerRegistry $indexerRegistry */
$indexerRegistry = $objectManager->get(IndexerRegistry::class);
$stockIndexer = $indexerRegistry->get('cataloginventory_stock');
$stockIndexer->reindexAll();
$nativeIndexer = $indexerRegistry->get('catalogsearch_fulltext');
/** @var ValidationService $validationService */
$validationService = $objectManager->get(ValidationService::class);
/** @var ConnectionManager $connectionManager */
$connectionManager = $objectManager->get(ConnectionManager::class);
$searchConnection = $connectionManager->getConnection();
if (!$searchConnection instanceof \Magento\OpenSearch\Model\SearchClient) {
    throw new RuntimeException('The inventory fixture has no OpenSearch client.');
}
$openSearch = $searchConnection->getOpenSearchClient();

$applyStatus = static function (
    int $status,
    float $quantity,
    bool $expectedSalable
) use (
    $connection,
    $journalTable,
    $outboxTable,
    $sourceItemFactory,
    $sourceItemsSave,
    $sku,
    $productId,
    $nativeIndexer,
    $outboxRepository,
    $priorityConsumer,
    $validationService,
    $generationId,
    $openSearch,
    $generation
): int {
    $beforeBoundary = (int)$connection->fetchOne(
        $connection->select()
            ->from($journalTable, [new Zend_Db_Expr('COALESCE(MAX(change_id), 0)')])
    );
    /** @var SourceItemInterface $sourceItem */
    $sourceItem = $sourceItemFactory->create();
    $sourceItem->setSourceCode('default');
    $sourceItem->setSku($sku);
    $sourceItem->setQuantity($quantity);
    $sourceItem->setStatus($status);
    $sourceItemsSave->execute([$sourceItem]);

    $changes = $connection->fetchAll(
        $connection->select()
            ->from($journalTable)
            ->where('change_id > ?', $beforeBoundary)
            ->where('entity_id = ?', $productId)
            ->where('reason = ?', 'inventory_salability_change')
            ->order('change_id ASC')
    );
    if (count($changes) !== 1 || (string)$changes[0]['operation'] !== 'REFRESH') {
        throw new RuntimeException('The salability transition did not capture one REFRESH change.');
    }
    $changeId = (int)$changes[0]['change_id'];
    $embeddingJobs = (int)$connection->fetchOne(
        $connection->select()
            ->from($outboxTable, [new Zend_Db_Expr('COUNT(*)')])
            ->where('latest_change_id = ?', $changeId)
            ->where('lane = ?', 'EMBEDDING')
    );
    if ($embeddingJobs !== 0) {
        throw new RuntimeException('A salability-only change created encoder work.');
    }

    $nativeIndexer->reindexList([$productId]);
    $priorityJobs = array_map('strval', $connection->fetchCol(
        $connection->select()
            ->from($outboxTable, ['job_id'])
            ->where('latest_change_id = ?', $changeId)
            ->where('lane = ?', 'PRIORITY')
            ->where('state NOT IN (?)', ['COMPLETED', 'SUPERSEDED'])
            ->order('generation_id ASC')
    ));
    if ($priorityJobs === []) {
        throw new RuntimeException('The salability transition has no priority work.');
    }
    foreach ($priorityJobs as $jobId) {
        if ((string)$outboxRepository->get($jobId)['state'] === 'PENDING') {
            $outboxRepository->markPublished($jobId);
        }
        $priorityConsumer->process($jobId);
    }
    if (!(bool)$validationService->validate($generationId)['ready']) {
        throw new RuntimeException('Inventory catch-up did not return the generation to READY.');
    }
    $document = $openSearch->get([
        'index' => (string)$generation['physical_index'],
        'id' => (string)$productId,
    ]);
    if (($document['_source']['is_salable'] ?? null) !== $expectedSalable) {
        throw new RuntimeException('The hybrid document has the wrong salability state.');
    }

    return $changeId;
};

$inStockChangeId = $applyStatus(SourceItemInterface::STATUS_IN_STOCK, 100.0, true);
$outOfStockChangeId = $applyStatus(SourceItemInterface::STATUS_OUT_OF_STOCK, 0.0, false);
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
    throw new RuntimeException('Inventory refresh changed or tombstoned the semantic embedding.');
}

printf(
    "Inventory salability REFRESH changes %d/%d preserved generation %d embedding revision %d.\n",
    $inStockChangeId,
    $outOfStockChangeId,
    $generationId,
    (int)$embeddingAfter['revision']
);
