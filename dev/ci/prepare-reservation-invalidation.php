<?php
declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Generation\ValidationService;
use MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository;
use MageOS\OpenSearchHybrid\Model\Queue\PriorityConsumer;
use Magento\Catalog\Api\ProductRepositoryInterface;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\ResourceConnection;
use Magento\Framework\App\State;
use Magento\Framework\Indexer\IndexerRegistry;
use Magento\InventoryApi\Api\Data\SourceItemInterface;
use Magento\InventoryApi\Api\Data\SourceItemInterfaceFactory;
use Magento\InventoryApi\Api\SourceItemsSaveInterface;
use Magento\InventorySalesApi\Api\Data\ItemToSellInterfaceFactory;
use Magento\InventorySalesApi\Api\Data\SalesChannelInterface;
use Magento\InventorySalesApi\Api\Data\SalesChannelInterfaceFactory;
use Magento\InventorySalesApi\Api\Data\SalesEventInterface;
use Magento\InventorySalesApi\Api\Data\SalesEventInterfaceFactory;
use Magento\InventorySalesApi\Api\PlaceReservationsForSalesEventInterface;

$fixtureRoot = $argv[1] ?? '';
$manifestPath = $argv[2] ?? '';
if ($fixtureRoot === '' || !is_file($fixtureRoot . '/app/bootstrap.php') || $manifestPath === '') {
    fwrite(STDERR, "Provide the installed Mage-OS fixture root and reservation manifest path.\n");
    exit(2);
}

require $fixtureRoot . '/app/bootstrap.php';
$bootstrap = Bootstrap::create(BP, $_SERVER);
$objectManager = $bootstrap->getObjectManager();
try {
    $objectManager->get(State::class)->setAreaCode('global');
} catch (\Magento\Framework\Exception\LocalizedException) {
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
        ->where('generation.state IN (?)', ['READY', 'CATCHING_UP'])
        ->where('progress.seeding_state = ?', 'COMPLETED')
        ->order('generation.generation_id DESC')
        ->limit(1)
);
if (!is_array($generation)) {
    throw new RuntimeException('The reservation fixture requires a ready generation.');
}
$generationId = (int)$generation['generation_id'];

/** @var ProductRepositoryInterface $productRepository */
$productRepository = $objectManager->get(ProductRepositoryInterface::class);
$product = $productRepository->get('mageos-hybrid-resume-02', false, $storeId, true);
$productId = (int)$product->getId();
$sku = (string)$product->getSku();

/** @var SourceItemInterfaceFactory $sourceItemFactory */
$sourceItemFactory = $objectManager->get(SourceItemInterfaceFactory::class);
/** @var SourceItemInterface $sourceItem */
$sourceItem = $sourceItemFactory->create();
$sourceItem->setSourceCode('default');
$sourceItem->setSku($sku);
$sourceItem->setQuantity(1.0);
$sourceItem->setStatus(SourceItemInterface::STATUS_IN_STOCK);
/** @var SourceItemsSaveInterface $sourceItemsSave */
$sourceItemsSave = $objectManager->get(SourceItemsSaveInterface::class);
$sourceItemsSave->execute([$sourceItem]);

/** @var IndexerRegistry $indexerRegistry */
$indexerRegistry = $objectManager->get(IndexerRegistry::class);
$indexerRegistry->get('catalogsearch_fulltext')->reindexList([$productId]);
/** @var OutboxRepository $outboxRepository */
$outboxRepository = $objectManager->get(OutboxRepository::class);
/** @var PriorityConsumer $priorityConsumer */
$priorityConsumer = $objectManager->get(PriorityConsumer::class);
$priorityJobs = array_map('strval', $connection->fetchCol(
    $connection->select()
        ->from($outboxTable, ['job_id'])
        ->where('store_id = ?', $storeId)
        ->where('lane = ?', 'PRIORITY')
        ->where('state NOT IN (?)', ['COMPLETED', 'SUPERSEDED'])
        ->order('created_at ASC')
));
foreach ($priorityJobs as $jobId) {
    if ((string)$outboxRepository->get($jobId)['state'] === 'PENDING') {
        $outboxRepository->markPublished($jobId);
    }
    $priorityConsumer->process($jobId);
}
/** @var ValidationService $validationService */
$validationService = $objectManager->get(ValidationService::class);
if (!(bool)$validationService->validate($generationId)['ready']) {
    throw new RuntimeException('The reservation fixture could not prepare a ready in-stock product.');
}
$embedding = $connection->fetchRow(
    $connection->select()
        ->from($embeddingTable)
        ->where('store_id = ?', $storeId)
        ->where('product_id = ?', $productId)
        ->where('generation_id = ?', $generationId)
);
if (!is_array($embedding) || (string)$embedding['status'] !== 'COMPLETE') {
    throw new RuntimeException('The reservation fixture has no semantic embedding to preserve.');
}
$beforeBoundary = (int)$connection->fetchOne(
    $connection->select()->from($journalTable, [new Zend_Db_Expr('COALESCE(MAX(change_id), 0)')])
);

/** @var ItemToSellInterfaceFactory $itemFactory */
$itemFactory = $objectManager->get(ItemToSellInterfaceFactory::class);
$item = $itemFactory->create(['sku' => $sku, 'qty' => -1.0]);
/** @var SalesChannelInterfaceFactory $channelFactory */
$channelFactory = $objectManager->get(SalesChannelInterfaceFactory::class);
$channel = $channelFactory->create(['data' => [
    'type' => SalesChannelInterface::TYPE_WEBSITE,
    'code' => 'base',
]]);
/** @var SalesEventInterfaceFactory $eventFactory */
$eventFactory = $objectManager->get(SalesEventInterfaceFactory::class);
$event = $eventFactory->create([
    'type' => SalesEventInterface::EVENT_ORDER_PLACED,
    'objectType' => SalesEventInterface::OBJECT_TYPE_ORDER,
    'objectId' => 'mageos-hybrid-reservation-fixture',
]);
/** @var PlaceReservationsForSalesEventInterface $placeReservation */
$placeReservation = $objectManager->get(PlaceReservationsForSalesEventInterface::class);
$placeReservation->execute([$item], $channel, $event);

$manifest = [
    'store_id' => $storeId,
    'product_id' => $productId,
    'sku' => $sku,
    'generation_id' => $generationId,
    'physical_index' => (string)$generation['physical_index'],
    'before_boundary' => $beforeBoundary,
    'embedding_revision' => (int)$embedding['revision'],
    'source_hash' => (string)$embedding['source_hash'],
    'vector_hash' => (string)$embedding['vector_hash'],
];
if (file_put_contents($manifestPath, json_encode($manifest, JSON_THROW_ON_ERROR)) === false) {
    throw new RuntimeException('The reservation fixture manifest could not be written.');
}

printf("Placed reservation for %s and queued stock %d salability processing.\n", $sku, $storeId);
