<?php
declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository;
use MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository;
use MageOS\OpenSearchHybrid\Model\Queue\DataPlanePublisher;
use MageOS\OpenSearchHybrid\Model\Store\ReadinessLatch;
use Magento\Catalog\Api\Data\TierPriceInterface;
use Magento\Catalog\Api\Data\TierPriceInterfaceFactory;
use Magento\Catalog\Api\ProductRepositoryInterface;
use Magento\Catalog\Api\TierPriceStorageInterface;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\ResourceConnection;
use Magento\Framework\App\State;
use Magento\Framework\Indexer\IndexerRegistry;

if ($argc !== 3) {
    fwrite(STDERR, "Usage: php prepare-active-readiness.php /path/to/mageos /path/to/manifest.json\n");
    exit(2);
}

$fixtureRoot = rtrim((string)$argv[1], DIRECTORY_SEPARATOR);
$manifestPath = (string)$argv[2];
if (!is_file($fixtureRoot . '/app/bootstrap.php')) {
    fwrite(STDERR, "Mage-OS bootstrap not found: {$fixtureRoot}\n");
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
$stateTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_store_state');
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
        ->where('generation.result_contract_accepted = ?', 1)
        ->where('progress.seeding_state = ?', 'COMPLETED')
        ->order('generation.generation_id DESC')
        ->limit(1)
);
if (!is_array($generation)) {
    throw new RuntimeException('The active-readiness drill requires one validated ready generation.');
}
$pendingChanges = (int)$connection->fetchOne(
    $connection->select()
        ->from($journalTable, [new Zend_Db_Expr('COUNT(*)')])
        ->where('store_id = ?', $storeId)
        ->where('native_state != ? OR hybrid_state != ?', 'COMPLETED')
);
if ($pendingChanges !== 0) {
    throw new RuntimeException('The active-readiness drill requires a fully acknowledged journal.');
}
$generationId = (int)$generation['generation_id'];
$boundary = (int)$connection->fetchOne(
    $connection->select()
        ->from($journalTable, [new Zend_Db_Expr('COALESCE(MAX(change_id), 0)')])
        ->where('store_id = ?', $storeId)
);
$connection->beginTransaction();
try {
    $connection->update(
        $generationTable,
        ['state' => 'RETAINED'],
        ['store_id = ?' => $storeId, 'state = ?' => 'ACTIVE']
    );
    $connection->update(
        $generationTable,
        ['state' => 'ACTIVE'],
        ['generation_id = ?' => $generationId, 'state = ?' => 'READY']
    );
    $connection->insertOnDuplicate(
        $stateTable,
        [
            'store_id' => $storeId,
            'activation_mode' => 'HYBRID',
            'active_generation_id' => $generationId,
            'required_watermark' => $boundary,
            'native_watermark' => $boundary,
            'hybrid_watermark' => $boundary,
            'readiness_latch' => 1,
            'cache_version' => 1,
        ],
        [
            'activation_mode',
            'active_generation_id',
            'required_watermark',
            'native_watermark',
            'hybrid_watermark',
            'readiness_latch',
            'cache_version' => new Zend_Db_Expr('cache_version + 1'),
        ]
    );
    $connection->commit();
} catch (Throwable $throwable) {
    $connection->rollBack();
    throw $throwable;
}

/** @var ReadinessLatch $readinessLatch */
$readinessLatch = $objectManager->get(ReadinessLatch::class);
/** @var GenerationRepository $generationRepository */
$generationRepository = $objectManager->get(GenerationRepository::class);
$active = $generationRepository->activeForStore($storeId);
if (!$readinessLatch->isOpen($storeId)
    || !is_array($active)
    || (int)$active['generation_id'] !== $generationId
) {
    throw new RuntimeException('The deterministic active generation did not begin with open readiness.');
}

/** @var ProductRepositoryInterface $productRepository */
$productRepository = $objectManager->get(ProductRepositoryInterface::class);
$product = $productRepository->get('mageos-hybrid-resume-02', false, $storeId, true);
$productId = (int)$product->getId();
$beforeBoundary = (int)$connection->fetchOne(
    $connection->select()
        ->from($journalTable, [new Zend_Db_Expr('COALESCE(MAX(change_id), 0)')])
        ->where('store_id = ?', $storeId)
);

/** @var TierPriceInterfaceFactory $tierPriceFactory */
$tierPriceFactory = $objectManager->get(TierPriceInterfaceFactory::class);
/** @var TierPriceInterface $tierPrice */
$tierPrice = $tierPriceFactory->create();
$tierPrice->setSku((string)$product->getSku());
$tierPrice->setPrice(6.75);
$tierPrice->setPriceType(TierPriceInterface::PRICE_TYPE_FIXED);
$tierPrice->setWebsiteId(0);
$tierPrice->setCustomerGroup('all groups');
$tierPrice->setQuantity(1.0);
/** @var TierPriceStorageInterface $tierPriceStorage */
$tierPriceStorage = $objectManager->get(TierPriceStorageInterface::class);
if ($tierPriceStorage->update([$tierPrice]) !== []) {
    throw new RuntimeException('The active-readiness tier-price update failed validation.');
}
/** @var IndexerRegistry $indexerRegistry */
$indexerRegistry = $objectManager->get(IndexerRegistry::class);
$indexerRegistry->get('catalog_product_price')->getView()->update();

$changes = $connection->fetchAll(
    $connection->select()
        ->from($journalTable)
        ->where('change_id > ?', $beforeBoundary)
        ->where('store_id = ?', $storeId)
        ->where('entity_type = ?', 'PRODUCT')
        ->where('entity_id = ?', $productId)
        ->where('reason = ?', 'product_price_index')
        ->order('change_id ASC')
);
if (count($changes) !== 1
    || (string)$changes[0]['operation'] !== 'REFRESH'
    || (string)$changes[0]['native_state'] !== 'COMPLETED'
    || (string)$changes[0]['hybrid_state'] !== 'PENDING'
) {
    throw new RuntimeException('The active price change did not stop after its native acknowledgement.');
}
$changeId = (int)$changes[0]['change_id'];
$priorityJobs = $connection->fetchAll(
    $connection->select()
        ->from($outboxTable)
        ->where('generation_id = ?', $generationId)
        ->where('latest_change_id = ?', $changeId)
        ->where('lane = ?', 'PRIORITY')
        ->where('state NOT IN (?)', ['COMPLETED', 'SUPERSEDED'])
);
$embeddingJobs = (int)$connection->fetchOne(
    $connection->select()
        ->from($outboxTable, [new Zend_Db_Expr('COUNT(*)')])
        ->where('generation_id = ?', $generationId)
        ->where('latest_change_id = ?', $changeId)
        ->where('lane = ?', 'EMBEDDING')
);
if (count($priorityJobs) !== 1 || $embeddingJobs !== 0) {
    throw new RuntimeException('The active price change did not create exactly one priority-only job.');
}
$jobId = (string)$priorityJobs[0]['job_id'];
/** @var OutboxRepository $outboxRepository */
$outboxRepository = $objectManager->get(OutboxRepository::class);
if ((string)$outboxRepository->get($jobId)['state'] === 'PENDING') {
    $outboxRepository->markPublished($jobId);
}
/** @var DataPlanePublisher $publisher */
$publisher = $objectManager->get(DataPlanePublisher::class);
$publisher->publish($jobId, 'PRIORITY');
$state = $connection->fetchRow(
    $connection->select()->from($stateTable)->where('store_id = ?', $storeId)
);
if (!is_array($state)
    || (int)$state['required_watermark'] !== $changeId
    || (int)$state['native_watermark'] < $changeId
    || (int)$state['hybrid_watermark'] >= $changeId
    || (bool)$state['readiness_latch']
    || $readinessLatch->isOpen($storeId)
    || $generationRepository->activeForStore($storeId) !== null
) {
    throw new RuntimeException('Native-only acknowledgement did not keep the active store closed.');
}

$manifest = [
    'store_id' => $storeId,
    'generation_id' => $generationId,
    'product_id' => $productId,
    'change_id' => $changeId,
    'priority_job_id' => $jobId,
];
if (file_put_contents(
    $manifestPath,
    json_encode($manifest, JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR) . "\n"
) === false) {
    throw new RuntimeException("Could not write active-readiness manifest: {$manifestPath}");
}

printf(
    "Active generation %d closed at change %d and remained native-only after native acknowledgement.\n",
    $generationId,
    $changeId
);
