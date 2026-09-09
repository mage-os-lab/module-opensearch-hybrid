<?php

declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Generation\BuildService;
use MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository;
use MageOS\OpenSearchHybrid\Model\Queue\DataPlanePublisher;
use Magento\Catalog\Api\ProductRepositoryInterface;
use Magento\Catalog\Model\ProductFactory;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\ResourceConnection;
use Magento\Framework\App\State;

if ($argc !== 3) {
    fwrite(STDERR, "Usage: php prepare-queue-isolation.php /path/to/mageos /path/to/manifest.json\n");
    exit(2);
}

$mageOsRoot = rtrim((string)$argv[1], DIRECTORY_SEPARATOR);
$manifestPath = (string)$argv[2];
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

$storeId = 1;
$backlogSize = 500;

/** @var BuildService $buildService */
$buildService = $objectManager->get(BuildService::class);
/** @var OutboxRepository $outboxRepository */
$outboxRepository = $objectManager->get(OutboxRepository::class);
/** @var DataPlanePublisher $publisher */
$publisher = $objectManager->get(DataPlanePublisher::class);
/** @var ProductRepositoryInterface $productRepository */
$productRepository = $objectManager->get(ProductRepositoryInterface::class);
/** @var ProductFactory $productFactory */
$productFactory = $objectManager->get(ProductFactory::class);
/** @var ResourceConnection $resourceConnection */
$resourceConnection = $objectManager->get(ResourceConnection::class);

$generationId = $buildService->buildFake($storeId);
$product = $productFactory->create();
$product->setTypeId('simple');
$product->setAttributeSetId($product->getDefaultAttributeSetId());
$product->setSku('mageos-hybrid-ci-product');
$product->setName('Mage-OS Hybrid CI Product');
$product->setDescription('Product used to prove atomic hybrid change capture.');
$product->setPrice(19.99);
$product->setWeight(1);
$product->setStatus(1);
$product->setVisibility(4);
$product->setWebsiteIds([1]);
$product = $productRepository->save($product);
$productId = (int)$product->getId();

$connection = $resourceConnection->getConnection();
$journalTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_change_journal');
$outboxTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_outbox');
$storeStateTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_store_state');
$journal = $connection->fetchRow(
    $connection->select()
        ->from($journalTable)
        ->where('store_id = ?', $storeId)
        ->where('entity_type = ?', 'PRODUCT')
        ->where('entity_id = ?', $productId)
        ->where('operation = ?', 'UPSERT')
        ->order('change_id DESC')
        ->limit(1)
);
if (!is_array($journal) || (string)$journal['native_state'] !== 'PENDING') {
    throw new RuntimeException('Scheduled product save did not create a pending native journal record.');
}
$watermark = (int)$journal['change_id'];
$storeState = $connection->fetchRow(
    $connection->select()
        ->from($storeStateTable)
        ->where('store_id = ?', $storeId)
);
if (!is_array($storeState)
    || (int)$storeState['required_watermark'] !== $watermark
    || (int)$storeState['native_watermark'] >= $watermark
    || (bool)$storeState['readiness_latch']
) {
    throw new RuntimeException('Pending native indexing did not keep the readiness latch closed.');
}
$jobs = $connection->fetchAll(
    $connection->select()
        ->from($outboxTable)
        ->where('store_id = ?', $storeId)
        ->where('generation_id = ?', $generationId)
        ->where('latest_change_id = ?', $watermark)
);
$jobsByLane = [];
foreach ($jobs as $job) {
    $jobsByLane[(string)$job['lane']][] = $job;
}
if (count($jobsByLane['EMBEDDING'] ?? []) !== 1 || count($jobsByLane['PRIORITY'] ?? []) !== 1) {
    throw new RuntimeException('Product save did not create one embedding and one priority job.');
}
$embeddingJobId = (string)$jobsByLane['EMBEDDING'][0]['job_id'];
$priorityJobId = (string)$jobsByLane['PRIORITY'][0]['job_id'];
if ((string)$outboxRepository->get($embeddingJobId)['state'] !== 'PUBLISHED'
    || (string)$outboxRepository->get($priorityJobId)['state'] !== 'PUBLISHED'
) {
    throw new RuntimeException('Product-save jobs were not published after transaction commit.');
}
for ($message = 0; $message < $backlogSize; $message++) {
    $publisher->publish($embeddingJobId, 'EMBEDDING');
}

$publisher->publish($priorityJobId, 'PRIORITY');

$manifest = [
    'store_id' => $storeId,
    'generation_id' => $generationId,
    'product_id' => $productId,
    'product_sku' => (string)$product->getSku(),
    'watermark' => $watermark,
    'backlog_size' => $backlogSize,
    'embedding_job_id' => $embeddingJobId,
    'priority_job_id' => $priorityJobId,
];
$encodedManifest = json_encode($manifest, JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR);
if (file_put_contents($manifestPath, $encodedManifest . "\n") === false) {
    throw new RuntimeException("Could not write queue-isolation manifest: {$manifestPath}");
}

fwrite(
    STDOUT,
    sprintf(
        "Captured a scheduled product save, closed readiness, and published %d embedding messages plus one correctness message for generation %d.\n",
        $backlogSize,
        $generationId
    )
);
