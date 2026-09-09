<?php

declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Config;
use MageOS\OpenSearchHybrid\Model\Generation\BuildRefreshService;
use MageOS\OpenSearchHybrid\Model\Generation\BuildService;
use MageOS\OpenSearchHybrid\Model\Generation\ValidationService;
use MageOS\OpenSearchHybrid\Model\OpenSearch\IndexManager;
use MageOS\OpenSearchHybrid\Model\Queue\EmbeddingConsumer;
use Magento\Catalog\Api\ProductRepositoryInterface;
use Magento\Catalog\Model\ProductFactory;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\Config\ReinitableConfigInterface;
use Magento\Framework\App\Config\Storage\WriterInterface;
use Magento\Framework\App\ResourceConnection;
use Magento\Framework\App\State;

if ($argc !== 2) {
    fwrite(STDERR, "Usage: php assert-build-resume.php /path/to/mageos\n");
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
try {
    $objectManager->get(State::class)->setAreaCode('global');
} catch (\Magento\Framework\Exception\LocalizedException) {
    // Another bootstrap participant already set the area code.
}

$storeId = 1;
/** @var WriterInterface $configWriter */
$configWriter = $objectManager->get(WriterInterface::class);
$configWriter->save(Config::XML_PATH_BATCH_SIZE, '2');
$configWriter->save(Config::XML_PATH_MAX_BATCHES, '1');
/** @var ReinitableConfigInterface $reinitableConfig */
$reinitableConfig = $objectManager->get(ReinitableConfigInterface::class);
$reinitableConfig->reinit();
/** @var ProductFactory $productFactory */
$productFactory = $objectManager->get(ProductFactory::class);
/** @var ProductRepositoryInterface $productRepository */
$productRepository = $objectManager->get(ProductRepositoryInterface::class);
for ($number = 1; $number <= 5; $number++) {
    $product = $productFactory->create();
    $product->setTypeId('simple');
    $product->setAttributeSetId($product->getDefaultAttributeSetId());
    $product->setSku(sprintf('mageos-hybrid-resume-%02d', $number));
    $product->setName(sprintf('Resumable Build Product %02d', $number));
    $product->setDescription('Product used to prove crash-safe keyset seeding.');
    $product->setPrice(10 + $number);
    $product->setWeight(1);
    $product->setStatus(1);
    $product->setVisibility(4);
    $product->setWebsiteIds([1]);
    $productRepository->save($product);
}

/** @var BuildService $buildService */
$buildService = $objectManager->get(BuildService::class);
$generationId = $buildService->buildFake($storeId);
/** @var ResourceConnection $resourceConnection */
$resourceConnection = $objectManager->get(ResourceConnection::class);
$connection = $resourceConnection->getConnection();
$generationTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_generation');
$progressTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_generation_progress');
$outboxTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_outbox');
$itemTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_outbox_item');
$progress = $connection->fetchRow(
    $connection->select()->from($progressTable)->where('generation_id = ?', $generationId)
);
$generation = $connection->fetchRow(
    $connection->select()->from($generationTable)->where('generation_id = ?', $generationId)
);
/** @var BuildRefreshService $buildRefreshService */
$buildRefreshService = $objectManager->get(BuildRefreshService::class);
/** @var IndexManager $indexManager */
$indexManager = $objectManager->get(IndexManager::class);
if (!is_array($progress)
    || !is_array($generation)
    || (string)$progress['refresh_state'] !== 'SUPPRESSED'
    || (string)$progress['build_refresh_interval'] !== '-1'
    || (string)$progress['normal_refresh_interval'] !== '1s'
    || (bool)$indexManager->validateGeneration($generation)['refresh_interval_restored']
) {
    throw new RuntimeException('The inactive build did not durably suppress periodic refresh.');
}
if (!is_array($progress)
    || (string)$progress['seeding_state'] !== 'PENDING'
    || (int)$progress['seeded_products'] !== 2
) {
    throw new RuntimeException('The bounded build did not stop after its first two-product batch.');
}
$pause = $buildService->pauseFake($generationId);
$repeatPause = $buildService->pauseFake($generationId);
$pausedProgress = $connection->fetchRow(
    $connection->select()->from($progressTable)->where('generation_id = ?', $generationId)
);
if ((string)$pause['seeding_state'] !== 'PAUSED'
    || (string)$repeatPause['seeding_state'] !== 'PAUSED'
    || !is_array($pausedProgress)
    || (string)$pausedProgress['seeding_state'] !== 'PAUSED'
    || (string)$pausedProgress['refresh_state'] !== 'SUPPRESSED'
) {
    throw new RuntimeException('The build did not persist its explicit pause state.');
}
$firstResume = $buildService->resumeFake($generationId);

/** @var EmbeddingConsumer $embeddingConsumer */
$embeddingConsumer = $objectManager->get(EmbeddingConsumer::class);
$sawCapacityPause = (string)$firstResume['seeding_state'] === 'PAUSED_CAPACITY';
for ($iteration = 0; $iteration < 20; $iteration++) {
    $progress = $connection->fetchRow(
        $connection->select()->from($progressTable)->where('generation_id = ?', $generationId)
    );
    if (is_array($progress) && (string)$progress['seeding_state'] === 'COMPLETED') {
        break;
    }
    $jobIds = array_map('strval', $connection->fetchCol(
        $connection->select()
            ->from($outboxTable, ['job_id'])
            ->where('generation_id = ?', $generationId)
            ->where('lane = ?', 'EMBEDDING')
            ->where('state NOT IN (?)', ['COMPLETED', 'SUPERSEDED'])
            ->order('created_at ASC')
    ));
    if (count($jobIds) !== 1) {
        throw new RuntimeException(sprintf(
            'The outstanding-batch cap expected one resumable job, found %d.',
            count($jobIds)
        ));
    }
    $embeddingConsumer->process($jobIds[0]);
    $report = $buildService->resumeFake($generationId);
    if ((string)$report['seeding_state'] === 'PAUSED_CAPACITY') {
        $sawCapacityPause = true;
    }
}
if (!$sawCapacityPause) {
    throw new RuntimeException('The resume probe never observed PAUSED_CAPACITY.');
}
$progress = $connection->fetchRow(
    $connection->select()->from($progressTable)->where('generation_id = ?', $generationId)
);
$generation = $connection->fetchRow(
    $connection->select()->from($generationTable)->where('generation_id = ?', $generationId)
);
if (!is_array($progress) || !is_array($generation)) {
    throw new RuntimeException('The resumed build lost its generation or progress record.');
}
$distinctSeeded = (int)$connection->fetchOne(
    $connection->select()
        ->from(['item' => $itemTable], [new Zend_Db_Expr('COUNT(DISTINCT item.product_id)')])
        ->joinInner(['job' => $outboxTable], 'job.job_id = item.job_id', [])
        ->where('job.generation_id = ?', $generationId)
        ->where('job.lane = ?', 'EMBEDDING')
);
if ((string)$progress['seeding_state'] !== 'COMPLETED'
    || (int)$progress['seeded_products'] !== (int)$generation['coverage_total']
    || $distinctSeeded !== (int)$generation['coverage_total']
    || (int)$generation['coverage_complete'] !== (int)$generation['coverage_total']
) {
    throw new RuntimeException('Resumed keyset seeding lost, duplicated, or failed to encode product work.');
}

/** @var ValidationService $validationService */
$validationService = $objectManager->get(ValidationService::class);
$validation = $validationService->validate($generationId);
$progress = $connection->fetchRow(
    $connection->select()->from($progressTable)->where('generation_id = ?', $generationId)
);
$generation = $connection->fetchRow(
    $connection->select()->from($generationTable)->where('generation_id = ?', $generationId)
);
if (!is_array($progress)
    || !is_array($generation)
    || (string)$progress['refresh_state'] !== 'RESTORED'
    || !$buildRefreshService->isRestored($generationId)
    || ($validation['checks']['build_refresh_restored'] ?? false) !== true
    || !(bool)$indexManager->validateGeneration($generation)['refresh_interval_restored']
) {
    throw new RuntimeException('Validation did not restore and verify the normal refresh interval.');
}

$configWriter->save(Config::XML_PATH_BATCH_SIZE, '64');
$configWriter->save(Config::XML_PATH_MAX_BATCHES, '8');
$reinitableConfig->reinit();

fwrite(
    STDOUT,
    sprintf(
        "Generation %d resumed through bounded batches and completed %d unique products.\n",
        $generationId,
        $distinctSeeded
    )
);
