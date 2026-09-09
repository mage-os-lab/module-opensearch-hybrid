<?php
declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract;
use MageOS\OpenSearchHybrid\Model\Generation\ValidationService;
use MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository;
use MageOS\OpenSearchHybrid\Model\Queue\EmbeddingConsumer;
use MageOS\OpenSearchHybrid\Model\Queue\PriorityConsumer;
use Magento\Catalog\Api\ProductRepositoryInterface;
use Magento\Catalog\Model\ProductFactory;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\ResourceConnection;
use Magento\Framework\App\State;
use Magento\Framework\Indexer\IndexerRegistry;
use Magento\Framework\Registry;

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
$generationId = (int)$connection->fetchOne(
    $connection->select()
        ->from(['generation' => $generationTable], ['generation_id'])
        ->joinInner(
            ['progress' => $progressTable],
            'progress.generation_id = generation.generation_id',
            []
        )
        ->where('generation.store_id = ?', $storeId)
        ->where('generation.state IN (?)', ['BUILDING', 'CATCHING_UP', 'READY'])
        ->where('progress.seeding_state = ?', 'COMPLETED')
        ->where('generation.coverage_complete = generation.coverage_total')
        ->where('generation.coverage_failed = ?', 0)
        ->order('generation.generation_id DESC')
        ->limit(1)
);
if ($generationId <= 0) {
    throw new RuntimeException('The catch-up fixture requires a fully covered seeded generation.');
}

/** @var OutboxRepository $outboxRepository */
$outboxRepository = $objectManager->get(OutboxRepository::class);
/** @var PriorityConsumer $priorityConsumer */
$priorityConsumer = $objectManager->get(PriorityConsumer::class);
/** @var EmbeddingConsumer $embeddingConsumer */
$embeddingConsumer = $objectManager->get(EmbeddingConsumer::class);
/** @var IndexerRegistry $indexerRegistry */
$indexerRegistry = $objectManager->get(IndexerRegistry::class);
$nativeIndexer = $indexerRegistry->get('catalogsearch_fulltext');

$pendingPriorityJobs = array_map('strval', $connection->fetchCol(
    $connection->select()
        ->from($outboxTable, ['job_id'])
        ->where('store_id = ?', $storeId)
        ->where('lane = ?', 'PRIORITY')
        ->where('state NOT IN (?)', ['COMPLETED', 'SUPERSEDED'])
        ->order('created_at ASC')
));
foreach ($pendingPriorityJobs as $jobId) {
    if ((string)$outboxRepository->get($jobId)['state'] === 'PENDING') {
        $outboxRepository->markPublished($jobId);
    }
    $priorityConsumer->process($jobId);
}
$nativeIndexer->reindexAll();

$processChange = static function (
    int $changeId,
    int $productId,
    bool $expectEmbedding
) use (
    $connection,
    $outboxTable,
    $generationId,
    $nativeIndexer,
    $outboxRepository,
    $priorityConsumer,
    $embeddingConsumer
): void {
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
        throw new RuntimeException(sprintf('Change %d has no correctness-priority work.', $changeId));
    }
    foreach ($priorityJobs as $jobId) {
        if ((string)$outboxRepository->get($jobId)['state'] === 'PENDING') {
            $outboxRepository->markPublished($jobId);
        }
        $priorityConsumer->process($jobId);
    }
    $embeddingJobs = array_map('strval', $connection->fetchCol(
        $connection->select()
            ->from($outboxTable, ['job_id'])
            ->where('generation_id = ?', $generationId)
            ->where('latest_change_id = ?', $changeId)
            ->where('lane = ?', 'EMBEDDING')
            ->where('state NOT IN (?)', ['COMPLETED', 'SUPERSEDED'])
    ));
    if ($expectEmbedding && count($embeddingJobs) !== 1) {
        throw new RuntimeException(sprintf('Change %d did not create one target embedding job.', $changeId));
    }
    if (!$expectEmbedding && $embeddingJobs !== []) {
        throw new RuntimeException(sprintf('Delete change %d unexpectedly created embedding work.', $changeId));
    }
    foreach ($embeddingJobs as $jobId) {
        if ((string)$outboxRepository->get($jobId)['state'] === 'PENDING') {
            $outboxRepository->markPublished($jobId);
        }
        $embeddingConsumer->process($jobId);
    }
};

/** @var ProductFactory $productFactory */
$productFactory = $objectManager->get(ProductFactory::class);
/** @var ProductRepositoryInterface $productRepository */
$productRepository = $objectManager->get(ProductRepositoryInterface::class);
$product = $productFactory->create();
$product->setTypeId('simple');
$product->setAttributeSetId($product->getDefaultAttributeSetId());
$product->setSku('mageos-hybrid-catch-up-product');
$product->setName('Catch-up Product');
$product->setDescription('Product created after full-build seeding completed.');
$product->setPrice(39.99);
$product->setWeight(1);
$product->setStatus(1);
$product->setVisibility(4);
$product->setWebsiteIds([1]);
$product = $productRepository->save($product);
$productId = (int)$product->getId();
$createChangeId = (int)$connection->fetchOne(
    $connection->select()
        ->from($journalTable, [new Zend_Db_Expr('MAX(change_id)')])
        ->where('store_id = ?', $storeId)
        ->where('entity_id = ?', $productId)
);
$processChange($createChangeId, $productId, true);

/** @var ValidationService $validationService */
$validationService = $objectManager->get(ValidationService::class);
/** @var RetrievalContract $retrievalContract */
$retrievalContract = $objectManager->get(RetrievalContract::class);
$firstValidation = $validationService->validate(
    $generationId,
    $retrievalContract->resultContractDigest()
);
if (!(bool)$firstValidation['ready'] || !(bool)$firstValidation['result_contract_accepted']) {
    $failedChecks = array_keys(array_filter(
        $firstValidation['checks'] ?? [],
        static fn ($passed): bool => $passed !== true
    ));
    throw new RuntimeException(sprintf(
        'The generation did not become ready after create catch-up. Failed checks: %s.',
        $failedChecks === [] ? 'none' : implode(', ', $failedChecks)
    ));
}

$product = $productRepository->getById($productId, false, $storeId, true);
$product->setName('Renamed Catch-up Product');
$productRepository->save($product);
$updateChangeId = (int)$connection->fetchOne(
    $connection->select()
        ->from($journalTable, [new Zend_Db_Expr('MAX(change_id)')])
        ->where('store_id = ?', $storeId)
        ->where('entity_id = ?', $productId)
);
$invalidatedGeneration = $connection->fetchRow(
    $connection->select()->from($generationTable)->where('generation_id = ?', $generationId)
);
if (!is_array($invalidatedGeneration)
    || (string)$invalidatedGeneration['state'] !== 'CATCHING_UP'
    || $invalidatedGeneration['validation_report'] !== null
) {
    throw new RuntimeException('New work did not invalidate the ready generation.');
}
$processChange($updateChangeId, $productId, true);
$secondValidation = $validationService->validate($generationId);
if (!(bool)$secondValidation['ready'] || !(bool)$secondValidation['result_contract_accepted']) {
    throw new RuntimeException('Revalidation did not preserve accepted result-contract identity.');
}

/** @var Registry $registry */
$registry = $objectManager->get(Registry::class);
$alreadySecure = (bool)$registry->registry('isSecureArea');
if (!$alreadySecure) {
    $registry->register('isSecureArea', true);
}
try {
    $productRepository->deleteById((string)$product->getSku());
} finally {
    if (!$alreadySecure) {
        $registry->unregister('isSecureArea');
    }
}
$deleteChangeId = (int)$connection->fetchOne(
    $connection->select()
        ->from($journalTable, [new Zend_Db_Expr('MAX(change_id)')])
        ->where('store_id = ?', $storeId)
        ->where('entity_id = ?', $productId)
);
$processChange($deleteChangeId, $productId, false);
$deleteValidation = $validationService->validate($generationId);
$generation = $connection->fetchRow(
    $connection->select()->from($generationTable)->where('generation_id = ?', $generationId)
);
$embedding = $connection->fetchRow(
    $connection->select()
        ->from($embeddingTable)
        ->where('store_id = ?', $storeId)
        ->where('product_id = ?', $productId)
        ->where('generation_id = ?', $generationId)
);
if (!(bool)$deleteValidation['ready']
    || !is_array($generation)
    || (int)$generation['coverage_complete'] !== (int)$generation['coverage_total']
    || !is_array($embedding)
    || (string)$embedding['status'] !== 'DELETED'
) {
    throw new RuntimeException('Delete catch-up did not reconcile coverage and embedding tombstone state.');
}

printf(
    "Generation %d caught up create, rename, and delete through boundary %d with coverage %d.\n",
    $generationId,
    $deleteChangeId,
    (int)$generation['coverage_total']
);
