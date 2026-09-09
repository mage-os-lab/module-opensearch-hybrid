<?php
declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Generation\ValidationService;
use MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository;
use MageOS\OpenSearchHybrid\Model\Queue\EmbeddingConsumer;
use MageOS\OpenSearchHybrid\Model\Queue\PriorityConsumer;
use Magento\Catalog\Api\ProductRepositoryInterface;
use Magento\Catalog\Model\Product\Action as ProductAction;
use Magento\Elasticsearch\SearchAdapter\ConnectionManager;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\ResourceConnection;
use Magento\Framework\App\State;
use Magento\Framework\Indexer\IndexerRegistry;

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
    throw new RuntimeException('The source invalidation fixture requires a ready generation.');
}
$generationId = (int)$generation['generation_id'];

/** @var OutboxRepository $outboxRepository */
$outboxRepository = $objectManager->get(OutboxRepository::class);
/** @var PriorityConsumer $priorityConsumer */
$priorityConsumer = $objectManager->get(PriorityConsumer::class);
/** @var EmbeddingConsumer $embeddingConsumer */
$embeddingConsumer = $objectManager->get(EmbeddingConsumer::class);
/** @var IndexerRegistry $indexerRegistry */
$indexerRegistry = $objectManager->get(IndexerRegistry::class);
$nativeIndexer = $indexerRegistry->get('catalogsearch_fulltext');

$processChange = static function (
    int $changeId,
    int $productId,
    bool $expectEmbedding
) use (
    $connection,
    $outboxTable,
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
        throw new RuntimeException(sprintf('Source change %d has no priority work.', $changeId));
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
            ->where('latest_change_id = ?', $changeId)
            ->where('lane = ?', 'EMBEDDING')
            ->where('state NOT IN (?)', ['COMPLETED', 'SUPERSEDED'])
            ->order('generation_id ASC')
    ));
    if ($expectEmbedding && $embeddingJobs === []) {
        throw new RuntimeException(sprintf('Source change %d has no embedding work.', $changeId));
    }
    if (!$expectEmbedding && $embeddingJobs !== []) {
        throw new RuntimeException(sprintf('Delete source change %d created embedding work.', $changeId));
    }
    foreach ($embeddingJobs as $jobId) {
        if ((string)$outboxRepository->get($jobId)['state'] === 'PENDING') {
            $outboxRepository->markPublished($jobId);
        }
        $embeddingConsumer->process($jobId);
    }
};

$latestChange = static function (int $productId, string $reason) use ($connection, $journalTable): array {
    $change = $connection->fetchRow(
        $connection->select()
            ->from($journalTable)
            ->where('entity_id = ?', $productId)
            ->where('reason = ?', $reason)
            ->order('change_id DESC')
            ->limit(1)
    );
    if (!is_array($change)) {
        throw new RuntimeException(sprintf('No %s journal change was captured.', $reason));
    }

    return $change;
};

/** @var ProductRepositoryInterface $productRepository */
$productRepository = $objectManager->get(ProductRepositoryInterface::class);
$product = $productRepository->get('mageos-hybrid-resume-01', false, $storeId, true);
$productId = (int)$product->getId();
/** @var ProductAction $productAction */
$productAction = $objectManager->get(ProductAction::class);
$productAction->updateAttributes(
    [$productId],
    ['name' => 'Bulk Source Invalidation Product'],
    $storeId
);
$bulkChange = $latestChange($productId, 'product_attribute_bulk_update');
if ((string)$bulkChange['operation'] !== 'UPSERT') {
    throw new RuntimeException('Bulk product attributes did not capture an UPSERT.');
}
$processChange((int)$bulkChange['change_id'], $productId, true);

/** @var ValidationService $validationService */
$validationService = $objectManager->get(ValidationService::class);
$bulkValidation = $validationService->validate($generationId);
if (!(bool)$bulkValidation['ready']) {
    throw new RuntimeException('Bulk attribute invalidation did not return the generation to READY.');
}

$productAction->updateWebsites([$productId], [1], 'remove');
$removeChange = $latestChange($productId, 'product_website_remove');
if ((string)$removeChange['operation'] !== 'DELETE') {
    throw new RuntimeException('Website removal did not capture a DELETE.');
}
$processChange((int)$removeChange['change_id'], $productId, false);
$removeValidation = $validationService->validate($generationId);
$deletedEmbedding = $connection->fetchRow(
    $connection->select()
        ->from($embeddingTable)
        ->where('store_id = ?', $storeId)
        ->where('product_id = ?', $productId)
        ->where('generation_id = ?', $generationId)
);
if (!(bool)$removeValidation['ready']
    || !is_array($deletedEmbedding)
    || (string)$deletedEmbedding['status'] !== 'DELETED'
) {
    throw new RuntimeException('Website removal did not tombstone the store-scoped embedding.');
}

$productAction->updateWebsites([$productId], [1], 'add');
$addChange = $latestChange($productId, 'product_website_add');
if ((string)$addChange['operation'] !== 'UPSERT') {
    throw new RuntimeException('Website addition did not capture an UPSERT.');
}
$processChange((int)$addChange['change_id'], $productId, true);
$addValidation = $validationService->validate($generationId);
$restoredEmbedding = $connection->fetchRow(
    $connection->select()
        ->from($embeddingTable)
        ->where('store_id = ?', $storeId)
        ->where('product_id = ?', $productId)
        ->where('generation_id = ?', $generationId)
);
if (!(bool)$addValidation['ready']
    || !is_array($restoredEmbedding)
    || (string)$restoredEmbedding['status'] !== 'COMPLETE'
) {
    throw new RuntimeException('Website addition did not restore the store-scoped embedding.');
}

$product = $productRepository->getById($productId, false, $storeId, true);
$product->setWebsiteIds([]);
$productRepository->save($product);
$repositoryRemoveChange = $latestChange($productId, 'product_resource_save');
if ((string)$repositoryRemoveChange['operation'] !== 'DELETE') {
    throw new RuntimeException('Repository website removal lost the previous store scope.');
}
$processChange((int)$repositoryRemoveChange['change_id'], $productId, false);
if (!(bool)$validationService->validate($generationId)['ready']) {
    throw new RuntimeException('Repository website removal did not return the generation to READY.');
}

$product = $productRepository->getById($productId, false, $storeId, true);
$product->setWebsiteIds([1]);
$productRepository->save($product);
$repositoryAddChange = $latestChange($productId, 'product_resource_save');
if ((string)$repositoryAddChange['operation'] !== 'UPSERT') {
    throw new RuntimeException('Repository website addition did not restore the store scope.');
}
$processChange((int)$repositoryAddChange['change_id'], $productId, true);
if (!(bool)$validationService->validate($generationId)['ready']) {
    throw new RuntimeException('Repository website addition did not return the generation to READY.');
}

/** @var ConnectionManager $connectionManager */
$connectionManager = $objectManager->get(ConnectionManager::class);
$searchConnection = $connectionManager->getConnection();
if (!$searchConnection instanceof \Magento\OpenSearch\Model\SearchClient) {
    throw new RuntimeException('The source invalidation fixture has no OpenSearch client.');
}
$document = $searchConnection->getOpenSearchClient()->get([
    'index' => (string)$generation['physical_index'],
    'id' => (string)$productId,
]);
if (($document['_source']['title'] ?? null) !== 'Bulk Source Invalidation Product') {
    throw new RuntimeException('The restored hybrid document did not retain the bulk title change.');
}

printf(
    "Bulk change %d, mass website changes %d/%d, and repository website changes %d/%d completed for generation %d.\n",
    (int)$bulkChange['change_id'],
    (int)$removeChange['change_id'],
    (int)$addChange['change_id'],
    (int)$repositoryRemoveChange['change_id'],
    (int)$repositoryAddChange['change_id'],
    $generationId
);
