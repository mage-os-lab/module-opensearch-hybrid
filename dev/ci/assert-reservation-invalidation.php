<?php
declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Generation\ValidationService;
use MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository;
use MageOS\OpenSearchHybrid\Model\Queue\PriorityConsumer;
use Magento\Elasticsearch\SearchAdapter\ConnectionManager;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\ResourceConnection;
use Magento\Framework\App\State;
use Magento\Framework\Indexer\IndexerRegistry;

$fixtureRoot = $argv[1] ?? '';
$manifestPath = $argv[2] ?? '';
if ($fixtureRoot === '' || !is_file($fixtureRoot . '/app/bootstrap.php') || !is_file($manifestPath)) {
    fwrite(STDERR, "Provide the installed Mage-OS fixture root and reservation manifest path.\n");
    exit(2);
}
$manifest = json_decode((string)file_get_contents($manifestPath), true, 512, JSON_THROW_ON_ERROR);

require $fixtureRoot . '/app/bootstrap.php';
$bootstrap = Bootstrap::create(BP, $_SERVER);
$objectManager = $bootstrap->getObjectManager();
try {
    $objectManager->get(State::class)->setAreaCode('global');
} catch (\Magento\Framework\Exception\LocalizedException) {
}

/** @var ResourceConnection $resourceConnection */
$resourceConnection = $objectManager->get(ResourceConnection::class);
$connection = $resourceConnection->getConnection();
$journalTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_change_journal');
$outboxTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_outbox');
$embeddingTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_embedding');
$changes = $connection->fetchAll(
    $connection->select()
        ->from($journalTable)
        ->where('change_id > ?', (int)$manifest['before_boundary'])
        ->where('store_id = ?', (int)$manifest['store_id'])
        ->where('entity_id = ?', (int)$manifest['product_id'])
        ->where('reason = ?', 'inventory_reservation_change')
        ->order('change_id ASC')
);
if (count($changes) !== 1 || (string)$changes[0]['operation'] !== 'REFRESH') {
    throw new RuntimeException('The reservation consumer did not capture one priority REFRESH.');
}
$changeId = (int)$changes[0]['change_id'];
$embeddingJobs = (int)$connection->fetchOne(
    $connection->select()
        ->from($outboxTable, [new Zend_Db_Expr('COUNT(*)')])
        ->where('latest_change_id = ?', $changeId)
        ->where('lane = ?', 'EMBEDDING')
);
if ($embeddingJobs !== 0) {
    throw new RuntimeException('A reservation-only update created encoder work.');
}

/** @var IndexerRegistry $indexerRegistry */
$indexerRegistry = $objectManager->get(IndexerRegistry::class);
$indexerRegistry->get('catalogsearch_fulltext')->reindexList([(int)$manifest['product_id']]);
/** @var OutboxRepository $outboxRepository */
$outboxRepository = $objectManager->get(OutboxRepository::class);
/** @var PriorityConsumer $priorityConsumer */
$priorityConsumer = $objectManager->get(PriorityConsumer::class);
$priorityJobs = array_map('strval', $connection->fetchCol(
    $connection->select()
        ->from($outboxTable, ['job_id'])
        ->where('latest_change_id = ?', $changeId)
        ->where('lane = ?', 'PRIORITY')
        ->where('state NOT IN (?)', ['COMPLETED', 'SUPERSEDED'])
        ->order('generation_id ASC')
));
if ($priorityJobs === []) {
    throw new RuntimeException('The reservation capture has no priority work.');
}
foreach ($priorityJobs as $jobId) {
    if ((string)$outboxRepository->get($jobId)['state'] === 'PENDING') {
        $outboxRepository->markPublished($jobId);
    }
    $priorityConsumer->process($jobId);
}
/** @var ValidationService $validationService */
$validationService = $objectManager->get(ValidationService::class);
if (!(bool)$validationService->validate((int)$manifest['generation_id'])['ready']) {
    throw new RuntimeException('Reservation catch-up did not return the generation to READY.');
}

/** @var ConnectionManager $connectionManager */
$connectionManager = $objectManager->get(ConnectionManager::class);
$searchConnection = $connectionManager->getConnection();
if (!$searchConnection instanceof \Magento\OpenSearch\Model\SearchClient) {
    throw new RuntimeException('The reservation fixture has no OpenSearch client.');
}
$document = $searchConnection->getOpenSearchClient()->get([
    'index' => (string)$manifest['physical_index'],
    'id' => (string)$manifest['product_id'],
]);
if (($document['_source']['is_salable'] ?? null) !== false) {
    throw new RuntimeException('The reservation-driven hybrid document remained salable.');
}
$embedding = $connection->fetchRow(
    $connection->select()
        ->from($embeddingTable)
        ->where('store_id = ?', (int)$manifest['store_id'])
        ->where('product_id = ?', (int)$manifest['product_id'])
        ->where('generation_id = ?', (int)$manifest['generation_id'])
);
if (!is_array($embedding)
    || (int)$embedding['revision'] !== (int)$manifest['embedding_revision']
    || !hash_equals((string)$manifest['source_hash'], (string)$embedding['source_hash'])
    || !hash_equals((string)$manifest['vector_hash'], (string)$embedding['vector_hash'])
) {
    throw new RuntimeException('Reservation refresh changed the semantic embedding.');
}

printf(
    "Reservation queue change %d refreshed generation %d without encoder work.\n",
    $changeId,
    (int)$manifest['generation_id']
);
