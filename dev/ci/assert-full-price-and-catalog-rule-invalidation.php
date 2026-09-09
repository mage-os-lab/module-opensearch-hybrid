<?php
declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Generation\ValidationService;
use MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository;
use MageOS\OpenSearchHybrid\Model\Queue\PriorityConsumer;
use Magento\Catalog\Api\ProductRepositoryInterface;
use Magento\CatalogRule\Model\ResourceModel\Rule as RuleResource;
use Magento\CatalogRule\Model\ResourceModel\Rule\CollectionFactory as RuleCollectionFactory;
use Magento\CatalogRule\Model\Rule\Action\Collection as RuleActionCollection;
use Magento\CatalogRule\Model\Rule\Condition\Combine as RuleConditionCombine;
use Magento\CatalogRule\Model\RuleFactory;
use Magento\Customer\Model\ResourceModel\Group\CollectionFactory as CustomerGroupCollectionFactory;
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
        ->where('generation.state IN (?)', ['READY', 'CATCHING_UP'])
        ->where('progress.seeding_state = ?', 'COMPLETED')
        ->order('generation.generation_id DESC')
        ->limit(1)
);
if (!is_array($generation)) {
    throw new RuntimeException('The full-price fixture requires a ready generation.');
}
$generationId = (int)$generation['generation_id'];

/** @var ProductRepositoryInterface $productRepository */
$productRepository = $objectManager->get(ProductRepositoryInterface::class);
$product = $productRepository->get('mageos-hybrid-resume-02', false, $storeId, true);
$productId = (int)$product->getId();
/** @var OutboxRepository $outboxRepository */
$outboxRepository = $objectManager->get(OutboxRepository::class);
/** @var PriorityConsumer $priorityConsumer */
$priorityConsumer = $objectManager->get(PriorityConsumer::class);
/** @var ValidationService $validationService */
$validationService = $objectManager->get(ValidationService::class);
$processPriorityAfter = static function (int $changeBoundary, string $context) use (
    $connection,
    $outboxTable,
    $outboxRepository,
    $priorityConsumer
): int {
    $jobIds = array_map('strval', $connection->fetchCol(
        $connection->select()
            ->from($outboxTable, ['job_id'])
            ->where('latest_change_id > ?', $changeBoundary)
            ->where('lane = ?', 'PRIORITY')
            ->where('state NOT IN (?)', ['COMPLETED', 'SUPERSEDED'])
            ->order('latest_change_id ASC')
            ->order('generation_id ASC')
    ));
    if ($jobIds === []) {
        throw new RuntimeException($context . ' has no priority work.');
    }
    foreach ($jobIds as $jobId) {
        if ((string)$outboxRepository->get($jobId)['state'] === 'PENDING') {
            $outboxRepository->markPublished($jobId);
        }
        $priorityConsumer->process($jobId);
    }

    return count($jobIds);
};
$changeBoundary = static fn (): int => (int)$connection->fetchOne(
    $connection->select()
        ->from($journalTable, [new Zend_Db_Expr('COALESCE(MAX(change_id), 0)')])
);
if ((string)$generation['state'] === 'CATCHING_UP') {
    $processPriorityAfter(0, 'Interrupted price fixture recovery');
    if (!(bool)$validationService->validate($generationId)['ready']) {
        throw new RuntimeException('Interrupted price fixture recovery did not return the generation to READY.');
    }
}

$embeddingBefore = $connection->fetchRow(
    $connection->select()
        ->from($embeddingTable)
        ->where('store_id = ?', $storeId)
        ->where('product_id = ?', $productId)
        ->where('generation_id = ?', $generationId)
);
if (!is_array($embeddingBefore) || (string)$embeddingBefore['status'] !== 'COMPLETE') {
    throw new RuntimeException('The full-price fixture product has no completed embedding to preserve.');
}

/** @var IndexerRegistry $indexerRegistry */
$indexerRegistry = $objectManager->get(IndexerRegistry::class);
$priceIndexer = $indexerRegistry->get('catalog_product_price');
$catalogRuleIndexer = $indexerRegistry->get('catalogrule_rule');
$catalogProductRuleIndexer = $indexerRegistry->get('catalogrule_product');
$catalogRuleIndexer->setScheduled(false);
$catalogProductRuleIndexer->setScheduled(false);
$priceIndexer->setScheduled(false);
$fullBoundary = $changeBoundary();
/** @var RuleFactory $ruleFactory */
$ruleFactory = $objectManager->get(RuleFactory::class);
/** @var RuleResource $ruleResource */
$ruleResource = $objectManager->get(RuleResource::class);
/** @var RuleCollectionFactory $ruleCollectionFactory */
$ruleCollectionFactory = $objectManager->get(RuleCollectionFactory::class);
foreach ($ruleCollectionFactory->create()->addFieldToFilter(
    'name',
    'MageOS Hybrid Fixture 50 Percent'
) as $existingRule) {
    $ruleResource->delete($existingRule);
}
$catalogRuleIndexer->reindexAll();
$priceIndexer->reindexAll();
$fullChanges = $connection->fetchAll(
    $connection->select()
        ->from($journalTable)
        ->where('change_id > ?', $fullBoundary)
        ->where('entity_id = ?', $productId)
        ->where('reason = ?', 'product_price_full_reindex')
        ->order('change_id ASC')
);
if (count($fullChanges) !== 1
    || (string)$fullChanges[0]['operation'] !== 'REFRESH'
    || (string)$fullChanges[0]['native_state'] !== 'COMPLETED'
) {
    throw new RuntimeException('Full price reindex did not capture one native-complete product REFRESH.');
}
$fullEmbeddingJobs = (int)$connection->fetchOne(
    $connection->select()
        ->from($outboxTable, [new Zend_Db_Expr('COUNT(*)')])
        ->where('latest_change_id > ?', $fullBoundary)
        ->where('lane = ?', 'EMBEDDING')
);
if ($fullEmbeddingJobs !== 0) {
    throw new RuntimeException('Full price reindex created encoder work.');
}
$fullJobCount = $processPriorityAfter($fullBoundary, 'Full price reindex');
if (!(bool)$validationService->validate($generationId)['ready']) {
    throw new RuntimeException('Full price reindex catch-up did not return the generation to READY.');
}

/** @var ConnectionManager $connectionManager */
$connectionManager = $objectManager->get(ConnectionManager::class);
$searchConnection = $connectionManager->getConnection();
if (!$searchConnection instanceof \Magento\OpenSearch\Model\SearchClient) {
    throw new RuntimeException('The full-price fixture has no OpenSearch client.');
}
$searchClient = $searchConnection->getOpenSearchClient();
$beforeRuleDocument = $searchClient->get([
    'index' => (string)$generation['physical_index'],
    'id' => (string)$productId,
]);
$beforeRuleRows = $beforeRuleDocument['_source']['customer_group_prices'] ?? null;
if (!is_array($beforeRuleRows) || $beforeRuleRows === []) {
    throw new RuntimeException('Full price reindex did not render canonical price rows.');
}

/** @var CustomerGroupCollectionFactory $customerGroupCollectionFactory */
$customerGroupCollectionFactory = $objectManager->get(CustomerGroupCollectionFactory::class);
$customerGroupIds = array_map('intval', $customerGroupCollectionFactory->create()->getAllIds());
if ($customerGroupIds === []) {
    throw new RuntimeException('The catalog-rule fixture found no customer groups.');
}
$rule = $ruleFactory->create();
$rule->setData([
    'name' => 'MageOS Hybrid Fixture 50 Percent',
    'description' => 'Proves catalog-rule price changes reach hybrid documents.',
    'is_active' => 1,
    'website_ids' => [1],
    'customer_group_ids' => $customerGroupIds,
    'stop_rules_processing' => 1,
    'simple_action' => 'by_percent',
    'discount_amount' => 50,
    'sort_order' => 0,
]);
$rule->getConditions()->loadArray([
    'type' => RuleConditionCombine::class,
    'aggregator' => 'all',
    'value' => '1',
]);
$rule->getActions()->loadArray([
    'type' => RuleActionCollection::class,
]);
$ruleBoundary = $changeBoundary();
$ruleResource->save($rule);
if ((int)$rule->getId() <= 0) {
    throw new RuntimeException('The catalog-rule fixture did not persist its rule.');
}
$matchingProductIds = array_map('intval', array_keys($rule->getMatchingProductIds()));
if (!in_array($productId, $matchingProductIds, true)) {
    throw new RuntimeException(sprintf(
        'The catalog-rule fixture did not match product %d among %d products.',
        $productId,
        count($matchingProductIds)
    ));
}
$ruleChanges = $connection->fetchAll(
    $connection->select()
        ->from($journalTable)
        ->where('change_id > ?', $ruleBoundary)
        ->where('entity_id = ?', $productId)
        ->where('reason = ?', 'product_price_index')
        ->order('change_id ASC')
);
if (count($ruleChanges) !== 1
    || (string)$ruleChanges[0]['operation'] !== 'REFRESH'
    || (string)$ruleChanges[0]['native_state'] !== 'COMPLETED'
) {
    throw new RuntimeException('Catalog-rule save did not capture one native-complete product REFRESH.');
}
$ruleEmbeddingJobs = (int)$connection->fetchOne(
    $connection->select()
        ->from($outboxTable, [new Zend_Db_Expr('COUNT(*)')])
        ->where('latest_change_id > ?', $ruleBoundary)
        ->where('lane = ?', 'EMBEDDING')
);
if ($ruleEmbeddingJobs !== 0) {
    throw new RuntimeException('Catalog-rule price change created encoder work.');
}
$processPriorityAfter($ruleBoundary, 'Catalog-rule price change');
if (!(bool)$validationService->validate($generationId)['ready']) {
    throw new RuntimeException('Catalog-rule catch-up did not return the generation to READY.');
}

$afterRuleDocument = $searchClient->get([
    'index' => (string)$generation['physical_index'],
    'id' => (string)$productId,
]);
$afterRuleRows = $afterRuleDocument['_source']['customer_group_prices'] ?? null;
if (!is_array($afterRuleRows) || count($afterRuleRows) !== count($beforeRuleRows)) {
    throw new RuntimeException('Catalog-rule refresh changed the canonical price-row cardinality.');
}
$discountedPrice = round((float)$beforeRuleRows[0]['regular_price'] * 0.50, 4);
foreach ($afterRuleRows as $priceRow) {
    if (round((float)$priceRow['final_price'], 4) !== $discountedPrice) {
        throw new RuntimeException('Hybrid price rows do not contain the catalog-rule final price.');
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
    throw new RuntimeException('Full or catalog-rule price refresh changed the semantic embedding.');
}

$cleanupBoundary = $changeBoundary();
$ruleResource->delete($rule);
$catalogRuleIndexer->reindexAll();
$priceIndexer->reindexAll();
$processPriorityAfter($cleanupBoundary, 'Catalog-rule cleanup');
if (!(bool)$validationService->validate($generationId)['ready']) {
    throw new RuntimeException('Catalog-rule cleanup did not return the generation to READY.');
}

printf(
    "Full price reindex emitted %d bounded priority jobs; catalog rule %d indexed final_price %.2f without encoder work.\n",
    $fullJobCount,
    (int)$rule->getId(),
    $discountedPrice
);
