<?php
declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Document\ProductDocumentFactory;
use MageOS\OpenSearchHybrid\Model\Generation\ValidationService;
use MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository;
use MageOS\OpenSearchHybrid\Model\Queue\EmbeddingConsumer;
use MageOS\OpenSearchHybrid\Model\Queue\PriorityConsumer;
use Magento\Catalog\Api\ProductAttributeOptionManagementInterface;
use Magento\Catalog\Api\ProductAttributeOptionUpdateInterface;
use Magento\Catalog\Api\ProductRepositoryInterface;
use Magento\Elasticsearch\SearchAdapter\ConnectionManager;
use Magento\Eav\Api\Data\AttributeOptionInterfaceFactory;
use Magento\Eav\Api\Data\AttributeOptionLabelInterfaceFactory;
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
$generation = $connection->fetchRow(
    $connection->select()
        ->from(['generation' => $generationTable])
        ->joinInner(
            ['progress' => $progressTable],
            'progress.generation_id = generation.generation_id',
            []
        )
        ->where('generation.store_id = ?', $storeId)
        ->where('generation.state IN (?)', ['BUILDING', 'CATCHING_UP', 'READY'])
        ->where('progress.seeding_state = ?', 'COMPLETED')
        ->order('generation.generation_id DESC')
        ->limit(1)
);
if (!is_array($generation)) {
    throw new RuntimeException('The attribute option fixture requires a seeded writable generation.');
}
$generationId = (int)$generation['generation_id'];

/** @var ProductAttributeOptionManagementInterface $optionManagement */
$optionManagement = $objectManager->get(ProductAttributeOptionManagementInterface::class);
/** @var AttributeOptionInterfaceFactory $optionFactory */
$optionFactory = $objectManager->get(AttributeOptionInterfaceFactory::class);
$adminLabel = 'Hybrid Fixture Azure';
$optionId = (int)$connection->fetchOne(
    $connection->select()
        ->from(['option' => $resourceConnection->getTableName('eav_attribute_option')], ['option_id'])
        ->joinInner(
            ['value' => $resourceConnection->getTableName('eav_attribute_option_value')],
            'value.option_id = option.option_id',
            []
        )
        ->joinInner(
            ['attribute' => $resourceConnection->getTableName('eav_attribute')],
            'attribute.attribute_id = option.attribute_id',
            []
        )
        ->where('attribute.attribute_code = ?', 'color')
        ->where('value.store_id = ?', 0)
        ->where('value.value = ?', $adminLabel)
        ->limit(1)
);
if ($optionId <= 0) {
    $newOption = $optionFactory->create();
    $newOption->setLabel($adminLabel);
    $newOption->setSortOrder(50);
    $newOption->setIsDefault(false);
    $optionId = (int)$optionManagement->add('color', $newOption);
}
if ($optionId <= 0) {
    throw new RuntimeException('The fixture could not create a color option.');
}

$attributeId = (int)$connection->fetchOne(
    $connection->select()
        ->from($resourceConnection->getTableName('eav_attribute'), ['attribute_id'])
        ->where('attribute_code = ?', 'color')
);
$attributeSetId = (int)$connection->fetchOne(
    $connection->select()
        ->from($resourceConnection->getTableName('catalog_product_entity'), ['attribute_set_id'])
        ->where('sku = ?', 'mageos-hybrid-resume-02')
);
$attributeGroupId = (int)$connection->fetchOne(
    $connection->select()
        ->from($resourceConnection->getTableName('eav_attribute_group'), ['attribute_group_id'])
        ->where('attribute_set_id = ?', $attributeSetId)
        ->order('sort_order ASC')
        ->limit(1)
);
if ($attributeId <= 0 || $attributeSetId <= 0 || $attributeGroupId <= 0) {
    throw new RuntimeException('The fixture could not resolve the color attribute-set placement.');
}
$connection->insertOnDuplicate(
    $resourceConnection->getTableName('eav_entity_attribute'),
    [
        'entity_type_id' => (int)$connection->fetchOne(
            $connection->select()
                ->from($resourceConnection->getTableName('eav_attribute'), ['entity_type_id'])
                ->where('attribute_id = ?', $attributeId)
        ),
        'attribute_set_id' => $attributeSetId,
        'attribute_group_id' => $attributeGroupId,
        'attribute_id' => $attributeId,
        'sort_order' => 100,
    ],
    ['attribute_group_id', 'sort_order']
);
$objectManager->get(\Magento\Eav\Model\Config::class)->clear();

/** @var ProductRepositoryInterface $productRepository */
$productRepository = $objectManager->get(ProductRepositoryInterface::class);
$product = $productRepository->get('mageos-hybrid-resume-02', false, 0, true);
$productId = (int)$product->getId();
$beforeAssignmentBoundary = (int)$connection->fetchOne(
    $connection->select()
        ->from($journalTable, [new Zend_Db_Expr('COALESCE(MAX(change_id), 0)')])
);
$product->setData('color', $optionId);
$productRepository->save($product);
$assignmentChangeId = (int)$connection->fetchOne(
    $connection->select()
        ->from($journalTable, [new Zend_Db_Expr('MAX(change_id)')])
        ->where('change_id > ?', $beforeAssignmentBoundary)
        ->where('store_id = ?', $storeId)
        ->where('entity_id = ?', $productId)
        ->where('reason = ?', 'product_resource_save')
);
if ($assignmentChangeId <= 0) {
    throw new RuntimeException('The color assignment did not create a product change.');
}

/** @var OutboxRepository $outboxRepository */
$outboxRepository = $objectManager->get(OutboxRepository::class);
/** @var PriorityConsumer $priorityConsumer */
$priorityConsumer = $objectManager->get(PriorityConsumer::class);
/** @var EmbeddingConsumer $embeddingConsumer */
$embeddingConsumer = $objectManager->get(EmbeddingConsumer::class);
$processJobs = static function (int $changeId, string $lane) use (
    $connection,
    $outboxTable,
    $outboxRepository,
    $priorityConsumer,
    $embeddingConsumer
): array {
    $jobIds = array_map('strval', $connection->fetchCol(
        $connection->select()
            ->from($outboxTable, ['job_id'])
            ->where('latest_change_id = ?', $changeId)
            ->where('lane = ?', $lane)
            ->where('state NOT IN (?)', ['COMPLETED', 'SUPERSEDED'])
            ->order('generation_id ASC')
    ));
    foreach ($jobIds as $jobId) {
        if ((string)$outboxRepository->get($jobId)['state'] === 'PENDING') {
            $outboxRepository->markPublished($jobId);
        }
        if ($lane === 'PRIORITY') {
            $priorityConsumer->process($jobId);
        } else {
            $embeddingConsumer->process($jobId);
        }
    }

    return $jobIds;
};
/** @var IndexerRegistry $indexerRegistry */
$indexerRegistry = $objectManager->get(IndexerRegistry::class);
$indexerRegistry->get('catalogsearch_fulltext')->reindexList([$productId]);
$processJobs($assignmentChangeId, 'PRIORITY');
$processJobs($assignmentChangeId, 'EMBEDDING');

/** @var ProductDocumentFactory $documentFactory */
$documentFactory = $objectManager->get(ProductDocumentFactory::class);
$beforeDocument = $documentFactory->create($productId, $storeId);
$currentStoreLabel = $connection->fetchOne(
    $connection->select()
        ->from($resourceConnection->getTableName('eav_attribute_option_value'), ['value'])
        ->where('option_id = ?', $optionId)
        ->where('store_id = ?', $storeId)
);
$beforeLabel = is_string($currentStoreLabel) && $currentStoreLabel !== ''
    ? $currentStoreLabel
    : $adminLabel;
if (!str_contains((string)$beforeDocument['features'], $beforeLabel)
    || !str_contains((string)$beforeDocument['rendered'], $beforeLabel)
) {
    throw new RuntimeException(sprintf(
        'The frozen semantic_feature_attributes registry omitted color option %s from %s.',
        (string)$productRepository->getById($productId, false, $storeId, true)->getData('color'),
        json_encode($beforeDocument, JSON_THROW_ON_ERROR)
    ));
}

/** @var ProductAttributeOptionUpdateInterface $optionUpdate */
$optionUpdate = $objectManager->get(ProductAttributeOptionUpdateInterface::class);
/** @var AttributeOptionLabelInterfaceFactory $labelFactory */
$labelFactory = $objectManager->get(AttributeOptionLabelInterfaceFactory::class);
$storeLabelText = $beforeLabel === 'Hybrid Fixture Cobalt'
    ? 'Hybrid Fixture Indigo'
    : 'Hybrid Fixture Cobalt';
$storeLabel = $labelFactory->create();
$storeLabel->setStoreId($storeId);
$storeLabel->setLabel($storeLabelText);
$updatedOption = $optionFactory->create();
$updatedOption->setLabel($adminLabel);
$updatedOption->setSortOrder(50);
$updatedOption->setIsDefault(false);
$updatedOption->setStoreLabels([$storeLabel]);
$beforeOptionBoundary = (int)$connection->fetchOne(
    $connection->select()
        ->from($journalTable, [new Zend_Db_Expr('COALESCE(MAX(change_id), 0)')])
);
if (!$optionUpdate->update('color', $optionId, $updatedOption)) {
    throw new RuntimeException('The color option store label update failed.');
}
$changes = $connection->fetchAll(
    $connection->select()
        ->from($journalTable)
        ->where('change_id > ?', $beforeOptionBoundary)
        ->where('store_id = ?', $storeId)
        ->where('entity_id = ?', $productId)
        ->where('reason = ?', 'attribute_option_label_change')
        ->order('change_id ASC')
);
if (count($changes) !== 1 || (string)$changes[0]['operation'] !== 'UPSERT') {
    throw new RuntimeException('The mapped option label update did not create one product UPSERT.');
}
$changeId = (int)$changes[0]['change_id'];
$priorityJobs = $processJobs($changeId, 'PRIORITY');
$embeddingJobs = array_map('strval', $connection->fetchCol(
    $connection->select()
        ->from($outboxTable, ['job_id'])
        ->where('latest_change_id = ?', $changeId)
        ->where('lane = ?', 'EMBEDDING')
        ->where('state NOT IN (?)', ['COMPLETED', 'SUPERSEDED'])
));
if ($priorityJobs === [] || $embeddingJobs === []) {
    throw new RuntimeException('The option label update did not create PRIORITY and EMBEDDING work.');
}

/** @var ConnectionManager $connectionManager */
$connectionManager = $objectManager->get(ConnectionManager::class);
$searchConnection = $connectionManager->getConnection();
if (!$searchConnection instanceof \Magento\OpenSearch\Model\SearchClient) {
    throw new RuntimeException('The option label fixture has no OpenSearch client.');
}
$client = $searchConnection->getOpenSearchClient();
$priorityDocument = $client->get([
    'index' => (string)$generation['physical_index'],
    'id' => (string)$productId,
])['_source'];
if (!str_contains((string)$priorityDocument['features'], $storeLabelText)
    || (bool)$priorityDocument['embedding_eligible']
    || hash_equals((string)$beforeDocument['source_hash'], (string)$priorityDocument['source_hash'])
) {
    throw new RuntimeException('Priority indexing did not expose the new label with an ineligible stale vector.');
}

$processJobs($changeId, 'EMBEDDING');
$indexerRegistry->get('catalogsearch_fulltext')->reindexList([$productId]);
$completedDocument = $client->get([
    'index' => (string)$generation['physical_index'],
    'id' => (string)$productId,
])['_source'];
$journal = $connection->fetchRow(
    $connection->select()->from($journalTable)->where('change_id = ?', $changeId)
);
if (!(bool)$completedDocument['embedding_eligible']
    || !str_contains((string)$completedDocument['features'], $storeLabelText)
    || !hash_equals((string)$priorityDocument['source_hash'], (string)$completedDocument['source_hash'])
    || !is_array($journal)
    || (string)$journal['native_state'] !== 'COMPLETED'
    || (string)$journal['hybrid_state'] !== 'COMPLETED'
) {
    throw new RuntimeException('Option-label catch-up did not finish both correctness boundaries.');
}
/** @var ValidationService $validationService */
$validationService = $objectManager->get(ValidationService::class);
if (!(bool)$validationService->validate($generationId)['ready']) {
    throw new RuntimeException('Option-label catch-up did not return its target generation to READY.');
}

printf(
    "Color option label change %d updated product %d, invalidated its stale vector, and re-encoded source_hash %s.\n",
    $changeId,
    $productId,
    (string)$completedDocument['source_hash']
);
