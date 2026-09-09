<?php

declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Document\CategoryTextResolver;
use MageOS\OpenSearchHybrid\Model\Document\ProductDocumentFactory;
use MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository;
use MageOS\OpenSearchHybrid\Model\Queue\EmbeddingConsumer;
use MageOS\OpenSearchHybrid\Model\Queue\PriorityConsumer;
use Magento\Catalog\Api\CategoryLinkManagementInterface;
use Magento\Catalog\Api\CategoryRepositoryInterface;
use Magento\Catalog\Api\ProductRepositoryInterface;
use Magento\Catalog\Model\CategoryFactory;
use Magento\Catalog\Model\ProductFactory;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\ResourceConnection;
use Magento\Framework\App\State;
use Magento\Framework\Indexer\IndexerRegistry;
use Magento\Store\Model\StoreManagerInterface;

if ($argc !== 2) {
    fwrite(STDERR, "Usage: php assert-dependent-invalidation.php /path/to/mageos\n");
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
/** @var ResourceConnection $resourceConnection */
$resourceConnection = $objectManager->get(ResourceConnection::class);
$connection = $resourceConnection->getConnection();
$generationTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_generation');
$journalTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_change_journal');
$outboxTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_outbox');
$itemTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_outbox_item');
$generationId = (int)$connection->fetchOne(
    $connection->select()
        ->from($generationTable, [new Zend_Db_Expr('MAX(generation_id)')])
        ->where('store_id = ?', $storeId)
        ->where('state = ?', 'BUILDING')
);
if ($generationId <= 0) {
    throw new RuntimeException('The dependent-invalidation probe requires a building generation.');
}

/** @var StoreManagerInterface $storeManager */
$storeManager = $objectManager->get(StoreManagerInterface::class);
/** @var CategoryRepositoryInterface $categoryRepository */
$categoryRepository = $objectManager->get(CategoryRepositoryInterface::class);
/** @var CategoryFactory $categoryFactory */
$categoryFactory = $objectManager->get(CategoryFactory::class);
$rootCategoryId = (int)$storeManager->getStore($storeId)->getRootCategoryId();
$rootCategory = $categoryRepository->get($rootCategoryId, 0);
$category = $categoryFactory->create();
$category->setName('Hybrid Fixture Category');
$category->setIsActive(true);
$category->setIncludeInMenu(true);
$category->setParentId($rootCategoryId);
$category->setPath(rtrim((string)$rootCategory->getPath(), '/') . '/');
$category->setStoreId(0);
$category = $categoryRepository->save($category);

/** @var ProductFactory $productFactory */
$productFactory = $objectManager->get(ProductFactory::class);
/** @var ProductRepositoryInterface $productRepository */
$productRepository = $objectManager->get(ProductRepositoryInterface::class);
$product = $productFactory->create();
$product->setTypeId('simple');
$product->setAttributeSetId($product->getDefaultAttributeSetId());
$product->setSku('mageos-hybrid-dependent-product');
$product->setName('Dependent Invalidation Product');
$product->setDescription('Product used to prove dependent semantic invalidation.');
$product->setPrice(29.99);
$product->setWeight(1);
$product->setStatus(1);
$product->setVisibility(4);
$product->setWebsiteIds([1]);
$product = $productRepository->save($product);
$productId = (int)$product->getId();
/** @var CategoryLinkManagementInterface $categoryLinks */
$categoryLinks = $objectManager->get(CategoryLinkManagementInterface::class);
$categoryLinks->assignProductToCategories((string)$product->getSku(), [(int)$category->getId()]);

$directJournal = $connection->fetchRow(
    $connection->select()
        ->from($journalTable)
        ->where('store_id = ?', $storeId)
        ->where('entity_id = ?', $productId)
        ->where('reason = ?', 'product_resource_save')
        ->order('change_id DESC')
        ->limit(1)
);
if (!is_array($directJournal)) {
    throw new RuntimeException('The dependent product did not create its direct capture record.');
}
$directChangeId = (int)$directJournal['change_id'];

$category = $categoryRepository->get((int)$category->getId(), 0);
$category->setName('Renamed Hybrid Fixture Category');
$categoryRepository->save($category);
$categoryJournal = $connection->fetchRow(
    $connection->select()
        ->from($journalTable)
        ->where('store_id = ?', $storeId)
        ->where('entity_id = ?', $productId)
        ->where('reason = ?', 'category_name_save')
        ->order('change_id DESC')
        ->limit(1)
);
if (!is_array($categoryJournal)) {
    throw new RuntimeException('A category name change did not capture its descendant product.');
}
$categoryChangeId = (int)$categoryJournal['change_id'];
$categoryJobs = $connection->fetchAll(
    $connection->select()
        ->from($outboxTable)
        ->where('generation_id = ?', $generationId)
        ->where('latest_change_id = ?', $categoryChangeId)
        ->order('lane ASC')
);
$jobsByLane = [];
foreach ($categoryJobs as $job) {
    $jobsByLane[(string)$job['lane']] = $job;
}
if (!isset($jobsByLane['PRIORITY'], $jobsByLane['EMBEDDING'])) {
    throw new RuntimeException('Category invalidation did not create both data-plane jobs.');
}
if ((int)$jobsByLane['PRIORITY']['earliest_change_id'] > $directChangeId) {
    throw new RuntimeException('Priority coalescing lost the superseded direct journal boundary.');
}
$oldItems = $connection->fetchAll(
    $connection->select()
        ->from(['item' => $itemTable], ['state'])
        ->joinInner(['job' => $outboxTable], 'job.job_id = item.job_id', [])
        ->where('job.generation_id = ?', $generationId)
        ->where('job.latest_change_id = ?', $directChangeId)
        ->where('item.product_id = ?', $productId)
);
if ($oldItems === []) {
    throw new RuntimeException('The coalescing probe could not find direct product work.');
}
foreach ($oldItems as $oldItem) {
    if ((string)$oldItem['state'] !== 'SUPERSEDED') {
        throw new RuntimeException('Newer category work did not supersede an older product item.');
    }
}

/** @var CategoryTextResolver $categoryTextResolver */
$categoryTextResolver = $objectManager->get(CategoryTextResolver::class);
$categoryPaths = $categoryTextResolver->resolve([(int)$category->getId()], $storeId);
if (!str_contains(implode(' | ', $categoryPaths), 'Renamed Hybrid Fixture Category')) {
    throw new RuntimeException('The store-scoped category path did not contain the renamed category.');
}
/** @var ProductDocumentFactory $documentFactory */
$documentFactory = $objectManager->get(ProductDocumentFactory::class);
$document = $documentFactory->create($productId, $storeId);
if (!str_contains((string)$document['category'], 'Renamed Hybrid Fixture Category')
    || !str_contains((string)$document['rendered'], 'Renamed Hybrid Fixture Category')
) {
    throw new RuntimeException('The semantic product document did not include the renamed category path.');
}

/** @var IndexerRegistry $indexerRegistry */
$indexerRegistry = $objectManager->get(IndexerRegistry::class);
$indexerRegistry->get('catalogsearch_fulltext')->reindexList([$productId]);
/** @var OutboxRepository $outboxRepository */
$outboxRepository = $objectManager->get(OutboxRepository::class);
/** @var PriorityConsumer $priorityConsumer */
$priorityConsumer = $objectManager->get(PriorityConsumer::class);
$priorityConsumer->process((string)$jobsByLane['PRIORITY']['job_id']);
/** @var EmbeddingConsumer $embeddingConsumer */
$embeddingConsumer = $objectManager->get(EmbeddingConsumer::class);
$embeddingConsumer->process((string)$jobsByLane['EMBEDDING']['job_id']);

$journals = $connection->fetchAll(
    $connection->select()
        ->from($journalTable, ['change_id', 'native_state', 'hybrid_state'])
        ->where('change_id IN (?)', [$directChangeId, $categoryChangeId])
);
foreach ($journals as $journal) {
    if ((string)$journal['native_state'] !== 'COMPLETED'
        || (string)$journal['hybrid_state'] !== 'COMPLETED'
    ) {
        throw new RuntimeException(sprintf(
            'Dependent journal %d did not reach both correctness boundaries.',
            (int)$journal['change_id']
        ));
    }
}
if ((string)$outboxRepository->get((string)$jobsByLane['PRIORITY']['job_id'])['state'] !== 'COMPLETED'
    || (string)$outboxRepository->get((string)$jobsByLane['EMBEDDING']['job_id'])['state'] !== 'COMPLETED'
) {
    throw new RuntimeException('Dependent invalidation jobs did not complete idempotently.');
}

fwrite(
    STDOUT,
    sprintf(
        "Category name change captured product %d, coalesced stale work, and completed change %d.\n",
        $productId,
        $categoryChangeId
    )
);
