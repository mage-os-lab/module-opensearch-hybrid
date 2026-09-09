<?php

declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Change\ChangeJournalRepository;
use MageOS\OpenSearchHybrid\Model\Config;
use MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract;
use MageOS\OpenSearchHybrid\Model\Generation\BuildService;
use MageOS\OpenSearchHybrid\Model\Generation\StoreScopeFingerprint;
use MageOS\OpenSearchHybrid\Model\Generation\ValidationService;
use MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository;
use MageOS\OpenSearchHybrid\Model\Queue\EmbeddingConsumer;
use Magento\Framework\Api\DataObjectHelper;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\Cache\TypeListInterface;
use Magento\Framework\App\Config\ReinitableConfigInterface;
use Magento\Framework\App\Config\Storage\WriterInterface;
use Magento\Framework\App\ResourceConnection;
use Magento\Framework\App\State;
use Magento\Framework\Indexer\IndexerRegistry;
use Magento\InventoryApi\Api\Data\SourceInterface;
use Magento\InventoryApi\Api\Data\SourceInterfaceFactory;
use Magento\InventoryApi\Api\Data\SourceItemInterface;
use Magento\InventoryApi\Api\Data\SourceItemInterfaceFactory;
use Magento\InventoryApi\Api\Data\StockInterface;
use Magento\InventoryApi\Api\Data\StockInterfaceFactory;
use Magento\InventoryApi\Api\Data\StockSourceLinkInterface;
use Magento\InventoryApi\Api\Data\StockSourceLinkInterfaceFactory;
use Magento\InventoryApi\Api\SourceRepositoryInterface;
use Magento\InventoryApi\Api\SourceItemsSaveInterface;
use Magento\InventoryApi\Api\StockRepositoryInterface;
use Magento\InventoryApi\Api\StockSourceLinksDeleteInterface;
use Magento\InventoryApi\Api\StockSourceLinksSaveInterface;
use Magento\InventorySalesApi\Api\Data\SalesChannelInterface;
use Magento\InventorySalesApi\Api\Data\SalesChannelInterfaceFactory;

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
}

$storeId = 1;
/** @var ResourceConnection $resourceConnection */
$resourceConnection = $objectManager->get(ResourceConnection::class);
$connection = $resourceConnection->getConnection();
$generationTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_generation');
$journalTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_change_journal');
$outboxTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_outbox');
$progressTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_generation_progress');
$priorGeneration = $connection->fetchRow(
    $connection->select()
        ->from($generationTable)
        ->where('store_id = ?', $storeId)
        ->where('state = ?', 'READY')
        ->order('generation_id DESC')
        ->limit(1)
);
if (!is_array($priorGeneration)) {
    throw new RuntimeException('The topology fixture requires a ready generation.');
}
$priorGenerationId = (int)$priorGeneration['generation_id'];
/** @var StoreScopeFingerprint $fingerprint */
$fingerprint = $objectManager->get(StoreScopeFingerprint::class);
$originalDigest = $fingerprint->digest($storeId);
if (!hash_equals((string)$priorGeneration['scope_digest'], $originalDigest)) {
    throw new RuntimeException('The topology fixture did not start at the frozen inventory scope.');
}
$beforeBoundary = (int)$connection->fetchOne(
    $connection->select()->from($journalTable, [new Zend_Db_Expr('COALESCE(MAX(change_id), 0)')])
);

/** @var DataObjectHelper $dataObjectHelper */
$dataObjectHelper = $objectManager->get(DataObjectHelper::class);
/** @var SourceInterfaceFactory $sourceFactory */
$sourceFactory = $objectManager->get(SourceInterfaceFactory::class);
/** @var SourceRepositoryInterface $sourceRepository */
$sourceRepository = $objectManager->get(SourceRepositoryInterface::class);
$createSource = static function (
    string $code,
    string $name
) use (
    $sourceFactory,
    $sourceRepository,
    $dataObjectHelper
): SourceInterface {
    $source = $sourceFactory->create();
    $dataObjectHelper->populateWithArray($source, [
        SourceInterface::SOURCE_CODE => $code,
        SourceInterface::NAME => $name,
        SourceInterface::ENABLED => true,
        SourceInterface::DESCRIPTION => 'OpenSearch Hybrid topology acceptance source.',
        SourceInterface::COUNTRY_ID => 'US',
        SourceInterface::POSTCODE => '46204',
        SourceInterface::USE_DEFAULT_CARRIER_CONFIG => 1,
        SourceInterface::CARRIER_LINKS => [],
    ], SourceInterface::class);
    $sourceRepository->save($source);

    return $source;
};
$primarySource = $createSource('mageos-hybrid-primary', 'MageOS Hybrid Primary');
$secondarySource = $createSource('mageos-hybrid-secondary', 'MageOS Hybrid Secondary');

/** @var StockInterfaceFactory $stockFactory */
$stockFactory = $objectManager->get(StockInterfaceFactory::class);
/** @var StockRepositoryInterface $stockRepository */
$stockRepository = $objectManager->get(StockRepositoryInterface::class);
$stock = $stockFactory->create();
$stock->setName('MageOS Hybrid Stock');
$stockId = $stockRepository->save($stock);
$stock = $stockRepository->get($stockId);

/** @var StockSourceLinkInterfaceFactory $linkFactory */
$linkFactory = $objectManager->get(StockSourceLinkInterfaceFactory::class);
/** @var StockSourceLinksSaveInterface $linksSave */
$linksSave = $objectManager->get(StockSourceLinksSaveInterface::class);
/** @var StockSourceLinksDeleteInterface $linksDelete */
$linksDelete = $objectManager->get(StockSourceLinksDeleteInterface::class);
$createLink = static function (
    string $sourceCode,
    int $priority
) use (
    $linkFactory,
    $stockId
): StockSourceLinkInterface {
    return $linkFactory->create(['data' => [
        StockSourceLinkInterface::STOCK_ID => $stockId,
        StockSourceLinkInterface::SOURCE_CODE => $sourceCode,
        StockSourceLinkInterface::PRIORITY => $priority,
    ]]);
};
$primaryLink = $createLink((string)$primarySource->getSourceCode(), 1);
$linksSave->execute([$primaryLink]);

/** @var SalesChannelInterfaceFactory $channelFactory */
$channelFactory = $objectManager->get(SalesChannelInterfaceFactory::class);
$channel = $channelFactory->create();
$channel->setType(SalesChannelInterface::TYPE_WEBSITE);
$channel->setCode('base');
$extensionAttributes = $stock->getExtensionAttributes();
$extensionAttributes->setSalesChannels([$channel]);
$stock->setExtensionAttributes($extensionAttributes);
$stockRepository->save($stock);
$assignedDigest = $fingerprint->digest($storeId);
if (hash_equals($originalDigest, $assignedDigest)) {
    throw new RuntimeException('Website stock reassignment did not change the frozen inventory scope.');
}
$failedGeneration = $connection->fetchRow(
    $connection->select()->from($generationTable)->where('generation_id = ?', $priorGenerationId)
);
if (!is_array($failedGeneration) || (string)$failedGeneration['state'] !== 'FAILED') {
    throw new RuntimeException('Website stock reassignment did not fail the ready generation.');
}

$primarySource = $sourceRepository->get((string)$primarySource->getSourceCode());
$primarySource->setEnabled(false);
$sourceRepository->save($primarySource);
$disabledDigest = $fingerprint->digest($storeId);
if (hash_equals($assignedDigest, $disabledDigest)) {
    throw new RuntimeException('Source disablement did not change the frozen inventory scope.');
}
$primarySource->setEnabled(true);
$sourceRepository->save($primarySource);
if (!hash_equals($assignedDigest, $fingerprint->digest($storeId))) {
    throw new RuntimeException('Source re-enablement did not restore the inventory scope.');
}

$secondaryLink = $createLink((string)$secondarySource->getSourceCode(), 2);
$linksSave->execute([$secondaryLink]);
$linkedDigest = $fingerprint->digest($storeId);
if (hash_equals($assignedDigest, $linkedDigest)) {
    throw new RuntimeException('Source link creation did not change the frozen inventory scope.');
}
$linksDelete->execute([$secondaryLink]);
if (!hash_equals($assignedDigest, $fingerprint->digest($storeId))) {
    throw new RuntimeException('Source link deletion did not restore the inventory scope.');
}

$topologyChanges = $connection->fetchAll(
    $connection->select()
        ->from($journalTable)
        ->where('change_id > ?', $beforeBoundary)
        ->where('store_id = ?', $storeId)
        ->where('entity_type IN (?)', ['INVENTORY_SALES_CHANNEL', 'INVENTORY_SOURCE', 'INVENTORY_STOCK'])
        ->order('change_id ASC')
);
$reasonCounts = array_count_values(array_column($topologyChanges, 'reason'));
$expectedReasons = [
    'inventory_sales_channel_change' => 1,
    'inventory_source_status_change' => 2,
    'inventory_stock_source_links_save' => 1,
    'inventory_stock_source_links_delete' => 1,
];
foreach ($expectedReasons as $reason => $count) {
    if (($reasonCounts[$reason] ?? 0) !== $count) {
        throw new RuntimeException(sprintf('Topology reason %s was not captured %d time(s).', $reason, $count));
    }
}
foreach ($topologyChanges as $change) {
    if (
        (string)$change['operation'] !== 'REBUILD'
        || (string)$change['native_state'] !== 'COMPLETED'
        || (string)$change['hybrid_state'] !== 'PENDING'
    ) {
        throw new RuntimeException('An inventory topology invalidation has the wrong correctness state.');
    }
}

/** @var SourceItemInterfaceFactory $sourceItemFactory */
$sourceItemFactory = $objectManager->get(SourceItemInterfaceFactory::class);
/** @var SourceItemInterface $sourceItem */
$sourceItem = $sourceItemFactory->create();
$sourceItem->setSourceCode((string)$primarySource->getSourceCode());
$sourceItem->setSku('mageos-hybrid-resume-02');
$sourceItem->setQuantity(100.0);
$sourceItem->setStatus(SourceItemInterface::STATUS_IN_STOCK);
/** @var SourceItemsSaveInterface $sourceItemsSave */
$sourceItemsSave = $objectManager->get(SourceItemsSaveInterface::class);
$sourceItemsSave->execute([$sourceItem]);

/** @var WriterInterface $configWriter */
$configWriter = $objectManager->get(WriterInterface::class);
$configWriter->save(Config::XML_PATH_ENCODER_ENDPOINT, 'http://10.255.255.1');
/** @var TypeListInterface $cacheTypes */
$cacheTypes = $objectManager->get(TypeListInterface::class);
$cacheTypes->cleanType('config');
/** @var ReinitableConfigInterface $reinitableConfig */
$reinitableConfig = $objectManager->get(ReinitableConfigInterface::class);
$reinitableConfig->reinit();

/** @var IndexerRegistry $indexerRegistry */
$indexerRegistry = $objectManager->get(IndexerRegistry::class);
$indexerRegistry->get('inventory')->reindexAll();
$indexerRegistry->get('cataloginventory_stock')->reindexAll();
$indexerRegistry->get('catalogsearch_fulltext')->reindexAll();
/** @var BuildService $buildService */
$buildService = $objectManager->get(BuildService::class);
$replacementGenerationId = $buildService->buildFake($storeId);
/** @var EmbeddingConsumer $embeddingConsumer */
$embeddingConsumer = $objectManager->get(EmbeddingConsumer::class);
/** @var OutboxRepository $outboxRepository */
$outboxRepository = $objectManager->get(OutboxRepository::class);
for ($iteration = 0; $iteration < 20; $iteration++) {
    $jobIds = array_map('strval', $connection->fetchCol(
        $connection->select()
            ->from($outboxTable, ['job_id'])
            ->where('generation_id = ?', $replacementGenerationId)
            ->where('lane = ?', 'EMBEDDING')
            ->where('state NOT IN (?)', ['COMPLETED', 'SUPERSEDED'])
            ->order('created_at ASC')
    ));
    foreach ($jobIds as $jobId) {
        if ((string)$outboxRepository->get($jobId)['state'] === 'PENDING') {
            $outboxRepository->markPublished($jobId);
        }
        $embeddingConsumer->process($jobId);
    }
    $progress = $connection->fetchRow(
        $connection->select()
            ->from($progressTable)
            ->where('generation_id = ?', $replacementGenerationId)
    );
    if (is_array($progress) && (string)$progress['seeding_state'] === 'COMPLETED' && $jobIds === []) {
        break;
    }
    $buildService->resumeFake($replacementGenerationId);
}
/** @var ValidationService $validationService */
$validationService = $objectManager->get(ValidationService::class);
/** @var RetrievalContract $retrievalContract */
$retrievalContract = $objectManager->get(RetrievalContract::class);
$validation = $validationService->validate(
    $replacementGenerationId,
    $retrievalContract->resultContractDigest()
);
if (!(bool)$validation['ready']) {
    throw new RuntimeException('The post-topology replacement generation did not validate.');
}
$replacement = $connection->fetchRow(
    $connection->select()->from($generationTable)->where('generation_id = ?', $replacementGenerationId)
);
if (
    !is_array($replacement)
    || !hash_equals((string)$replacement['scope_digest'], $fingerprint->digest($storeId))
) {
    throw new RuntimeException('The replacement generation did not freeze the final inventory topology.');
}
/** @var ChangeJournalRepository $journalRepository */
$journalRepository = $objectManager->get(ChangeJournalRepository::class);
$journalRepository->acknowledgeGenerationInvalidations(
    $storeId,
    (int)$replacement['captured_change_id']
);

printf(
    "Inventory topology failed generation %d and replacement %d froze stock %d with %d invalidations.\n",
    $priorGenerationId,
    $replacementGenerationId,
    $stockId,
    count($topologyChanges)
);
