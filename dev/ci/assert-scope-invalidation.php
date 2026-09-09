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
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\Config\ReinitableConfigInterface;
use Magento\Framework\App\Config\Value;
use Magento\Framework\App\ResourceConnection;
use Magento\Framework\App\State;
use Magento\Store\Api\StoreRepositoryInterface;

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
$journalTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_change_journal');
$stateTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_store_state');
$outboxTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_outbox');
$progressTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_generation_progress');
$configTable = $resourceConnection->getTableName('core_config_data');
$priorGeneration = $connection->fetchRow(
    $connection->select()
        ->from($generationTable)
        ->where('store_id = ?', $storeId)
        ->where('state = ?', 'READY')
        ->order('generation_id DESC')
        ->limit(1)
);
if (!is_array($priorGeneration)) {
    throw new RuntimeException('The scope fixture requires a ready generation to invalidate.');
}
$priorGenerationId = (int)$priorGeneration['generation_id'];
/** @var StoreScopeFingerprint $fingerprint */
$fingerprint = $objectManager->get(StoreScopeFingerprint::class);
if (!hash_equals((string)$priorGeneration['scope_digest'], $fingerprint->digest($storeId))) {
    throw new RuntimeException('The ready generation did not start with the current scope digest.');
}
$activeBoundary = (int)$connection->fetchOne(
    $connection->select()
        ->from($journalTable, [new Zend_Db_Expr('COALESCE(MAX(change_id), 0)')])
        ->where('store_id = ?', $storeId)
);
$connection->update(
    $generationTable,
    ['state' => 'ACTIVE'],
    ['generation_id = ?' => $priorGenerationId, 'state = ?' => 'READY']
);
$connection->insertOnDuplicate(
    $stateTable,
    [
        'store_id' => $storeId,
        'activation_mode' => 'HYBRID',
        'active_generation_id' => $priorGenerationId,
        'required_watermark' => $activeBoundary,
        'native_watermark' => $activeBoundary,
        'hybrid_watermark' => $activeBoundary,
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

$beforeBoundary = (int)$connection->fetchOne(
    $connection->select()->from($journalTable, [new Zend_Db_Expr('COALESCE(MAX(change_id), 0)')])
);
/** @var StoreRepositoryInterface $storeRepository */
$storeRepository = $objectManager->get(StoreRepositoryInterface::class);
$store = $storeRepository->getById($storeId);
/** @var \Magento\Store\Model\ResourceModel\Store $storeResource */
$storeResource = $objectManager->get(\Magento\Store\Model\ResourceModel\Store::class);
$originalName = (string)$store->getName();
$store->setName($originalName . ' Scope Probe');
$storeResource->save($store);
$store->setName($originalName);
$storeResource->save($store);

$configRow = $connection->fetchRow(
    $connection->select()
        ->from($configTable)
        ->where('scope = ?', 'default')
        ->where('scope_id = ?', 0)
        ->where('path = ?', Config::XML_PATH_ENCODER_ENDPOINT)
        ->limit(1)
);
/** @var \Magento\Config\Model\ResourceModel\Config\Data $configResource */
$configResource = $objectManager->get(\Magento\Config\Model\ResourceModel\Config\Data::class);
/** @var Value $configValue */
$configValue = $objectManager->create(Value::class);
if (is_array($configRow)) {
    $configResource->load($configValue, (int)$configRow['config_id']);
} else {
    $configValue->setScope('default');
    $configValue->setScopeId(0);
    $configValue->setPath(Config::XML_PATH_ENCODER_ENDPOINT);
}
$originalEndpoint = is_array($configRow) ? (string)$configRow['value'] : null;
$configValue->setValue('http://127.0.0.1:18181/fixture-scope-change');
$configResource->save($configValue);
if ($originalEndpoint === null) {
    $configResource->delete($configValue);
} else {
    $configValue->setValue($originalEndpoint);
    $configResource->save($configValue);
}
/** @var ReinitableConfigInterface $reinitableConfig */
$reinitableConfig = $objectManager->get(ReinitableConfigInterface::class);
$reinitableConfig->reinit();

$scopeChanges = $connection->fetchAll(
    $connection->select()
        ->from($journalTable)
        ->where('change_id > ?', $beforeBoundary)
        ->where('store_id = ?', $storeId)
        ->where('entity_type IN (?)', ['STORE', 'CONFIGURATION'])
        ->order('change_id ASC')
);
$entityTypes = array_values(array_unique(array_column($scopeChanges, 'entity_type')));
sort($entityTypes);
if ($entityTypes !== ['CONFIGURATION', 'STORE']) {
    throw new RuntimeException('Store and configuration changes were not both captured.');
}
foreach ($scopeChanges as $scopeChange) {
    if ((string)$scopeChange['operation'] !== 'REBUILD'
        || (string)$scopeChange['native_state'] !== 'COMPLETED'
        || (string)$scopeChange['hybrid_state'] !== 'PENDING'
    ) {
        throw new RuntimeException('A scope invalidation has the wrong correctness state.');
    }
}
$failedGeneration = $connection->fetchRow(
    $connection->select()->from($generationTable)->where('generation_id = ?', $priorGenerationId)
);
$storeState = $connection->fetchRow(
    $connection->select()->from($stateTable)->where('store_id = ?', $storeId)
);
if (!is_array($failedGeneration)
    || (string)$failedGeneration['state'] !== 'FAILED'
    || !is_array($storeState)
    || (bool)$storeState['readiness_latch']
) {
    throw new RuntimeException('The scope change did not fail the candidate and close readiness.');
}
$unresolvedOldJobs = (int)$connection->fetchOne(
    $connection->select()
        ->from($outboxTable, [new Zend_Db_Expr('COUNT(*)')])
        ->where('generation_id = ?', $priorGenerationId)
        ->where('state NOT IN (?)', ['COMPLETED', 'SUPERSEDED'])
);
if ($unresolvedOldJobs !== 0) {
    throw new RuntimeException('Invalidating the candidate left unresolved generation work.');
}

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
if (!(bool)$validation['ready'] || !(bool)$validation['checks']['scope_digest']) {
    throw new RuntimeException('The replacement generation did not validate the restored store scope.');
}
$replacement = $connection->fetchRow(
    $connection->select()->from($generationTable)->where('generation_id = ?', $replacementGenerationId)
);
if (!is_array($replacement)
    || !hash_equals((string)$replacement['scope_digest'], $fingerprint->digest($storeId))
) {
    throw new RuntimeException('The replacement generation did not freeze the current scope digest.');
}

/** @var ChangeJournalRepository $journalRepository */
$journalRepository = $objectManager->get(ChangeJournalRepository::class);
$journalRepository->acknowledgeGenerationInvalidations(
    $storeId,
    (int)$replacement['captured_change_id']
);
$pendingScopeChanges = (int)$connection->fetchOne(
    $connection->select()
        ->from($journalTable, [new Zend_Db_Expr('COUNT(*)')])
        ->where('change_id > ?', $beforeBoundary)
        ->where('store_id = ?', $storeId)
        ->where('entity_type != ?', 'PRODUCT')
        ->where('hybrid_state != ?', 'COMPLETED')
);
if ($pendingScopeChanges !== 0) {
    throw new RuntimeException('Replacement activation acknowledgement left scope changes pending.');
}

printf(
    "Store and configuration invalidation failed generation %d; replacement %d validated scope digest %s.\n",
    $priorGenerationId,
    $replacementGenerationId,
    (string)$replacement['scope_digest']
);
