<?php
declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Generation\GenerationCleanupService;
use MageOS\OpenSearchHybrid\Model\OpenSearch\IndexManager;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\ResourceConnection;
use Magento\Framework\App\State;

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
/** @var ResourceConnection $resource */
$resource = $objectManager->get(ResourceConnection::class);
$connection = $resource->getConnection();
$generationTable = $resource->getTableName('mageos_opensearch_hybrid_generation');
$stateTable = $resource->getTableName('mageos_opensearch_hybrid_store_state');
$activationTable = $resource->getTableName('mageos_opensearch_hybrid_activation');
$cleanupTable = $resource->getTableName('mageos_opensearch_hybrid_cleanup');
$embeddingTable = $resource->getTableName('mageos_opensearch_hybrid_embedding');
$progressTable = $resource->getTableName('mageos_opensearch_hybrid_generation_progress');
$outboxTable = $resource->getTableName('mageos_opensearch_hybrid_outbox');
$itemTable = $resource->getTableName('mageos_opensearch_hybrid_outbox_item');

/** @var GenerationCleanupService $cleanupService */
$cleanupService = $objectManager->get(GenerationCleanupService::class);
$activeGenerationId = (int)$connection->fetchOne(
    $connection->select()->from($stateTable, ['active_generation_id'])->where('store_id = ?', $storeId)
);
if ($activeGenerationId < 1) {
    throw new RuntimeException('The cleanup probe requires an active generation reference.');
}
$activePreview = $cleanupService->preview($storeId, $activeGenerationId);
if ((bool)$activePreview['eligible']
    || $activePreview['protection_reasons'] !== ['active_generation']
    || $activePreview['confirmation_token'] !== null
) {
    throw new RuntimeException('Cleanup did not refuse the active generation.');
}

$retainedGenerationId = (int)$connection->fetchOne(
    $connection->select()
        ->from($generationTable, ['generation_id'])
        ->where('store_id = ?', $storeId)
        ->where('generation_id != ?', $activeGenerationId)
        ->where('is_fake = ?', 1)
        ->order('generation_id DESC')
        ->limit(1)
);
if ($retainedGenerationId < 1) {
    throw new RuntimeException('The cleanup probe requires a retained-generation candidate.');
}
$connection->update(
    $generationTable,
    ['state' => 'RETAINED', 'updated_at' => new Zend_Db_Expr('UTC_TIMESTAMP()')],
    ['generation_id = ?' => $retainedGenerationId]
);
$retainedPreview = $cleanupService->preview($storeId, $retainedGenerationId);
if ((bool)$retainedPreview['eligible']
    || $retainedPreview['protection_reasons'] !== ['rollback_retention_window']
) {
    throw new RuntimeException('Cleanup did not protect a rollback-retained generation.');
}
$connection->update(
    $generationTable,
    ['updated_at' => new Zend_Db_Expr('DATE_SUB(UTC_TIMESTAMP(), INTERVAL 15 DAY)')],
    ['generation_id = ?' => $retainedGenerationId]
);
$expiredPreview = $cleanupService->preview($storeId, $retainedGenerationId);
if (!(bool)$expiredPreview['eligible'] || !is_string($expiredPreview['confirmation_token'])) {
    throw new RuntimeException('Cleanup did not release a retained generation after its retention window.');
}

$target = $connection->fetchRow(
    $connection->select()
        ->from($generationTable)
        ->where('store_id = ?', $storeId)
        ->where('is_fake = ?', 0)
        ->order('generation_id DESC')
        ->limit(1)
);
if (!is_array($target)) {
    throw new RuntimeException('The cleanup probe requires the production-gated generation.');
}
$generationId = (int)$target['generation_id'];
$connection->update($generationTable, ['state' => 'FAILED'], ['generation_id = ?' => $generationId]);
$connection->insert($activationTable, [
    'store_id' => $storeId,
    'generation_id' => $generationId,
    'prior_generation_id' => null,
    'mode' => 'NATIVE',
    'status' => 'COMPLETED',
    'contract_digest' => $target['result_contract_digest'],
    'physical_index' => $target['physical_index'],
    'high_watermark' => (int)$target['captured_change_id'],
    'validation_report' => null,
    'actor' => 'fixture',
    'completed_at' => new Zend_Db_Expr('UTC_TIMESTAMP()'),
]);
$activationId = (int)$connection->lastInsertId($activationTable);
$expectedPreservedActivationRows = (int)$connection->fetchOne(
    $connection->select()
        ->from($activationTable, [new Zend_Db_Expr('COUNT(*)')])
        ->where('generation_id = ?', $generationId)
        ->orWhere('prior_generation_id = ?', $generationId)
);

$preview = $cleanupService->preview($storeId, $generationId);
$deleteRows = [];
foreach ($preview['database']['delete'] as $rowImpact) {
    $deleteRows[(string)$rowImpact['table']] = (int)$rowImpact['rows'];
}
$expectedRows = [
    $itemTable => (int)$connection->fetchOne(
        $connection->select()
            ->from(['item' => $itemTable], [new Zend_Db_Expr('COUNT(*)')])
            ->joinInner(['job' => $outboxTable], 'job.job_id = item.job_id', [])
            ->where('job.generation_id = ?', $generationId)
    ),
    $outboxTable => (int)$connection->fetchOne(
        $connection->select()->from($outboxTable, [new Zend_Db_Expr('COUNT(*)')])
            ->where('generation_id = ?', $generationId)
    ),
    $embeddingTable => (int)$connection->fetchOne(
        $connection->select()->from($embeddingTable, [new Zend_Db_Expr('COUNT(*)')])
            ->where('generation_id = ?', $generationId)
    ),
    $progressTable => (int)$connection->fetchOne(
        $connection->select()->from($progressTable, [new Zend_Db_Expr('COUNT(*)')])
            ->where('generation_id = ?', $generationId)
    ),
    $generationTable => 1,
];
if (!(bool)$preview['eligible']
    || !is_string($preview['confirmation_token'])
    || $deleteRows !== $expectedRows
    || !(bool)$preview['opensearch']['indices'][0]['exists']
    || $preview['opensearch']['pipelines'][0]['action'] !== 'preserve_shared_contract_resource'
    || $preview['encoder_artifacts'][0]['action'] !== 'retain_external_operator_managed'
    || (int)$preview['database']['preserve'][0]['rows'] !== $expectedPreservedActivationRows
) {
    throw new RuntimeException('Cleanup preview did not bind the exact database and external impact.');
}

$auditCountBefore = (int)$connection->fetchOne(
    $connection->select()->from($cleanupTable, [new Zend_Db_Expr('COUNT(*)')])
);
try {
    $cleanupService->cleanup($storeId, $generationId, 'cleanup-wrong-token');
    throw new RuntimeException('Cleanup accepted a stale or incorrect confirmation token.');
} catch (InvalidArgumentException $exception) {
    if (!str_contains($exception->getMessage(), 'does not match')) {
        throw $exception;
    }
}
if ((int)$connection->fetchOne(
    $connection->select()->from($cleanupTable, [new Zend_Db_Expr('COUNT(*)')])
) !== $auditCountBefore
    || (int)$connection->fetchOne(
        $connection->select()->from($generationTable, [new Zend_Db_Expr('COUNT(*)')])
            ->where('generation_id = ?', $generationId)
    ) !== 1
) {
    throw new RuntimeException('Rejected cleanup changed generation or audit state.');
}

$report = $cleanupService->cleanup($storeId, $generationId, (string)$preview['confirmation_token']);
/** @var IndexManager $indexManager */
$indexManager = $objectManager->get(IndexManager::class);
$audit = $connection->fetchRow(
    $connection->select()->from($cleanupTable)->where('cleanup_id = ?', (int)$report['cleanup_audit_id'])
);
$remainingChildren = 0;
foreach ([$embeddingTable, $progressTable, $outboxTable] as $table) {
    $remainingChildren += (int)$connection->fetchOne(
        $connection->select()->from($table, [new Zend_Db_Expr('COUNT(*)')])
            ->where('generation_id = ?', $generationId)
    );
}
if ((string)$report['status'] !== 'completed'
    || (int)$connection->fetchOne(
        $connection->select()->from($generationTable, [new Zend_Db_Expr('COUNT(*)')])
            ->where('generation_id = ?', $generationId)
    ) !== 0
    || $remainingChildren !== 0
    || $indexManager->indexExists((string)$target['physical_index'])
    || !is_array($audit)
    || (string)$audit['status'] !== 'COMPLETED'
    || (int)$connection->fetchOne(
        $connection->select()->from($activationTable, [new Zend_Db_Expr('COUNT(*)')])
            ->where('activation_id = ?', $activationId)
    ) !== 1
) {
    throw new RuntimeException('Confirmed cleanup did not delete narrowly and preserve its audits.');
}

printf(
    "Cleanup generation %d deleted its exact index and rows under audit %d.\n",
    $generationId,
    (int)$report['cleanup_audit_id']
);
