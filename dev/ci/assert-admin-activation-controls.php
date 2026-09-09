<?php
declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Activation\ActivationService;
use MageOS\OpenSearchHybrid\Api\DocumentEncoderInterface;
use MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract;
use MageOS\OpenSearchHybrid\Model\Config;
use MageOS\OpenSearchHybrid\Model\Generation\ValidationService;
use MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository;
use MageOS\OpenSearchHybrid\Model\Queue\EmbeddingConsumer;
use MageOS\OpenSearchHybrid\Model\Queue\EmbeddingJobProcessor;
use MageOS\OpenSearchHybrid\Model\Vector\DeterministicVector;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\Cache\TypeListInterface;
use Magento\Framework\App\Config\Storage\WriterInterface;
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
    $objectManager->get(State::class)->setAreaCode('adminhtml');
} catch (\Magento\Framework\Exception\LocalizedException) {
}

/** @var ResourceConnection $resourceConnection */
$resourceConnection = $objectManager->get(ResourceConnection::class);
$connection = $resourceConnection->getConnection();
$generationTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_generation');
$progressTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_generation_progress');
$embeddingTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_embedding');
$outboxTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_outbox');
$stateTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_store_state');
$activationTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_activation');
$rollbackSetSnapshot = static function (int $generationId) use (
    $connection,
    $generationTable,
    $progressTable,
    $embeddingTable,
    $outboxTable
): array {
    $generation = $connection->fetchRow(
        $connection->select()->from($generationTable, [
            'generation_id',
            'store_id',
            'state',
            'model_id',
            'model_revision',
            'dimension',
            'recipe_version',
            'contract_version',
            'contract_digest',
            'result_contract_digest',
            'mapping_digest',
            'pipeline_digest',
            'physical_index',
            'pipeline_id',
            'encoder_endpoint',
            'encoder_identity_digest',
            'scope_digest',
            'captured_change_id',
            'coverage_total',
            'coverage_complete',
            'coverage_failed',
            'result_contract_accepted',
            'is_fake',
            'validation_report',
        ])->where('generation_id = ?', $generationId)
    );
    $progress = $connection->fetchRow(
        $connection->select()->from($progressTable, [
            'generation_id',
            'captured_boundary',
            'processed_boundary',
            'last_seeded_product_id',
            'seeded_products',
            'seeding_state',
            'build_refresh_interval',
            'normal_refresh_interval',
            'refresh_state',
        ])->where('generation_id = ?', $generationId)
    );
    $embeddings = $connection->fetchAll(
        $connection->select()->from($embeddingTable, [
            'embedding_id',
            'product_id',
            'model_id',
            'model_revision',
            'dimension',
            'recipe_version',
            'source_hash',
            'vector_hash',
            'stored_vector_sha256' => new Zend_Db_Expr('SHA2(vector, 256)'),
            'status',
            'revision',
        ])->where('generation_id = ?', $generationId)->order('embedding_id ASC')
    );
    $outbox = $connection->fetchAll(
        $connection->select()->from($outboxTable, [
            'job_id',
            'lane',
            'operation',
            'state',
            'revision',
            'earliest_change_id',
            'latest_change_id',
            'publish_count',
            'attempt_count',
            'replay_count',
            'last_error_class',
            'last_diagnostic',
        ])->where('generation_id = ?', $generationId)->order('job_id ASC')
    );

    if (!is_array($generation) || !is_array($progress) || $embeddings === []) {
        throw new RuntimeException('The retained rollback fixture is missing part of its rollback set.');
    }

    return [
        'generation' => $generation,
        'progress' => $progress,
        'embeddings_sha256' => hash(
            'sha256',
            json_encode($embeddings, JSON_UNESCAPED_SLASHES | JSON_THROW_ON_ERROR)
        ),
        'embedding_count' => count($embeddings),
        'outbox_sha256' => hash(
            'sha256',
            json_encode($outbox, JSON_UNESCAPED_SLASHES | JSON_THROW_ON_ERROR)
        ),
        'outbox_count' => count($outbox),
    ];
};
$fixtureState = $connection->fetchRow(
    $connection->select()->from($stateTable)->where('store_id = ?', 1)
);
if (!is_array($fixtureState)
    || (string)$fixtureState['activation_mode'] !== 'HYBRID'
    || $fixtureState['active_generation_id'] === null
) {
    throw new RuntimeException('The Admin rollback fixture requires one active hybrid store pointer.');
}
$fixtureGenerationId = (int)$fixtureState['active_generation_id'];
$retainedGenerationId = (int)$connection->fetchOne(
    $connection->select()
        ->from($generationTable, ['generation_id'])
        ->where('store_id = ?', 1)
        ->where('is_fake = ?', 0)
        ->where('state IN (?)', ['BUILDING', 'CATCHING_UP', 'READY'])
        ->order('generation_id DESC')
        ->limit(1)
);
if ($retainedGenerationId < 1) {
    throw new RuntimeException('The Admin rollback fixture requires one production-gated generation.');
}
$productionEndpoint = (string)$connection->fetchOne(
    $connection->select()
        ->from($generationTable, ['encoder_endpoint'])
        ->where('generation_id = ?', $retainedGenerationId)
);
if ($productionEndpoint === '') {
    throw new RuntimeException('The production rollback fixture generation has no encoder endpoint.');
}
/** @var WriterInterface $configWriter */
$configWriter = $objectManager->get(WriterInterface::class);
/** @var TypeListInterface $cacheTypes */
$cacheTypes = $objectManager->get(TypeListInterface::class);
$configWriter->save(Config::XML_PATH_ENCODER_ENDPOINT, $productionEndpoint);
$cacheTypes->cleanType('config');

/** @var DeterministicVector $deterministicVector */
$deterministicVector = $objectManager->get(DeterministicVector::class);
$fixtureDocumentEncoder = new class($deterministicVector) implements DocumentEncoderInterface {
    public function __construct(private readonly DeterministicVector $deterministicVector)
    {
    }

    public function encode(array $documents, array $generation): array
    {
        if ((bool)$generation['is_fake'] || $documents === []) {
            throw new RuntimeException('The production rollback fixture received an invalid encoder request.');
        }

        return array_map(
            fn (string $document): array => $this->deterministicVector->encode($document),
            $documents
        );
    }
};
/** @var EmbeddingJobProcessor $processor */
$processor = $objectManager->create(
    EmbeddingJobProcessor::class,
    ['documentEncoder' => $fixtureDocumentEncoder]
);
/** @var EmbeddingConsumer $consumer */
$consumer = $objectManager->create(EmbeddingConsumer::class, ['processor' => $processor]);
/** @var OutboxRepository $outboxRepository */
$outboxRepository = $objectManager->get(OutboxRepository::class);
$embeddingJobs = $connection->fetchAll(
    $connection->select()
        ->from($outboxTable, ['job_id', 'state'])
        ->where('generation_id = ?', $retainedGenerationId)
        ->where('lane = ?', 'EMBEDDING')
        ->order('job_id ASC')
);
if ($embeddingJobs === []) {
    throw new RuntimeException('The production rollback fixture has no document work to process.');
}
foreach ($embeddingJobs as $job) {
    if ((string)$job['state'] === 'PENDING') {
        $outboxRepository->markPublished((string)$job['job_id']);
    }
    $consumer->process((string)$job['job_id']);
}
$unresolvedJobs = (int)$connection->fetchOne(
    $connection->select()
        ->from($outboxTable, [new Zend_Db_Expr('COUNT(*)')])
        ->where('generation_id = ?', $retainedGenerationId)
        ->where('state NOT IN (?)', ['COMPLETED', 'SUPERSEDED'])
);
if ($unresolvedJobs !== 0) {
    throw new RuntimeException('The production rollback fixture left document work unresolved.');
}

/** @var RetrievalContract $retrievalContract */
$retrievalContract = $objectManager->get(RetrievalContract::class);
/** @var ValidationService $validationService */
$validationService = $objectManager->get(ValidationService::class);
$validation = $validationService->validate(
    $retainedGenerationId,
    $retrievalContract->resultContractDigest()
);
if (($validation['ready'] ?? false) !== true
    || ($validation['result_contract_accepted'] ?? false) !== true
) {
    throw new RuntimeException('The production rollback fixture generation did not validate.');
}
$auditCount = (int)$connection->fetchOne(
    $connection->select()->from($activationTable, [new Zend_Db_Expr('COUNT(*)')])
);

/** @var ActivationService $activationService */
$activationService = $objectManager->get(ActivationService::class);
$activationPreview = $activationService->previewActivation(1, $retainedGenerationId);
$activationReport = $activationService->activateConfirmed(
    1,
    $retainedGenerationId,
    (string)$activationPreview['confirmation_token'],
    'admin_user_id:7'
);
$activationAudit = $connection->fetchRow(
    $connection->select()
        ->from($activationTable)
        ->where('activation_id = ?', (int)$activationReport['activation_id'])
);
if ((string)$activationReport['status'] !== 'activated'
    || (int)$activationReport['generation_id'] !== $retainedGenerationId
    || (int)$activationReport['prior_generation_id'] !== $fixtureGenerationId
    || !is_array($activationAudit)
    || (string)$activationAudit['mode'] !== 'HYBRID'
    || (string)$activationAudit['status'] !== 'COMPLETED'
) {
    throw new RuntimeException('The production rollback fixture did not activate under an exact audit.');
}
$beforeState = $connection->fetchRow(
    $connection->select()->from($stateTable)->where('store_id = ?', 1)
);
$beforeRollbackSet = $rollbackSetSnapshot($retainedGenerationId);
$rollbackAuditCount = (int)$connection->fetchOne(
    $connection->select()->from($activationTable, [new Zend_Db_Expr('COUNT(*)')])
);
$preview = $activationService->previewNativeRollback(1);
if ((string)$preview['operation'] !== 'rollback_native'
    || (string)$preview['human_confirmation'] !== 'native'
    || (int)$preview['current_generation']['generation_id'] !== (int)$beforeState['active_generation_id']
    || !array_key_exists('coverage_complete', $preview['current_generation'])
    || !array_key_exists('validation_report', $preview['current_generation'])
    || preg_match('/^rollback-native-1-[0-9a-f]{64}$/D', (string)$preview['confirmation_token']) !== 1
) {
    throw new RuntimeException('The Admin native rollback preview omitted exact generation evidence.');
}
try {
    $activationService->rollbackToNativeConfirmed(1, 'rollback-native-1-' . str_repeat('0', 64), 'admin_user_id:7');
    throw new RuntimeException('A stale Admin activation confirmation was accepted.');
} catch (InvalidArgumentException $exception) {
    if (!str_contains($exception->getMessage(), 'current exact preview')) {
        throw $exception;
    }
}
$unchangedAuditCount = (int)$connection->fetchOne(
    $connection->select()->from($activationTable, [new Zend_Db_Expr('COUNT(*)')])
);
if ($unchangedAuditCount !== $rollbackAuditCount) {
    throw new RuntimeException('A rejected Admin activation confirmation created an audit row.');
}

$report = $activationService->rollbackToNativeConfirmed(
    1,
    (string)$preview['confirmation_token'],
    'admin_user_id:7'
);
$afterState = $connection->fetchRow(
    $connection->select()->from($stateTable)->where('store_id = ?', 1)
);
$nativeAudit = $connection->fetchRow(
    $connection->select()
        ->from($activationTable)
        ->where('activation_id = ?', (int)$report['activation_id'])
);
if ((string)$report['status'] !== 'rolled_back_to_native'
    || !is_array($afterState)
    || (string)$afterState['activation_mode'] !== 'NATIVE'
    || $afterState['active_generation_id'] !== null
    || !is_array($nativeAudit)
    || (int)$nativeAudit['prior_generation_id'] !== $retainedGenerationId
    || $nativeAudit['generation_id'] !== null
    || (string)$nativeAudit['actor'] !== 'admin_user_id:7'
    || (string)$nativeAudit['mode'] !== 'NATIVE'
    || (string)$nativeAudit['status'] !== 'COMPLETED'
) {
    throw new RuntimeException('The confirmed Admin rollback did not apply or preserve its actor audit.');
}

$generationPreview = $activationService->previewGenerationRollback(1, $retainedGenerationId);
if ((string)$generationPreview['operation'] !== 'rollback_generation'
    || (string)$generationPreview['human_confirmation'] !== (string)$retainedGenerationId
    || $generationPreview['current_generation'] !== null
    || (string)$generationPreview['target']['activation_mode'] !== 'HYBRID'
    || (int)$generationPreview['target']['generation']['generation_id'] !== $retainedGenerationId
    || (string)$generationPreview['target']['generation']['state'] !== 'RETAINED'
    || preg_match(
        '/^rollback-generation-1-[0-9a-f]{64}$/D',
        (string)$generationPreview['confirmation_token']
    ) !== 1
) {
    throw new RuntimeException('The Admin retained-generation preview omitted exact rollback-set evidence.');
}
try {
    $activationService->rollbackToGenerationConfirmed(
        1,
        $retainedGenerationId,
        'rollback-generation-1-' . str_repeat('0', 64),
        'admin_user_id:7'
    );
    throw new RuntimeException('A stale retained-generation confirmation was accepted.');
} catch (InvalidArgumentException $exception) {
    if (!str_contains($exception->getMessage(), 'current exact preview')) {
        throw $exception;
    }
}
$restoreReport = $activationService->rollbackToGenerationConfirmed(
    1,
    $retainedGenerationId,
    (string)$generationPreview['confirmation_token'],
    'admin_user_id:7'
);
$restoredState = $connection->fetchRow(
    $connection->select()->from($stateTable)->where('store_id = ?', 1)
);
$restoreAudit = $connection->fetchRow(
    $connection->select()
        ->from($activationTable)
        ->where('activation_id = ?', (int)$restoreReport['activation_id'])
);
$afterRestoreSet = $rollbackSetSnapshot($retainedGenerationId);
$finalAuditCount = (int)$connection->fetchOne(
    $connection->select()->from($activationTable, [new Zend_Db_Expr('COUNT(*)')])
);
if ((string)$restoreReport['status'] !== 'rolled_back_to_generation'
    || (int)$restoreReport['generation_id'] !== $retainedGenerationId
    || $restoreReport['prior_generation_id'] !== null
    || !is_array($restoredState)
    || (string)$restoredState['activation_mode'] !== 'HYBRID'
    || (int)$restoredState['active_generation_id'] !== $retainedGenerationId
    || !(bool)$restoredState['readiness_latch']
    || !is_array($restoreAudit)
    || (int)$restoreAudit['generation_id'] !== $retainedGenerationId
    || $restoreAudit['prior_generation_id'] !== null
    || (string)$restoreAudit['actor'] !== 'admin_user_id:7'
    || (string)$restoreAudit['mode'] !== 'HYBRID'
    || (string)$restoreAudit['status'] !== 'COMPLETED'
    || $finalAuditCount !== $auditCount + 3
    || $afterRestoreSet !== $beforeRollbackSet
) {
    throw new RuntimeException('The retained generation was not restored intact under an exact audit.');
}

$configWriter->delete(Config::XML_PATH_ENCODER_ENDPOINT);
$cacheTypes->cleanType('config');
$connection->beginTransaction();
try {
    $connection->update(
        $generationTable,
        ['state' => 'READY'],
        ['generation_id = ?' => $retainedGenerationId, 'state = ?' => 'ACTIVE']
    );
    $connection->update(
        $generationTable,
        ['state' => 'ACTIVE'],
        ['generation_id = ?' => $fixtureGenerationId, 'state = ?' => 'RETAINED']
    );
    $connection->update(
        $stateTable,
        [
            'activation_mode' => (string)$fixtureState['activation_mode'],
            'active_generation_id' => $fixtureGenerationId,
            'required_watermark' => (int)$fixtureState['required_watermark'],
            'native_watermark' => (int)$fixtureState['native_watermark'],
            'hybrid_watermark' => (int)$fixtureState['hybrid_watermark'],
            'readiness_latch' => (int)$fixtureState['readiness_latch'],
            'cache_version' => new Zend_Db_Expr('cache_version + 1'),
        ],
        ['store_id = ?' => 1]
    );
    $connection->commit();
} catch (Throwable $throwable) {
    $connection->rollBack();
    throw $throwable;
}

printf(
    "Admin rollback restored production generation %d without rebuilding %d embeddings under audits %d and %d.\n",
    $retainedGenerationId,
    (int)$afterRestoreSet['embedding_count'],
    (int)$report['activation_id'],
    (int)$restoreReport['activation_id']
);
