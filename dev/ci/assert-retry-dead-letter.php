<?php
declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Config;
use MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\Config\ReinitableConfigInterface;
use Magento\Framework\App\Config\Storage\WriterInterface;
use Magento\Framework\App\ResourceConnection;

$fixtureRoot = $argv[1] ?? '';
if ($fixtureRoot === '' || !is_file($fixtureRoot . '/app/bootstrap.php')) {
    fwrite(STDERR, "Provide the installed Mage-OS fixture root.\n");
    exit(2);
}

require $fixtureRoot . '/app/bootstrap.php';
$bootstrap = Bootstrap::create(BP, $_SERVER);
$objectManager = $bootstrap->getObjectManager();

/** @var WriterInterface $configWriter */
$configWriter = $objectManager->get(WriterInterface::class);
/** @var ReinitableConfigInterface $config */
$config = $objectManager->get(ReinitableConfigInterface::class);
/** @var ResourceConnection $resourceConnection */
$resourceConnection = $objectManager->get(ResourceConnection::class);
/** @var OutboxRepository $outboxRepository */
$outboxRepository = $objectManager->get(OutboxRepository::class);

$configWriter->save(Config::XML_PATH_JOB_MAX_ATTEMPTS, '2');
$config->reinit();

$connection = $resourceConnection->getConnection();
$productTable = $resourceConnection->getTableName('catalog_product_entity');
$productId = (int)$connection->fetchOne(
    $connection->select()->from($productTable, ['entity_id'])->order('entity_id ASC')->limit(1)
);
if ($productId <= 0) {
    throw new RuntimeException('The retry fixture requires at least one catalog product.');
}
$generationId = (int)$connection->fetchOne(
    $connection->select()
        ->from($resourceConnection->getTableName('mageos_opensearch_hybrid_generation'), ['generation_id'])
        ->where('store_id = ?', 1)
        ->order('generation_id DESC')
        ->limit(1)
);
if ($generationId <= 0) {
    throw new RuntimeException('The retry fixture requires a hybrid generation.');
}

$coalesceJobId = $outboxRepository->create(1, $generationId, 'EMBEDDING', 'UPSERT', [$productId], 1);
$coalescedBoundary = $outboxRepository->supersedeProductJobs(
    1,
    $generationId,
    'EMBEDDING',
    $productId,
    2
);
$coalesced = $outboxRepository->get($coalesceJobId);
if ($coalescedBoundary !== 0 || (string)$coalesced['state'] !== 'SUPERSEDED') {
    throw new RuntimeException('A newer revision did not supersede boundary-zero full-build work.');
}
$connection->delete(
    $resourceConnection->getTableName('mageos_opensearch_hybrid_outbox'),
    ['job_id = ?' => $coalesceJobId]
);

$jobId = $outboxRepository->create(1, null, 'EMBEDDING', 'UPSERT', [$productId], 1);
$outboxTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_outbox');

$outboxRepository->markPublished($jobId);
if (!$outboxRepository->claim($jobId, 'retry-fixture:first', 30)) {
    throw new RuntimeException('The first retry fixture claim failed.');
}
$terminal = $outboxRepository->retry($jobId, new RuntimeException('fixture failure'), 1);
$first = $outboxRepository->get($jobId);
if ($terminal || (string)$first['state'] !== 'RETRY' || (int)$first['attempt_count'] !== 1) {
    throw new RuntimeException('The first failed attempt did not enter RETRY.');
}

$connection->update($outboxTable, ['next_eligible_at' => null], ['job_id = ?' => $jobId]);
$outboxRepository->markPublished($jobId);
if (!$outboxRepository->claim($jobId, 'retry-fixture:second', 30)) {
    throw new RuntimeException('The terminal retry fixture claim failed.');
}
$terminal = $outboxRepository->retry($jobId, new RuntimeException('fixture failure'), 1);
$dead = $outboxRepository->get($jobId);
if (!$terminal || (string)$dead['state'] !== 'DEAD' || (int)$dead['attempt_count'] !== 2) {
    throw new RuntimeException('The final failed attempt did not enter DEAD.');
}

$replayed = $outboxRepository->prepareManualRetry($jobId);
if ((string)$replayed['state'] !== 'PENDING'
    || (int)$replayed['attempt_count'] !== 0
    || (int)$replayed['replay_count'] !== 1
) {
    throw new RuntimeException('Manual replay did not reset attempts and increment replay_count.');
}

$connection->delete($outboxTable, ['job_id = ?' => $jobId]);
$configWriter->save(Config::XML_PATH_JOB_MAX_ATTEMPTS, '5');
$config->reinit();

printf("Job %s reached RETRY and DEAD, then entered audited manual replay.\n", $jobId);
