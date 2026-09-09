<?php
declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository;
use MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository;
use MageOS\OpenSearchHybrid\Model\Store\ReadinessLatch;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\ResourceConnection;

if ($argc !== 3) {
    fwrite(STDERR, "Usage: php assert-active-readiness.php /path/to/mageos /path/to/manifest.json\n");
    exit(2);
}

$fixtureRoot = rtrim((string)$argv[1], DIRECTORY_SEPARATOR);
$manifestPath = (string)$argv[2];
if (!is_file($fixtureRoot . '/app/bootstrap.php') || !is_file($manifestPath)) {
    fwrite(STDERR, "Provide the installed Mage-OS fixture and active-readiness manifest.\n");
    exit(2);
}

require $fixtureRoot . '/app/bootstrap.php';
$manifest = json_decode((string)file_get_contents($manifestPath), true, 512, JSON_THROW_ON_ERROR);
$bootstrap = Bootstrap::create(BP, $_SERVER);
$objectManager = $bootstrap->getObjectManager();
/** @var ResourceConnection $resourceConnection */
$resourceConnection = $objectManager->get(ResourceConnection::class);
$connection = $resourceConnection->getConnection();
$journal = $connection->fetchRow(
    $connection->select()
        ->from($resourceConnection->getTableName('mageos_opensearch_hybrid_change_journal'))
        ->where('change_id = ?', (int)$manifest['change_id'])
);
$state = $connection->fetchRow(
    $connection->select()
        ->from($resourceConnection->getTableName('mageos_opensearch_hybrid_store_state'))
        ->where('store_id = ?', (int)$manifest['store_id'])
);
/** @var OutboxRepository $outboxRepository */
$outboxRepository = $objectManager->get(OutboxRepository::class);
$job = $outboxRepository->get((string)$manifest['priority_job_id']);
/** @var ReadinessLatch $readinessLatch */
$readinessLatch = $objectManager->get(ReadinessLatch::class);
/** @var GenerationRepository $generationRepository */
$generationRepository = $objectManager->get(GenerationRepository::class);
$active = $generationRepository->activeForStore((int)$manifest['store_id']);
if (!is_array($journal)
    || (string)$journal['native_state'] !== 'COMPLETED'
    || (string)$journal['hybrid_state'] !== 'COMPLETED'
    || (string)$job['state'] !== 'COMPLETED'
    || !is_array($state)
    || (int)$state['required_watermark'] !== (int)$manifest['change_id']
    || (int)$state['native_watermark'] < (int)$manifest['change_id']
    || (int)$state['hybrid_watermark'] < (int)$manifest['change_id']
    || !(bool)$state['readiness_latch']
    || !$readinessLatch->isOpen((int)$manifest['store_id'])
    || !is_array($active)
    || (int)$active['generation_id'] !== (int)$manifest['generation_id']
) {
    throw new RuntimeException('Hybrid acknowledgement did not reopen the exact active generation.');
}

printf(
    "Active generation %d reopened only after native and hybrid reached change %d.\n",
    (int)$manifest['generation_id'],
    (int)$manifest['change_id']
);
