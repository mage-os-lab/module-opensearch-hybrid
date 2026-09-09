<?php

declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository;
use MageOS\OpenSearchHybrid\Model\Queue\Topics;
use MageOS\OpenSearchHybrid\Model\Store\ReadinessLatch;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\ResourceConnection;
use PhpAmqpLib\Connection\AMQPStreamConnection;

if ($argc !== 4) {
    fwrite(
        STDERR,
        "Usage: php assert-queue-isolation.php /path/to/mageos /path/to/manifest.json elapsed-seconds\n"
    );
    exit(2);
}

$mageOsRoot = rtrim((string)$argv[1], DIRECTORY_SEPARATOR);
$manifestPath = (string)$argv[2];
$elapsedSeconds = filter_var($argv[3], FILTER_VALIDATE_INT);
$bootstrapFile = $mageOsRoot . '/app/bootstrap.php';
if (!is_file($bootstrapFile)) {
    fwrite(STDERR, "Mage-OS bootstrap not found: {$bootstrapFile}\n");
    exit(2);
}
if (!is_file($manifestPath)) {
    fwrite(STDERR, "Queue-isolation manifest not found: {$manifestPath}\n");
    exit(2);
}
if ($elapsedSeconds === false || $elapsedSeconds < 0) {
    fwrite(STDERR, "The correctness-consumer elapsed time is invalid.\n");
    exit(2);
}

require $bootstrapFile;

/** @var array<string, int|string> $manifest */
$manifest = json_decode((string)file_get_contents($manifestPath), true, 512, JSON_THROW_ON_ERROR);
$bootstrap = Bootstrap::create(BP, $_SERVER);
$objectManager = $bootstrap->getObjectManager();
/** @var OutboxRepository $outboxRepository */
$outboxRepository = $objectManager->get(OutboxRepository::class);
/** @var ResourceConnection $resourceConnection */
$resourceConnection = $objectManager->get(ResourceConnection::class);
/** @var ReadinessLatch $readinessLatch */
$readinessLatch = $objectManager->get(ReadinessLatch::class);

$embeddingJob = $outboxRepository->get((string)$manifest['embedding_job_id']);
$priorityJob = $outboxRepository->get((string)$manifest['priority_job_id']);
$connection = $resourceConnection->getConnection();
$journal = $connection->fetchRow(
    $connection->select()
        ->from($resourceConnection->getTableName('mageos_opensearch_hybrid_change_journal'))
        ->where('change_id = ?', (int)$manifest['watermark'])
);
$state = $connection->fetchRow(
    $connection->select()
        ->from($resourceConnection->getTableName('mageos_opensearch_hybrid_store_state'))
        ->where('store_id = ?', (int)$manifest['store_id'])
);
if (!is_array($state)) {
    throw new RuntimeException('The queue-isolation probe did not create store readiness state.');
}
if (!is_array($journal)) {
    throw new RuntimeException('The queue-isolation probe did not create a product change journal row.');
}

$amqp = new AMQPStreamConnection(
    '127.0.0.1',
    35672,
    'mageos',
    'mageos_ci_only',
    'mageos_hybrid_ci'
);
$channel = $amqp->channel();
try {
    [, $embeddingDepth] = $channel->queue_declare(Topics::EMBEDDING_QUEUE, true);
    [, $priorityDepth] = $channel->queue_declare(Topics::PRIORITY_QUEUE, true);
} finally {
    $channel->close();
    $amqp->close();
}

$failures = [];
if ((string)$priorityJob['state'] !== 'COMPLETED') {
    $failures[] = sprintf('Correctness job state is %s, not COMPLETED.', (string)$priorityJob['state']);
}
if ((string)$embeddingJob['state'] !== 'PUBLISHED') {
    $failures[] = sprintf('Embedding job state is %s, not PUBLISHED.', (string)$embeddingJob['state']);
}
if ($embeddingDepth < (int)$manifest['backlog_size']) {
    $failures[] = sprintf(
        'Embedding queue depth is %d, below the expected backlog of %d.',
        $embeddingDepth,
        (int)$manifest['backlog_size']
    );
}
if ($priorityDepth !== 0) {
    $failures[] = sprintf('Correctness queue depth is %d after its consumer exited.', $priorityDepth);
}
if ($elapsedSeconds > 20) {
    $failures[] = sprintf('Correctness consumer took %d seconds, above the 20-second bound.', $elapsedSeconds);
}
if ((int)$state['required_watermark'] !== (int)$manifest['watermark']) {
    $failures[] = 'The required watermark does not match the queued correctness revision.';
}
if ((int)$state['hybrid_watermark'] !== (int)$manifest['watermark']) {
    $failures[] = 'The hybrid watermark did not advance with the completed correctness job.';
}
if ((string)$journal['native_state'] !== 'COMPLETED'
    || (string)$journal['hybrid_state'] !== 'COMPLETED'
) {
    $failures[] = 'The product change was not acknowledged by both native and hybrid indexing.';
}
if ((int)$state['native_watermark'] !== (int)$manifest['watermark']) {
    $failures[] = 'The native watermark did not advance after native indexing completed.';
}
if ((bool)$state['readiness_latch'] || $readinessLatch->isOpen((int)$manifest['store_id'])) {
    $failures[] = 'The readiness latch opened for an inactive deterministic generation.';
}

if ($failures !== []) {
    fwrite(STDERR, implode("\n", $failures) . "\n");
    exit(1);
}

fwrite(
    STDOUT,
    sprintf(
        "Product save reached both correctness watermarks in %d seconds while %d embedding messages remained queued.\n",
        $elapsedSeconds,
        $embeddingDepth
    )
);
