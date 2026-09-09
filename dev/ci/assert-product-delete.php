<?php

declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository;
use MageOS\OpenSearchHybrid\Model\Queue\Topics;
use Magento\Catalog\Api\ProductRepositoryInterface;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\ResourceConnection;
use Magento\Framework\Exception\NoSuchEntityException;
use PhpAmqpLib\Connection\AMQPStreamConnection;

if ($argc !== 4) {
    fwrite(STDERR, "Usage: php assert-product-delete.php /path/to/mageos /path/to/manifest.json elapsed-seconds\n");
    exit(2);
}

$mageOsRoot = rtrim((string)$argv[1], DIRECTORY_SEPARATOR);
$manifestPath = (string)$argv[2];
$elapsedSeconds = filter_var($argv[3], FILTER_VALIDATE_INT);
$bootstrapFile = $mageOsRoot . '/app/bootstrap.php';
if (!is_file($bootstrapFile) || !is_file($manifestPath) || $elapsedSeconds === false) {
    fwrite(STDERR, "Mage-OS bootstrap, manifest, or elapsed time is invalid.\n");
    exit(2);
}

require $bootstrapFile;

/** @var array<string, int|string> $manifest */
$manifest = json_decode((string)file_get_contents($manifestPath), true, 512, JSON_THROW_ON_ERROR);
$bootstrap = Bootstrap::create(BP, $_SERVER);
$objectManager = $bootstrap->getObjectManager();
/** @var ProductRepositoryInterface $productRepository */
$productRepository = $objectManager->get(ProductRepositoryInterface::class);
/** @var ResourceConnection $resourceConnection */
$resourceConnection = $objectManager->get(ResourceConnection::class);
/** @var OutboxRepository $outboxRepository */
$outboxRepository = $objectManager->get(OutboxRepository::class);

$productMissing = false;
try {
    $productRepository->get((string)$manifest['product_sku'], false, null, true);
} catch (NoSuchEntityException) {
    $productMissing = true;
}
$connection = $resourceConnection->getConnection();
$journal = $connection->fetchRow(
    $connection->select()
        ->from($resourceConnection->getTableName('mageos_opensearch_hybrid_change_journal'))
        ->where('change_id = ?', (int)$manifest['delete_change_id'])
);
$state = $connection->fetchRow(
    $connection->select()
        ->from($resourceConnection->getTableName('mageos_opensearch_hybrid_store_state'))
        ->where('store_id = ?', (int)$manifest['store_id'])
);
$deleteJob = $outboxRepository->get((string)$manifest['delete_job_id']);
$embeddingJob = $outboxRepository->get((string)$manifest['embedding_job_id']);
$embedding = $connection->fetchRow(
    $connection->select()
        ->from($resourceConnection->getTableName('mageos_opensearch_hybrid_embedding'))
        ->where('store_id = ?', (int)$manifest['store_id'])
        ->where('product_id = ?', (int)$manifest['product_id'])
        ->where('generation_id = ?', (int)$manifest['generation_id'])
);

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
if (!$productMissing) {
    $failures[] = 'Deleted fixture product is still loadable.';
}
if (!is_array($journal)
    || (string)$journal['operation'] !== 'DELETE'
    || (string)$journal['native_state'] !== 'COMPLETED'
    || (string)$journal['hybrid_state'] !== 'COMPLETED'
) {
    $failures[] = 'Deletion journal row is not complete in both indexing lanes.';
}
if ((string)$deleteJob['state'] !== 'COMPLETED') {
    $failures[] = 'Deletion priority job did not complete.';
}
if ((string)$embeddingJob['state'] !== 'SUPERSEDED') {
    $failures[] = 'Deleted product embedding work was not superseded.';
}
if (!is_array($embedding)
    || (string)$embedding['status'] !== 'DELETED'
    || (int)$embedding['revision'] !== (int)$manifest['delete_change_id']
) {
    $failures[] = 'Deleted product embedding state has no durable revision tombstone.';
}
if (!is_array($state)
    || (int)$state['required_watermark'] !== (int)$manifest['delete_change_id']
    || (int)$state['native_watermark'] !== (int)$manifest['delete_change_id']
    || (int)$state['hybrid_watermark'] !== (int)$manifest['delete_change_id']
) {
    $failures[] = 'Deletion did not advance all correctness watermarks.';
}
if ($embeddingDepth < (int)$manifest['backlog_size']) {
    $failures[] = 'Embedding backlog was consumed while processing the deletion.';
}
if ($priorityDepth !== 0) {
    $failures[] = 'Correctness queue is not empty after deletion processing.';
}
if ((int)$elapsedSeconds > 20) {
    $failures[] = 'Deletion correctness processing exceeded 20 seconds.';
}
if ($failures !== []) {
    fwrite(STDERR, implode("\n", $failures) . "\n");
    exit(1);
}

fwrite(
    STDOUT,
    sprintf(
        "Deletion tombstone completed in %d seconds while %d embedding messages remained queued.\n",
        (int)$elapsedSeconds,
        $embeddingDepth
    )
);
