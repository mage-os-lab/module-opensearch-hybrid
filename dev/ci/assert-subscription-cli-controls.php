<?php
declare(strict_types=1);

use MageOS\AsyncEvents\Api\AsyncEventRepositoryInterface;
use MageOS\AsyncEvents\Api\Data\AsyncEventInterface;
use MageOS\AsyncEvents\Model\ResourceModel\AsyncEvent as AsyncEventResource;
use MageOS\OpenSearchHybrid\Console\EventsInstallCommand;
use MageOS\OpenSearchHybrid\Model\AsyncEvent\LifecyclePublisher;
use MageOS\OpenSearchHybrid\Model\AsyncEvent\SubscriptionReconciler;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\ResourceConnection;
use Magento\Framework\App\State;
use Magento\Framework\Exception\LocalizedException;
use Symfony\Component\Console\Tester\CommandTester;

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
} catch (LocalizedException) {
    // Another bootstrap participant already set the area code.
}

/** @var ResourceConnection $resourceConnection */
$resourceConnection = $objectManager->get(ResourceConnection::class);
$connection = $resourceConnection->getConnection();
$table = $resourceConnection->getTableName('async_event_subscriber');
/** @var SubscriptionReconciler $reconciler */
$reconciler = $objectManager->get(SubscriptionReconciler::class);
$reconciler->reconcile(1);

$original = $connection->fetchRow(
    $connection->select()
        ->from($table)
        ->where('metadata = ?', SubscriptionReconciler::METADATA)
        ->where('store_id = ?', 1)
        ->order('subscription_id ASC')
        ->limit(1)
);
if (!is_array($original)) {
    throw new RuntimeException('The subscription CLI fixture requires one module-owned subscription.');
}
$subscriptionId = (int)$original['subscription_id'];

$runCommand = static function (array $arguments) use ($objectManager): array {
    $tester = new CommandTester($objectManager->create(EventsInstallCommand::class));
    $exitCode = $tester->execute($arguments);
    if ($exitCode !== 0) {
        throw new RuntimeException(sprintf(
            'Subscription CLI exited %d: %s',
            $exitCode,
            trim($tester->getDisplay())
        ));
    }
    $report = json_decode(trim($tester->getDisplay()), true, 512, JSON_THROW_ON_ERROR);
    if (!is_array($report)) {
        throw new RuntimeException('The subscription CLI did not return a JSON object.');
    }

    return $report;
};

$initialPreview = $runCommand([
    '--store' => '1',
    '--dry-run' => true,
]);
if ((string)$initialPreview['operation'] !== 'reconcile_async_event_subscription'
    || (string)$initialPreview['mode'] !== 'preview'
    || (string)$initialPreview['change']['canonical_action'] !== 'none'
    || !is_string($initialPreview['confirmation_token'])
) {
    throw new RuntimeException('The subscription CLI preview omitted exact current-state evidence.');
}

$implicitTester = new CommandTester($objectManager->create(EventsInstallCommand::class));
if ($implicitTester->execute(['--store' => '1']) !== 2
    || !str_contains($implicitTester->getDisplay(), '--dry-run')
    || !str_contains($implicitTester->getDisplay(), '--confirm')
) {
    throw new RuntimeException('The subscription CLI accepted an implicit mutation.');
}

/** @var AsyncEventRepositoryInterface $repository */
$repository = $objectManager->get(AsyncEventRepositoryInterface::class);
$saveCandidate = $repository->get($subscriptionId);
if (!$saveCandidate instanceof AsyncEventInterface) {
    throw new RuntimeException('The subscription repository did not return a mutable event model.');
}
$saveCandidate->setMetadata('cleared-before-save');
try {
    $repository->save($saveCandidate, false);
    throw new RuntimeException('The repository allowed module subscription ownership to be cleared.');
} catch (LocalizedException $exception) {
    if (!str_contains($exception->getMessage(), 'cannot be edited directly')) {
        throw $exception;
    }
}

$deleteCandidate = $repository->get($subscriptionId);
if (!$deleteCandidate instanceof AsyncEventInterface) {
    throw new RuntimeException('The subscription repository did not return a deletable event model.');
}
$deleteCandidate->setMetadata('cleared-before-delete');
/** @var AsyncEventResource $eventResource */
$eventResource = $objectManager->get(AsyncEventResource::class);
try {
    $eventResource->delete($deleteCandidate);
    throw new RuntimeException('The resource model allowed module subscription ownership to be cleared.');
} catch (LocalizedException $exception) {
    if (!str_contains($exception->getMessage(), 'cannot be deleted directly')) {
        throw $exception;
    }
}

$duplicateId = null;
$probeFailure = null;
try {
    $connection->update(
        $table,
        ['event_name' => 'mageos.opensearch_hybrid.drifted'],
        ['subscription_id = ?' => $subscriptionId]
    );
    $connection->insert($table, [
        'event_name' => LifecyclePublisher::EVENT_NAME,
        'recipient_url' => SubscriptionReconciler::RECIPIENT,
        'verification_token' => '',
        'status' => 1,
        'subscribed_at' => new Zend_Db_Expr('UTC_TIMESTAMP()'),
        'metadata' => SubscriptionReconciler::METADATA,
        'store_id' => 1,
    ]);
    $duplicateId = (int)$connection->lastInsertId($table);

    try {
        $runCommand([
            '--store' => '1',
            '--confirm' => (string)$initialPreview['confirmation_token'],
        ]);
        throw new RuntimeException('The CLI accepted a stale subscription confirmation.');
    } catch (InvalidArgumentException $exception) {
        if (!str_contains($exception->getMessage(), 'current exact preview')) {
            throw $exception;
        }
    }
    $unchangedDrift = $connection->fetchRow(
        $connection->select()
            ->from($table, ['event_name', 'status'])
            ->where('subscription_id = ?', $subscriptionId)
    );
    $unchangedDuplicateStatus = (int)$connection->fetchOne(
        $connection->select()
            ->from($table, ['status'])
            ->where('subscription_id = ?', $duplicateId)
    );
    if (!is_array($unchangedDrift)
        || (string)$unchangedDrift['event_name'] !== 'mageos.opensearch_hybrid.drifted'
        || $unchangedDuplicateStatus !== 1
    ) {
        throw new RuntimeException('A rejected subscription confirmation changed database state.');
    }

    $repairPreview = $runCommand([
        '--store' => '1',
        '--dry-run' => true,
    ]);
    if ((string)$repairPreview['change']['canonical_action'] !== 'repair'
        || $repairPreview['change']['canonical_fields'] !== ['event_name']
        || !in_array($duplicateId, $repairPreview['change']['duplicate_subscription_ids'], true)
    ) {
        throw new RuntimeException('The subscription preview did not report exact drift and duplicate impact.');
    }
    $applied = $runCommand([
        '--store' => '1',
        '--confirm' => (string)$repairPreview['confirmation_token'],
    ]);
    $repaired = $connection->fetchRow(
        $connection->select()
            ->from($table, ['event_name', 'recipient_url', 'verification_token', 'status'])
            ->where('subscription_id = ?', $subscriptionId)
    );
    $duplicateStatus = (int)$connection->fetchOne(
        $connection->select()
            ->from($table, ['status'])
            ->where('subscription_id = ?', $duplicateId)
    );
    if ((string)$applied['mode'] !== 'applied'
        || (int)$applied['subscription_id'] !== $subscriptionId
        || !is_array($repaired)
        || (string)$repaired['event_name'] !== LifecyclePublisher::EVENT_NAME
        || (string)$repaired['recipient_url'] !== SubscriptionReconciler::RECIPIENT
        || (string)$repaired['verification_token'] !== (string)$original['verification_token']
        || (int)$repaired['status'] !== 1
        || $duplicateStatus !== 0
    ) {
        throw new RuntimeException('The confirmed subscription repair did not apply its exact preview.');
    }
} catch (Throwable $throwable) {
    $probeFailure = $throwable;
} finally {
    if ($duplicateId !== null) {
        $connection->delete($table, ['subscription_id = ?' => $duplicateId]);
    }
    $connection->update(
        $table,
        [
            'event_name' => (string)$original['event_name'],
            'recipient_url' => (string)$original['recipient_url'],
            'verification_token' => (string)$original['verification_token'],
            'status' => (int)$original['status'],
            'subscribed_at' => $original['subscribed_at'],
            'metadata' => (string)$original['metadata'],
            'store_id' => (int)$original['store_id'],
        ],
        ['subscription_id = ?' => $subscriptionId]
    );
}
if ($probeFailure instanceof Throwable) {
    throw $probeFailure;
}

printf(
    "Subscription %d required state-bound CLI confirmation and resisted repository mutation bypasses.\n",
    $subscriptionId
);
