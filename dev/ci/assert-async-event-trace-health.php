<?php
declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Operations\OperationalMetricsService;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\ResourceConnection;
use Magento\Framework\App\State;

$fixtureRoot = $argv[1] ?? '';
$storeId = filter_var($argv[2] ?? '1', FILTER_VALIDATE_INT, [
    'options' => ['min_range' => 1],
]);
if ($fixtureRoot === '' || !is_file($fixtureRoot . '/app/bootstrap.php') || $storeId === false) {
    fwrite(STDERR, "Provide the installed Mage-OS fixture root and a positive store ID.\n");
    exit(2);
}

require $fixtureRoot . '/app/bootstrap.php';
$bootstrap = Bootstrap::create(BP, $_SERVER);
$objectManager = $bootstrap->getObjectManager();
$state = $objectManager->get(State::class);
try {
    $state->setAreaCode('global');
} catch (\Magento\Framework\Exception\LocalizedException) {
    if ($state->getAreaCode() !== 'global') {
        throw new RuntimeException('Trace health verification requires the global area.');
    }
}

/** @var OperationalMetricsService $metricsService */
$metricsService = $objectManager->get(OperationalMetricsService::class);
/** @var ResourceConnection $resourceConnection */
$resourceConnection = $objectManager->get(ResourceConnection::class);
$connection = $resourceConnection->getConnection();
$traceTable = $resourceConnection->getTableName('async_event_subscriber_log');
$baseline = $metricsService->get($storeId);
$baselineSubscription = null;
foreach ($baseline['subscriptions'] as $subscription) {
    if ((bool)$subscription['healthy']) {
        $baselineSubscription = $subscription;
        break;
    }
}
if (!is_array($baselineSubscription)) {
    throw new RuntimeException('Trace health verification requires a module-owned subscription.');
}
$subscriptionId = (int)$baselineSubscription['subscription_id'];
$baselineFailures = (int)$baselineSubscription['trace_failure_count'];
$baselineUnresolved = (int)$baselineSubscription['trace_unresolved_failure_count'];
$recoveredUuid = 'metrics-recovered-' . bin2hex(random_bytes(8));
$failedUuid = 'metrics-failed-' . bin2hex(random_bytes(8));
$subscriptionMetrics = static function (array $metrics) use ($subscriptionId): array {
    foreach ($metrics['subscriptions'] as $subscription) {
        if ((int)$subscription['subscription_id'] === $subscriptionId) {
            return $subscription;
        }
    }
    throw new RuntimeException('The module-owned subscription disappeared during trace verification.');
};
$hasHandoffFailureAlert = static function (array $metrics): bool {
    foreach ($metrics['alerts'] as $alert) {
        if ((string)$alert['code'] === 'async_event_handoff_failure') {
            return true;
        }
    }

    return false;
};
try {
    $connection->insertMultiple($traceTable, [
        [
            'uuid' => $recoveredUuid,
            'subscription_id' => $subscriptionId,
            'success' => 0,
            'response_data' => 'fixture failure',
        ],
        [
            'uuid' => $recoveredUuid,
            'subscription_id' => $subscriptionId,
            'success' => 1,
            'response_data' => 'fixture recovery',
        ],
    ]);
    $recoveredMetrics = $metricsService->get($storeId);
    $recoveredSubscription = $subscriptionMetrics($recoveredMetrics);
    if ((int)$recoveredSubscription['trace_failure_count'] !== $baselineFailures + 1
        || (int)$recoveredSubscription['trace_unresolved_failure_count'] !== $baselineUnresolved
        || $hasHandoffFailureAlert($recoveredMetrics) !== $hasHandoffFailureAlert($baseline)
    ) {
        throw new RuntimeException('A recovered Async Events trace remained operationally unresolved.');
    }

    $connection->insert($traceTable, [
        'uuid' => $failedUuid,
        'subscription_id' => $subscriptionId,
        'success' => 0,
        'response_data' => 'fixture terminal failure',
    ]);
    $failedMetrics = $metricsService->get($storeId);
    $failedSubscription = $subscriptionMetrics($failedMetrics);
    if ((int)$failedSubscription['trace_unresolved_failure_count'] !== $baselineUnresolved + 1
        || !$hasHandoffFailureAlert($failedMetrics)
    ) {
        throw new RuntimeException('A terminal Async Events trace did not fail operational health.');
    }
} finally {
    $connection->delete($traceTable, ['uuid IN (?)' => [$recoveredUuid, $failedUuid]]);
}

printf(
    "Async Events trace health preserved %d historical failure(s) and %d unresolved trace(s).\n",
    $baselineFailures,
    $baselineUnresolved
);
