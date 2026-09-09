<?php
declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\AsyncEvent\LifecyclePublisher;
use MageOS\OpenSearchHybrid\Model\AsyncEvent\SubscriptionReconciler;
use MageOS\OpenSearchHybrid\Model\DoctorService;
use MageOS\OpenSearchHybrid\Model\Operations\OperationalMetricsService;
use MageOS\OpenSearchHybrid\Model\StatusService;
use MageOS\OpenSearchHybrid\Model\Store\ReadinessLatch;
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

/** @var OperationalMetricsService $metricsService */
$metricsService = $objectManager->get(OperationalMetricsService::class);
$snapshot = $metricsService->get(1);
$alertCodes = array_column($snapshot['alerts'], 'code');
if ((int)$snapshot['schema_version'] !== 1
    || $snapshot['scope']['store_id'] !== 1
    || $snapshot['readiness'] === []
    || $snapshot['generations'] === []
    || $snapshot['work'] === []
    || $snapshot['subscriptions'] === []
    || in_array('active_generation_not_ready', $alertCodes, true)
    || !in_array('async_event_trace_cleanup_disabled', $alertCodes, true)
    || (bool)$snapshot['async_event_trace_cleanup']['bounded']
    || $snapshot['external_metrics_required'] === []
) {
    throw new RuntimeException('The operations snapshot omitted required durable-state metrics or alerts.');
}
foreach ($snapshot['subscriptions'] as $subscription) {
    if (!(bool)$subscription['healthy']
        || (string)$subscription['event_name'] !== LifecyclePublisher::EVENT_NAME
        || (string)$subscription['recipient_url'] !== SubscriptionReconciler::RECIPIENT
    ) {
        throw new RuntimeException('A module-owned subscription is missing or has definition drift.');
    }
}
$activeGeneration = null;
foreach ($snapshot['generations'] as $generation) {
    if ((bool)$generation['active_for_store']) {
        $activeGeneration = $generation;
        break;
    }
}
if (!is_array($activeGeneration) || !(bool)$activeGeneration['contract_current']) {
    throw new RuntimeException('The active generation did not report a current retrieval contract.');
}
$completedCleanupEvents = 0;
foreach ($snapshot['cleanup_events'] as $event) {
    if ((string)$event['status'] === 'COMPLETED') {
        $completedCleanupEvents += (int)$event['events'];
    }
}
if ($completedCleanupEvents !== 1) {
    throw new RuntimeException('The operations snapshot did not count the completed cleanup audit.');
}

/** @var StatusService $statusService */
$statusService = $objectManager->get(StatusService::class);
$status = $statusService->get(1);
if ((int)($status['schema_version'] ?? 0) !== 2
    || ($status['operations']['scope']['store_id'] ?? null) !== 1
    || !($status['capabilities']['opensearch']['supported'] ?? false)
    || !($status['capabilities']['base64_vector_ingestion']['qualified'] ?? false)
    || ($status['similarity'][0]['state'] ?? null) !== 'DISABLED'
) {
    throw new RuntimeException('Status output did not include its store-scoped operations snapshot.');
}

/** @var ResourceConnection $resourceConnection */
$resourceConnection = $objectManager->get(ResourceConnection::class);
$connection = $resourceConnection->getConnection();
$generationTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_generation');
$stateTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_store_state');
$originalState = $connection->fetchRow(
    $connection->select()->from($stateTable)->where('store_id = ?', 1)
);
if (!is_array($originalState) || !(bool)$originalState['readiness_latch']) {
    throw new RuntimeException('The operations fixture requires the active-readiness drill to finish open.');
}
try {
    /** @var ReadinessLatch $readinessLatch */
    $readinessLatch = $objectManager->get(ReadinessLatch::class);
    $readinessLatch->requireWatermark(1, (int)$originalState['required_watermark'] + 1);
    $unready = $metricsService->get(1);
    if (!in_array('active_generation_not_ready', array_column($unready['alerts'], 'code'), true)) {
        throw new RuntimeException('The operations snapshot did not expose a closed active readiness latch.');
    }
    /** @var DoctorService $doctorService */
    $doctorService = $objectManager->get(DoctorService::class);
    $doctor = $doctorService->diagnose();
    if ((bool)$doctor['checks']['operational_state']['ok']
        || (bool)$doctor['checks']['async_event_trace_cleanup']['ok']
    ) {
        throw new RuntimeException('Doctor did not fail closed on active unreadiness and unbounded trace retention.');
    }
} finally {
    $connection->update(
        $stateTable,
        [
            'required_watermark' => (int)$originalState['required_watermark'],
            'native_watermark' => (int)$originalState['native_watermark'],
            'hybrid_watermark' => (int)$originalState['hybrid_watermark'],
            'readiness_latch' => (int)$originalState['readiness_latch'],
            'cache_version' => new Zend_Db_Expr('cache_version + 1'),
        ],
        ['store_id = ?' => 1]
    );
}
try {
    $connection->update(
        $generationTable,
        ['pipeline_digest' => str_repeat('0', 64)],
        ['generation_id = ?' => (int)$activeGeneration['generation_id']]
    );
    $drifted = $metricsService->get(1);
    if (!in_array('active_generation_contract_drift', array_column($drifted['alerts'], 'code'), true)) {
        throw new RuntimeException('The operations snapshot did not expose active contract drift.');
    }
} finally {
    $connection->update(
        $generationTable,
        ['pipeline_digest' => (string)$activeGeneration['pipeline_digest']],
        ['generation_id = ?' => (int)$activeGeneration['generation_id']]
    );
}

printf(
    "Operations snapshot emitted %d readiness row(s), %d work row(s), and %d alert(s).\n",
    count($snapshot['readiness']),
    count($snapshot['work']),
    count($snapshot['alerts'])
);
