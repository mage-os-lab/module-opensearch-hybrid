<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model;

class DoctorService
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Config $config,
        private readonly \MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract $retrievalContract,
        private readonly \Magento\Framework\Search\EngineResolverInterface $engineResolver,
        private readonly \Magento\Framework\App\DeploymentConfig $deploymentConfig,
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \MageOS\OpenSearchHybrid\Model\Operations\OperationalMetricsService $metricsService,
        private readonly \MageOS\OpenSearchHybrid\Model\OpenSearch\ClusterVersionProvider $clusterVersionProvider,
        private readonly \MageOS\OpenSearchHybrid\Model\OpenSearch\VersionPolicy $versionPolicy,
        private readonly \MageOS\OpenSearchHybrid\Model\OpenSearch\VectorWireQualification $wireQualification,
        private readonly \MageOS\OpenSearchHybrid\Model\Similarity\SimilarityStatusService $similarityStatusService
    ) {
    }

    public function diagnose(): array
    {
        $checks = [];
        $checks['component_identity'] = [
            'ok' => $this->retrievalContract->digest() !== '',
            'value' => 'MageOS_OpenSearchHybrid',
        ];
        $engine = (string)$this->engineResolver->getCurrentSearchEngine();
        $checks['global_wrapper'] = [
            'ok' => $engine === 'mageos_opensearch_hybrid',
            'value' => $engine,
            'required_for_build' => false,
        ];
        $queue = $this->deploymentConfig->getConfigData('queue');
        $amqp = is_array($queue) ? ($queue['amqp'] ?? null) : null;
        $checks['rabbitmq'] = [
            'ok' => is_array($amqp) && !empty($amqp['host']),
            'value' => is_array($amqp) && !empty($amqp['host']) ? 'configured' : 'missing',
        ];
        $openSearchVersion = null;
        try {
            $openSearchVersion = $this->clusterVersionProvider->current();
            $checks['opensearch_version'] = [
                'ok' => $this->versionPolicy->isSupported($openSearchVersion),
                'value' => $openSearchVersion,
                'required' => \MageOS\OpenSearchHybrid\Model\OpenSearch\VersionPolicy::SUPPORTED_RANGE,
            ];
        } catch (\Throwable $throwable) {
            $checks['opensearch_version'] = [
                'ok' => false,
                'value' => $throwable::class,
                'required' => \MageOS\OpenSearchHybrid\Model\OpenSearch\VersionPolicy::SUPPORTED_RANGE,
            ];
        }
        $checks['base64_vector_ingestion'] = [
            'ok' => true,
            'value' => $this->wireQualification->get($openSearchVersion),
        ];
        try {
            $endpoint = $this->config->encoderEndpoint();
            $checks['encoder_endpoint'] = ['ok' => true, 'value' => $endpoint];
        } catch (\Throwable $throwable) {
            $checks['encoder_endpoint'] = ['ok' => false, 'value' => $throwable::class];
        }
        $connection = $this->resourceConnection->getConnection();
        $subscriberTable = $this->resourceConnection->getTableName('async_event_subscriber');
        $subscriptionCount = (int)$connection->fetchOne(
            $connection->select()
                ->from($subscriberTable, [new \Zend_Db_Expr('COUNT(*)')])
                ->where('metadata = ?', \MageOS\OpenSearchHybrid\Model\AsyncEvent\SubscriptionReconciler::METADATA)
                ->where('status = ?', 1)
        );
        $checks['subscriptions'] = ['ok' => $subscriptionCount > 0, 'value' => $subscriptionCount];
        $operations = $this->metricsService->get();
        $blockingAlerts = [];
        foreach ($operations['alerts'] as $alert) {
            if (in_array($alert['severity'], ['critical', 'high'], true)) {
                $blockingAlerts[] = $alert;
            }
        }
        $checks['operational_state'] = [
            'ok' => $blockingAlerts === [],
            'value' => $blockingAlerts,
        ];
        $checks['async_event_trace_cleanup'] = [
            'ok' => (bool)$operations['async_event_trace_cleanup']['bounded'],
            'value' => $operations['async_event_trace_cleanup'],
        ];
        $activeCompatibility = [];
        foreach ($operations['readiness'] as $readiness) {
            if ($readiness['activation_mode'] !== 'HYBRID') {
                continue;
            }
            $generation = null;
            foreach ($operations['generations'] as $candidate) {
                if ($candidate['store_id'] === $readiness['store_id'] && $candidate['active_for_store']) {
                    $generation = $candidate;
                    break;
                }
            }
            $activeCompatibility[] = [
                'store_id' => $readiness['store_id'],
                'generation_id' => $readiness['active_generation_id'],
                'ready' => $readiness['ready'],
                'contract_current' => is_array($generation) && $generation['contract_current'],
            ];
        }
        $checks['active_generation_compatible'] = [
            'ok' => !in_array(false, array_map(
                static fn (array $state): bool => $state['ready'] && $state['contract_current'],
                $activeCompatibility
            ), true),
            'value' => $activeCompatibility,
        ];
        $radialSimilarity = [];
        $stateTable = $this->resourceConnection->getTableName('mageos_opensearch_hybrid_store_state');
        $storeIds = $connection->fetchCol(
            $connection->select()->from($stateTable, ['store_id'])->order('store_id ASC')
        );
        foreach ($storeIds as $storeId) {
            $radialSimilarity[] = $this->similarityStatusService->get((int)$storeId);
        }
        $checks['radial_similarity'] = [
            'ok' => !in_array(
                \MageOS\OpenSearchHybrid\Model\Similarity\SimilarityStatusService::STATE_FAILED_CLOSED,
                array_column($radialSimilarity, 'state'),
                true
            ),
            'value' => $radialSimilarity,
            'required_for_catalog_search' => false,
        ];

        return [
            'healthy' => !in_array(false, array_column($checks, 'ok'), true),
            'checks' => $checks,
        ];
    }
}
