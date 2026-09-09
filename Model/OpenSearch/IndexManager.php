<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\OpenSearch;

class IndexManager
{
    public function __construct(
        private readonly \Magento\Elasticsearch\SearchAdapter\ConnectionManager $connectionManager,
        private readonly \MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract $retrievalContract,
        private readonly \Magento\Framework\Serialize\Serializer\Json $json,
        private readonly \MageOS\OpenSearchHybrid\Model\OpenSearch\ClusterVersionProvider $clusterVersionProvider,
        private readonly \MageOS\OpenSearchHybrid\Model\OpenSearch\VersionPolicy $versionPolicy,
        private readonly \MageOS\OpenSearchHybrid\Model\OpenSearch\BulkPayloadBuilder $bulkPayloadBuilder
    ) {
    }

    public function ensureGeneration(array $generation): void
    {
        $this->assertSupportedVersion();
        $client = $this->client();
        $index = (string)$generation['physical_index'];
        if (!$client->indices()->exists(['index' => $index])) {
            $contract = $this->retrievalContract->get();
            $client->indices()->create([
                'index' => $index,
                'body' => [
                    'settings' => $contract['index_settings'],
                    'mappings' => $contract['mapping'],
                ],
            ]);
        }
        $pipeline = $this->retrievalContract->get()['pipeline'];
        $future = $client->transport->performRequest(
            'PUT',
            '/_search/pipeline/' . rawurlencode((string)$generation['pipeline_id']),
            [],
            $this->json->serialize($pipeline)
        );
        $client->transport->resultOrFuture($future);
    }

    public function indexDocument(array $generation, array $document, ?array $vector, int $revision): void
    {
        $this->indexDocuments($generation, [[
            'document' => $document,
            'vector' => $vector,
            'revision' => $revision,
        ]]);
    }

    public function indexDocuments(array $generation, array $records): void
    {
        if ($records === []) {
            return;
        }
        $this->assertSupportedVersion();
        $body = $this->bulkPayloadBuilder->build($generation, $records);
        $response = $this->client()->bulk(['body' => $body, 'refresh' => false]);
        $items = $response['items'] ?? null;
        if (!is_array($items) || count($items) !== count($records)) {
            throw new \RuntimeException('OpenSearch returned an invalid bulk indexing response.');
        }
        $failures = 0;
        foreach ($items as $item) {
            $result = is_array($item) ? ($item['index'] ?? null) : null;
            $status = is_array($result) ? (int)($result['status'] ?? 0) : 0;
            if (($status >= 200 && $status < 300) || $status === 409) {
                continue;
            }
            $failures++;
        }
        if ($failures > 0) {
            throw new \RuntimeException(sprintf(
                'OpenSearch bulk indexing failed for %d document(s).',
                $failures
            ));
        }
    }

    public function assertSupportedVersion(): void
    {
        $this->versionPolicy->assertSupported($this->clusterVersionProvider->current());
    }

    public function deleteDocument(array $generation, int $productId, int $revision): void
    {
        try {
            $this->client()->delete([
                'index' => (string)$generation['physical_index'],
                'id' => (string)$productId,
                'version' => max(1, $revision),
                'version_type' => 'external_gte',
                'refresh' => false,
            ]);
        } catch (
            \OpenSearch\Common\Exceptions\Missing404Exception
            | \OpenSearch\Common\Exceptions\Conflict409Exception
        ) {
            return;
        }
    }

    public function count(string $index): int
    {
        $response = $this->client()->count(['index' => $index]);

        return (int)($response['count'] ?? 0);
    }

    public function indexExists(string $index): bool
    {
        return (bool)$this->client()->indices()->exists(['index' => $index]);
    }

    public function deleteGenerationIndex(array $generation): void
    {
        try {
            $this->client()->indices()->delete(['index' => (string)$generation['physical_index']]);
        } catch (\OpenSearch\Common\Exceptions\Missing404Exception) {
            return;
        }
    }

    public function refreshGeneration(array $generation): void
    {
        $this->client()->indices()->refresh(['index' => (string)$generation['physical_index']]);
    }

    public function setRefreshInterval(array $generation, string $refreshInterval): void
    {
        if (!in_array((string)($generation['state'] ?? ''), ['BUILDING', 'CATCHING_UP'], true)) {
            throw new \RuntimeException(
                'Build refresh settings can change only on an inactive building generation.'
            );
        }
        if (!in_array($refreshInterval, ['-1', '1s'], true)) {
            throw new \InvalidArgumentException('The generation refresh interval is not allowlisted.');
        }
        $index = (string)$generation['physical_index'];
        $client = $this->client();
        $response = $client->indices()->putSettings([
            'index' => $index,
            'body' => [
                'index' => [
                    'refresh_interval' => $refreshInterval,
                ],
            ],
        ]);
        if (!is_array($response) || !(bool)($response['acknowledged'] ?? false)) {
            throw new \RuntimeException('OpenSearch did not acknowledge the refresh interval change.');
        }
        $settings = $client->indices()->getSettings(['index' => $index]);
        $actual = $settings[$index]['settings']['index']['refresh_interval'] ?? null;
        if ((string)$actual !== $refreshInterval) {
            throw new \RuntimeException('OpenSearch did not apply the requested refresh interval.');
        }
    }

    public function validateGeneration(array $generation): array
    {
        $client = $this->client();
        $index = (string)$generation['physical_index'];
        if (!$client->indices()->exists(['index' => $index])) {
            return [
                'index_exists' => false,
                'index_count' => 0,
                'mapping' => false,
                'settings' => false,
                'pipeline' => false,
                'document_identity' => false,
                'refresh_interval_restored' => false,
            ];
        }
        $contract = $this->retrievalContract->get();
        $mappingResponse = $client->indices()->getMapping(['index' => $index]);
        $actualMapping = $mappingResponse[$index]['mappings'] ?? null;
        $settingsResponse = $client->indices()->getSettings(['index' => $index]);
        $actualSettings = $settingsResponse[$index]['settings']['index'] ?? null;
        $pipelineId = (string)$generation['pipeline_id'];
        try {
            $future = $client->transport->performRequest(
                'GET',
                '/_search/pipeline/' . rawurlencode($pipelineId)
            );
            $pipelineResponse = $client->transport->resultOrFuture($future);
            $actualPipeline = is_array($pipelineResponse) ? ($pipelineResponse[$pipelineId] ?? null) : null;
        } catch (\OpenSearch\Common\Exceptions\Missing404Exception) {
            $actualPipeline = null;
        }
        $identityResponse = $client->count([
            'index' => $index,
            'body' => [
                'query' => [
                    'bool' => [
                        'filter' => [
                            ['term' => ['store_id' => (int)$generation['store_id']]],
                            ['term' => ['generation_id' => (int)$generation['generation_id']]],
                            ['term' => ['model_revision' => (string)$generation['model_revision']]],
                            ['term' => ['embedding_eligible' => true]],
                        ],
                    ],
                ],
            ],
        ]);
        $indexCount = $this->count($index);

        return [
            'index_exists' => true,
            'index_count' => $indexCount,
            'mapping' => is_array($actualMapping)
                && hash_equals(
                    $this->retrievalContract->mappingDigest(),
                    $this->retrievalContract->digestValue($actualMapping)
                ),
            'settings' => is_array($actualSettings)
                && $this->settingsMatch($actualSettings, $contract['index_settings']['index']),
            'pipeline' => is_array($actualPipeline)
                && hash_equals(
                    $this->retrievalContract->pipelineDigest(),
                    $this->retrievalContract->digestValue($actualPipeline)
                ),
            'document_identity' => (int)($identityResponse['count'] ?? 0) === $indexCount,
            'refresh_interval_restored' => is_array($actualSettings)
                && (string)($actualSettings['refresh_interval'] ?? '1s') === '1s',
        ];
    }

    private function settingsMatch(array $actual, array $expected): bool
    {
        foreach ($expected as $key => $value) {
            if (array_key_exists($key, $actual)) {
                $actualValue = $actual[$key];
                $found = true;
            } else {
                $path = explode('.', (string)$key);
                $actualValue = null;
                $found = false;
                for ($prefixLength = count($path) - 1; $prefixLength >= 1; $prefixLength--) {
                    $prefix = implode('.', array_slice($path, 0, $prefixLength));
                    if (!array_key_exists($prefix, $actual)) {
                        continue;
                    }
                    $actualValue = $actual[$prefix];
                    foreach (array_slice($path, $prefixLength) as $part) {
                        if (!is_array($actualValue) || !array_key_exists($part, $actualValue)) {
                            $actualValue = null;
                            continue 2;
                        }
                        $actualValue = $actualValue[$part];
                    }
                    $found = true;
                    break;
                }
            }
            if (!$found) {
                return false;
            }
            if (is_bool($value)) {
                if (filter_var($actualValue, FILTER_VALIDATE_BOOL, FILTER_NULL_ON_FAILURE) !== $value) {
                    return false;
                }
                continue;
            }
            if ((string)$actualValue !== (string)$value) {
                return false;
            }
        }

        return true;
    }

    private function client(): \OpenSearch\Client
    {
        $connection = $this->connectionManager->getConnection();
        if (!$connection instanceof \Magento\OpenSearch\Model\SearchClient) {
            throw new \RuntimeException('The active search connection is not OpenSearch.');
        }

        return $connection->getOpenSearchClient();
    }
}
