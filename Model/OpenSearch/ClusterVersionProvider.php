<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\OpenSearch;

class ClusterVersionProvider
{
    private ?string $version = null;

    public function __construct(
        private readonly \Magento\Elasticsearch\SearchAdapter\ConnectionManager $connectionManager
    ) {
    }

    public function current(): string
    {
        if ($this->version !== null) {
            return $this->version;
        }
        $connection = $this->connectionManager->getConnection();
        if (!$connection instanceof \Magento\OpenSearch\Model\SearchClient) {
            throw new \RuntimeException('The active search connection is not OpenSearch.');
        }
        $response = $connection->getOpenSearchClient()->info();
        $version = $response['version']['number'] ?? null;
        if (!is_string($version) || $version === '') {
            throw new \RuntimeException('OpenSearch did not return a version number.');
        }

        $this->version = $version;

        return $this->version;
    }
}
