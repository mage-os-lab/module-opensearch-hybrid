<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\OpenSearch;

class HybridTransport
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Config $config,
        private readonly \Magento\Elasticsearch\Model\Config $nativeOpenSearchConfig,
        private readonly \MageOS\OpenSearchHybrid\Model\Http\CurlJsonClient $client
    ) {
    }

    public function search(string $index, ?string $pipelineId, array $request): array
    {
        $options = $this->nativeOpenSearchConfig->prepareClientOptions();
        $hostname = trim((string)$options['hostname']);
        if (!str_contains($hostname, '://')) {
            $hostname = 'http://' . $hostname;
        }
        $port = (int)($options['port'] ?? 0);
        $parts = parse_url($hostname);
        if (!is_array($parts) || !isset($parts['scheme'], $parts['host'])) {
            throw new \RuntimeException('The native OpenSearch endpoint is invalid.');
        }
        $baseUrl = sprintf('%s://%s', $parts['scheme'], $parts['host']);
        if ($port > 0) {
            $baseUrl .= ':' . $port;
        }
        $headers = [];
        if ((int)($options['enableAuth'] ?? 0) === 1) {
            $headers['Authorization'] = 'Basic ' . base64_encode(
                (string)$options['username'] . ':' . (string)$options['password']
            );
        }

        $url = $baseUrl . '/' . rawurlencode($index) . '/_search';
        if ($pipelineId !== null && $pipelineId !== '') {
            $url .= '?search_pipeline=' . rawurlencode($pipelineId);
        }

        return $this->client->post(
            $url,
            $request,
            $this->config->openSearchConnectTimeoutMs(),
            $this->config->openSearchReadTimeoutMs(),
            $headers
        );
    }
}
