<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Http;

class CurlJsonClient
{
    private const MAX_RESPONSE_BYTES = 2_000_000;

    public function __construct(
        private readonly \Magento\Framework\HTTP\Client\CurlFactory $curlFactory,
        private readonly \Magento\Framework\Serialize\Serializer\Json $json
    ) {
    }

    public function post(
        string $url,
        array $payload,
        int $connectTimeoutMs,
        int $readTimeoutMs,
        array $headers = []
    ): array {
        $client = $this->curlFactory->create();
        $client->setHeaders(array_merge(['Content-Type' => 'application/json'], $headers));
        $client->setOption(CURLOPT_CONNECTTIMEOUT_MS, $connectTimeoutMs);
        $client->setOption(CURLOPT_TIMEOUT_MS, $readTimeoutMs);
        $client->setOption(CURLOPT_FOLLOWLOCATION, false);
        $client->setOption(CURLOPT_MAXREDIRS, 0);
        $client->post($url, $this->json->serialize($payload));
        $status = (int)$client->getStatus();
        $body = (string)$client->getBody();
        if ($status < 200 || $status >= 300) {
            throw new \RuntimeException(sprintf('HTTP service returned status %d.', $status));
        }
        if (strlen($body) > self::MAX_RESPONSE_BYTES) {
            throw new \RuntimeException('HTTP service response exceeded the configured limit.');
        }
        $decoded = $this->json->unserialize($body);
        if (!is_array($decoded)) {
            throw new \RuntimeException('HTTP service returned an invalid JSON object.');
        }

        return $decoded;
    }

    public function get(
        string $url,
        int $connectTimeoutMs,
        int $readTimeoutMs,
        array $headers = []
    ): array {
        $client = $this->curlFactory->create();
        $client->setHeaders($headers);
        $client->setOption(CURLOPT_CONNECTTIMEOUT_MS, $connectTimeoutMs);
        $client->setOption(CURLOPT_TIMEOUT_MS, $readTimeoutMs);
        $client->setOption(CURLOPT_FOLLOWLOCATION, false);
        $client->setOption(CURLOPT_MAXREDIRS, 0);
        $client->get($url);
        $status = (int)$client->getStatus();
        $body = (string)$client->getBody();
        if ($status < 200 || $status >= 300) {
            throw new \RuntimeException(sprintf('HTTP service returned status %d.', $status));
        }
        if (strlen($body) > self::MAX_RESPONSE_BYTES) {
            throw new \RuntimeException('HTTP service response exceeded the configured limit.');
        }
        $decoded = $this->json->unserialize($body);
        if (!is_array($decoded)) {
            throw new \RuntimeException('HTTP service returned an invalid JSON object.');
        }

        return $decoded;
    }
}
