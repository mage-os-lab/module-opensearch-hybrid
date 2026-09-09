<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Encoder;

class HttpQueryEncoder implements \MageOS\OpenSearchHybrid\Api\QueryEncoderInterface
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Config $config,
        private readonly \MageOS\OpenSearchHybrid\Model\Http\CurlJsonClient $client,
        private readonly \MageOS\OpenSearchHybrid\Model\Vector\VectorValidator $vectorValidator
    ) {
    }

    public function encode(string $query, array $generation): array
    {
        $requestId = hash('sha256', (string)$generation['generation_id'] . "\0" . $query);
        $headers = [];
        $token = $this->config->encoderToken();
        if ($token !== '') {
            $headers['Authorization'] = 'Bearer ' . $token;
        }
        $response = $this->client->post(
            $this->config->validateEncoderEndpoint((string)$generation['encoder_endpoint']) . '/v1/embed/query',
            [
                'request_id' => $requestId,
                'model_id' => (string)$generation['model_id'],
                'model_revision' => (string)$generation['model_revision'],
                'dimension' => (int)$generation['dimension'],
                'recipe_version' => (string)$generation['recipe_version'],
                'encoder_identity_digest' => (string)$generation['encoder_identity_digest'],
                'query' => $query,
            ],
            $this->config->encoderConnectTimeoutMs(),
            $this->config->encoderQueryTimeoutMs(),
            $headers
        );
        if (($response['request_id'] ?? null) !== $requestId
            || ($response['model_id'] ?? null) !== (string)$generation['model_id']
            || ($response['model_revision'] ?? null) !== (string)$generation['model_revision']
            || (int)($response['dimension'] ?? 0) !== (int)$generation['dimension']
            || ($response['recipe_version'] ?? null) !== (string)$generation['recipe_version']
            || ($response['encoder_identity_digest'] ?? null) !== (string)$generation['encoder_identity_digest']
            || ($response['production_eligible'] ?? false) !== true
            || !isset($response['vectors'][0])
            || !is_array($response['vectors'][0])
        ) {
            throw new \RuntimeException('The encoder response identity does not match the active generation.');
        }
        $vector = $response['vectors'][0];
        $this->vectorValidator->validate($vector, (int)$generation['dimension']);

        return array_map('floatval', $vector);
    }
}
