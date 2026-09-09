<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Encoder;

class HttpDocumentEncoder implements \MageOS\OpenSearchHybrid\Api\DocumentEncoderInterface
{
    private const MAX_BATCH_SIZE = 64;

    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Config $config,
        private readonly \MageOS\OpenSearchHybrid\Model\Http\CurlJsonClient $client,
        private readonly \MageOS\OpenSearchHybrid\Model\Vector\VectorValidator $vectorValidator
    ) {
    }

    public function encode(array $documents, array $generation): array
    {
        $documents = array_values($documents);
        if ($documents === []
            || count(array_filter($documents, 'is_string')) !== count($documents)
        ) {
            throw new \InvalidArgumentException('The document encoder requires a non-empty string batch.');
        }
        if (count($documents) > self::MAX_BATCH_SIZE) {
            throw new \InvalidArgumentException(sprintf(
                'The document encoder accepts at most %d documents per batch.',
                self::MAX_BATCH_SIZE
            ));
        }
        $vectors = [];
        foreach (array_chunk($documents, $this->config->batchSize()) as $batch) {
            array_push($vectors, ...$this->encodeBatch($batch, $generation));
        }

        return $vectors;
    }

    private function encodeBatch(array $documents, array $generation): array
    {
        $documentDigests = array_map(
            static fn (string $document): string => hash('sha256', $document),
            $documents
        );
        $requestId = hash(
            'sha256',
            (string)$generation['generation_id'] . "\0" . implode("\0", $documentDigests)
        );
        $headers = [];
        $token = $this->config->encoderToken();
        if ($token !== '') {
            $headers['Authorization'] = 'Bearer ' . $token;
        }
        $response = $this->client->post(
            $this->config->validateEncoderEndpoint((string)$generation['encoder_endpoint'])
                . '/v1/embed/documents',
            [
                'request_id' => $requestId,
                'model_id' => (string)$generation['model_id'],
                'model_revision' => (string)$generation['model_revision'],
                'dimension' => (int)$generation['dimension'],
                'recipe_version' => (string)$generation['recipe_version'],
                'encoder_identity_digest' => (string)$generation['encoder_identity_digest'],
                'documents' => $documents,
            ],
            $this->config->encoderConnectTimeoutMs(),
            $this->config->documentTimeoutSeconds() * 1000,
            $headers
        );
        $vectors = $response['vectors'] ?? null;
        if (($response['request_id'] ?? null) !== $requestId
            || ($response['model_id'] ?? null) !== (string)$generation['model_id']
            || ($response['model_revision'] ?? null) !== (string)$generation['model_revision']
            || (int)($response['dimension'] ?? 0) !== (int)$generation['dimension']
            || ($response['recipe_version'] ?? null) !== (string)$generation['recipe_version']
            || ($response['encoder_identity_digest'] ?? null)
                !== (string)$generation['encoder_identity_digest']
            || ($response['production_eligible'] ?? false) !== true
            || !is_array($vectors)
            || count($vectors) !== count($documents)
        ) {
            throw new \RuntimeException('The document encoder response identity or batch shape is invalid.');
        }
        foreach ($vectors as $vector) {
            if (!is_array($vector)) {
                throw new \RuntimeException('The document encoder returned a non-vector batch item.');
            }
            $this->vectorValidator->validate($vector, (int)$generation['dimension']);
        }

        return array_map(
            static fn (array $vector): array => array_map('floatval', $vector),
            $vectors
        );
    }
}
