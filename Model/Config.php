<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model;

class Config
{
    public const XML_PATH_ENABLED = 'mageos_opensearch_hybrid/general/enabled';
    public const XML_PATH_INCLUDED_STORES = 'mageos_opensearch_hybrid/general/included_stores';
    public const XML_PATH_ENCODER_ENDPOINT = 'mageos_opensearch_hybrid/encoder/endpoint';
    public const XML_PATH_ENCODER_ALLOWED_HOSTS = 'mageos_opensearch_hybrid/encoder/allowed_hosts';
    public const XML_PATH_ENCODER_TOKEN = 'mageos_opensearch_hybrid/encoder/api_token';
    public const XML_PATH_ENCODER_CONNECT_TIMEOUT = 'mageos_opensearch_hybrid/encoder/connect_timeout_ms';
    public const XML_PATH_ENCODER_QUERY_TIMEOUT = 'mageos_opensearch_hybrid/encoder/query_timeout_ms';
    public const XML_PATH_DOCUMENT_TIMEOUT = 'mageos_opensearch_hybrid/encoder/document_timeout_seconds';
    public const XML_PATH_BATCH_SIZE = 'mageos_opensearch_hybrid/encoder/batch_size';
    public const XML_PATH_MAX_BATCHES = 'mageos_opensearch_hybrid/encoder/max_outstanding_batches';
    public const XML_PATH_OS_CONNECT_TIMEOUT = 'mageos_opensearch_hybrid/opensearch/connect_timeout_ms';
    public const XML_PATH_OS_READ_TIMEOUT = 'mageos_opensearch_hybrid/opensearch/read_timeout_ms';
    public const XML_PATH_SIMILARITY_ENABLED = 'mageos_opensearch_hybrid/similarity/enabled';
    public const XML_PATH_JOB_MAX_ATTEMPTS = 'mageos_opensearch_hybrid/reliability/job_max_attempts';
    public const XML_PATH_FAILURE_THRESHOLD = 'mageos_opensearch_hybrid/reliability/failure_threshold';
    public const XML_PATH_CIRCUIT_SECONDS = 'mageos_opensearch_hybrid/reliability/circuit_open_seconds';
    public const XML_PATH_RETENTION_DAYS = 'mageos_opensearch_hybrid/reliability/previous_generation_retention_days';
    public const XML_PATH_QUERY_LOGGING = 'mageos_opensearch_hybrid/privacy/query_logging';

    public function __construct(
        private readonly \Magento\Framework\App\Config\ScopeConfigInterface $scopeConfig,
        private readonly \Magento\Framework\Encryption\EncryptorInterface $encryptor,
        private readonly \MageOS\OpenSearchHybrid\Model\Security\EndpointValidator $endpointValidator
    ) {
    }

    public function isEnabled(?int $storeId = null): bool
    {
        return $this->scopeConfig->isSetFlag(
            self::XML_PATH_ENABLED,
            \Magento\Store\Model\ScopeInterface::SCOPE_STORE,
            $storeId
        );
    }

    public function isStoreIncluded(int $storeId): bool
    {
        $configured = (string)$this->scopeConfig->getValue(
            self::XML_PATH_INCLUDED_STORES,
            \Magento\Store\Model\ScopeInterface::SCOPE_STORE,
            $storeId
        );
        if (trim($configured) === '') {
            return true;
        }
        $stores = array_map('intval', array_filter(array_map('trim', explode(',', $configured))));

        return in_array($storeId, $stores, true);
    }

    public function encoderEndpoint(): string
    {
        $endpoint = (string)$this->scopeConfig->getValue(self::XML_PATH_ENCODER_ENDPOINT);

        return $this->validateEncoderEndpoint($endpoint);
    }

    public function validateEncoderEndpoint(string $endpoint): string
    {
        $allowedHosts = array_values(array_filter(array_map(
            'trim',
            explode(',', (string)$this->scopeConfig->getValue(self::XML_PATH_ENCODER_ALLOWED_HOSTS))
        )));

        return $this->endpointValidator->validate(trim($endpoint), $allowedHosts);
    }

    public function encoderToken(): string
    {
        $encrypted = (string)$this->scopeConfig->getValue(self::XML_PATH_ENCODER_TOKEN);
        if ($encrypted === '') {
            return '';
        }

        return $this->encryptor->decrypt($encrypted);
    }

    public function encoderConnectTimeoutMs(): int
    {
        return $this->boundedInt(self::XML_PATH_ENCODER_CONNECT_TIMEOUT, 10, 5000);
    }

    public function encoderQueryTimeoutMs(): int
    {
        return $this->boundedInt(self::XML_PATH_ENCODER_QUERY_TIMEOUT, 25, 10000);
    }

    public function documentTimeoutSeconds(): int
    {
        return $this->boundedInt(self::XML_PATH_DOCUMENT_TIMEOUT, 1, 300);
    }

    public function batchSize(): int
    {
        return $this->boundedInt(self::XML_PATH_BATCH_SIZE, 1, 64);
    }

    public function maxOutstandingBatches(): int
    {
        return $this->boundedInt(self::XML_PATH_MAX_BATCHES, 1, 128);
    }

    public function openSearchConnectTimeoutMs(): int
    {
        return $this->boundedInt(self::XML_PATH_OS_CONNECT_TIMEOUT, 10, 5000);
    }

    public function openSearchReadTimeoutMs(): int
    {
        return $this->boundedInt(self::XML_PATH_OS_READ_TIMEOUT, 25, 10000);
    }

    public function isSimilarityEnabled(int $storeId): bool
    {
        return $this->scopeConfig->isSetFlag(
            self::XML_PATH_SIMILARITY_ENABLED,
            \Magento\Store\Model\ScopeInterface::SCOPE_STORE,
            $storeId
        );
    }

    public function jobMaxAttempts(): int
    {
        return $this->boundedInt(self::XML_PATH_JOB_MAX_ATTEMPTS, 1, 100);
    }

    public function failureThreshold(): int
    {
        return $this->boundedInt(self::XML_PATH_FAILURE_THRESHOLD, 1, 100);
    }

    public function circuitOpenSeconds(): int
    {
        return $this->boundedInt(self::XML_PATH_CIRCUIT_SECONDS, 1, 3600);
    }

    public function retentionDays(): int
    {
        return $this->boundedInt(self::XML_PATH_RETENTION_DAYS, 1, 365);
    }

    public function isQueryLoggingEnabled(): bool
    {
        return $this->scopeConfig->isSetFlag(self::XML_PATH_QUERY_LOGGING);
    }

    private function boundedInt(string $path, int $minimum, int $maximum): int
    {
        $value = (int)$this->scopeConfig->getValue($path);
        if ($value < $minimum || $value > $maximum) {
            throw new \UnexpectedValueException(sprintf('Configuration value %s is outside its safe range.', $path));
        }

        return $value;
    }
}
