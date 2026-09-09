<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Reliability;

class CircuitBreaker
{
    private const CACHE_PREFIX = 'mageos_opensearch_hybrid_circuit_';
    private const CIRCUITS = ['catalog', 'similarity'];

    public function __construct(
        private readonly \Magento\Framework\App\CacheInterface $cache,
        private readonly \Magento\Framework\Serialize\Serializer\Json $json,
        private readonly \MageOS\OpenSearchHybrid\Model\Config $config
    ) {
    }

    public function isOpen(int $storeId, string $circuit = 'catalog'): bool
    {
        $state = $this->read($storeId, $circuit);

        return (int)($state['open_until'] ?? 0) > time();
    }

    public function success(int $storeId, string $circuit = 'catalog'): void
    {
        $this->cache->remove($this->cacheKey($storeId, $circuit));
    }

    public function failure(int $storeId, string $circuit = 'catalog'): void
    {
        $state = $this->read($storeId, $circuit);
        $failures = (int)($state['failures'] ?? 0) + 1;
        $openUntil = $failures >= $this->config->failureThreshold()
            ? time() + $this->config->circuitOpenSeconds()
            : 0;
        $this->cache->save(
            $this->json->serialize(['failures' => $failures, 'open_until' => $openUntil]),
            $this->cacheKey($storeId, $circuit),
            [],
            max(60, $this->config->circuitOpenSeconds())
        );
    }

    private function read(int $storeId, string $circuit): array
    {
        $cached = $this->cache->load($this->cacheKey($storeId, $circuit));
        if (!is_string($cached) || $cached === '') {
            return [];
        }
        $state = $this->json->unserialize($cached);

        return is_array($state) ? $state : [];
    }

    private function cacheKey(int $storeId, string $circuit): string
    {
        if ($storeId <= 0 || !in_array($circuit, self::CIRCUITS, true)) {
            throw new \InvalidArgumentException('The circuit identity is invalid.');
        }

        return self::CACHE_PREFIX . ($circuit === 'catalog' ? '' : $circuit . '_') . $storeId;
    }
}
