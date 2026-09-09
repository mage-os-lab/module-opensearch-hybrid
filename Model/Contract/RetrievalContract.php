<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Contract;

class RetrievalContract
{
    private ?array $contract = null;

    public function __construct(
        private readonly \Magento\Framework\Module\Dir\Reader $moduleDirReader,
        private readonly \Magento\Framework\Serialize\Serializer\Json $json
    ) {
    }

    public function get(): array
    {
        if ($this->contract === null) {
            $path = $this->moduleDirReader->getModuleDir('etc', 'MageOS_OpenSearchHybrid')
                . '/opensearch_hybrid_contract.json';
            $contents = file_get_contents($path);
            if ($contents === false) {
                throw new \RuntimeException('Unable to read the OpenSearch Hybrid retrieval contract.');
            }
            $contract = $this->json->unserialize($contents);
            if (!is_array($contract)) {
                throw new \RuntimeException('The OpenSearch Hybrid retrieval contract is invalid.');
            }
            $this->contract = $contract;
        }

        return $this->contract;
    }

    public function digest(): string
    {
        return hash('sha256', $this->canonicalJson($this->get()));
    }

    public function resultContractDigest(): string
    {
        $resultContract = $this->get()['result_contract'] ?? null;
        if (!is_array($resultContract)) {
            throw new \RuntimeException('The result contract is missing.');
        }

        return hash('sha256', $this->canonicalJson($resultContract));
    }

    public function pipelineDigest(): string
    {
        $pipeline = $this->get()['pipeline'] ?? null;
        if (!is_array($pipeline)) {
            throw new \RuntimeException('The pipeline contract is missing.');
        }

        return hash('sha256', $this->canonicalJson($pipeline));
    }

    public function mappingDigest(): string
    {
        $mapping = $this->get()['mapping'] ?? null;
        if (!is_array($mapping)) {
            throw new \RuntimeException('The mapping contract is missing.');
        }

        return hash('sha256', $this->canonicalJson($mapping));
    }

    public function digestValue(array $value): string
    {
        return hash('sha256', $this->canonicalJson($value));
    }

    public function pipelineId(): string
    {
        return 'mageos-opensearch-hybrid-' . substr($this->pipelineDigest(), 0, 24);
    }

    public function physicalIndexName(int $storeId, int $generationId): string
    {
        return sprintf(
            'mageos-opensearch-hybrid-s%d-g%d-%s',
            $storeId,
            $generationId,
            substr($this->digest(), 0, 16)
        );
    }

    private function canonicalJson(array $value): string
    {
        $canonical = $this->sortRecursively($value);
        $json = json_encode(
            $canonical,
            JSON_THROW_ON_ERROR | JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE
        );

        return $json;
    }

    private function sortRecursively(array $value): array
    {
        if (!array_is_list($value)) {
            ksort($value, SORT_STRING);
        }
        foreach ($value as $key => $item) {
            if (is_array($item)) {
                $value[$key] = $this->sortRecursively($item);
            }
        }

        return $value;
    }
}
