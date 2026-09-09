<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Document;

class SemanticAttributeRegistry
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract $retrievalContract
    ) {
    }

    public function brandCodes(): array
    {
        return $this->codes('lexical_brand_attributes');
    }

    public function featureCodes(): array
    {
        return $this->codes('semantic_feature_attributes');
    }

    public function isMapped(string $attributeCode): bool
    {
        return in_array($attributeCode, array_merge($this->brandCodes(), $this->featureCodes()), true);
    }

    private function codes(string $key): array
    {
        $codes = $this->retrievalContract->get()['model'][$key] ?? null;
        if (!is_array($codes) || $codes === []) {
            throw new \RuntimeException(sprintf('The retrieval contract model.%s registry is invalid.', $key));
        }
        foreach ($codes as $code) {
            if (!is_string($code) || !preg_match('/^[a-z][a-z0-9_]*$/', $code)) {
                throw new \RuntimeException(sprintf('The retrieval contract model.%s registry is invalid.', $key));
            }
        }

        return array_values(array_unique($codes));
    }
}
