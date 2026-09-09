<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Document;

class AttributeTextResolver
{
    public function __construct(
        private readonly \Magento\Eav\Model\Config $eavConfig,
        private readonly \MageOS\OpenSearchHybrid\Model\Document\SemanticAttributeRegistry $registry
    ) {
    }

    public function resolve(\Magento\Catalog\Api\Data\ProductInterface $product, int $storeId): array
    {
        $brand = '';
        foreach ($this->registry->brandCodes() as $attributeCode) {
            $value = $this->value($product, $attributeCode, $storeId);
            if ($value !== '') {
                $brand = $value;
                break;
            }
        }
        $features = [];
        foreach ($this->registry->featureCodes() as $attributeCode) {
            $value = $this->value($product, $attributeCode, $storeId);
            if ($value !== '') {
                $features[] = $value;
            }
        }
        $features = array_values(array_unique($features));

        return [
            'brand' => $brand,
            'features' => implode(' | ', $features),
            'semantic' => $features,
        ];
    }

    private function value(
        \Magento\Catalog\Api\Data\ProductInterface $product,
        string $attributeCode,
        int $storeId
    ): string {
        $attribute = $this->eavConfig->getAttribute(\Magento\Catalog\Model\Product::ENTITY, $attributeCode);
        if (!$attribute->getId()) {
            return '';
        }
        $attribute->setStoreId($storeId);
        $value = $attribute->getFrontend()->getValue($product);
        if (is_array($value)) {
            $value = implode(', ', $value);
        }
        if (!is_scalar($value)) {
            return '';
        }
        $text = html_entity_decode((string)$value, ENT_QUOTES | ENT_HTML5, 'UTF-8');
        $text = strip_tags($text);
        $text = preg_replace('/\s+/u', ' ', $text);

        return trim(is_string($text) ? $text : '');
    }
}
