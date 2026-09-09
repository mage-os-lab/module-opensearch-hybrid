<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Document;

class ProductDocumentFactory
{
    public function __construct(
        private readonly \Magento\Catalog\Api\ProductRepositoryInterface $productRepository,
        private readonly \MageOS\OpenSearchHybrid\Model\Text\Canonicalizer $canonicalizer,
        private readonly \MageOS\OpenSearchHybrid\Model\Text\DocumentRenderer $documentRenderer,
        private readonly \MageOS\OpenSearchHybrid\Model\Document\CategoryTextResolver $categoryTextResolver,
        private readonly \MageOS\OpenSearchHybrid\Model\Document\AttributeTextResolver $attributeTextResolver,
        private readonly \MageOS\OpenSearchHybrid\Model\Document\PriceIndexResolver $priceIndexResolver,
        private readonly \Magento\InventoryCatalog\Model\GetStockIdForByStoreId $getStockIdForStore,
        private readonly \Magento\InventorySalesApi\Api\AreProductsSalableInterface $areProductsSalable
    ) {
    }

    public function create(int $productId, int $storeId): array
    {
        $product = $this->productRepository->getById($productId, false, $storeId, true);
        $title = (string)$product->getName();
        $description = (string)($product->getData('description') ?: $product->getData('short_description'));
        $categories = $this->categoryTextResolver->resolve($product->getCategoryIds(), $storeId);
        $attributeText = $this->attributeTextResolver->resolve($product, $storeId);
        $attributes = array_filter(
            array_merge(
                [(string)$product->getTypeId(), implode(' | ', $categories)],
                $attributeText['semantic']
            ),
            static fn (string $value): bool => trim($value) !== ''
        );
        $rendered = $this->documentRenderer->render($title, $description, $attributes);
        $stockId = $this->getStockIdForStore->execute($storeId);
        $salability = $this->areProductsSalable->execute([(string)$product->getSku()], $stockId);
        $isSalable = isset($salability[0]) && $salability[0]->isSalable();

        return [
            'entity_id' => $productId,
            'store_id' => $storeId,
            'sku' => (string)$product->getSku(),
            'sku_normalized' => $this->canonicalizer->normalize((string)$product->getSku()),
            'title' => $title,
            'title_normalized' => $this->canonicalizer->normalize($title),
            'description' => $description,
            'product_class' => (string)$product->getTypeId(),
            'category' => implode(' | ', $categories),
            'brand' => $attributeText['brand'],
            'features' => $attributeText['features'],
            'visibility' => (int)$product->getVisibility(),
            'status' => (int)$product->getStatus(),
            'price' => (float)$product->getPrice(),
            'customer_group_prices' => $this->priceIndexResolver->resolve($productId, $storeId),
            'stock_id' => $stockId,
            'is_salable' => $isSalable,
            'attributes' => $attributes,
            'rendered' => $rendered,
            'source_hash' => hash('sha256', $rendered),
        ];
    }
}
