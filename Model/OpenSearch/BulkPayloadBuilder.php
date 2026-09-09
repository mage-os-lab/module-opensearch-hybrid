<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\OpenSearch;

class BulkPayloadBuilder
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Vector\VectorValidator $vectorValidator
    ) {
    }

    public function build(array $generation, array $records): array
    {
        $body = [];
        foreach ($records as $record) {
            if (!is_array($record)
                || !isset($record['document'], $record['revision'])
                || !is_array($record['document'])
            ) {
                throw new \InvalidArgumentException('A bulk index record is invalid.');
            }
            $vector = $record['vector'] ?? null;
            if ($vector !== null && !is_array($vector)) {
                throw new \InvalidArgumentException('A bulk index record vector is invalid.');
            }
            $document = $record['document'];
            $revision = (int)$record['revision'];
            $body[] = [
                'index' => [
                    '_index' => (string)$generation['physical_index'],
                    '_id' => (string)$document['entity_id'],
                    'version' => max(1, $revision),
                    'version_type' => 'external_gte',
                ],
            ];
            $body[] = $this->documentSource($generation, $document, $vector, $revision);
        }

        return $body;
    }

    private function documentSource(array $generation, array $document, ?array $vector, int $revision): array
    {
        $source = [
            'entity_id' => (int)$document['entity_id'],
            'store_id' => (int)$document['store_id'],
            'generation_id' => (int)$generation['generation_id'],
            'sku' => (string)$document['sku'],
            'sku_normalized' => (string)$document['sku_normalized'],
            'title' => (string)$document['title'],
            'title_normalized' => (string)$document['title_normalized'],
            'description' => (string)$document['description'],
            'product_class' => (string)$document['product_class'],
            'category' => (string)$document['category'],
            'brand' => (string)$document['brand'],
            'features' => (string)$document['features'],
            'visibility' => (int)$document['visibility'],
            'status' => (int)$document['status'],
            'price' => (float)$document['price'],
            'customer_group_prices' => $document['customer_group_prices'],
            'stock_id' => (int)$document['stock_id'],
            'is_salable' => (bool)$document['is_salable'],
            'embedding_eligible' => $vector !== null,
            'source_hash' => (string)$document['source_hash'],
            'model_revision' => (string)$generation['model_revision'],
            'revision' => $revision,
        ];
        if ($vector !== null) {
            $source['embedding'] = $this->vectorValidator->toBase64Float32(
                $vector,
                (int)$generation['dimension']
            );
        }

        return $source;
    }
}
