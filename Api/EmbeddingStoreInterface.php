<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Api;

interface EmbeddingStoreInterface
{
    public function save(
        int $storeId,
        int $productId,
        int $generationId,
        string $sourceHash,
        array $vector,
        int $revision
    ): bool;

    public function delete(int $storeId, int $productId, int $generationId, int $revision): bool;

    public function getVectorForSource(
        int $storeId,
        int $productId,
        int $generationId,
        string $sourceHash
    ): ?array;
}
