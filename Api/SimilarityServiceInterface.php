<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Api;

interface SimilarityServiceInterface
{
    /**
     * Return bounded product similarity results or the caller's declared fallback.
     *
     * @param int $storeId Trusted Magento store scope.
     * @param int $productId Seed product ID.
     * @param int $limit Maximum result count.
     * @param string $thresholdProfileId Server-controlled threshold profile ID.
     * @param array $fallbackProductIds Caller-declared native fallback product IDs.
     * @return array Bounded result items and fallback metadata.
     */
    public function searchByProduct(
        int $storeId,
        int $productId,
        int $limit,
        string $thresholdProfileId,
        array $fallbackProductIds = []
    ): array;
}
