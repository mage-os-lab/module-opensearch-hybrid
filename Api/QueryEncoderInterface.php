<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Api;

interface QueryEncoderInterface
{
    /**
     * Encode one storefront query for the exact active generation identity.
     *
     * @return float[]
     */
    public function encode(string $query, array $generation): array;
}
