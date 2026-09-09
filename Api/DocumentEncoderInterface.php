<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Api;

interface DocumentEncoderInterface
{
    /**
     * Encode one bounded document batch for the exact target generation identity.
     *
     * @return array<int, array<int, float>>
     */
    public function encode(array $documents, array $generation): array;
}
