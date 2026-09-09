<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\OpenSearch;

class VersionPolicy
{
    public const MINIMUM = '3.8.0';
    public const NEXT_MINOR = '3.9.0';
    public const SUPPORTED_RANGE = '>=3.8.0 <3.9.0';

    public function isSupported(string $version): bool
    {
        return version_compare($version, self::MINIMUM, '>=')
            && version_compare($version, self::NEXT_MINOR, '<');
    }

    public function assertSupported(string $version): void
    {
        if (!$this->isSupported($version)) {
            throw new \RuntimeException(sprintf(
                'OpenSearch %s is unsupported. OpenSearch Hybrid requires %s.',
                $version,
                self::SUPPORTED_RANGE
            ));
        }
    }
}
