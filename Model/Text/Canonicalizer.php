<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Text;

class Canonicalizer
{
    public const VERSION = 'nfkc-ws-lower-v1';

    public function normalize(string $value): string
    {
        if (!class_exists(\Normalizer::class)) {
            throw new \RuntimeException('The intl extension is required for canonicalization.');
        }
        $normalized = \Normalizer::normalize($value, \Normalizer::FORM_KC);
        if ($normalized === false) {
            throw new \InvalidArgumentException('The value could not be normalized as UTF-8.');
        }
        $collapsed = preg_replace('/\s+/u', ' ', trim($normalized));
        if ($collapsed === null) {
            throw new \InvalidArgumentException('The value contains invalid UTF-8.');
        }

        return mb_strtolower($collapsed, 'UTF-8');
    }
}
