<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Vector;

class DeterministicVector
{
    public const DIMENSION = 256;

    public function encode(string $text): array
    {
        $vector = [];
        $sumSquares = 0.0;
        for ($index = 0; $index < self::DIMENSION; $index++) {
            $bytes = hash('sha256', $text . ':' . $index, true);
            $parts = unpack('nvalue', substr($bytes, 0, 2));
            $value = (((float)$parts['value'] / 65535.0) * 2.0) - 1.0;
            $vector[] = $value;
            $sumSquares += $value * $value;
        }
        if ($sumSquares === 0.0) {
            throw new \RuntimeException('The deterministic vector has zero magnitude.');
        }
        $magnitude = sqrt($sumSquares);

        return array_map(static fn (float $value): float => $value / $magnitude, $vector);
    }
}
