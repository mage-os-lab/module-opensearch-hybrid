<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Vector;

class VectorValidator
{
    private const NORMALIZATION_TOLERANCE = 0.001;

    public function validate(array $vector, int $dimension): void
    {
        if (count($vector) !== $dimension) {
            throw new \InvalidArgumentException('The encoder returned the wrong vector dimension.');
        }
        $sumSquares = 0.0;
        foreach ($vector as $value) {
            if (!is_int($value) && !is_float($value)) {
                throw new \InvalidArgumentException('The encoder returned a non-numeric vector value.');
            }
            $floatValue = (float)$value;
            if (!is_finite($floatValue)) {
                throw new \InvalidArgumentException('The encoder returned a non-finite vector value.');
            }
            $sumSquares += $floatValue * $floatValue;
        }
        if (abs(sqrt($sumSquares) - 1.0) > self::NORMALIZATION_TOLERANCE) {
            throw new \InvalidArgumentException('The encoder returned a non-normalized vector.');
        }
    }

    public function toFloat32Binary(array $vector, int $dimension): string
    {
        $this->validate($vector, $dimension);

        return pack('g*', ...array_map(static fn (int|float $value): float => (float)$value, $vector));
    }

    public function toBase64Float32(array $vector, int $dimension): string
    {
        return base64_encode($this->toFloat32Binary($vector, $dimension));
    }

    public function fromBase64Float32(string $encoded, int $dimension): array
    {
        $binary = base64_decode($encoded, true);
        if ($dimension <= 0 || $binary === false || strlen($binary) !== $dimension * 4) {
            throw new \InvalidArgumentException('The stored vector bytes are invalid.');
        }
        $vector = unpack('g*', $binary);
        if (!is_array($vector)) {
            throw new \InvalidArgumentException('The stored vector bytes could not be decoded.');
        }
        $vector = array_values($vector);
        $this->validate($vector, $dimension);

        return $vector;
    }
}
