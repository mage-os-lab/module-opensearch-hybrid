<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Text;

class DocumentRenderer
{
    public const RECIPE_VERSION = 'mageos-v1';

    public function render(string $title, string $description, array $attributes): string
    {
        $trimmedAttributes = array_map('trim', $attributes);
        $presentAttributes = array_filter(
            $trimmedAttributes,
            static fn (string $value): bool => $value !== ''
        );
        $attributeText = implode('. ', array_values($presentAttributes));

        return sprintf('%s. %s. %s', trim($title), trim($description), $attributeText);
    }

    public function sourceHash(string $title, string $description, array $attributes): string
    {
        return hash('sha256', $this->render($title, $description, $attributes));
    }
}
