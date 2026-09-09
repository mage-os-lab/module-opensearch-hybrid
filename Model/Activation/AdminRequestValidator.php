<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Activation;

class AdminRequestValidator
{
    private const OPERATIONS = ['activate', 'rollback_native', 'rollback_generation'];

    public function preview(mixed $operation, mixed $store, mixed $generation): array
    {
        $operation = $this->operation($operation);
        $storeId = $this->positiveInt($store, 'Store ID');
        $generationId = $operation === 'rollback_native'
            ? null
            : $this->positiveInt($generation, 'Generation ID');

        return [
            'operation' => $operation,
            'store_id' => $storeId,
            'generation_id' => $generationId,
        ];
    }

    public function apply(
        mixed $operation,
        mixed $store,
        mixed $generation,
        mixed $humanConfirmation,
        mixed $confirmationToken
    ): array {
        $input = $this->preview($operation, $store, $generation);
        $expectedHumanConfirmation = $input['operation'] === 'rollback_native'
            ? 'native'
            : (string)$input['generation_id'];
        if (!is_string($humanConfirmation)
            || !hash_equals($expectedHumanConfirmation, trim($humanConfirmation))
        ) {
            throw new \InvalidArgumentException('The typed confirmation does not match the exact target.');
        }
        if (!is_string($confirmationToken)
            || preg_match('/^[a-z-]+-[1-9][0-9]*-[0-9a-f]{64}$/D', $confirmationToken) !== 1
        ) {
            throw new \InvalidArgumentException('The exact preview confirmation token is missing or invalid.');
        }
        $input['confirmation_token'] = $confirmationToken;

        return $input;
    }

    private function operation(mixed $operation): string
    {
        if (!is_string($operation) || !in_array($operation, self::OPERATIONS, true)) {
            throw new \InvalidArgumentException('Select one supported activation or rollback operation.');
        }

        return $operation;
    }

    private function positiveInt(mixed $value, string $label): int
    {
        if (is_bool($value) || is_array($value) || is_object($value)) {
            throw new \InvalidArgumentException($label . ' must be a positive integer.');
        }
        $validated = filter_var($value, FILTER_VALIDATE_INT, ['options' => ['min_range' => 1]]);
        if ($validated === false) {
            throw new \InvalidArgumentException($label . ' must be a positive integer.');
        }

        return (int)$validated;
    }
}
