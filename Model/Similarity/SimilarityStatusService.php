<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Similarity;

class SimilarityStatusService
{
    public const STATE_DISABLED = 'DISABLED';
    public const STATE_ENABLED_UNCALIBRATED = 'ENABLED_UNCALIBRATED';
    public const STATE_READY = 'READY';
    public const STATE_FAILED_CLOSED = 'FAILED_CLOSED';

    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Config $config,
        private readonly \MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository $generationRepository,
        private readonly \MageOS\OpenSearchHybrid\Model\Similarity\ThresholdProfileRegistry $profileRegistry,
        private readonly \MageOS\OpenSearchHybrid\Model\Reliability\CircuitBreaker $circuitBreaker
    ) {
    }

    public function get(int $storeId): array
    {
        if ($storeId <= 0) {
            throw new \InvalidArgumentException('Similarity status requires a positive store ID.');
        }
        $enabled = $this->config->isSimilarityEnabled($storeId);
        if (!$enabled) {
            return $this->result(
                $storeId,
                self::STATE_DISABLED,
                false,
                null,
                null,
                $this->emptyProfiles(),
                'not_checked'
            );
        }
        $generation = $this->generationRepository->activeForStore($storeId);
        if ($generation === null || !(bool)($generation['result_contract_accepted'] ?? false)) {
            return $this->result(
                $storeId,
                self::STATE_FAILED_CLOSED,
                false,
                'active_generation_unavailable',
                null,
                $this->emptyProfiles(),
                'not_checked'
            );
        }
        $generationId = (int)$generation['generation_id'];
        $profiles = $this->profileRegistry->inspect($generation, $storeId);
        if ($profiles['invalid_configuration_count'] > 0 || $profiles['invalid_profile_ids'] !== []) {
            return $this->result(
                $storeId,
                self::STATE_FAILED_CLOSED,
                false,
                'threshold_profile_invalid',
                $generationId,
                $profiles,
                'not_checked'
            );
        }
        if ($profiles['accepted_profile_ids'] === []) {
            return $this->result(
                $storeId,
                self::STATE_ENABLED_UNCALIBRATED,
                false,
                'threshold_profile_unavailable',
                $generationId,
                $profiles,
                'not_checked'
            );
        }
        try {
            $circuitOpen = $this->circuitBreaker->isOpen($storeId, 'similarity');
        } catch (\Throwable) {
            return $this->result(
                $storeId,
                self::STATE_FAILED_CLOSED,
                false,
                'circuit_state_unavailable',
                $generationId,
                $profiles,
                'not_checked'
            );
        }
        if ($circuitOpen) {
            return $this->result(
                $storeId,
                self::STATE_FAILED_CLOSED,
                false,
                'circuit_open',
                $generationId,
                $profiles,
                'open'
            );
        }

        return $this->result(
            $storeId,
            self::STATE_READY,
            true,
            null,
            $generationId,
            $profiles,
            'closed'
        );
    }

    private function result(
        int $storeId,
        string $state,
        bool $ready,
        ?string $reason,
        ?int $generationId,
        array $profiles,
        string $circuitState
    ): array {
        return [
            'store_id' => $storeId,
            'state' => $state,
            'ready' => $ready,
            'reason' => $reason,
            'generation_id' => $generationId,
            'circuit_state' => $circuitState,
            'profiles' => $profiles,
        ];
    }

    private function emptyProfiles(): array
    {
        return [
            'configured_profile_count' => 0,
            'applicable_profile_ids' => [],
            'accepted_profile_ids' => [],
            'invalid_profile_ids' => [],
            'invalid_configuration_count' => 0,
        ];
    }
}
