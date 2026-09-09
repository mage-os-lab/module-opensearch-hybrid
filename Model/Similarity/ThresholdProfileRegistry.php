<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Similarity;

class ThresholdProfileRegistry
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract $retrievalContract,
        private readonly array $profiles = []
    ) {
    }

    public function resolve(string $profileId, array $generation, int $storeId): array
    {
        if (preg_match('/^[a-z0-9][a-z0-9._-]{0,63}$/', $profileId) !== 1
            || !isset($this->profiles[$profileId])
            || !is_array($this->profiles[$profileId])
        ) {
            throw new \RuntimeException('The similarity threshold profile is unavailable.');
        }
        $profile = $this->profiles[$profileId];
        $contract = $this->retrievalContract->get();
        $storeIds = $profile['store_ids'] ?? null;
        $minScore = $profile['min_score'] ?? null;
        if (($profile['id'] ?? null) !== $profileId
            || ($profile['calibration_status'] ?? null) !== 'ACCEPTED'
            || ($profile['model_id'] ?? null) !== (string)$generation['model_id']
            || ($profile['model_revision'] ?? null) !== (string)$generation['model_revision']
            || ($profile['dimension'] ?? null) !== (int)$generation['dimension']
            || ($profile['similarity'] ?? null) !== (string)$contract['model']['similarity']
            || ($profile['source_recipe_version'] ?? null) !== (string)$generation['recipe_version']
            || !is_array($storeIds)
            || !in_array($storeId, $storeIds, true)
            || (!is_int($minScore) && !is_float($minScore))
            || !is_finite((float)$minScore)
            || (float)$minScore <= 0.0
            || (float)$minScore > 1.0
            || !is_string($profile['use_case'] ?? null)
            || trim((string)$profile['use_case']) === ''
            || preg_match('/^[0-9a-f]{64}$/', (string)($profile['judgment_set_sha256'] ?? '')) !== 1
            || !$this->isCalibrationDate((string)($profile['calibration_date'] ?? ''))
            || !is_string($profile['primary_metric'] ?? null)
            || trim((string)$profile['primary_metric']) === ''
        ) {
            throw new \RuntimeException('The similarity threshold profile identity is incompatible.');
        }
        foreach ($storeIds as $profileStoreId) {
            if (!is_int($profileStoreId) || $profileStoreId <= 0) {
                throw new \RuntimeException('The similarity threshold profile store scope is invalid.');
            }
        }
        $profile['min_score'] = (float)$minScore;
        $profile['store_ids'] = array_values(array_unique($storeIds));

        return $profile;
    }

    public function inspect(array $generation, int $storeId): array
    {
        $applicableProfileIds = [];
        $acceptedProfileIds = [];
        $invalidProfileIds = [];
        $invalidConfigurationCount = 0;
        foreach ($this->profiles as $profileId => $profile) {
            if (!is_string($profileId)
                || preg_match('/^[a-z0-9][a-z0-9._-]{0,63}$/', $profileId) !== 1
                || !is_array($profile)
            ) {
                $invalidConfigurationCount++;
                continue;
            }
            $storeIds = $profile['store_ids'] ?? null;
            if (!is_array($storeIds)) {
                $invalidConfigurationCount++;
                continue;
            }
            foreach ($storeIds as $profileStoreId) {
                if (!is_int($profileStoreId) || $profileStoreId <= 0) {
                    $invalidConfigurationCount++;
                    continue 2;
                }
            }
            if (!in_array($storeId, $storeIds, true)) {
                continue;
            }
            $applicableProfileIds[] = $profileId;
            try {
                $this->resolve($profileId, $generation, $storeId);
                $acceptedProfileIds[] = $profileId;
            } catch (\Throwable) {
                $invalidProfileIds[] = $profileId;
            }
        }
        sort($applicableProfileIds, SORT_STRING);
        sort($acceptedProfileIds, SORT_STRING);
        sort($invalidProfileIds, SORT_STRING);

        return [
            'configured_profile_count' => count($this->profiles),
            'applicable_profile_ids' => $applicableProfileIds,
            'accepted_profile_ids' => $acceptedProfileIds,
            'invalid_profile_ids' => $invalidProfileIds,
            'invalid_configuration_count' => $invalidConfigurationCount,
        ];
    }

    private function isCalibrationDate(string $date): bool
    {
        if (preg_match('/^(\d{4})-(\d{2})-(\d{2})$/', $date, $parts) !== 1) {
            return false;
        }

        return checkdate((int)$parts[2], (int)$parts[3], (int)$parts[1]);
    }
}
