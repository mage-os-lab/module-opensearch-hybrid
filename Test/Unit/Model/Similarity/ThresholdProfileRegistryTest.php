<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Similarity;

class ThresholdProfileRegistryTest extends \PHPUnit\Framework\TestCase
{
    public function testResolvesOnlyAnAcceptedExactGenerationProfile(): void
    {
        $contract = $this->createStub(\MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract::class);
        $contract->method('get')->willReturn(['model' => ['similarity' => 'cosine']]);
        $profile = $this->profile();
        $registry = new \MageOS\OpenSearchHybrid\Model\Similarity\ThresholdProfileRegistry(
            $contract,
            ['substitute-v1' => $profile]
        );

        self::assertSame($profile, $registry->resolve('substitute-v1', $this->generation(), 2));
    }

    public function testRejectsAProfileBoundToAnotherModelRevision(): void
    {
        $contract = $this->createStub(\MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract::class);
        $contract->method('get')->willReturn(['model' => ['similarity' => 'cosine']]);
        $profile = $this->profile();
        $profile['model_revision'] = 'other-revision';
        $registry = new \MageOS\OpenSearchHybrid\Model\Similarity\ThresholdProfileRegistry(
            $contract,
            ['substitute-v1' => $profile]
        );
        $this->expectException(\RuntimeException::class);
        $this->expectExceptionMessage('identity is incompatible');

        $registry->resolve('substitute-v1', $this->generation(), 2);
    }

    public function testRejectsAnUnregisteredCallerSelectedProfile(): void
    {
        $contract = $this->createStub(\MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract::class);
        $registry = new \MageOS\OpenSearchHybrid\Model\Similarity\ThresholdProfileRegistry($contract);
        $this->expectException(\RuntimeException::class);
        $this->expectExceptionMessage('unavailable');

        $registry->resolve('caller-score', $this->generation(), 2);
    }

    public function testInspectionSeparatesAcceptedApplicableAndInvalidProfiles(): void
    {
        $contract = $this->createStub(\MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract::class);
        $contract->method('get')->willReturn(['model' => ['similarity' => 'cosine']]);
        $accepted = $this->profile();
        $otherStore = $this->profile();
        $otherStore['id'] = 'other-store';
        $otherStore['store_ids'] = [3];
        $invalid = $this->profile();
        $invalid['id'] = 'invalid-revision';
        $invalid['model_revision'] = 'wrong';
        $registry = new \MageOS\OpenSearchHybrid\Model\Similarity\ThresholdProfileRegistry(
            $contract,
            [
                'substitute-v1' => $accepted,
                'other-store' => $otherStore,
                'invalid-revision' => $invalid,
                'broken-config' => 'not-an-array',
            ]
        );

        self::assertSame([
            'configured_profile_count' => 4,
            'applicable_profile_ids' => ['invalid-revision', 'substitute-v1'],
            'accepted_profile_ids' => ['substitute-v1'],
            'invalid_profile_ids' => ['invalid-revision'],
            'invalid_configuration_count' => 1,
        ], $registry->inspect($this->generation(), 2));
    }

    private function profile(): array
    {
        return [
            'id' => 'substitute-v1',
            'calibration_status' => 'ACCEPTED',
            'model_id' => 'fixture/model',
            'model_revision' => 'revision-1',
            'dimension' => 2,
            'similarity' => 'cosine',
            'source_recipe_version' => 'recipe-1',
            'store_ids' => [2],
            'use_case' => 'substitute',
            'judgment_set_sha256' => str_repeat('a', 64),
            'calibration_date' => '2026-08-26',
            'primary_metric' => 'radial_recall',
            'min_score' => 0.91,
        ];
    }

    private function generation(): array
    {
        return [
            'model_id' => 'fixture/model',
            'model_revision' => 'revision-1',
            'dimension' => 2,
            'recipe_version' => 'recipe-1',
        ];
    }
}
