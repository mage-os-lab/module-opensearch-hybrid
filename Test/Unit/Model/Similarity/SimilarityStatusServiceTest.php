<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Similarity;

class SimilarityStatusServiceTest extends \PHPUnit\Framework\TestCase
{
    public function testDisabledFeatureDoesNotResolveAGenerationOrCircuit(): void
    {
        $config = $this->createStub(\MageOS\OpenSearchHybrid\Model\Config::class);
        $config->method('isSimilarityEnabled')->willReturn(false);
        $generations = $this->createMock(
            \MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository::class
        );
        $generations->expects($this->never())->method('activeForStore');
        $circuit = $this->createMock(\MageOS\OpenSearchHybrid\Model\Reliability\CircuitBreaker::class);
        $circuit->expects($this->never())->method('isOpen');

        $result = $this->service($config, $generations, null, $circuit)->get(2);

        self::assertSame('DISABLED', $result['state']);
        self::assertFalse($result['ready']);
        self::assertSame('not_checked', $result['circuit_state']);
    }

    public function testEnabledFeatureWithoutAProfileIsReportedAsUncalibrated(): void
    {
        $config = $this->createStub(\MageOS\OpenSearchHybrid\Model\Config::class);
        $config->method('isSimilarityEnabled')->willReturn(true);
        $generations = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository::class
        );
        $generations->method('activeForStore')->willReturn($this->generation());
        $profiles = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Similarity\ThresholdProfileRegistry::class
        );
        $profiles->method('inspect')->willReturn($this->profiles());

        $result = $this->service($config, $generations, $profiles)->get(2);

        self::assertSame('ENABLED_UNCALIBRATED', $result['state']);
        self::assertSame('threshold_profile_unavailable', $result['reason']);
        self::assertSame(7, $result['generation_id']);
    }

    public function testAcceptedProfileAndClosedCircuitAreReady(): void
    {
        $config = $this->createStub(\MageOS\OpenSearchHybrid\Model\Config::class);
        $config->method('isSimilarityEnabled')->willReturn(true);
        $generations = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository::class
        );
        $generations->method('activeForStore')->willReturn($this->generation());
        $profiles = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Similarity\ThresholdProfileRegistry::class
        );
        $profileState = $this->profiles();
        $profileState['applicable_profile_ids'] = ['substitute-v1'];
        $profileState['accepted_profile_ids'] = ['substitute-v1'];
        $profiles->method('inspect')->willReturn($profileState);
        $circuit = $this->createMock(\MageOS\OpenSearchHybrid\Model\Reliability\CircuitBreaker::class);
        $circuit->expects($this->once())->method('isOpen')->with(2, 'similarity')->willReturn(false);

        $result = $this->service($config, $generations, $profiles, $circuit)->get(2);

        self::assertSame('READY', $result['state']);
        self::assertTrue($result['ready']);
        self::assertSame(['substitute-v1'], $result['profiles']['accepted_profile_ids']);
        self::assertSame('closed', $result['circuit_state']);
    }

    public function testInvalidApplicableProfileFailsClosedWithoutCheckingCircuit(): void
    {
        $config = $this->createStub(\MageOS\OpenSearchHybrid\Model\Config::class);
        $config->method('isSimilarityEnabled')->willReturn(true);
        $generations = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository::class
        );
        $generations->method('activeForStore')->willReturn($this->generation());
        $profiles = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Similarity\ThresholdProfileRegistry::class
        );
        $profileState = $this->profiles();
        $profileState['applicable_profile_ids'] = ['substitute-v1'];
        $profileState['invalid_profile_ids'] = ['substitute-v1'];
        $profiles->method('inspect')->willReturn($profileState);
        $circuit = $this->createMock(\MageOS\OpenSearchHybrid\Model\Reliability\CircuitBreaker::class);
        $circuit->expects($this->never())->method('isOpen');

        $result = $this->service($config, $generations, $profiles, $circuit)->get(2);

        self::assertSame('FAILED_CLOSED', $result['state']);
        self::assertSame('threshold_profile_invalid', $result['reason']);
    }

    public function testUnavailableCircuitStateFailsClosed(): void
    {
        $config = $this->createStub(\MageOS\OpenSearchHybrid\Model\Config::class);
        $config->method('isSimilarityEnabled')->willReturn(true);
        $generations = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository::class
        );
        $generations->method('activeForStore')->willReturn($this->generation());
        $profiles = $this->createStub(
            \MageOS\OpenSearchHybrid\Model\Similarity\ThresholdProfileRegistry::class
        );
        $profileState = $this->profiles();
        $profileState['applicable_profile_ids'] = ['substitute-v1'];
        $profileState['accepted_profile_ids'] = ['substitute-v1'];
        $profiles->method('inspect')->willReturn($profileState);
        $circuit = $this->createStub(\MageOS\OpenSearchHybrid\Model\Reliability\CircuitBreaker::class);
        $circuit->method('isOpen')->willThrowException(new \RuntimeException('cache unavailable'));

        $result = $this->service($config, $generations, $profiles, $circuit)->get(2);

        self::assertSame('FAILED_CLOSED', $result['state']);
        self::assertSame('circuit_state_unavailable', $result['reason']);
        self::assertSame('not_checked', $result['circuit_state']);
    }

    private function service(
        \MageOS\OpenSearchHybrid\Model\Config $config,
        \MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository $generations,
        ?\MageOS\OpenSearchHybrid\Model\Similarity\ThresholdProfileRegistry $profiles = null,
        ?\MageOS\OpenSearchHybrid\Model\Reliability\CircuitBreaker $circuit = null
    ): \MageOS\OpenSearchHybrid\Model\Similarity\SimilarityStatusService {
        return new \MageOS\OpenSearchHybrid\Model\Similarity\SimilarityStatusService(
            $config,
            $generations,
            $profiles ?? $this->createStub(
                \MageOS\OpenSearchHybrid\Model\Similarity\ThresholdProfileRegistry::class
            ),
            $circuit ?? $this->createStub(\MageOS\OpenSearchHybrid\Model\Reliability\CircuitBreaker::class)
        );
    }

    private function generation(): array
    {
        return [
            'generation_id' => 7,
            'result_contract_accepted' => 1,
        ];
    }

    private function profiles(): array
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
