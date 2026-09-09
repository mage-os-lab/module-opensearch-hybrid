<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Activation;

class AdminRequestValidatorTest extends \PHPUnit\Framework\TestCase
{
    private \MageOS\OpenSearchHybrid\Model\Activation\AdminRequestValidator $validator;

    protected function setUp(): void
    {
        $this->validator = new \MageOS\OpenSearchHybrid\Model\Activation\AdminRequestValidator();
    }

    public function testActivationApplyRequiresTheExactGenerationAndPreviewToken(): void
    {
        $token = 'activate-3-' . str_repeat('a', 64);

        self::assertSame(
            [
                'operation' => 'activate',
                'store_id' => 3,
                'generation_id' => 8,
                'confirmation_token' => $token,
            ],
            $this->validator->apply('activate', '3', '8', '8', $token)
        );
    }

    public function testNativeRollbackRequiresTheExactNativeConfirmation(): void
    {
        $token = 'rollback-native-3-' . str_repeat('b', 64);

        self::assertSame(
            [
                'operation' => 'rollback_native',
                'store_id' => 3,
                'generation_id' => null,
                'confirmation_token' => $token,
            ],
            $this->validator->apply('rollback_native', '3', null, 'native', $token)
        );
    }

    public function testRetainedRollbackRejectsAConfirmationForAnotherGeneration(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('typed confirmation');

        $this->validator->apply(
            'rollback_generation',
            '3',
            '8',
            '7',
            'rollback-generation-3-' . str_repeat('c', 64)
        );
    }

    public function testPreviewRejectsImplicitOrInvalidTargets(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('Generation ID');

        $this->validator->preview('activate', '3', null);
    }

    public function testApplyRejectsAnUnboundToken(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('preview confirmation token');

        $this->validator->apply('activate', '3', '8', '8', 'activate-3-unbound');
    }
}
