<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\OpenSearch;

class VectorWireQualificationTest extends \PHPUnit\Framework\TestCase
{
    public function testBindsTheAcceptedEvidenceToTheExactRuntime(): void
    {
        $qualification = new \MageOS\OpenSearchHybrid\Model\OpenSearch\VectorWireQualification();

        $exact = $qualification->get('3.8.0');
        $otherPatch = $qualification->get('3.8.1');

        self::assertTrue($exact['qualified']);
        self::assertSame('float32_le_base64', $exact['wire_format']);
        self::assertTrue($exact['current_runtime_exactly_qualified']);
        self::assertFalse($otherPatch['current_runtime_exactly_qualified']);
        self::assertMatchesRegularExpression('/^[0-9a-f]{64}$/', $exact['artifact_sha256']);
        self::assertMatchesRegularExpression('/^[0-9a-f]{40}$/', $exact['source_commit']);
    }
}
