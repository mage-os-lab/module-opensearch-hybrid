<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\OpenSearch;

class VectorWireQualification
{
    private const QUALIFIED_VERSION = '3.8.0';
    private const ARTIFACT_SHA256 = '4bc9ac88aa09ed0a2979b01f87626a176911ae66d31245bd33193c65860e00d4';
    private const SOURCE_COMMIT = 'e08d05dae2f47099d81a0346fb6951759c6c4e34';

    public function get(?string $currentVersion = null): array
    {
        return [
            'qualified' => true,
            'wire_format' => 'float32_le_base64',
            'qualified_opensearch_version' => self::QUALIFIED_VERSION,
            'current_runtime_exactly_qualified' => $currentVersion === self::QUALIFIED_VERSION,
            'artifact_sha256' => self::ARTIFACT_SHA256,
            'source_commit' => self::SOURCE_COMMIT,
        ];
    }
}
