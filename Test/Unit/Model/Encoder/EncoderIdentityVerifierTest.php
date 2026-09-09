<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Encoder;

class EncoderIdentityVerifierTest extends \PHPUnit\Framework\TestCase
{
    public function testAcceptsOnlyThePinnedQualifiedAmd64Identity(): void
    {
        $verifier = new \MageOS\OpenSearchHybrid\Model\Encoder\EncoderIdentityVerifier(
            $this->retrievalContract()
        );

        $verified = $verifier->verify($this->identity(), str_repeat('a', 64));

        self::assertSame(str_repeat('a', 64), $verified['encoder_identity_digest']);
        self::assertSame('linux/amd64', $verified['architecture']);
    }

    #[\PHPUnit\Framework\Attributes\DataProvider('invalidIdentityProvider')]
    public function testRejectsUnpinnedIneligibleOrIncompleteIdentity(
        array $overrides,
        string $expectedDigest,
        string $message
    ): void {
        $identity = array_replace_recursive($this->identity(), $overrides);
        $verifier = new \MageOS\OpenSearchHybrid\Model\Encoder\EncoderIdentityVerifier(
            $this->retrievalContract()
        );

        $this->expectException(\RuntimeException::class);
        $this->expectExceptionMessage($message);

        $verifier->verify($identity, $expectedDigest);
    }

    public static function invalidIdentityProvider(): array
    {
        return [
            'operator digest mismatch' => [[], str_repeat('b', 64), 'pinned digest'],
            'development runtime' => [['production_eligible' => false], str_repeat('a', 64), 'not qualified'],
            'wrong architecture' => [['architecture' => 'linux/arm64'], str_repeat('a', 64), 'amd64'],
            'contract mismatch' => [
                ['model_contract_digest' => str_repeat('c', 64)],
                str_repeat('a', 64),
                'model contract',
            ],
            'registry name in canonical deployment' => [
                [
                    'deployment' => ['image' => 'registry.example/namespace/encoder'],
                    'identity_manifest' => [
                        'deployment' => ['image' => 'registry.example/namespace/encoder'],
                    ],
                ],
                str_repeat('a', 64),
                'deployment identity',
            ],
            'split query and document models' => [
                [
                    'artifacts' => [
                        'query_model' => [
                            'path' => 'query.onnx',
                            'sha256' => 'f' . str_repeat('e', 63),
                            'size' => 10,
                        ],
                    ],
                    'identity_manifest' => [
                        'artifacts' => [
                            'query_model' => [
                                'path' => 'query.onnx',
                                'sha256' => 'f' . str_repeat('e', 63),
                                'size' => 10,
                            ],
                        ],
                    ],
                ],
                str_repeat('a', 64),
                'shared fp32 model',
            ],
        ];
    }

    public function testRejectsIncompleteQualificationEvidence(): void
    {
        $identity = $this->identity();
        $identity['qualification']['evidence_artifact_roles'] = ['amd64_parity'];
        $identity['identity_manifest']['qualification'] = $identity['qualification'];
        $verifier = new \MageOS\OpenSearchHybrid\Model\Encoder\EncoderIdentityVerifier(
            $this->retrievalContract()
        );

        $this->expectException(\RuntimeException::class);
        $this->expectExceptionMessage('qualification evidence');

        $verifier->verify($identity, str_repeat('a', 64));
    }

    private function retrievalContract(): \MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract
    {
        $contract = $this->createStub(\MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract::class);
        $contract->method('get')->willReturn([
            'model' => [
                'id' => 'Snowflake/snowflake-arctic-embed-m-v2.0',
                'revision' => '95c2741480856aa9666782eb4afe11959938017f',
                'dimension' => 256,
                'similarity' => 'cosine',
                'query_prefix' => 'query: ',
                'query_template' => '{query}',
                'document_template' => '{title}. {description}. {attributes}',
                'source_recipe_version' => 'mageos-v1',
                'lexical_brand_attributes' => ['manufacturer'],
                'semantic_feature_attributes' => ['color', 'material'],
            ],
        ]);
        $contract->method('digestValue')->willReturnCallback(
            static fn (array $value): string => isset($value['schema_version'])
                ? str_repeat('a', 64)
                : str_repeat('d', 64)
        );

        return $contract;
    }

    private function identity(): array
    {
        $artifact = ['path' => 'artifact', 'sha256' => str_repeat('e', 64), 'size' => 10];

        $manifest = [
            'schema_version' => 2,
            'api_version' => 'v1',
            'model' => [
                'id' => 'Snowflake/snowflake-arctic-embed-m-v2.0',
                'revision' => '95c2741480856aa9666782eb4afe11959938017f',
                'dimension' => 256,
                'similarity' => 'cosine',
                'normalization' => 'l2_after_truncation',
                'truncate_to_dimension' => 256,
                'max_sequence_length' => 512,
            ],
            'recipe' => [
                'version' => 'mageos-v1',
                'query_prefix' => 'query: ',
                'query_template' => '{query}',
                'document_prefix' => '',
                'document_template' => '{title}. {description}. {attributes}',
                'lexical_brand_attributes' => ['manufacturer'],
                'semantic_feature_attributes' => ['color', 'material'],
            ],
            'runtime' => [
                'architecture' => 'linux/amd64',
                'implementation' => 'mageos_onnxruntime_tokenizers_v1',
                'execution_provider' => 'CPUExecutionProvider',
                'query_precision' => 'fp32',
                'document_precision' => 'fp32',
                'query_batch_size' => 1,
                'max_batch_size' => 64,
                'versions' => [
                    'python' => '3.13.7',
                    'numpy' => '2.5.2',
                    'onnxruntime' => '1.22.1',
                    'tokenizers' => '0.22.2',
                ],
            ],
            'deployment' => [
                'kind' => 'oci_image',
                'artifact_digest' => 'sha256:' . str_repeat('1', 64),
                'base_artifact_digest' => 'sha256:' . str_repeat('2', 64),
            ],
            'artifacts' => [
                'document_model' => $artifact,
                'query_model' => $artifact,
                'tokenizer' => $artifact,
                'model_config' => $artifact,
                'amd64_parity' => $artifact,
                'quality_guard' => $artifact,
                'self_retrieval' => $artifact,
                'reference_vectors' => $artifact,
                'latency_concurrency_four' => $artifact,
            ],
            'qualification' => [
                'status' => 'qualified',
                'evidence_artifact_roles' => [
                    'amd64_parity',
                    'quality_guard',
                    'self_retrieval',
                    'reference_vectors',
                    'latency_concurrency_four',
                ],
            ],
        ];

        return [
            'schema_version' => 2,
            'api_version' => 'v1',
            'model_id' => 'Snowflake/snowflake-arctic-embed-m-v2.0',
            'model_revision' => '95c2741480856aa9666782eb4afe11959938017f',
            'dimension' => 256,
            'recipe_version' => 'mageos-v1',
            'model_contract_digest' => str_repeat('d', 64),
            'encoder_identity_digest' => str_repeat('a', 64),
            'production_eligible' => true,
            'architecture' => 'linux/amd64',
            'runtime' => $manifest['runtime'],
            'deployment' => $manifest['deployment'],
            'artifacts' => $manifest['artifacts'],
            'qualification' => $manifest['qualification'],
            'identity_manifest' => $manifest,
        ];
    }
}
