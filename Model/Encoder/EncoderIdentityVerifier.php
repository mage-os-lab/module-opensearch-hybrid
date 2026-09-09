<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Encoder;

class EncoderIdentityVerifier
{
    private const REQUIRED_ARTIFACT_ROLES = [
        'document_model',
        'query_model',
        'tokenizer',
        'model_config',
        'amd64_parity',
        'quality_guard',
        'self_retrieval',
        'reference_vectors',
        'latency_concurrency_four',
    ];
    private const REQUIRED_EVIDENCE_ROLES = [
        'amd64_parity',
        'quality_guard',
        'self_retrieval',
        'reference_vectors',
        'latency_concurrency_four',
    ];

    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract $retrievalContract
    ) {
    }

    public function verify(array $identity, string $expectedDigest): array
    {
        if (!preg_match('/^[0-9a-f]{64}$/D', $expectedDigest)) {
            throw new \InvalidArgumentException('The pinned digest must be 64 lowercase hexadecimal characters.');
        }
        $manifest = $identity['identity_manifest'] ?? null;
        if (!$this->hasExactKeys($identity, [
            'schema_version',
            'api_version',
            'model_id',
            'model_revision',
            'dimension',
            'recipe_version',
            'model_contract_digest',
            'encoder_identity_digest',
            'production_eligible',
            'architecture',
            'runtime',
            'deployment',
            'artifacts',
            'qualification',
            'identity_manifest',
        ])
            || !is_array($manifest)
            || !$this->hasExactKeys($manifest, [
                'schema_version',
                'api_version',
                'model',
                'recipe',
                'runtime',
                'deployment',
                'artifacts',
                'qualification',
            ])
            || ($identity['encoder_identity_digest'] ?? null) !== $expectedDigest
            || !hash_equals($expectedDigest, $this->retrievalContract->digestValue($manifest))
        ) {
            throw new \RuntimeException('The encoder identity does not match the operator-pinned digest.');
        }
        if (($identity['production_eligible'] ?? null) !== true
            || ($identity['schema_version'] ?? null) !== 2
            || ($identity['api_version'] ?? null) !== 'v1'
            || ($manifest['schema_version'] ?? null) !== 2
            || ($manifest['api_version'] ?? null) !== 'v1'
        ) {
            throw new \RuntimeException('The encoder identity is not qualified for production.');
        }
        $this->verifyModelContract($identity, $manifest);
        $this->verifyRuntime($identity, $manifest);
        $this->verifyDeployment($identity, $manifest);
        $this->verifyArtifacts($identity, $manifest);

        return $identity;
    }

    private function verifyModelContract(array $identity, array $manifest): void
    {
        $model = $manifest['model'] ?? null;
        $recipe = $manifest['recipe'] ?? null;
        if (!is_array($model)
            || !$this->hasExactKeys($model, [
                'id',
                'revision',
                'dimension',
                'similarity',
                'normalization',
                'truncate_to_dimension',
                'max_sequence_length',
            ])
            || !is_array($recipe)
            || !$this->hasExactKeys($recipe, [
                'version',
                'query_prefix',
                'query_template',
                'document_prefix',
                'document_template',
                'lexical_brand_attributes',
                'semantic_feature_attributes',
            ])
        ) {
            throw new \RuntimeException('The encoder model contract is missing.');
        }
        $modelContract = [
            'id' => $model['id'] ?? null,
            'revision' => $model['revision'] ?? null,
            'dimension' => $model['dimension'] ?? null,
            'similarity' => $model['similarity'] ?? null,
            'query_prefix' => $recipe['query_prefix'] ?? null,
            'query_template' => $recipe['query_template'] ?? null,
            'document_template' => $recipe['document_template'] ?? null,
            'source_recipe_version' => $recipe['version'] ?? null,
            'lexical_brand_attributes' => $recipe['lexical_brand_attributes'] ?? null,
            'semantic_feature_attributes' => $recipe['semantic_feature_attributes'] ?? null,
        ];
        $contractModel = $this->retrievalContract->get()['model'] ?? null;
        $contractDigest = $this->retrievalContract->digestValue($modelContract);
        if (!is_array($contractModel)
            || $modelContract !== $contractModel
            || ($identity['model_contract_digest'] ?? null) !== $contractDigest
            || ($identity['model_id'] ?? null) !== $modelContract['id']
            || ($identity['model_revision'] ?? null) !== $modelContract['revision']
            || ($identity['dimension'] ?? null) !== $modelContract['dimension']
            || ($identity['recipe_version'] ?? null) !== $modelContract['source_recipe_version']
            || ($model['normalization'] ?? null) !== 'l2_after_truncation'
            || ($model['truncate_to_dimension'] ?? null) !== 256
            || ($model['max_sequence_length'] ?? null) !== 512
            || ($recipe['document_prefix'] ?? null) !== ''
        ) {
            throw new \RuntimeException('The encoder model contract does not match the module contract.');
        }
    }

    private function verifyRuntime(array $identity, array $manifest): void
    {
        $runtime = $identity['runtime'] ?? null;
        if (!is_array($runtime)
            || !$this->hasExactKeys($runtime, [
                'architecture',
                'implementation',
                'execution_provider',
                'query_precision',
                'document_precision',
                'query_batch_size',
                'max_batch_size',
                'versions',
            ])
            || ($manifest['runtime'] ?? null) !== $runtime
            || ($identity['architecture'] ?? null) !== 'linux/amd64'
            || ($runtime['architecture'] ?? null) !== 'linux/amd64'
            || ($runtime['execution_provider'] ?? null) !== 'CPUExecutionProvider'
            || ($runtime['query_precision'] ?? null) !== 'fp32'
            || ($runtime['document_precision'] ?? null) !== 'fp32'
            || ($runtime['query_batch_size'] ?? null) !== 1
            || ($runtime['max_batch_size'] ?? null) !== 64
            || !$this->isBoundedText($runtime['implementation'] ?? null, 128)
        ) {
            throw new \RuntimeException('The encoder runtime is not the qualified linux amd64 contract.');
        }
        $versions = $runtime['versions'] ?? null;
        $versionKeys = is_array($versions) ? array_keys($versions) : [];
        sort($versionKeys, SORT_STRING);
        if (!is_array($versions)
            || $versionKeys !== ['numpy', 'onnxruntime', 'python', 'tokenizers']
            || count(array_filter(
                $versions,
                fn (mixed $value): bool => $this->isBoundedText($value, 64)
            )) !== 4
        ) {
            throw new \RuntimeException('The encoder runtime versions are incomplete.');
        }
    }

    private function verifyDeployment(array $identity, array $manifest): void
    {
        $deployment = $identity['deployment'] ?? null;
        if (!is_array($deployment)
            || !$this->hasExactKeys($deployment, ['kind', 'artifact_digest', 'base_artifact_digest'])
            || ($manifest['deployment'] ?? null) !== $deployment
            || ($deployment['kind'] ?? null) !== 'oci_image'
            || !$this->isArtifactDigest($deployment['artifact_digest'] ?? null)
            || !$this->isArtifactDigest($deployment['base_artifact_digest'] ?? null)
        ) {
            throw new \RuntimeException('The encoder deployment identity is incomplete.');
        }
    }

    private function verifyArtifacts(array $identity, array $manifest): void
    {
        $artifacts = $identity['artifacts'] ?? null;
        $qualification = $identity['qualification'] ?? null;
        if (!is_array($artifacts)
            || ($manifest['artifacts'] ?? null) !== $artifacts
            || !is_array($qualification)
            || ($manifest['qualification'] ?? null) !== $qualification
        ) {
            throw new \RuntimeException('The encoder artifact identity is incomplete.');
        }
        $roles = array_keys($artifacts);
        sort($roles, SORT_STRING);
        $required = self::REQUIRED_ARTIFACT_ROLES;
        sort($required, SORT_STRING);
        if ($roles !== $required) {
            throw new \RuntimeException('The encoder artifact identity is incomplete.');
        }
        if (($artifacts['query_model'] ?? null) !== ($artifacts['document_model'] ?? null)) {
            throw new \RuntimeException('The encoder artifact identity must use one shared fp32 model.');
        }
        foreach ($artifacts as $role => $artifact) {
            if (!is_array($artifact)
                || !$this->hasExactKeys($artifact, ['path', 'sha256', 'size'])
                || !$this->isBoundedText($artifact['path'] ?? null, 1024)
                || str_starts_with($artifact['path'], '/')
                || str_contains($artifact['path'], '..')
                || !is_string($artifact['sha256'] ?? null)
                || !preg_match('/^[0-9a-f]{64}$/D', $artifact['sha256'])
                || !is_int($artifact['size'] ?? null)
                || $artifact['size'] <= 0
            ) {
                throw new \RuntimeException(sprintf('The encoder artifact %s is invalid.', $role));
            }
        }
        if (($qualification['status'] ?? null) !== 'qualified'
            || !$this->hasExactKeys($qualification, ['status', 'evidence_artifact_roles'])
            || ($qualification['evidence_artifact_roles'] ?? null) !== self::REQUIRED_EVIDENCE_ROLES
        ) {
            throw new \RuntimeException('The encoder qualification evidence is incomplete.');
        }
    }

    private function isBoundedText(mixed $value, int $maximumBytes): bool
    {
        return is_string($value) && $value !== '' && strlen($value) <= $maximumBytes;
    }

    private function isArtifactDigest(mixed $value): bool
    {
        return is_string($value) && preg_match('/^sha256:[0-9a-f]{64}$/D', $value) === 1;
    }

    private function hasExactKeys(array $value, array $expected): bool
    {
        $keys = array_keys($value);
        sort($keys, SORT_STRING);
        sort($expected, SORT_STRING);

        return $keys === $expected;
    }
}
