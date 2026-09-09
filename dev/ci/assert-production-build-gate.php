<?php
declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Config;
use MageOS\OpenSearchHybrid\Model\Encoder\EncoderIdentityClient;
use MageOS\OpenSearchHybrid\Model\Generation\BuildService;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\Cache\TypeListInterface;
use Magento\Framework\App\Config\Storage\WriterInterface;
use Magento\Framework\App\ResourceConnection;

if ($argc !== 2) {
    fwrite(STDERR, "Usage: php assert-production-build-gate.php /path/to/mageos\n");
    exit(2);
}

$mageOsRoot = rtrim((string)$argv[1], DIRECTORY_SEPARATOR);
$bootstrapFile = $mageOsRoot . '/app/bootstrap.php';
if (!is_file($bootstrapFile)) {
    fwrite(STDERR, "Mage-OS bootstrap not found: {$bootstrapFile}\n");
    exit(2);
}
require $bootstrapFile;

$bootstrap = Bootstrap::create(BP, $_SERVER);
$objectManager = $bootstrap->getObjectManager();
$writer = $objectManager->get(WriterInterface::class);
$cacheTypes = $objectManager->get(TypeListInterface::class);
$writer->save(Config::XML_PATH_ENCODER_ENDPOINT, 'http://10.0.0.20:8080');
$cacheTypes->cleanType('config');

$digest = str_repeat('a', 64);
$identity = [
    'model_id' => 'Snowflake/snowflake-arctic-embed-m-v2.0',
    'model_revision' => '95c2741480856aa9666782eb4afe11959938017f',
    'dimension' => 256,
    'recipe_version' => 'mageos-v1',
    'encoder_identity_digest' => $digest,
    'architecture' => 'linux/amd64',
    'deployment' => [
        'artifact_digest' => 'sha256:' . str_repeat('b', 64),
    ],
];
$resource = $objectManager->get(ResourceConnection::class);
$connection = $resource->getConnection();
$generationTable = $resource->getTableName('mageos_opensearch_hybrid_generation');
$before = (int)$connection->fetchOne(
    $connection->select()->from($generationTable, [new Zend_Db_Expr('COUNT(*)')])
);

try {
    $rejectingClient = new class extends EncoderIdentityClient {
        public function __construct()
        {
        }

        public function fetchAt(string $endpoint, string $expectedDigest): array
        {
            throw new RuntimeException('Injected unverified encoder identity.');
        }
    };
    /** @var BuildService $rejectedBuild */
    $rejectedBuild = $objectManager->create(BuildService::class, ['identityClient' => $rejectingClient]);
    try {
        $rejectedBuild->buildProduction(1, $digest);
        throw new RuntimeException('An unverified production identity created a generation.');
    } catch (RuntimeException $exception) {
        if ($exception->getMessage() !== 'Injected unverified encoder identity.') {
            throw $exception;
        }
    }
    $afterRejected = (int)$connection->fetchOne(
        $connection->select()->from($generationTable, [new Zend_Db_Expr('COUNT(*)')])
    );
    if ($afterRejected !== $before) {
        throw new RuntimeException('Production identity verification did not precede the database write.');
    }

    $verifiedClient = new class($identity, $digest) extends EncoderIdentityClient {
        public int $calls = 0;

        public function __construct(
            private readonly array $identity,
            private readonly string $expectedDigest
        ) {
        }

        public function fetchAt(string $endpoint, string $expectedDigest): array
        {
            $this->calls++;
            if ($endpoint !== 'http://10.0.0.20:8080'
                || !hash_equals($this->expectedDigest, $expectedDigest)
            ) {
                throw new RuntimeException('Production build did not preserve the configured endpoint and pin.');
            }

            return $this->identity;
        }
    };
    /** @var BuildService $buildService */
    $buildService = $objectManager->create(BuildService::class, ['identityClient' => $verifiedClient]);
    $preview = $buildService->planProduction(1, $digest);
    $afterPreview = (int)$connection->fetchOne(
        $connection->select()->from($generationTable, [new Zend_Db_Expr('COUNT(*)')])
    );
    $eligibleProducts = (int)($preview['catalog']['eligible_products'] ?? -1);
    $batchSize = (int)($preview['workload']['batch_size'] ?? 0);
    $expectedBatches = $eligibleProducts === 0
        ? 0
        : intdiv($eligibleProducts + $batchSize - 1, $batchSize);
    if ($afterPreview !== $before
        || ($preview['mode'] ?? null) !== 'preview'
        || ($preview['mutation'] ?? null) !== 'none'
        || ($preview['encoder']['identity_digest'] ?? null) !== $digest
        || !is_string($preview['scope']['digest'] ?? null)
        || ($preview['workload']['expected_total_embedding_batches'] ?? null) !== $expectedBatches
        || ($preview['workload']['target_vectors'] ?? null) !== $eligibleProducts
        || ($preview['vector_payload']['raw_float32_bytes'] ?? null) !== $eligibleProducts * 1_024
        || ($preview['activation']['changes'] ?? null) !== 0
        || ($preview['activation']['current_mode'] ?? null)
            !== ($preview['activation']['planned_mode'] ?? null)
        || ($preview['activation']['current_generation_id'] ?? null)
            !== ($preview['activation']['planned_generation_id'] ?? null)
        || !is_string($preview['confirmation_token'] ?? null)
    ) {
        throw new RuntimeException('Production build preview was incomplete or changed generation state.');
    }
    try {
        $buildService->buildProductionConfirmed(
            1,
            $digest,
            'build-production-1-' . str_repeat('0', 64)
        );
        throw new RuntimeException('A stale production build confirmation created a generation.');
    } catch (InvalidArgumentException $exception) {
        if ($exception->getMessage()
            !== 'The production build confirmation does not match the current exact preview.'
        ) {
            throw $exception;
        }
    }
    $afterStaleConfirmation = (int)$connection->fetchOne(
        $connection->select()->from($generationTable, [new Zend_Db_Expr('COUNT(*)')])
    );
    if ($afterStaleConfirmation !== $before) {
        throw new RuntimeException('A stale production build confirmation changed generation state.');
    }
    $generationId = $buildService->buildProductionConfirmed(
        1,
        $digest,
        (string)$preview['confirmation_token']
    );
    $generation = $connection->fetchRow(
        $connection->select()->from($generationTable)->where('generation_id = ?', $generationId)
    );
    if (!is_array($generation)
        || (int)$generation['is_fake'] !== 0
        || (string)$generation['encoder_endpoint'] !== 'http://10.0.0.20:8080'
        || !hash_equals($digest, (string)$generation['encoder_identity_digest'])
        || (string)$generation['model_id'] !== $identity['model_id']
        || (string)$generation['model_revision'] !== $identity['model_revision']
        || !hash_equals((string)$preview['scope']['digest'], (string)$generation['scope_digest'])
        || $verifiedClient->calls !== 3
    ) {
        throw new RuntimeException('Verified production identity was not frozen into the generation.');
    }

    printf(
        "Production generation %d froze verified identity %s before seeding.\n",
        $generationId,
        $digest
    );
} finally {
    $writer->delete(Config::XML_PATH_ENCODER_ENDPOINT);
    $cacheTypes->cleanType('config');
}
