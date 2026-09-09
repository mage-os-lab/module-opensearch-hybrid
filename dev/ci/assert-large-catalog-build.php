<?php

declare(strict_types=1);

use Composer\InstalledVersions;
use MageOS\OpenSearchHybrid\Model\Change\SearchableProductResolver;
use MageOS\OpenSearchHybrid\Model\Config;
use MageOS\OpenSearchHybrid\Model\Contract\RetrievalContract;
use MageOS\OpenSearchHybrid\Model\Generation\BuildService;
use MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository;
use MageOS\OpenSearchHybrid\Model\Generation\ValidationService;
use MageOS\OpenSearchHybrid\Model\OpenSearch\ClusterVersionProvider;
use MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository;
use MageOS\OpenSearchHybrid\Model\Queue\EmbeddingConsumer;
use Magento\Catalog\Model\Product;
use Magento\Eav\Model\Config as EavConfig;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\ProductMetadataInterface;
use Magento\Framework\App\ResourceConnection;
use Magento\Framework\App\State;
use Magento\Framework\Indexer\IndexerRegistry;

const MAXIMUM_TARGET_PRODUCTS = 1_000_000;
const PHASE5_TARGETS = [100_000, 1_000_000];

if ($argc < 2 || $argc > 5) {
    fwrite(
        STDERR,
        "Usage: php assert-large-catalog-build.php /path/to/mageos [target-products] "
            . "[scale-confirmation] [evidence-output]\n"
    );
    exit(2);
}

$fixtureRoot = rtrim((string)$argv[1], DIRECTORY_SEPARATOR);
$targetProducts = isset($argv[2]) ? (int)$argv[2] : 10_000;
$scaleConfirmation = isset($argv[3]) ? (string)$argv[3] : '';
$evidenceOutput = isset($argv[4]) ? (string)$argv[4] : '';
$phase5Scale = in_array($targetProducts, PHASE5_TARGETS, true);
if (!is_file($fixtureRoot . '/app/bootstrap.php')) {
    fwrite(STDERR, "Mage-OS bootstrap not found: {$fixtureRoot}\n");
    exit(2);
}
if (
    $targetProducts < 100
    || $targetProducts > MAXIMUM_TARGET_PRODUCTS
    || ($targetProducts > 10_000 && !$phase5Scale)
) {
    fwrite(STDERR, "Target products must be 100 to 10000, 100000, or 1000000.\n");
    exit(2);
}
if ($phase5Scale && $scaleConfirmation !== sprintf('scale-%d', $targetProducts)) {
    fwrite(STDERR, "Phase 5 scale creation requires the exact target-bound confirmation.\n");
    exit(2);
}
if ($phase5Scale && $evidenceOutput === '') {
    fwrite(STDERR, "Phase 5 scale creation requires an evidence output path.\n");
    exit(2);
}
$packageManifestPath = getenv('MAGEOS_MODULE_PACKAGE_MANIFEST');
if ($evidenceOutput !== '' && (
    !is_string($packageManifestPath)
    || !str_starts_with($packageManifestPath, DIRECTORY_SEPARATOR)
    || !is_file($packageManifestPath)
    || is_link($packageManifestPath)
)) {
    fwrite(STDERR, "Phase 5 scale creation requires the verified package manifest.\n");
    exit(2);
}
if ($evidenceOutput !== '' && (
    !str_starts_with($evidenceOutput, DIRECTORY_SEPARATOR)
    || !is_dir(dirname($evidenceOutput))
    || file_exists($evidenceOutput)
    || is_link($evidenceOutput)
)) {
    fwrite(STDERR, "Evidence output must be a new absolute file under an existing directory.\n");
    exit(2);
}

require $fixtureRoot . '/app/bootstrap.php';
$bootstrap = Bootstrap::create(BP, $_SERVER);
$objectManager = $bootstrap->getObjectManager();
try {
    $objectManager->get(State::class)->setAreaCode('global');
} catch (\Magento\Framework\Exception\LocalizedException) {
}

$storeId = 1;
/** @var ResourceConnection $resourceConnection */
$resourceConnection = $objectManager->get(ResourceConnection::class);
$connection = $resourceConnection->getConnection();
/** @var SearchableProductResolver $productResolver */
$productResolver = $objectManager->get(SearchableProductResolver::class);
$writableGenerations = $connection->fetchAll(
    $connection->select()
        ->from(
            $resourceConnection->getTableName('mageos_opensearch_hybrid_generation'),
            ['generation_id', 'state']
        )
        ->where('state IN (?)', ['ACTIVE', 'BUILDING', 'CATCHING_UP', 'READY'])
        ->order('generation_id ASC')
);
if ($writableGenerations !== []) {
    throw new RuntimeException(
        'Large-catalog qualification requires an isolated fixture with no writable generation.'
    );
}
$eligibleBefore = $productResolver->countEligible($storeId);
if ($eligibleBefore > $targetProducts) {
    throw new RuntimeException('The fixture already exceeds the requested large-catalog target.');
}

$productTable = $resourceConnection->getTableName('catalog_product_entity');
$websiteTable = $resourceConnection->getTableName('catalog_product_website');
$sourceItemTable = $resourceConnection->getTableName('inventory_source_item');
$legacyStockItemTable = $resourceConnection->getTableName('cataloginventory_stock_item');
$fixtureProductsBefore = (int)$connection->fetchOne(
    $connection->select()
        ->from($productTable, [new Zend_Db_Expr('COUNT(*)')])
        ->where('sku LIKE ?', 'mageos-hybrid-large-%')
);
if ($phase5Scale && $fixtureProductsBefore !== 0) {
    throw new RuntimeException(
        'Phase 5 scale qualification requires a fresh fixture without prior large-catalog products.'
    );
}
$entityTypeId = (int)$connection->fetchOne(
    $connection->select()
        ->from($resourceConnection->getTableName('eav_entity_type'), ['entity_type_id'])
        ->where('entity_type_code = ?', Product::ENTITY)
);
$attributeSetId = (int)$connection->fetchOne(
    $connection->select()
        ->from($resourceConnection->getTableName('eav_attribute_set'), ['attribute_set_id'])
        ->where('entity_type_id = ?', $entityTypeId)
        ->where('attribute_set_name = ?', 'Default')
        ->limit(1)
);
if ($entityTypeId < 1 || $attributeSetId < 1) {
    throw new RuntimeException('The fixture has no default catalog product attribute set.');
}

/** @var EavConfig $eavConfig */
$eavConfig = $objectManager->get(EavConfig::class);
$attributeIds = [];
foreach (['name', 'url_key', 'status', 'visibility', 'tax_class_id', 'price', 'weight', 'description'] as $code) {
    $attribute = $eavConfig->getAttribute(Product::ENTITY, $code);
    if (!(int)$attribute->getId()) {
        throw new RuntimeException(sprintf('Required fixture attribute %s does not exist.', $code));
    }
    $attributeIds[$code] = (int)$attribute->getId();
}

$websiteId = (int)$connection->fetchOne(
    $connection->select()
        ->from($resourceConnection->getTableName('store'), ['website_id'])
        ->where('store_id = ?', $storeId)
);
$websiteCode = (string)$connection->fetchOne(
    $connection->select()
        ->from($resourceConnection->getTableName('store_website'), ['code'])
        ->where('website_id = ?', $websiteId)
);
$sourceCode = (string)$connection->fetchOne(
    $connection->select()
        ->from(['channel' => $resourceConnection->getTableName('inventory_stock_sales_channel')], [])
        ->joinInner(
            ['link' => $resourceConnection->getTableName('inventory_source_stock_link')],
            'link.stock_id = channel.stock_id',
            ['source_code']
        )
        ->joinInner(
            ['source' => $resourceConnection->getTableName('inventory_source')],
            'source.source_code = link.source_code',
            []
        )
        ->where('channel.type = ?', 'website')
        ->where('channel.code = ?', $websiteCode)
        ->where('source.enabled = ?', 1)
        ->order('link.priority ASC')
        ->limit(1)
);
if ($websiteId < 1 || $sourceCode === '') {
    throw new RuntimeException('The fixture store has no enabled inventory source.');
}

$missing = $targetProducts - $eligibleBefore;
$nextProductId = (int)$connection->fetchOne(
    $connection->select()->from($productTable, [new Zend_Db_Expr('COALESCE(MAX(entity_id), 0) + 1')])
);
$insertStartedAt = hrtime(true);
for ($offset = 0; $offset < $missing; $offset += 250) {
    $batchSize = min(250, $missing - $offset);
    $entities = [];
    $websites = [];
    $sourceItems = [];
    $legacyStockItems = [];
    $varcharValues = [];
    $intValues = [];
    $decimalValues = [];
    $textValues = [];
    for ($index = 0; $index < $batchSize; $index++) {
        $productId = $nextProductId + $offset + $index;
        $sku = sprintf('mageos-hybrid-large-%07d', $productId);
        $name = sprintf('MageOS Hybrid Large Catalog Product %07d', $productId);
        $entities[] = [
            'entity_id' => $productId,
            'attribute_set_id' => $attributeSetId,
            'type_id' => 'simple',
            'sku' => $sku,
            'has_options' => 0,
            'required_options' => 0,
        ];
        $websites[] = ['product_id' => $productId, 'website_id' => $websiteId];
        $sourceItems[] = [
            'source_code' => $sourceCode,
            'sku' => $sku,
            'quantity' => 100.0,
            'status' => 1,
        ];
        $legacyStockItems[] = [
            'product_id' => $productId,
            'stock_id' => 1,
            'qty' => 100.0,
            'is_in_stock' => 1,
            'website_id' => 0,
        ];
        $varcharValues[] = [
            'attribute_id' => $attributeIds['name'],
            'store_id' => 0,
            'entity_id' => $productId,
            'value' => $name,
        ];
        $varcharValues[] = [
            'attribute_id' => $attributeIds['url_key'],
            'store_id' => 0,
            'entity_id' => $productId,
            'value' => $sku,
        ];
        foreach (['status' => 1, 'visibility' => 4, 'tax_class_id' => 2] as $code => $value) {
            $intValues[] = [
                'attribute_id' => $attributeIds[$code],
                'store_id' => 0,
                'entity_id' => $productId,
                'value' => $value,
            ];
        }
        foreach (['price' => 19.99, 'weight' => 1.0] as $code => $value) {
            $decimalValues[] = [
                'attribute_id' => $attributeIds[$code],
                'store_id' => 0,
                'entity_id' => $productId,
                'value' => $value,
            ];
        }
        $textValues[] = [
            'attribute_id' => $attributeIds['description'],
            'store_id' => 0,
            'entity_id' => $productId,
            'value' => 'Deterministic large-catalog product for OpenSearch Hybrid build qualification.',
        ];
    }
    $connection->beginTransaction();
    try {
        $connection->insertMultiple($productTable, $entities);
        $connection->insertMultiple($websiteTable, $websites);
        $connection->insertMultiple($sourceItemTable, $sourceItems);
        $connection->insertMultiple($legacyStockItemTable, $legacyStockItems);
        $connection->insertMultiple(
            $resourceConnection->getTableName('catalog_product_entity_varchar'),
            $varcharValues
        );
        $connection->insertMultiple(
            $resourceConnection->getTableName('catalog_product_entity_int'),
            $intValues
        );
        $connection->insertMultiple(
            $resourceConnection->getTableName('catalog_product_entity_decimal'),
            $decimalValues
        );
        $connection->insertMultiple(
            $resourceConnection->getTableName('catalog_product_entity_text'),
            $textValues
        );
        $connection->commit();
    } catch (Throwable $throwable) {
        $connection->rollBack();
        throw $throwable;
    }
    printf("Created %d of %d large-catalog products.\n", min($offset + $batchSize, $missing), $missing);
}
$insertSeconds = (hrtime(true) - $insertStartedAt) / 1_000_000_000;

/** @var IndexerRegistry $indexerRegistry */
$indexerRegistry = $objectManager->get(IndexerRegistry::class);
$nativeReindexStartedAt = hrtime(true);
foreach (['inventory', 'cataloginventory_stock', 'catalog_product_price', 'catalogsearch_fulltext'] as $indexerId) {
    $indexerRegistry->get($indexerId)->reindexAll();
}
$nativeReindexSeconds = (hrtime(true) - $nativeReindexStartedAt) / 1_000_000_000;
$eligibleAfter = $productResolver->countEligible($storeId);
if ($eligibleAfter !== $targetProducts) {
    throw new RuntimeException(sprintf(
        'Large-catalog fixture resolved %d eligible products instead of %d.',
        $eligibleAfter,
        $targetProducts
    ));
}
$fixtureProductsAfter = (int)$connection->fetchOne(
    $connection->select()
        ->from($productTable, [new Zend_Db_Expr('COUNT(*)')])
        ->where('sku LIKE ?', 'mageos-hybrid-large-%')
);
if ($fixtureProductsAfter !== $fixtureProductsBefore + $missing) {
    throw new RuntimeException('The large-catalog fixture prefix count is incomplete.');
}

/** @var BuildService $buildService */
$buildService = $objectManager->get(BuildService::class);
/** @var EmbeddingConsumer $embeddingConsumer */
$embeddingConsumer = $objectManager->get(EmbeddingConsumer::class);
/** @var OutboxRepository $outboxRepository */
$outboxRepository = $objectManager->get(OutboxRepository::class);
/** @var Config $moduleConfig */
$moduleConfig = $objectManager->get(Config::class);
$buildStartedAt = hrtime(true);
$generationId = $buildService->buildFake($storeId);
$processedJobs = 0;
$maximumIterations = (int)ceil($targetProducts / max(1, $moduleConfig->batchSize())) + 100;
$progressInterval = $phase5Scale ? 500 : 40;
for ($iteration = 0; $iteration < $maximumIterations; $iteration++) {
    $jobIds = array_map('strval', $connection->fetchCol(
        $connection->select()
            ->from($resourceConnection->getTableName('mageos_opensearch_hybrid_outbox'), ['job_id'])
            ->where('generation_id = ?', $generationId)
            ->where('lane = ?', 'EMBEDDING')
            ->where('state NOT IN (?)', ['COMPLETED', 'SUPERSEDED'])
            ->order('created_at ASC')
    ));
    foreach ($jobIds as $jobId) {
        if ((string)$outboxRepository->get($jobId)['state'] === 'PENDING') {
            $outboxRepository->markPublished($jobId);
        }
        $embeddingConsumer->process($jobId);
        $processedJobs++;
    }
    $report = $buildService->resumeFake($generationId);
    if ($processedJobs > 0 && $processedJobs % $progressInterval === 0) {
        printf("Processed %d embedding batches for generation %d.\n", $processedJobs, $generationId);
    }
    if ((string)$report['seeding_state'] === 'COMPLETED' && $jobIds === []) {
        break;
    }
}

/** @var ValidationService $validationService */
$validationService = $objectManager->get(ValidationService::class);
/** @var RetrievalContract $retrievalContract */
$retrievalContract = $objectManager->get(RetrievalContract::class);
$validation = $validationService->validate(
    $generationId,
    $retrievalContract->resultContractDigest()
);
/** @var GenerationRepository $generationRepository */
$generationRepository = $objectManager->get(GenerationRepository::class);
$generation = $generationRepository->get($generationId);
$buildSeconds = (hrtime(true) - $buildStartedAt) / 1_000_000_000;
if (
    !(bool)$validation['ready']
    || (string)$generation['state'] !== 'READY'
    || (int)$generation['coverage_total'] !== $targetProducts
    || (int)$generation['coverage_complete'] !== $targetProducts
    || (int)$generation['coverage_failed'] !== 0
) {
    $failedChecks = array_keys(array_filter(
        $validation['checks'],
        static fn (bool $passed): bool => !$passed
    ));
    throw new RuntimeException(sprintf(
        'The %d-product generation failed validation checks: %s.',
        $targetProducts,
        implode(', ', $failedChecks)
    ));
}

if ($evidenceOutput !== '') {
    /** @var ProductMetadataInterface $productMetadata */
    $productMetadata = $objectManager->get(ProductMetadataInterface::class);
    /** @var ClusterVersionProvider $clusterVersionProvider */
    $clusterVersionProvider = $objectManager->get(ClusterVersionProvider::class);
    $mageOsVersion = InstalledVersions::getPrettyVersion('mage-os/product-community-edition');
    if (!is_string($mageOsVersion) || $mageOsVersion === '') {
        throw new RuntimeException('Could not resolve the installed Mage-OS release identity.');
    }
    $configuredModuleRoot = getenv('MAGEOS_MODULE_INSTALL_ROOT');
    $moduleInstallRoot = is_string($configuredModuleRoot) && $configuredModuleRoot !== ''
        ? rtrim($configuredModuleRoot, DIRECTORY_SEPARATOR)
        : dirname(__DIR__, 2);
    $composerPath = $moduleInstallRoot . '/composer.json';
    if (!is_file($composerPath)) {
        throw new RuntimeException('The scale evidence module root has no Composer manifest.');
    }
    $composerDigest = hash_file('sha256', $composerPath);
    if (!is_string($composerDigest)) {
        throw new RuntimeException('Could not hash the module Composer manifest.');
    }
    $packageManifestBytes = file_get_contents((string)$packageManifestPath);
    $packageManifest = is_string($packageManifestBytes)
        ? json_decode($packageManifestBytes, true, 512, JSON_THROW_ON_ERROR)
        : null;
    $packageFiles = is_array($packageManifest) ? ($packageManifest['payload']['files'] ?? null) : null;
    $packageFileCount = is_array($packageManifest)
        ? (int)($packageManifest['payload']['file_count'] ?? 0)
        : 0;
    $packagePayloadDigest = is_array($packageManifest)
        ? (string)($packageManifest['payload']['sha256'] ?? '')
        : '';
    $packageArchiveDigest = is_array($packageManifest)
        ? (string)($packageManifest['archive']['sha256'] ?? '')
        : '';
    $manifestComposerDigest = '';
    if (is_array($packageFiles)) {
        foreach ($packageFiles as $packageFile) {
            if (is_array($packageFile) && ($packageFile['path'] ?? null) === 'composer.json') {
                $manifestComposerDigest = (string)($packageFile['sha256'] ?? '');
                break;
            }
        }
    }
    if (
        !is_array($packageManifest)
        || ($packageManifest['schema_version'] ?? null) !== 1
        || ($packageManifest['package']['name'] ?? null) !== 'mage-os/module-opensearch-hybrid'
        || ($packageManifest['package']['type'] ?? null) !== 'magento2-module'
        || !is_array($packageFiles)
        || $packageFileCount < 1
        || count($packageFiles) !== $packageFileCount
        || preg_match('/\A[a-f0-9]{64}\z/D', $packagePayloadDigest) !== 1
        || preg_match('/\A[a-f0-9]{64}\z/D', $packageArchiveDigest) !== 1
        || $manifestComposerDigest !== $composerDigest
    ) {
        throw new RuntimeException('The installed module does not match the scale package manifest.');
    }
    $evidence = [
        'schema_version' => 2,
        'status' => 'build_passed',
        'qualification_scope' => $phase5Scale ? 'phase5_scale' : 'ci_smoke',
        'catalog' => [
            'target_products' => $targetProducts,
            'eligible_before' => $eligibleBefore,
            'created_products' => $missing,
            'eligible_after' => $eligibleAfter,
            'fixture_products' => $fixtureProductsAfter,
        ],
        'runtime' => [
            'mageos_version' => $mageOsVersion,
            'magento_product_version' => $productMetadata->getVersion(),
            'php_version' => PHP_VERSION,
            'php_memory_limit' => (string)ini_get('memory_limit'),
            'opensearch_version' => $clusterVersionProvider->current(),
            'module_composer_sha256' => $composerDigest,
            'module_package_file_count' => $packageFileCount,
            'module_package_payload_sha256' => $packagePayloadDigest,
            'module_package_archive_sha256' => $packageArchiveDigest,
            'encoder_runtime' => 'deterministic_hash_fake_only',
        ],
        'generation' => [
            'generation_id' => $generationId,
            'state' => (string)$generation['state'],
            'coverage_total' => (int)$generation['coverage_total'],
            'coverage_complete' => (int)$generation['coverage_complete'],
            'coverage_failed' => (int)$generation['coverage_failed'],
            'processed_batches' => $processedJobs,
            'validation_ready' => (bool)$validation['ready'],
            'contract_digest' => (string)$generation['contract_digest'],
            'result_contract_digest' => (string)$generation['result_contract_digest'],
        ],
        'timings_seconds' => [
            'product_insert' => round($insertSeconds, 3),
            'native_reindex' => round($nativeReindexSeconds, 3),
            'hybrid_build' => round($buildSeconds, 3),
        ],
        'process' => [
            'peak_memory_bytes' => memory_get_peak_usage(true),
        ],
        'capacity_guidance' => [
            'production_encoder_included' => false,
            'eligible' => false,
            'reason' => 'synthetic_fixture_and_fake_encoder',
        ],
    ];
    $temporaryEvidence = $evidenceOutput . '.tmp-' . getmypid();
    $handle = fopen($temporaryEvidence, 'x');
    if ($handle === false) {
        throw new RuntimeException('Could not create the temporary scale evidence file.');
    }
    try {
        $encodedEvidence = json_encode(
            $evidence,
            JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_THROW_ON_ERROR
        ) . "\n";
        if (fwrite($handle, $encodedEvidence) !== strlen($encodedEvidence)) {
            throw new RuntimeException('Could not write the complete scale evidence file.');
        }
    } finally {
        fclose($handle);
    }
    if (!rename($temporaryEvidence, $evidenceOutput)) {
        throw new RuntimeException('Could not publish the scale build evidence file.');
    }
}

printf(
    "Large catalog inserted in %.1f seconds and natively reindexed in %.1f seconds; generation %d "
        . "indexed and validated %d products in %.1f seconds across %d batches.\n",
    $insertSeconds,
    $nativeReindexSeconds,
    $generationId,
    $targetProducts,
    $buildSeconds,
    $processedJobs
);
