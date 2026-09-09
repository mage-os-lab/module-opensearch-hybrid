<?php
declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Change\SearchableProductResolver;
use Magento\Catalog\Model\Product;
use Magento\Eav\Model\Config as EavConfig;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\ResourceConnection;
use Magento\Framework\App\State;
use Magento\Framework\Indexer\IndexerRegistry;

const MODE_PLAN = 'plan';
const MODE_APPLY = 'apply';
const MODE_ROLLBACK_PREVIEW = 'rollback-preview';
const MODE_ROLLBACK = 'rollback';
const TARGET_PRODUCTS = 50_000;
const SKU_PREFIX = 'mageos-radial-registered-';
const BATCH_SIZE = 250;

if ($argc < 4 || $argc > 5) {
    fwrite(
        STDERR,
        "Usage: php prepare-registered-catalog.php /path/to/mageos "
            . "plan|apply|rollback-preview|rollback store-id [confirmation-token]\n"
    );
    exit(2);
}

$fixtureRoot = rtrim((string)$argv[1], DIRECTORY_SEPARATOR);
$mode = (string)$argv[2];
$storeId = filter_var($argv[3], FILTER_VALIDATE_INT, ['options' => ['min_range' => 1]]);
$confirmationToken = isset($argv[4]) ? (string)$argv[4] : '';
if (!is_file($fixtureRoot . '/app/bootstrap.php')) {
    fwrite(STDERR, "Mage-OS bootstrap not found: {$fixtureRoot}\n");
    exit(2);
}
if (!in_array($mode, [MODE_PLAN, MODE_APPLY, MODE_ROLLBACK_PREVIEW, MODE_ROLLBACK], true)) {
    fwrite(STDERR, "The fixture operation mode is invalid.\n");
    exit(2);
}
if ($storeId === false) {
    fwrite(STDERR, "The fixture store ID must be a positive integer.\n");
    exit(2);
}
if (in_array($mode, [MODE_APPLY, MODE_ROLLBACK], true)
    && !preg_match('/^[0-9a-f]{64}$/D', $confirmationToken)
) {
    fwrite(STDERR, "Apply and rollback require the exact current confirmation token.\n");
    exit(2);
}

require $fixtureRoot . '/app/bootstrap.php';
$bootstrap = Bootstrap::create(BP, $_SERVER);
$objectManager = $bootstrap->getObjectManager();
try {
    $objectManager->get(State::class)->setAreaCode('global');
} catch (\Magento\Framework\Exception\LocalizedException) {
}

/** @var ResourceConnection $resourceConnection */
$resourceConnection = $objectManager->get(ResourceConnection::class);
$connection = $resourceConnection->getConnection();
/** @var SearchableProductResolver $productResolver */
$productResolver = $objectManager->get(SearchableProductResolver::class);
/** @var EavConfig $eavConfig */
$eavConfig = $objectManager->get(EavConfig::class);
/** @var IndexerRegistry $indexerRegistry */
$indexerRegistry = $objectManager->get(IndexerRegistry::class);

$tables = [
    'product' => $resourceConnection->getTableName('catalog_product_entity'),
    'website' => $resourceConnection->getTableName('catalog_product_website'),
    'source_item' => $resourceConnection->getTableName('inventory_source_item'),
    'stock_item' => $resourceConnection->getTableName('cataloginventory_stock_item'),
    'varchar' => $resourceConnection->getTableName('catalog_product_entity_varchar'),
    'int' => $resourceConnection->getTableName('catalog_product_entity_int'),
    'decimal' => $resourceConnection->getTableName('catalog_product_entity_decimal'),
    'text' => $resourceConnection->getTableName('catalog_product_entity_text'),
];
$websiteId = (int)$connection->fetchOne(
    $connection->select()
        ->from($resourceConnection->getTableName('store'), ['website_id'])
        ->where('store_id = ?', (int)$storeId)
);
if ($websiteId < 1) {
    throw new RuntimeException('The fixture store has no website.');
}
$sourceCode = resolveSourceCode($connection, $resourceConnection, $websiteId);
$attributeSetId = resolveAttributeSetId($connection, $resourceConnection);
$attributeIds = resolveAttributeIds($eavConfig);
$lockName = sprintf('mageos_radial_registered_catalog_store_%d', (int)$storeId);
$requireApplySafeState = in_array($mode, [MODE_PLAN, MODE_APPLY], true);

$state = fixtureState(
    $connection,
    $productResolver,
    $tables,
    (int)$storeId,
    $websiteId,
    $sourceCode,
    $requireApplySafeState
);
if ($mode === MODE_PLAN) {
    emit(planPayload('apply', $state));
    exit(0);
}
if ($mode === MODE_ROLLBACK_PREVIEW) {
    emit(planPayload('rollback', $state));
    exit(0);
}

$locked = (int)$connection->fetchOne('SELECT GET_LOCK(?, 0)', [$lockName]) === 1;
if (!$locked) {
    throw new RuntimeException('Another registered-catalog operation holds the store lock.');
}
try {
    $current = fixtureState(
        $connection,
        $productResolver,
        $tables,
        (int)$storeId,
        $websiteId,
        $sourceCode,
        $mode === MODE_APPLY
    );
    $action = $mode === MODE_APPLY ? 'apply' : 'rollback';
    $plan = planPayload($action, $current);
    if (!hash_equals((string)$plan['confirmation_token'], $confirmationToken)) {
        throw new RuntimeException('The confirmation token no longer matches the exact fixture impact.');
    }

    if ($mode === MODE_APPLY) {
        applyFixture(
            $connection,
            $tables,
            $attributeIds,
            $attributeSetId,
            $websiteId,
            $sourceCode,
            (int)$current['fixture_products'],
            (int)$current['products_to_create']
        );
    } else {
        rollbackFixture($connection, $tables);
    }
    reindex($indexerRegistry);

    $completed = fixtureState(
        $connection,
        $productResolver,
        $tables,
        (int)$storeId,
        $websiteId,
        $sourceCode,
        $mode === MODE_APPLY
    );
    if ($mode === MODE_APPLY && (int)$completed['eligible_products'] !== TARGET_PRODUCTS) {
        throw new RuntimeException(sprintf(
            'The registered fixture resolved %d eligible products instead of %d.',
            (int)$completed['eligible_products'],
            TARGET_PRODUCTS
        ));
    }
    if ($mode === MODE_ROLLBACK && (int)$completed['fixture_products'] !== 0) {
        throw new RuntimeException('The registered fixture rollback left prefixed products behind.');
    }
    emit([
        'schema_version' => 1,
        'status' => 'completed',
        'action' => $action,
        'state' => $completed,
    ]);
} finally {
    $connection->fetchOne('SELECT RELEASE_LOCK(?)', [$lockName]);
}

function fixtureState(
    \Magento\Framework\DB\Adapter\AdapterInterface $connection,
    SearchableProductResolver $productResolver,
    array $tables,
    int $storeId,
    int $websiteId,
    string $sourceCode,
    bool $requireApplySafeState
): array {
    $rows = $connection->fetchAll(
        $connection->select()
            ->from($tables['product'], ['entity_id', 'sku'])
            ->where('sku LIKE ?', SKU_PREFIX . '%')
            ->order('sku ASC')
    );
    $fixtureIds = array_map(static fn (array $row): int => (int)$row['entity_id'], $rows);
    $fixtureEligible = 0;
    foreach (array_chunk($fixtureIds, 1000) as $ids) {
        $fixtureEligible += count($productResolver->eligibleIds($storeId, $ids));
    }
    $fixtureInconsistencies = [];
    if (count($rows) !== $fixtureEligible) {
        $fixtureInconsistencies[] = 'not_every_prefixed_product_is_eligible';
    }
    foreach ($rows as $offset => $row) {
        $expectedSku = SKU_PREFIX . sprintf('%05d', $offset + 1);
        if ((string)$row['sku'] !== $expectedSku) {
            $fixtureInconsistencies[] = 'sku_sequence_is_not_contiguous';
            break;
        }
    }
    $fixtureConsistent = $fixtureInconsistencies === [];
    if ($requireApplySafeState && !$fixtureConsistent) {
        throw new RuntimeException(
            'The prefixed fixture is incomplete or inconsistent; use rollback-preview before continuing.'
        );
    }

    $eligibleProducts = $productResolver->countEligible($storeId);
    if ($requireApplySafeState && $eligibleProducts > TARGET_PRODUCTS) {
        throw new RuntimeException(sprintf(
            'Store %d already has %d eligible products, above the registered %d-product target.',
            $storeId,
            $eligibleProducts,
            TARGET_PRODUCTS
        ));
    }
    $identity = hash_init('sha256');
    foreach ($rows as $row) {
        hash_update($identity, sprintf("%d:%s\n", (int)$row['entity_id'], (string)$row['sku']));
    }

    return [
        'store_id' => $storeId,
        'website_id' => $websiteId,
        'source_code' => $sourceCode,
        'sku_prefix' => SKU_PREFIX,
        'target_eligible_products' => TARGET_PRODUCTS,
        'eligible_products' => $eligibleProducts,
        'unrelated_eligible_products' => $eligibleProducts - $fixtureEligible,
        'fixture_products' => count($rows),
        'fixture_eligible_products' => $fixtureEligible,
        'fixture_consistent' => $fixtureConsistent,
        'fixture_inconsistencies' => $fixtureInconsistencies,
        'products_to_create' => TARGET_PRODUCTS - $eligibleProducts,
        'fixture_min_entity_id' => $rows === [] ? null : (int)$rows[0]['entity_id'],
        'fixture_max_entity_id' => $rows === [] ? null : (int)$rows[array_key_last($rows)]['entity_id'],
        'fixture_identity_sha256' => hash_final($identity),
    ];
}

function planPayload(string $action, array $state): array
{
    $impact = [
        'schema_version' => 1,
        'action' => $action,
        'state' => $state,
        'database_scope' => [
            'catalog_product_entity',
            'catalog_product_website',
            'inventory_source_item',
            'cataloginventory_stock_item',
            'catalog_product_entity_varchar',
            'catalog_product_entity_int',
            'catalog_product_entity_decimal',
            'catalog_product_entity_text',
        ],
        'reindexers' => [
            'inventory',
            'cataloginventory_stock',
            'catalog_product_price',
            'catalogsearch_fulltext',
        ],
        'rollback_scope' => SKU_PREFIX . '*',
    ];
    $impact['confirmation_token'] = hash('sha256', canonicalJson($impact));
    return $impact;
}

function applyFixture(
    \Magento\Framework\DB\Adapter\AdapterInterface $connection,
    array $tables,
    array $attributeIds,
    int $attributeSetId,
    int $websiteId,
    string $sourceCode,
    int $existingFixtureProducts,
    int $productsToCreate
): void {
    for ($offset = 0; $offset < $productsToCreate; $offset += BATCH_SIZE) {
        $batchSize = min(BATCH_SIZE, $productsToCreate - $offset);
        $entities = [];
        $products = [];
        for ($index = 0; $index < $batchSize; $index++) {
            $ordinal = $existingFixtureProducts + $offset + $index + 1;
            $product = deterministicProduct($ordinal);
            $products[$product['sku']] = $product;
            $entities[] = [
                'attribute_set_id' => $attributeSetId,
                'type_id' => 'simple',
                'sku' => $product['sku'],
                'has_options' => 0,
                'required_options' => 0,
            ];
        }

        $connection->beginTransaction();
        try {
            $connection->insertMultiple($tables['product'], $entities);
            $entityRows = $connection->fetchAll(
                $connection->select()
                    ->from($tables['product'], ['entity_id', 'sku'])
                    ->where('sku IN (?)', array_keys($products))
            );
            if (count($entityRows) !== $batchSize) {
                throw new RuntimeException('The fixture batch did not create every product entity.');
            }
            insertProductRelations(
                $connection,
                $tables,
                $attributeIds,
                $websiteId,
                $sourceCode,
                $products,
                $entityRows
            );
            $connection->commit();
        } catch (Throwable $throwable) {
            $connection->rollBack();
            throw $throwable;
        }
        printf(
            "Created %d of %d registered-catalog products.\n",
            min($offset + $batchSize, $productsToCreate),
            $productsToCreate
        );
    }
}

function insertProductRelations(
    \Magento\Framework\DB\Adapter\AdapterInterface $connection,
    array $tables,
    array $attributeIds,
    int $websiteId,
    string $sourceCode,
    array $products,
    array $entityRows
): void {
    $websites = [];
    $sourceItems = [];
    $stockItems = [];
    $varcharValues = [];
    $intValues = [];
    $decimalValues = [];
    $textValues = [];
    foreach ($entityRows as $row) {
        $productId = (int)$row['entity_id'];
        $sku = (string)$row['sku'];
        $product = $products[$sku] ?? null;
        if (!is_array($product)) {
            throw new RuntimeException('The fixture batch product identity changed.');
        }
        $websites[] = ['product_id' => $productId, 'website_id' => $websiteId];
        $sourceItems[] = [
            'source_code' => $sourceCode,
            'sku' => $sku,
            'quantity' => 100.0,
            'status' => 1,
        ];
        $stockItems[] = [
            'product_id' => $productId,
            'stock_id' => 1,
            'qty' => 100.0,
            'is_in_stock' => 1,
            'website_id' => 0,
        ];
        foreach (['name', 'url_key'] as $code) {
            $varcharValues[] = attributeRow(
                $attributeIds[$code],
                $productId,
                (string)$product[$code]
            );
        }
        foreach (['status' => 1, 'visibility' => 4, 'tax_class_id' => 2] as $code => $value) {
            $intValues[] = attributeRow($attributeIds[$code], $productId, $value);
        }
        foreach (['price', 'weight'] as $code) {
            $decimalValues[] = attributeRow(
                $attributeIds[$code],
                $productId,
                (float)$product[$code]
            );
        }
        $textValues[] = attributeRow(
            $attributeIds['description'],
            $productId,
            (string)$product['description']
        );
    }
    $connection->insertMultiple($tables['website'], $websites);
    $connection->insertMultiple($tables['source_item'], $sourceItems);
    $connection->insertMultiple($tables['stock_item'], $stockItems);
    $connection->insertMultiple($tables['varchar'], $varcharValues);
    $connection->insertMultiple($tables['int'], $intValues);
    $connection->insertMultiple($tables['decimal'], $decimalValues);
    $connection->insertMultiple($tables['text'], $textValues);
}

function deterministicProduct(int $ordinal): array
{
    $brands = ['Aster', 'Boreal', 'Cinder', 'Drift', 'Ember', 'Fjord', 'Grove', 'Harbor'];
    $materials = ['Oak', 'Walnut', 'Linen', 'Wool', 'Brass', 'Stone', 'Leather', 'Ceramic'];
    $families = ['Chair', 'Table', 'Lamp', 'Rug', 'Cabinet', 'Bedding', 'Desk', 'Serveware'];
    $rooms = ['living room', 'dining room', 'bedroom', 'office', 'entryway', 'outdoor room'];
    $brand = $brands[($ordinal - 1) % count($brands)];
    $material = $materials[intdiv($ordinal - 1, count($brands)) % count($materials)];
    $family = $families[intdiv($ordinal - 1, count($brands) * count($materials)) % count($families)];
    $room = $rooms[intdiv($ordinal - 1, 17) % count($rooms)];
    $sku = SKU_PREFIX . sprintf('%05d', $ordinal);
    $name = sprintf('%s %s %s %05d', $brand, $material, $family, $ordinal);
    $description = $ordinal % 20 === 0
        ? ''
        : sprintf(
            '%s %s designed for the %s. Deterministic qualification item %05d with '
                . 'bounded material, room, and product-family variation.%s',
            $material,
            strtolower($family),
            $room,
            $ordinal,
            $ordinal % 13 === 0
                ? ' Extended care, dimensions, finish, construction, and placement details '
                    . 'exercise the long-description slice without copying customer content.'
                : ''
        );

    return [
        'sku' => $sku,
        'name' => $name,
        'url_key' => $sku,
        'description' => $description,
        'price' => 19.0 + (($ordinal * 37) % 98_000) / 100,
        'weight' => 0.5 + (($ordinal * 11) % 2500) / 100,
    ];
}

function rollbackFixture(
    \Magento\Framework\DB\Adapter\AdapterInterface $connection,
    array $tables
): void {
    while (true) {
        $rows = $connection->fetchAll(
            $connection->select()
                ->from($tables['product'], ['entity_id', 'sku'])
                ->where('sku LIKE ?', SKU_PREFIX . '%')
                ->order('entity_id ASC')
                ->limit(BATCH_SIZE)
        );
        if ($rows === []) {
            return;
        }
        $ids = array_map(static fn (array $row): int => (int)$row['entity_id'], $rows);
        $skus = array_map(static fn (array $row): string => (string)$row['sku'], $rows);
        $connection->beginTransaction();
        try {
            $connection->delete($tables['source_item'], ['sku IN (?)' => $skus]);
            $connection->delete($tables['stock_item'], ['product_id IN (?)' => $ids]);
            $connection->delete($tables['product'], ['entity_id IN (?)' => $ids]);
            $connection->commit();
        } catch (Throwable $throwable) {
            $connection->rollBack();
            throw $throwable;
        }
        printf("Removed %d registered-catalog products.\n", count($rows));
    }
}

function resolveSourceCode(
    \Magento\Framework\DB\Adapter\AdapterInterface $connection,
    ResourceConnection $resourceConnection,
    int $websiteId
): string {
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
    if ($websiteCode === '' || $sourceCode === '') {
        throw new RuntimeException('The fixture website has no enabled inventory source.');
    }
    return $sourceCode;
}

function resolveAttributeSetId(
    \Magento\Framework\DB\Adapter\AdapterInterface $connection,
    ResourceConnection $resourceConnection
): int {
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
    return $attributeSetId;
}

function resolveAttributeIds(EavConfig $eavConfig): array
{
    $ids = [];
    foreach (
        ['name', 'url_key', 'status', 'visibility', 'tax_class_id', 'price', 'weight', 'description']
        as $code
    ) {
        $attribute = $eavConfig->getAttribute(Product::ENTITY, $code);
        if ((int)$attribute->getId() < 1) {
            throw new RuntimeException(sprintf('Required fixture attribute %s does not exist.', $code));
        }
        $ids[$code] = (int)$attribute->getId();
    }
    return $ids;
}

function attributeRow(int $attributeId, int $productId, mixed $value): array
{
    return [
        'attribute_id' => $attributeId,
        'store_id' => 0,
        'entity_id' => $productId,
        'value' => $value,
    ];
}

function reindex(IndexerRegistry $indexerRegistry): void
{
    foreach (
        ['inventory', 'cataloginventory_stock', 'catalog_product_price', 'catalogsearch_fulltext']
        as $indexerId
    ) {
        $indexerRegistry->get($indexerId)->reindexAll();
    }
}

function canonicalJson(array $value): string
{
    $normalize = static function (mixed $item) use (&$normalize): mixed {
        if (!is_array($item)) {
            return $item;
        }
        if (!array_is_list($item)) {
            ksort($item, SORT_STRING);
        }
        return array_map($normalize, $item);
    };
    return (string)json_encode(
        $normalize($value),
        JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE | JSON_THROW_ON_ERROR
    );
}

function emit(array $value): void
{
    fwrite(
        STDOUT,
        json_encode(
            $value,
            JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE | JSON_THROW_ON_ERROR
        ) . "\n"
    );
}
