<?php
declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Document\PriceIndexResolver;
use MageOS\OpenSearchHybrid\Model\Generation\ValidationService;
use MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository;
use MageOS\OpenSearchHybrid\Model\Queue\PriorityConsumer;
use Magento\Catalog\Api\ProductRepositoryInterface;
use Magento\Catalog\Model\Indexer\Product\Price\DimensionModeConfiguration;
use Magento\Customer\Model\Indexer\CustomerGroupDimensionProvider;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\ResourceConnection;
use Magento\Framework\App\State;
use Magento\Store\Model\Indexer\WebsiteDimensionProvider;

$fixtureRoot = $argv[1] ?? '';
$mode = $argv[2] ?? '';
if ($fixtureRoot === '' || !is_file($fixtureRoot . '/app/bootstrap.php')) {
    fwrite(STDERR, "Provide the installed Mage-OS fixture root.\n");
    exit(2);
}
require $fixtureRoot . '/app/bootstrap.php';
$expectedDimensions = [
    DimensionModeConfiguration::DIMENSION_NONE => [],
    DimensionModeConfiguration::DIMENSION_WEBSITE => [WebsiteDimensionProvider::DIMENSION_NAME],
    DimensionModeConfiguration::DIMENSION_CUSTOMER_GROUP => [CustomerGroupDimensionProvider::DIMENSION_NAME],
    DimensionModeConfiguration::DIMENSION_WEBSITE_AND_CUSTOMER_GROUP => [
        WebsiteDimensionProvider::DIMENSION_NAME,
        CustomerGroupDimensionProvider::DIMENSION_NAME,
    ],
];
if (!array_key_exists($mode, $expectedDimensions)) {
    throw new RuntimeException('Provide a supported price dimension mode.');
}
$bootstrap = Bootstrap::create(BP, $_SERVER);
$objectManager = $bootstrap->getObjectManager();
try {
    $objectManager->get(State::class)->setAreaCode('global');
} catch (\Magento\Framework\Exception\LocalizedException) {
    // Another bootstrap participant already set the area code.
}

/** @var DimensionModeConfiguration $modeConfiguration */
$modeConfiguration = $objectManager->get(DimensionModeConfiguration::class);
if ($modeConfiguration->getDimensionConfiguration() !== $expectedDimensions[$mode]) {
    throw new RuntimeException('Magento did not activate the requested price dimension mode.');
}
/** @var ResourceConnection $resourceConnection */
$resourceConnection = $objectManager->get(ResourceConnection::class);
$connection = $resourceConnection->getConnection();
$generationTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_generation');
$outboxTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_outbox');
$generationId = (int)$connection->fetchOne(
    $connection->select()
        ->from($generationTable, ['generation_id'])
        ->where('store_id = ?', 1)
        ->where('state IN (?)', ['READY', 'CATCHING_UP'])
        ->order('generation_id DESC')
        ->limit(1)
);
if ($generationId <= 0) {
    throw new RuntimeException('The price dimension fixture found no writable generation.');
}
$priorityJobIds = array_map('strval', $connection->fetchCol(
    $connection->select()
        ->from($outboxTable, ['job_id'])
        ->where('lane = ?', 'PRIORITY')
        ->where('state NOT IN (?)', ['COMPLETED', 'SUPERSEDED'])
        ->order('latest_change_id ASC')
));
if ($priorityJobIds === []) {
    throw new RuntimeException('Full dimension-mode price reindex created no priority work.');
}
/** @var OutboxRepository $outboxRepository */
$outboxRepository = $objectManager->get(OutboxRepository::class);
/** @var PriorityConsumer $priorityConsumer */
$priorityConsumer = $objectManager->get(PriorityConsumer::class);
foreach ($priorityJobIds as $jobId) {
    if ((string)$outboxRepository->get($jobId)['state'] === 'PENDING') {
        $outboxRepository->markPublished($jobId);
    }
    $priorityConsumer->process($jobId);
}
/** @var ValidationService $validationService */
$validationService = $objectManager->get(ValidationService::class);
if (!(bool)$validationService->validate($generationId)['ready']) {
    throw new RuntimeException('Dimension-mode price catch-up did not return the generation to READY.');
}
/** @var ProductRepositoryInterface $productRepository */
$productRepository = $objectManager->get(ProductRepositoryInterface::class);
$product = $productRepository->get('mageos-hybrid-resume-02', false, 1, true);
/** @var PriceIndexResolver $priceResolver */
$priceResolver = $objectManager->get(PriceIndexResolver::class);
$priceRows = $priceResolver->resolve((int)$product->getId(), 1);
if (count($priceRows) !== 4) {
    throw new RuntimeException('The price resolver did not return all customer groups.');
}
$customerGroupIds = [];
foreach ($priceRows as $priceRow) {
    $customerGroupIds[] = (int)$priceRow['customer_group_id'];
    if ((int)$priceRow['website_id'] !== 1 || (float)$priceRow['tier_price'] !== 6.5) {
        throw new RuntimeException('The dimension-scoped price row is incorrect.');
    }
}
if (count(array_unique($customerGroupIds)) !== 4) {
    throw new RuntimeException('The price resolver returned duplicate customer groups.');
}

printf("Price dimension mode %s resolved four canonical customer-group rows.\n", $mode);
