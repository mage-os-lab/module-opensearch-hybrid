<?php

declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Change\SearchableProductResolver;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\ResourceConnection;
use Magento\Framework\App\State;

if ($argc < 3 || $argc > 5) {
    fwrite(
        STDERR,
        "Usage: php assert-large-catalog-graphql.php /path/to/mageos http://127.0.0.1:18080 "
            . "[target-products] [evidence-output]\n"
    );
    exit(2);
}

$fixtureRoot = rtrim((string)$argv[1], DIRECTORY_SEPARATOR);
$baseUrl = rtrim((string)$argv[2], '/');
$targetProducts = isset($argv[3]) ? (int)$argv[3] : 10_000;
$evidenceOutput = isset($argv[4]) ? (string)$argv[4] : '';
if (!is_file($fixtureRoot . '/app/bootstrap.php')) {
    fwrite(STDERR, "Mage-OS bootstrap not found: {$fixtureRoot}\n");
    exit(2);
}
$urlParts = parse_url($baseUrl);
if (
    !is_array($urlParts)
    || ($urlParts['scheme'] ?? '') !== 'http'
    || !in_array((string)($urlParts['host'] ?? ''), ['127.0.0.1', 'localhost', '::1'], true)
) {
    fwrite(STDERR, "The large-catalog storefront URL must use HTTP on loopback.\n");
    exit(2);
}
if (
    $targetProducts < 100
    || $targetProducts > 1_000_000
    || ($targetProducts > 10_000 && !in_array($targetProducts, [100_000, 1_000_000], true))
) {
    fwrite(STDERR, "Target products must be 100 to 10000, 100000, or 1000000.\n");
    exit(2);
}
if ($targetProducts > 10_000 && (
    $evidenceOutput === ''
    || !str_starts_with($evidenceOutput, DIRECTORY_SEPARATOR)
    || !is_file($evidenceOutput)
    || is_link($evidenceOutput)
)) {
    fwrite(STDERR, "Phase 5 GraphQL qualification requires the scale build evidence file.\n");
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
/** @var SearchableProductResolver $productResolver */
$productResolver = $objectManager->get(SearchableProductResolver::class);
$eligibleProducts = $productResolver->countEligible($storeId);
if ($eligibleProducts !== $targetProducts) {
    throw new RuntimeException(sprintf(
        'GraphQL qualification found %d eligible products instead of %d.',
        $eligibleProducts,
        $targetProducts
    ));
}

/** @var ResourceConnection $resourceConnection */
$resourceConnection = $objectManager->get(ResourceConnection::class);
$connection = $resourceConnection->getConnection();
$fixtureSkus = array_map('strval', $connection->fetchCol(
    $connection->select()
        ->from($resourceConnection->getTableName('catalog_product_entity'), ['sku'])
        ->where('sku LIKE ?', 'mageos-hybrid-large-%')
        ->order('sku ASC')
));
$fixtureCount = count($fixtureSkus);
if ($fixtureCount < 4 || $fixtureCount > $targetProducts) {
    throw new RuntimeException('The deterministic large-catalog GraphQL slice is incomplete or ambiguous.');
}

$requestTimeoutMs = $targetProducts > 10_000 ? 60_000 : 15_000;
$request = static function (string $url, string $query) use ($requestTimeoutMs): array {
    $lastError = '';
    for ($attempt = 1; $attempt <= 20; $attempt++) {
        $curl = curl_init($url);
        if ($curl === false) {
            throw new RuntimeException('Could not initialize the GraphQL HTTP client.');
        }
        curl_setopt_array($curl, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_FOLLOWLOCATION => false,
            CURLOPT_CONNECTTIMEOUT_MS => 1000,
            CURLOPT_TIMEOUT_MS => $requestTimeoutMs,
            CURLOPT_HTTPHEADER => [
                'Accept: application/json',
                'Content-Type: application/json',
                'Host: mageos-hybrid.test',
            ],
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode(['query' => $query], JSON_THROW_ON_ERROR),
        ]);
        $startedAt = hrtime(true);
        $body = curl_exec($curl);
        $elapsedMs = (hrtime(true) - $startedAt) / 1_000_000;
        $status = (int)curl_getinfo($curl, CURLINFO_RESPONSE_CODE);
        $errorCode = curl_errno($curl);
        $lastError = curl_error($curl);
        if (is_string($body)) {
            if ($status !== 200) {
                throw new RuntimeException(sprintf('GraphQL returned HTTP %d: %s', $status, $body));
            }
            $decoded = json_decode($body, true, 512, JSON_THROW_ON_ERROR);
            if (isset($decoded['errors']) || !is_array($decoded['data']['products'] ?? null)) {
                throw new RuntimeException('GraphQL returned errors or omitted the products result.');
            }

            return ['products' => $decoded['data']['products'], 'elapsed_ms' => $elapsedMs];
        }
        if ($errorCode !== CURLE_COULDNT_CONNECT) {
            throw new RuntimeException('GraphQL HTTP request failed: ' . $lastError);
        }
        if ($attempt < 20) {
            usleep(100_000);
        }
    }

    throw new RuntimeException('GraphQL HTTP request failed: ' . $lastError);
};

$query = static function (int $page, ?string $priceFilter = null): string {
    $filter = $priceFilter === null ? '' : sprintf(', filter: { price: %s }', $priceFilter);

    return sprintf(
        <<<'GRAPHQL'
{ products(
    search: "qualification"
    pageSize: 2
    currentPage: %d
    sort: { name: ASC }
    %s
) {
    total_count
    page_info { current_page page_size total_pages }
    items {
        sku
        name
        price_range { minimum_price { regular_price { value currency } } }
    }
    aggregations { attribute_code count options { count label value } }
} }
GRAPHQL,
        $page,
        $filter
    );
};

$assertPage = static function (
    array $products,
    int $expectedTotal,
    int $expectedPage,
    array $expectedSkus
): void {
    $pageInfo = $products['page_info'] ?? [];
    $items = $products['items'] ?? [];
    $actualSkus = is_array($items) ? array_map('strval', array_column($items, 'sku')) : [];
    if (
        (int)($products['total_count'] ?? -1) !== $expectedTotal
        || (int)($pageInfo['current_page'] ?? -1) !== $expectedPage
        || (int)($pageInfo['page_size'] ?? -1) !== 2
        || (int)($pageInfo['total_pages'] ?? -1) !== (int)ceil($expectedTotal / 2)
        || $actualSkus !== $expectedSkus
    ) {
        throw new RuntimeException(sprintf('GraphQL page %d did not preserve the native result contract.', $expectedPage));
    }
    foreach ($items as $item) {
        $price = $item['price_range']['minimum_price']['regular_price'] ?? [];
        if ((float)($price['value'] ?? 0.0) !== 19.99 || (string)($price['currency'] ?? '') !== 'USD') {
            throw new RuntimeException('GraphQL returned an unexpected large-catalog price dimension.');
        }
    }
};

$pageOne = $request($baseUrl . '/graphql', $query(1));
$pageTwo = $request($baseUrl . '/graphql', $query(2));
$assertPage($pageOne['products'], $fixtureCount, 1, array_slice($fixtureSkus, 0, 2));
$assertPage($pageTwo['products'], $fixtureCount, 2, array_slice($fixtureSkus, 2, 2));

$priceAggregation = null;
foreach ($pageOne['products']['aggregations'] ?? [] as $aggregation) {
    if (($aggregation['attribute_code'] ?? null) === 'price') {
        $priceAggregation = $aggregation;
        break;
    }
}
$facetCount = 0;
foreach ($priceAggregation['options'] ?? [] as $option) {
    $facetCount += (int)($option['count'] ?? 0);
}
if (!is_array($priceAggregation) || $facetCount !== $fixtureCount) {
    throw new RuntimeException('GraphQL price aggregations do not account for the result set.');
}

$included = $request($baseUrl . '/graphql', $query(1, '{ from: "19", to: "20" }'));
$assertPage($included['products'], $fixtureCount, 1, array_slice($fixtureSkus, 0, 2));
$excluded = $request($baseUrl . '/graphql', $query(1, '{ from: "20", to: "21" }'));
if (
    (int)($excluded['products']['total_count'] ?? -1) !== 0
    || ($excluded['products']['items'] ?? null) !== []
) {
    throw new RuntimeException('GraphQL price filtering did not exclude the deterministic result set.');
}

if ($evidenceOutput !== '') {
    $evidenceBytes = file_get_contents($evidenceOutput);
    $evidence = is_string($evidenceBytes)
        ? json_decode($evidenceBytes, true, 512, JSON_THROW_ON_ERROR)
        : null;
    if (
        !is_array($evidence)
        || ($evidence['schema_version'] ?? null) !== 2
        || ($evidence['status'] ?? null) !== 'build_passed'
        || (int)($evidence['catalog']['target_products'] ?? 0) !== $targetProducts
        || (int)($evidence['catalog']['fixture_products'] ?? 0) !== $fixtureCount
    ) {
        throw new RuntimeException('GraphQL qualification does not match the scale build evidence.');
    }
    $evidence['status'] = 'passed';
    $evidence['graphql'] = [
        'fixture_total' => $fixtureCount,
        'page_one_ms' => round((float)$pageOne['elapsed_ms'], 3),
        'page_two_ms' => round((float)$pageTwo['elapsed_ms'], 3),
        'included_price_filter_ms' => round((float)$included['elapsed_ms'], 3),
        'excluded_price_filter_ms' => round((float)$excluded['elapsed_ms'], 3),
        'stable_pages' => true,
        'price_filter' => true,
        'price_aggregation' => true,
    ];
    $temporaryEvidence = $evidenceOutput . '.tmp-' . getmypid();
    $handle = fopen($temporaryEvidence, 'x');
    if ($handle === false) {
        throw new RuntimeException('Could not create the temporary GraphQL evidence file.');
    }
    try {
        $encodedEvidence = json_encode(
            $evidence,
            JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_THROW_ON_ERROR
        ) . "\n";
        if (fwrite($handle, $encodedEvidence) !== strlen($encodedEvidence)) {
            throw new RuntimeException('Could not write complete GraphQL evidence.');
        }
    } finally {
        fclose($handle);
    }
    if (!rename($temporaryEvidence, $evidenceOutput)) {
        throw new RuntimeException('Could not publish the completed scale evidence file.');
    }
}

printf(
    "GraphQL preserved %d-product totals, two stable pages, price filtering, and aggregations "
        . "in %.1f/%.1f/%.1f/%.1f ms.\n",
    $fixtureCount,
    (float)$pageOne['elapsed_ms'],
    (float)$pageTwo['elapsed_ms'],
    (float)$included['elapsed_ms'],
    (float)$excluded['elapsed_ms']
);
