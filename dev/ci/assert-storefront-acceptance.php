<?php

declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\Config;
use MageOS\OpenSearchHybrid\Model\Generation\GenerationRepository;
use MageOS\OpenSearchHybrid\Model\Reliability\CircuitBreaker;
use MageOS\OpenSearchHybrid\Model\Store\ReadinessLatch;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\Cache\TypeListInterface;
use Magento\Framework\App\Config\ReinitableConfigInterface;
use Magento\Framework\App\Config\Storage\WriterInterface;
use Magento\Framework\App\ResourceConnection;
use Magento\Framework\App\State;

if ($argc !== 5) {
    fwrite(
        STDERR,
        "Usage: php assert-storefront-acceptance.php /path/to/mageos "
        . "http://127.0.0.1:18080 http://127.0.0.1:18081 /path/to/encoder-evidence.jsonl\n"
    );
    exit(2);
}

$fixtureRoot = rtrim((string)$argv[1], DIRECTORY_SEPARATOR);
$baseUrl = rtrim((string)$argv[2], '/');
$encoderBaseUrl = rtrim((string)$argv[3], '/');
$encoderEvidence = (string)$argv[4];
if (!is_file($fixtureRoot . '/app/bootstrap.php')) {
    fwrite(STDERR, "Mage-OS bootstrap not found: {$fixtureRoot}\n");
    exit(2);
}
if (filter_var($baseUrl, FILTER_VALIDATE_URL) === false
    || filter_var($encoderBaseUrl, FILTER_VALIDATE_URL) === false
) {
    fwrite(STDERR, "Provide valid storefront and encoder base URLs.\n");
    exit(2);
}
if (file_put_contents($encoderEvidence, '') === false) {
    fwrite(STDERR, "Could not initialize encoder evidence: {$encoderEvidence}\n");
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
$expectedSku = 'mageos-hybrid-resume-02';
$expectedTitle = 'Resumable Build Product 02';
$encoderBlackhole = 'http://10.255.255.1';
/** @var ResourceConnection $resourceConnection */
$resourceConnection = $objectManager->get(ResourceConnection::class);
$connection = $resourceConnection->getConnection();
$generationTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_generation');
$configTable = $resourceConnection->getTableName('core_config_data');
/** @var GenerationRepository $generationRepository */
$generationRepository = $objectManager->get(GenerationRepository::class);
$active = $generationRepository->activeForStore($storeId);
/** @var ReadinessLatch $readinessLatch */
$readinessLatch = $objectManager->get(ReadinessLatch::class);
if (
    !is_array($active)
    || !(bool)$active['result_contract_accepted']
    || !$readinessLatch->isOpen($storeId)
) {
    throw new RuntimeException('Storefront acceptance requires an accepted active generation with open readiness.');
}
$generationId = (int)$active['generation_id'];
$originalGenerationEndpoint = (string)$active['encoder_endpoint'];

$configPaths = [
    'catalog/search/engine',
    Config::XML_PATH_ENABLED,
    Config::XML_PATH_ENCODER_ALLOWED_HOSTS,
    Config::XML_PATH_FAILURE_THRESHOLD,
];
$originalConfig = [];
foreach ($configPaths as $path) {
    $row = $connection->fetchRow(
        $connection->select()
            ->from($configTable, ['value'])
            ->where('scope = ?', 'default')
            ->where('scope_id = ?', 0)
            ->where('path = ?', $path)
            ->limit(1)
    );
    $originalConfig[$path] = is_array($row) ? (string)$row['value'] : null;
}

/** @var WriterInterface $configWriter */
$configWriter = $objectManager->get(WriterInterface::class);
/** @var TypeListInterface $cacheTypes */
$cacheTypes = $objectManager->get(TypeListInterface::class);
/** @var ReinitableConfigInterface $reinitableConfig */
$reinitableConfig = $objectManager->get(ReinitableConfigInterface::class);

$request = static function (string $url, ?array $payload = null): array {
    $lastError = '';
    for ($attempt = 1; $attempt <= 20; $attempt++) {
        $curl = curl_init($url);
        if ($curl === false) {
            throw new RuntimeException('Could not initialize the storefront HTTP client.');
        }
        $headers = [
            'Accept: text/html,application/json',
            'Host: mageos-hybrid.test',
        ];
        curl_setopt_array($curl, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_FOLLOWLOCATION => false,
            CURLOPT_CONNECTTIMEOUT_MS => 1000,
            CURLOPT_TIMEOUT_MS => 5000,
            CURLOPT_ENCODING => '',
            CURLOPT_HTTPHEADER => $headers,
        ]);
        if ($payload !== null) {
            $encodedPayload = json_encode($payload, JSON_THROW_ON_ERROR);
            $headers[] = 'Content-Type: application/json';
            curl_setopt($curl, CURLOPT_HTTPHEADER, $headers);
            curl_setopt($curl, CURLOPT_POST, true);
            curl_setopt($curl, CURLOPT_POSTFIELDS, $encodedPayload);
        }
        $startedAt = hrtime(true);
        $body = curl_exec($curl);
        $elapsedMs = (hrtime(true) - $startedAt) / 1_000_000;
        $status = (int)curl_getinfo($curl, CURLINFO_RESPONSE_CODE);
        $errorCode = curl_errno($curl);
        $lastError = curl_error($curl);
        if (is_string($body)) {
            return ['status' => $status, 'body' => $body, 'elapsed_ms' => $elapsedMs];
        }
        if ($errorCode !== CURLE_COULDNT_CONNECT) {
            throw new RuntimeException('Storefront HTTP request failed: ' . $lastError);
        }
        if ($attempt < 20) {
            usleep(100_000);
        }
    }

    throw new RuntimeException('Storefront HTTP request failed: ' . $lastError);
};

$assertCatalogResponse = static function (array $response, string $context) use ($expectedTitle): void {
    if ((int)$response['status'] !== 200 || !str_contains((string)$response['body'], $expectedTitle)) {
        throw new RuntimeException(sprintf('%s did not return the expected catalog product.', $context));
    }
    if ((float)$response['elapsed_ms'] > 5000.0) {
        throw new RuntimeException(sprintf('%s exceeded the five-second fixture response bound.', $context));
    }
};

/** @var CircuitBreaker $circuitBreaker */
$circuitBreaker = $objectManager->get(CircuitBreaker::class);
try {
    $configWriter->save('catalog/search/engine', 'mageos_opensearch_hybrid');
    $configWriter->save(Config::XML_PATH_ENABLED, '1');
    $configWriter->save(Config::XML_PATH_ENCODER_ALLOWED_HOSTS, '127.0.0.1');
    $configWriter->save(Config::XML_PATH_FAILURE_THRESHOLD, '1');
    $cacheTypes->cleanType('config');
    $reinitableConfig->reinit();
    $connection->update(
        $generationTable,
        ['encoder_endpoint' => $encoderBaseUrl],
        ['generation_id = ?' => $generationId]
    );
    $circuitBreaker->success($storeId);

    $graphQlHybrid = $request($baseUrl . '/graphql', [
        'query' => '{ products(search: "build product", pageSize: 10)'
            . ' { total_count items { sku name } } }',
    ]);
    $graphQlHybridBody = json_decode((string)$graphQlHybrid['body'], true, 512, JSON_THROW_ON_ERROR);
    $graphQlHybridItems = $graphQlHybridBody['data']['products']['items'] ?? [];
    $graphQlHybridSkus = is_array($graphQlHybridItems) ? array_column($graphQlHybridItems, 'sku') : [];
    $encoderCalls = array_values(array_filter(
        file($encoderEvidence, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES) ?: []
    ));
    if (
        (int)$graphQlHybrid['status'] !== 200
        || isset($graphQlHybridBody['errors'])
        || !in_array($expectedSku, $graphQlHybridSkus, true)
        || $encoderCalls === []
        || $circuitBreaker->isOpen($storeId)
    ) {
        throw new RuntimeException('GraphQL hybrid search did not complete through the query encoder.');
    }

    $lumaHybridCallsBefore = count($encoderCalls);
    $lumaHybrid = $request(
        $baseUrl . '/catalogsearch/result/?q=' . rawurlencode('build product')
        . '&product_list_order=relevance'
    );
    $assertCatalogResponse($lumaHybrid, 'Luma hybrid search');
    $lumaHybridCallsAfter = count(array_values(array_filter(
        file($encoderEvidence, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES) ?: []
    )));
    if ($lumaHybridCallsAfter <= $lumaHybridCallsBefore || $circuitBreaker->isOpen($storeId)) {
        throw new RuntimeException('Luma hybrid search did not complete through the query encoder.');
    }

    $exactSku = $request($baseUrl . '/catalogsearch/result/?q=' . rawurlencode($expectedSku));
    $assertCatalogResponse($exactSku, 'Exact SKU search');
    if ($circuitBreaker->isOpen($storeId)) {
        throw new RuntimeException('Exact SKU search called the unavailable encoder instead of the lexical guard.');
    }

    $exactTitle = $request($baseUrl . '/catalogsearch/result/?q=' . rawurlencode($expectedTitle));
    $assertCatalogResponse($exactTitle, 'Exact title search');
    if ($circuitBreaker->isOpen($storeId)) {
        throw new RuntimeException('Exact title search called the unavailable encoder instead of the lexical guard.');
    }

    $connection->update(
        $generationTable,
        ['encoder_endpoint' => $encoderBlackhole],
        ['generation_id = ?' => $generationId]
    );

    $outageFallback = $request($baseUrl . '/catalogsearch/result/?q=' . rawurlencode('build product'));
    $assertCatalogResponse($outageFallback, 'Encoder outage fallback');
    if (!$circuitBreaker->isOpen($storeId)) {
        throw new RuntimeException('Encoder outage did not open the circuit after the configured failure threshold.');
    }

    $graphQl = $request($baseUrl . '/graphql', [
        'query' => sprintf(
            '{ products(search: "%s", pageSize: 10) { total_count items { sku name } } }',
            $expectedSku
        ),
    ]);
    $graphQlBody = json_decode((string)$graphQl['body'], true, 512, JSON_THROW_ON_ERROR);
    $graphQlItems = $graphQlBody['data']['products']['items'] ?? [];
    $graphQlSkus = is_array($graphQlItems) ? array_column($graphQlItems, 'sku') : [];
    if (
        (int)$graphQl['status'] !== 200
        || isset($graphQlBody['errors'])
        || !in_array($expectedSku, $graphQlSkus, true)
    ) {
        throw new RuntimeException('GraphQL native fallback did not return the expected catalog product.');
    }

    printf(
        "GraphQL hybrid %.1f ms; Luma hybrid %.1f ms, exact SKU %.1f ms, exact title %.1f ms, "
        . "encoder fallback %.1f ms; "
        . "GraphQL fallback returned %s.\n",
        (float)$graphQlHybrid['elapsed_ms'],
        (float)$lumaHybrid['elapsed_ms'],
        (float)$exactSku['elapsed_ms'],
        (float)$exactTitle['elapsed_ms'],
        (float)$outageFallback['elapsed_ms'],
        $expectedSku
    );
} finally {
    $circuitBreaker->success($storeId);
    $connection->update(
        $generationTable,
        ['encoder_endpoint' => $originalGenerationEndpoint],
        ['generation_id = ?' => $generationId]
    );
    foreach ($originalConfig as $path => $value) {
        if ($value === null) {
            $configWriter->delete($path);
        } else {
            $configWriter->save($path, $value);
        }
    }
    $cacheTypes->cleanType('config');
    $reinitableConfig->reinit();
}
