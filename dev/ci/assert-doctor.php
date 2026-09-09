<?php

declare(strict_types=1);

use MageOS\OpenSearchHybrid\Model\DoctorService;
use MageOS\OpenSearchHybrid\Model\OpenSearch\VersionPolicy;
use Magento\Framework\App\Bootstrap;

if ($argc !== 2) {
    fwrite(STDERR, "Usage: php assert-doctor.php /path/to/mageos\n");
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
$result = $bootstrap->getObjectManager()->get(DoctorService::class)->diagnose();
$checks = $result['checks'] ?? [];

$rabbitMq = $checks['rabbitmq'] ?? [];
if (($rabbitMq['ok'] ?? false) !== true || ($rabbitMq['value'] ?? null) !== 'configured') {
    fwrite(STDERR, "Doctor did not detect the configured RabbitMQ connection.\n");
    exit(1);
}

$openSearch = $checks['opensearch_version'] ?? [];
if (
    ($openSearch['ok'] ?? false) !== true
    || ($openSearch['value'] ?? null) !== '3.8.0'
    || ($openSearch['required'] ?? null) !== VersionPolicy::SUPPORTED_RANGE
) {
    fwrite(STDERR, "Doctor did not accept the OpenSearch 3.8 fixture.\n");
    exit(1);
}

$base64 = $checks['base64_vector_ingestion']['value'] ?? [];
if (($checks['base64_vector_ingestion']['ok'] ?? false) !== true
    || ($base64['qualified'] ?? false) !== true
    || ($base64['current_runtime_exactly_qualified'] ?? false) !== true
) {
    fwrite(STDERR, "Doctor did not report the exact registered Base64 qualification.\n");
    exit(1);
}

$radial = $checks['radial_similarity'] ?? [];
$radialValues = $radial['value'] ?? null;
$radialDisabled = is_array($radialValues);
if ($radialDisabled) {
    foreach ($radialValues as $radialValue) {
        if (!is_array($radialValue) || ($radialValue['state'] ?? null) !== 'DISABLED') {
            $radialDisabled = false;
            break;
        }
    }
}
if (($radial['ok'] ?? false) !== true || !$radialDisabled) {
    fwrite(STDERR, "Doctor did not report the safe disabled radial state.\n");
    exit(1);
}

fwrite(STDOUT, "Doctor detected RabbitMQ, accepted OpenSearch 3.8, and reported safe radial state.\n");
