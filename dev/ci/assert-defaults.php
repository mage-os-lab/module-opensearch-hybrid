<?php

declare(strict_types=1);

use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\Config\ScopeConfigInterface;

if ($argc !== 2) {
    fwrite(STDERR, "Usage: php assert-defaults.php /path/to/mageos\n");
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
$scopeConfig = $bootstrap->getObjectManager()->get(ScopeConfigInterface::class);
$enabled = $scopeConfig->getValue(
    'mageos_opensearch_hybrid/general/enabled',
    ScopeConfigInterface::SCOPE_TYPE_DEFAULT
);

if ((string)$enabled !== '0') {
    fwrite(STDERR, "OpenSearch Hybrid must remain disabled by default.\n");
    exit(1);
}

fwrite(STDOUT, "OpenSearch Hybrid is disabled by default.\n");
