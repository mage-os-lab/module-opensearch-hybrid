<?php

declare(strict_types=1);

if ($argc !== 3) {
    fwrite(STDERR, "Usage: php assert-package-version.php /path/to/mageos expected-version\n");
    exit(2);
}

$mageOsRoot = rtrim((string)$argv[1], DIRECTORY_SEPARATOR);
$expectedVersion = ltrim((string)$argv[2], 'v');
$lockPath = $mageOsRoot . '/composer.lock';
if (!is_file($lockPath)) {
    fwrite(STDERR, "Composer lock not found: {$lockPath}\n");
    exit(2);
}

$lock = json_decode((string)file_get_contents($lockPath), true, flags: JSON_THROW_ON_ERROR);
$packages = array_merge($lock['packages'] ?? [], $lock['packages-dev'] ?? []);
foreach ($packages as $package) {
    if (($package['name'] ?? null) !== 'mage-os/product-community-edition') {
        continue;
    }
    $actualVersion = ltrim((string)($package['version'] ?? ''), 'v');
    if ($actualVersion !== $expectedVersion) {
        throw new RuntimeException(
            "Expected mage-os/product-community-edition {$expectedVersion}, found {$actualVersion}."
        );
    }
    printf("Verified mage-os/product-community-edition %s in composer.lock.\n", $actualVersion);
    exit(0);
}

throw new RuntimeException('mage-os/product-community-edition is missing from composer.lock.');
