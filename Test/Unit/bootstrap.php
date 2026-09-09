<?php
declare(strict_types=1);

$mageOsRoot = getenv('MAGEOS_ROOT');
if (!is_string($mageOsRoot) || $mageOsRoot === '') {
    throw new RuntimeException('Set MAGEOS_ROOT to a Mage-OS 3.4 checkout before running module unit tests.');
}
require $mageOsRoot . '/app/autoload.php';

spl_autoload_register(static function (string $class): void {
    $prefix = 'MageOS\\OpenSearchHybrid\\';
    if (!str_starts_with($class, $prefix)) {
        return;
    }
    $relative = substr($class, strlen($prefix));
    $path = dirname(__DIR__, 2) . '/' . str_replace('\\', '/', $relative) . '.php';
    if (is_file($path)) {
        require $path;
    }
}, true, true);
