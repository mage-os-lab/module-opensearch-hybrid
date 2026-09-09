<?php
declare(strict_types=1);

use MageOS\OpenSearchHybrid\Console\ActivateCommand;
use MageOS\OpenSearchHybrid\Console\RollbackCommand;
use MageOS\OpenSearchHybrid\Model\Config;
use Magento\Framework\App\Bootstrap;
use Magento\Framework\App\Cache\TypeListInterface;
use Magento\Framework\App\Config\Storage\WriterInterface;
use Magento\Framework\App\ResourceConnection;
use Magento\Framework\App\State;
use Symfony\Component\Console\Tester\CommandTester;

$fixtureRoot = $argv[1] ?? '';
if ($fixtureRoot === '' || !is_file($fixtureRoot . '/app/bootstrap.php')) {
    fwrite(STDERR, "Provide the installed Mage-OS fixture root.\n");
    exit(2);
}

require $fixtureRoot . '/app/bootstrap.php';
$bootstrap = Bootstrap::create(BP, $_SERVER);
$objectManager = $bootstrap->getObjectManager();
try {
    $objectManager->get(State::class)->setAreaCode('global');
} catch (\Magento\Framework\Exception\LocalizedException) {
    // Another bootstrap participant already set the area code.
}

/** @var ResourceConnection $resourceConnection */
$resourceConnection = $objectManager->get(ResourceConnection::class);
$connection = $resourceConnection->getConnection();
$generationTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_generation');
$stateTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_store_state');
$activationTable = $resourceConnection->getTableName('mageos_opensearch_hybrid_activation');
$fixtureState = $connection->fetchRow(
    $connection->select()->from($stateTable)->where('store_id = ?', 1)
);
if (!is_array($fixtureState)
    || (string)$fixtureState['activation_mode'] !== 'HYBRID'
    || $fixtureState['active_generation_id'] === null
) {
    throw new RuntimeException('The CLI lifecycle fixture requires one active hybrid store pointer.');
}
$fixtureGenerationId = (int)$fixtureState['active_generation_id'];
$productionGenerationId = (int)$connection->fetchOne(
    $connection->select()
        ->from($generationTable, ['generation_id'])
        ->where('store_id = ?', 1)
        ->where('is_fake = ?', 0)
        ->where('state = ?', 'READY')
        ->order('generation_id DESC')
        ->limit(1)
);
if ($productionGenerationId < 1) {
    throw new RuntimeException('The CLI lifecycle fixture requires one ready production generation.');
}
$productionEndpoint = (string)$connection->fetchOne(
    $connection->select()
        ->from($generationTable, ['encoder_endpoint'])
        ->where('generation_id = ?', $productionGenerationId)
);
if ($productionEndpoint === '') {
    throw new RuntimeException('The production CLI lifecycle generation has no encoder endpoint.');
}
/** @var WriterInterface $configWriter */
$configWriter = $objectManager->get(WriterInterface::class);
/** @var TypeListInterface $cacheTypes */
$cacheTypes = $objectManager->get(TypeListInterface::class);
$configWriter->save(Config::XML_PATH_ENCODER_ENDPOINT, $productionEndpoint);
$cacheTypes->cleanType('config');

$runCommand = static function (string $commandClass, array $arguments) use ($objectManager): array {
    $tester = new CommandTester($objectManager->create($commandClass));
    $exitCode = $tester->execute($arguments);
    if ($exitCode !== 0) {
        throw new RuntimeException(sprintf(
            'CLI command %s exited %d: %s',
            $commandClass,
            $exitCode,
            trim($tester->getDisplay())
        ));
    }
    $report = json_decode(trim($tester->getDisplay()), true, 512, JSON_THROW_ON_ERROR);
    if (!is_array($report)) {
        throw new RuntimeException(sprintf('CLI command %s did not return a JSON object.', $commandClass));
    }

    return $report;
};

$auditCountBefore = (int)$connection->fetchOne(
    $connection->select()->from($activationTable, [new Zend_Db_Expr('COUNT(*)')])
);
$activationPreview = $runCommand(ActivateCommand::class, [
    '--store' => '1',
    '--generation' => (string)$productionGenerationId,
    '--dry-run' => true,
]);
if ((string)$activationPreview['operation'] !== 'activate'
    || (int)$activationPreview['store_id'] !== 1
    || (int)$activationPreview['target']['generation']['generation_id'] !== $productionGenerationId
    || !is_string($activationPreview['confirmation_token'])
) {
    throw new RuntimeException('The activation CLI preview omitted exact target evidence.');
}
try {
    $runCommand(ActivateCommand::class, [
        '--store' => '1',
        '--generation' => (string)$productionGenerationId,
        '--confirm' => 'activate-1-' . str_repeat('0', 64),
    ]);
    throw new RuntimeException('The activation CLI accepted a stale confirmation token.');
} catch (InvalidArgumentException $exception) {
    if (!str_contains($exception->getMessage(), 'current exact preview')) {
        throw $exception;
    }
}
$unchangedState = $connection->fetchRow(
    $connection->select()->from($stateTable)->where('store_id = ?', 1)
);
if ((int)$connection->fetchOne(
    $connection->select()->from($activationTable, [new Zend_Db_Expr('COUNT(*)')])
) !== $auditCountBefore
    || !is_array($unchangedState)
    || (int)$unchangedState['active_generation_id'] !== $fixtureGenerationId
) {
    throw new RuntimeException('A rejected CLI activation changed state or wrote an audit.');
}
$activationReport = $runCommand(ActivateCommand::class, [
    '--store' => '1',
    '--generation' => (string)$productionGenerationId,
    '--confirm' => (string)$activationPreview['confirmation_token'],
]);

$nativePreview = $runCommand(RollbackCommand::class, [
    '--store' => '1',
    '--dry-run' => true,
]);
$nativeReport = $runCommand(RollbackCommand::class, [
    '--store' => '1',
    '--confirm' => (string)$nativePreview['confirmation_token'],
]);

$generationPreview = $runCommand(RollbackCommand::class, [
    '--store' => '1',
    '--generation' => (string)$productionGenerationId,
    '--dry-run' => true,
]);
$generationReport = $runCommand(RollbackCommand::class, [
    '--store' => '1',
    '--generation' => (string)$productionGenerationId,
    '--confirm' => (string)$generationPreview['confirmation_token'],
]);

$finalState = $connection->fetchRow(
    $connection->select()->from($stateTable)->where('store_id = ?', 1)
);
$auditIds = [
    (int)$activationReport['activation_id'],
    (int)$nativeReport['activation_id'],
    (int)$generationReport['activation_id'],
];
$cliAudits = $connection->fetchAll(
    $connection->select()
        ->from($activationTable, ['activation_id', 'actor', 'status'])
        ->where('activation_id IN (?)', $auditIds)
        ->order('activation_id ASC')
);
if ((string)$activationReport['status'] !== 'activated'
    || (string)$nativeReport['status'] !== 'rolled_back_to_native'
    || (string)$generationReport['status'] !== 'rolled_back_to_generation'
    || !is_array($finalState)
    || (string)$finalState['activation_mode'] !== 'HYBRID'
    || (int)$finalState['active_generation_id'] !== $productionGenerationId
    || count($cliAudits) !== 3
    || count(array_filter(
        $cliAudits,
        static fn (array $audit): bool => (string)$audit['actor'] === 'cli'
            && (string)$audit['status'] === 'COMPLETED'
    )) !== 3
) {
    throw new RuntimeException('The CLI lifecycle did not apply its three exact audited transitions.');
}

$configWriter->delete(Config::XML_PATH_ENCODER_ENDPOINT);
$cacheTypes->cleanType('config');
$connection->beginTransaction();
try {
    $connection->update(
        $generationTable,
        ['state' => 'READY'],
        ['generation_id = ?' => $productionGenerationId, 'state = ?' => 'ACTIVE']
    );
    $connection->update(
        $generationTable,
        ['state' => 'ACTIVE'],
        ['generation_id = ?' => $fixtureGenerationId, 'state = ?' => 'RETAINED']
    );
    $connection->update(
        $stateTable,
        [
            'activation_mode' => (string)$fixtureState['activation_mode'],
            'active_generation_id' => $fixtureGenerationId,
            'required_watermark' => (int)$fixtureState['required_watermark'],
            'native_watermark' => (int)$fixtureState['native_watermark'],
            'hybrid_watermark' => (int)$fixtureState['hybrid_watermark'],
            'readiness_latch' => (int)$fixtureState['readiness_latch'],
            'cache_version' => new Zend_Db_Expr('cache_version + 1'),
        ],
        ['store_id = ?' => 1]
    );
    $connection->commit();
} catch (Throwable $throwable) {
    $connection->rollBack();
    throw $throwable;
}

$restoredState = $connection->fetchRow(
    $connection->select()->from($stateTable)->where('store_id = ?', 1)
);
if (!is_array($restoredState)
    || (string)$restoredState['activation_mode'] !== 'HYBRID'
    || (int)$restoredState['active_generation_id'] !== $fixtureGenerationId
    || (string)$connection->fetchOne(
        $connection->select()
            ->from($generationTable, ['state'])
            ->where('generation_id = ?', $productionGenerationId)
    ) !== 'READY'
) {
    throw new RuntimeException('The CLI lifecycle fixture did not restore the later probe baseline.');
}

printf(
    "Generation %d required state-bound CLI confirmation for activation and both rollback paths.\n",
    $productionGenerationId
);
