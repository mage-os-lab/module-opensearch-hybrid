<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Console;

class ReconcileCommand extends \Symfony\Component\Console\Command\Command
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Outbox\ReconciliationService $reconciliationService,
        private readonly \Magento\Framework\Serialize\Serializer\Json $json,
        ?string $name = null
    ) {
        parent::__construct($name);
    }

    protected function configure(): void
    {
        $this->setName('mage-os:opensearch-hybrid:reconcile')
            ->setDescription('Repair subscriptions, create eligible replacements, and republish stale work')
            ->addOption('store', null, \Symfony\Component\Console\Input\InputOption::VALUE_OPTIONAL, 'Store ID')
            ->addOption(
                'all-stores',
                null,
                \Symfony\Component\Console\Input\InputOption::VALUE_NONE,
                'Explicitly reconcile all stores'
            );
    }

    protected function execute(
        \Symfony\Component\Console\Input\InputInterface $input,
        \Symfony\Component\Console\Output\OutputInterface $output
    ): int {
        $rawStore = $input->getOption('store');
        $allStores = (bool)$input->getOption('all-stores');
        if (($rawStore === null && !$allStores) || ($rawStore !== null && $allStores)) {
            $output->writeln('<error>Provide exactly one --store or the explicit --all-stores option.</error>');
            return self::INVALID;
        }
        $storeId = $rawStore === null
            ? null
            : filter_var($rawStore, FILTER_VALIDATE_INT, ['options' => ['min_range' => 1]]);
        if ($storeId === false) {
            $output->writeln('<error>--store must be a positive integer.</error>');
            return self::INVALID;
        }
        $report = $this->reconciliationService->reconcile($storeId === null ? null : (int)$storeId);
        $output->writeln($this->json->serialize($report));

        return $report['failed_publications'] === 0 && $report['failed_replacements'] === 0
            ? self::SUCCESS
            : self::FAILURE;
    }
}
