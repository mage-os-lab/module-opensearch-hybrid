<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Console;

class EventsInstallCommand extends \Symfony\Component\Console\Command\Command
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\AsyncEvent\SubscriptionReconciler $reconciler,
        private readonly \Magento\Framework\Serialize\Serializer\Json $json,
        private readonly \Magento\Store\Model\StoreManagerInterface $storeManager,
        ?string $name = null
    ) {
        parent::__construct($name);
    }

    protected function configure(): void
    {
        $this->setName('mage-os:opensearch-hybrid:events:install')
            ->setDescription('Preview or reconcile the module-owned Async Events subscription for one store')
            ->addOption('store', null, \Symfony\Component\Console\Input\InputOption::VALUE_REQUIRED, 'Store ID')
            ->addOption(
                'dry-run',
                null,
                \Symfony\Component\Console\Input\InputOption::VALUE_NONE,
                'Print the exact subscription reconciliation preview without changing state'
            )
            ->addOption(
                'confirm',
                null,
                \Symfony\Component\Console\Input\InputOption::VALUE_OPTIONAL,
                'Exact confirmation token emitted by the current dry-run preview'
            );
    }

    protected function execute(
        \Symfony\Component\Console\Input\InputInterface $input,
        \Symfony\Component\Console\Output\OutputInterface $output
    ): int {
        $storeId = filter_var($input->getOption('store'), FILTER_VALIDATE_INT, ['options' => ['min_range' => 1]]);
        if ($storeId === false) {
            $output->writeln('<error>Provide one explicit positive --store ID.</error>');
            return self::INVALID;
        }
        $dryRun = (bool)$input->getOption('dry-run');
        $confirmation = $input->getOption('confirm');
        if ($dryRun === ($confirmation !== null)) {
            $output->writeln('<error>Use --dry-run for preview or --confirm with its exact token for apply.</error>');
            return self::INVALID;
        }
        try {
            $this->storeManager->getStore((int)$storeId);
        } catch (\Magento\Framework\Exception\NoSuchEntityException) {
            $output->writeln(sprintf('<error>Store %d does not exist.</error>', $storeId));
            return self::INVALID;
        }
        $report = $dryRun
            ? $this->reconciler->preview((int)$storeId)
            : $this->reconciler->reconcileConfirmed((int)$storeId, (string)$confirmation);
        $output->writeln($this->json->serialize($report));

        return self::SUCCESS;
    }
}
