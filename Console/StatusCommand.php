<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Console;

class StatusCommand extends \Symfony\Component\Console\Command\Command
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\StatusService $statusService,
        private readonly \Magento\Framework\Serialize\Serializer\Json $json,
        ?string $name = null
    ) {
        parent::__construct($name);
    }

    protected function configure(): void
    {
        $this->setName('mage-os:opensearch-hybrid:status')
            ->setDescription('Show OpenSearch Hybrid generations, readiness, and outbox state')
            ->addOption('store', null, \Symfony\Component\Console\Input\InputOption::VALUE_OPTIONAL, 'Store ID');
    }

    protected function execute(
        \Symfony\Component\Console\Input\InputInterface $input,
        \Symfony\Component\Console\Output\OutputInterface $output
    ): int {
        $rawStore = $input->getOption('store');
        $storeId = $rawStore === null ? null : filter_var($rawStore, FILTER_VALIDATE_INT, ['options' => ['min_range' => 1]]);
        if ($rawStore !== null && $storeId === false) {
            $output->writeln('<error>--store must be a positive integer.</error>');
            return self::INVALID;
        }
        $status = $this->statusService->get($storeId === null ? null : (int)$storeId);
        $output->writeln($this->json->serialize($status));

        return self::SUCCESS;
    }
}
