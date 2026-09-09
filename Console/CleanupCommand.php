<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Console;

use Symfony\Component\Console\Input\InputInterface;
use Symfony\Component\Console\Input\InputOption;
use Symfony\Component\Console\Output\OutputInterface;

class CleanupCommand extends \Symfony\Component\Console\Command\Command
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Generation\GenerationCleanupService $cleanupService,
        private readonly \Magento\Framework\Serialize\Serializer\Json $json,
        ?string $name = null
    ) {
        parent::__construct($name);
    }

    protected function configure(): void
    {
        $this->setName('mage-os:opensearch-hybrid:cleanup')
            ->setDescription('Preview or apply cleanup of one non-active hybrid generation')
            ->addOption('store', null, InputOption::VALUE_REQUIRED, 'Store ID')
            ->addOption('generation', null, InputOption::VALUE_REQUIRED, 'Generation ID')
            ->addOption('dry-run', null, InputOption::VALUE_NONE, 'Print exact impact without changing state')
            ->addOption(
                'confirm',
                null,
                InputOption::VALUE_OPTIONAL,
                'Exact confirmation token emitted by the current dry-run preview'
            );
    }

    protected function execute(InputInterface $input, OutputInterface $output): int
    {
        $storeId = $this->positiveInt($input->getOption('store'));
        $generationId = $this->positiveInt($input->getOption('generation'));
        if ($storeId === null || $generationId === null) {
            $output->writeln('<error>Provide one explicit positive --store and --generation ID.</error>');
            return self::INVALID;
        }
        $dryRun = (bool)$input->getOption('dry-run');
        $confirmation = $input->getOption('confirm');
        if ($dryRun === ($confirmation !== null)) {
            $output->writeln('<error>Use --dry-run for preview or --confirm with its exact token for apply.</error>');
            return self::INVALID;
        }

        $report = $dryRun
            ? $this->cleanupService->preview($storeId, $generationId)
            : $this->cleanupService->cleanup($storeId, $generationId, (string)$confirmation);
        $output->writeln($this->json->serialize($report));

        return self::SUCCESS;
    }

    private function positiveInt(mixed $value): ?int
    {
        $validated = filter_var($value, FILTER_VALIDATE_INT, ['options' => ['min_range' => 1]]);

        return $validated === false ? null : (int)$validated;
    }
}
