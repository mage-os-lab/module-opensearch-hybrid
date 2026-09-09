<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Console;

class ActivateCommand extends \Symfony\Component\Console\Command\Command
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Activation\ActivationService $activationService,
        private readonly \Magento\Framework\Serialize\Serializer\Json $json,
        ?string $name = null
    ) {
        parent::__construct($name);
    }

    protected function configure(): void
    {
        $this->setName('mage-os:opensearch-hybrid:activate')
            ->setDescription('Preview or atomically activate one validated production generation for one store')
            ->addOption('store', null, \Symfony\Component\Console\Input\InputOption::VALUE_REQUIRED, 'Store ID')
            ->addOption(
                'generation',
                null,
                \Symfony\Component\Console\Input\InputOption::VALUE_REQUIRED,
                'Generation ID'
            )
            ->addOption(
                'dry-run',
                null,
                \Symfony\Component\Console\Input\InputOption::VALUE_NONE,
                'Print the exact activation preview without changing state'
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
            ? $this->activationService->previewActivation($storeId, $generationId)
            : $this->activationService->activateConfirmed(
                $storeId,
                $generationId,
                (string)$confirmation,
                'cli'
            );
        $output->writeln($this->json->serialize($report));

        return self::SUCCESS;
    }

    private function positiveInt(mixed $value): ?int
    {
        $validated = filter_var($value, FILTER_VALIDATE_INT, ['options' => ['min_range' => 1]]);

        return $validated === false ? null : (int)$validated;
    }
}
