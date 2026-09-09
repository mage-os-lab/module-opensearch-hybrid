<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Console;

class RollbackCommand extends \Symfony\Component\Console\Command\Command
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
        $this->setName('mage-os:opensearch-hybrid:rollback')
            ->setDescription('Preview or atomically restore native search or one retained semantic generation')
            ->addOption('store', null, \Symfony\Component\Console\Input\InputOption::VALUE_REQUIRED, 'Store ID')
            ->addOption(
                'generation',
                null,
                \Symfony\Component\Console\Input\InputOption::VALUE_OPTIONAL,
                'Retained generation ID; omit for native search'
            )
            ->addOption(
                'dry-run',
                null,
                \Symfony\Component\Console\Input\InputOption::VALUE_NONE,
                'Print the exact rollback preview without changing state'
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
        $storeId = filter_var(
            $input->getOption('store'),
            FILTER_VALIDATE_INT,
            ['options' => ['min_range' => 1]]
        );
        if ($storeId === false) {
            $output->writeln('<error>Provide one explicit positive --store ID.</error>');
            return self::INVALID;
        }
        $rawGeneration = $input->getOption('generation');
        $generationId = $rawGeneration === null
            ? null
            : filter_var($rawGeneration, FILTER_VALIDATE_INT, ['options' => ['min_range' => 1]]);
        if ($generationId === false) {
            $output->writeln('<error>--generation must be a positive retained generation ID.</error>');
            return self::INVALID;
        }
        $dryRun = (bool)$input->getOption('dry-run');
        $confirmation = $input->getOption('confirm');
        if ($dryRun === ($confirmation !== null)) {
            $output->writeln('<error>Use --dry-run for preview or --confirm with its exact token for apply.</error>');
            return self::INVALID;
        }

        $report = $this->executeOperation(
            (int)$storeId,
            $generationId === null ? null : (int)$generationId,
            $dryRun,
            $confirmation === null ? null : (string)$confirmation
        );
        $output->writeln($this->json->serialize($report));

        return self::SUCCESS;
    }

    private function executeOperation(
        int $storeId,
        ?int $generationId,
        bool $dryRun,
        ?string $confirmation
    ): array {
        if ($generationId === null) {
            return $dryRun
                ? $this->activationService->previewNativeRollback($storeId)
                : $this->activationService->rollbackToNativeConfirmed($storeId, (string)$confirmation, 'cli');
        }

        return $dryRun
            ? $this->activationService->previewGenerationRollback($storeId, $generationId)
            : $this->activationService->rollbackToGenerationConfirmed(
                $storeId,
                $generationId,
                (string)$confirmation,
                'cli'
            );
    }
}
