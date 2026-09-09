<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Console;

class BuildCommand extends \Symfony\Component\Console\Command\Command
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Generation\BuildService $buildService,
        private readonly \Magento\Framework\Serialize\Serializer\Json $json,
        ?string $name = null
    ) {
        parent::__construct($name);
    }

    protected function configure(): void
    {
        $this->setName('mage-os:opensearch-hybrid:build')
            ->setDescription('Build a store-scoped OpenSearch Hybrid generation')
            ->addOption('store', null, \Symfony\Component\Console\Input\InputOption::VALUE_REQUIRED, 'Store ID')
            ->addOption(
                'resume',
                null,
                \Symfony\Component\Console\Input\InputOption::VALUE_REQUIRED,
                'Resume one existing generation ID'
            )
            ->addOption(
                'pause',
                null,
                \Symfony\Component\Console\Input\InputOption::VALUE_REQUIRED,
                'Pause seeding for one existing generation ID'
            )
            ->addOption(
                'fake',
                null,
                \Symfony\Component\Console\Input\InputOption::VALUE_NONE,
                'Use deterministic development vectors that can never be activated'
            )
            ->addOption(
                'encoder-identity',
                null,
                \Symfony\Component\Console\Input\InputOption::VALUE_REQUIRED,
                'Operator-pinned production encoder identity digest'
            )
            ->addOption(
                'dry-run',
                null,
                \Symfony\Component\Console\Input\InputOption::VALUE_NONE,
                'Print verified production build impact without changing state'
            )
            ->addOption(
                'confirm',
                null,
                \Symfony\Component\Console\Input\InputOption::VALUE_OPTIONAL,
                'Exact confirmation token emitted by the current production dry-run preview'
            );
    }

    protected function execute(
        \Symfony\Component\Console\Input\InputInterface $input,
        \Symfony\Component\Console\Output\OutputInterface $output
    ): int {
        $rawResume = $input->getOption('resume');
        $rawPause = $input->getOption('pause');
        $expectedIdentity = $input->getOption('encoder-identity');
        $dryRun = (bool)$input->getOption('dry-run');
        $confirmation = $input->getOption('confirm');
        if ($rawPause !== null) {
            if ($rawResume !== null
                || $input->getOption('store') !== null
                || $input->getOption('fake')
                || $expectedIdentity !== null
                || $dryRun
                || $confirmation !== null
            ) {
                $output->writeln('<error>Use --pause by itself.</error>');
                return self::INVALID;
            }
            $generationId = filter_var($rawPause, FILTER_VALIDATE_INT, ['options' => ['min_range' => 1]]);
            if ($generationId === false) {
                $output->writeln('<error>--pause must be a positive generation ID.</error>');
                return self::INVALID;
            }
            $report = $this->buildService->pause((int)$generationId);
            $output->writeln(sprintf(
                '<info>Generation %d seeding is paused after %d products.</info>',
                $generationId,
                (int)$report['seeded_products']
            ));

            return self::SUCCESS;
        }
        if ($rawResume !== null) {
            if ($input->getOption('store') !== null
                || $input->getOption('fake')
                || $expectedIdentity !== null
                || $dryRun
                || $confirmation !== null
            ) {
                $output->writeln('<error>Use --resume by itself.</error>');
                return self::INVALID;
            }
            $generationId = filter_var($rawResume, FILTER_VALIDATE_INT, ['options' => ['min_range' => 1]]);
            if ($generationId === false) {
                $output->writeln('<error>--resume must be a positive generation ID.</error>');
                return self::INVALID;
            }
            $report = $this->buildService->resume((int)$generationId);
            $output->writeln(sprintf(
                '<info>Generation %d seeding state: %s. Seeded %d products in this run.</info>',
                $generationId,
                (string)$report['seeding_state'],
                (int)$report['seeded_products']
            ));

            return self::SUCCESS;
        }
        $storeId = filter_var($input->getOption('store'), FILTER_VALIDATE_INT, ['options' => ['min_range' => 1]]);
        if ($storeId === false) {
            $output->writeln('<error>Provide one explicit positive --store ID.</error>');
            return self::INVALID;
        }
        if ($input->getOption('fake')) {
            if ($dryRun) {
                $output->writeln('<error>Production preview cannot be combined with --fake.</error>');
                return self::INVALID;
            }
            if ($expectedIdentity !== null) {
                $output->writeln('<error>Do not combine --fake with --encoder-identity.</error>');
                return self::INVALID;
            }
            if ($confirmation !== null) {
                $output->writeln('<error>Do not combine --fake with --confirm.</error>');
                return self::INVALID;
            }
            $generationId = $this->buildService->buildFake((int)$storeId);
            $output->writeln(sprintf(
                '<info>Created non-activatable fake generation %d for store %d.</info>',
                $generationId,
                $storeId
            ));

            return self::SUCCESS;
        }
        if (!is_string($expectedIdentity) || !preg_match('/^[0-9a-f]{64}$/D', $expectedIdentity)) {
            $output->writeln(
                '<error>Production builds require the exact lowercase digest with --encoder-identity.</error>'
            );
            return self::INVALID;
        }
        if ($dryRun === ($confirmation !== null)) {
            $output->writeln('<error>Use --dry-run for preview or --confirm with its exact token for apply.</error>');
            return self::INVALID;
        }
        if ($dryRun) {
            $output->writeln($this->json->serialize(
                $this->buildService->planProduction((int)$storeId, $expectedIdentity)
            ));

            return self::SUCCESS;
        }
        $generationId = $this->buildService->buildProductionConfirmed(
            (int)$storeId,
            $expectedIdentity,
            (string)$confirmation
        );
        $output->writeln(sprintf(
            '<info>Created production generation %d for store %d with pinned encoder identity %s.</info>',
            $generationId,
            $storeId,
            $expectedIdentity
        ));

        return self::SUCCESS;
    }
}
