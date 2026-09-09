<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Console;

class ValidateCommand extends \Symfony\Component\Console\Command\Command
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Generation\ValidationService $validationService,
        private readonly \Magento\Framework\Serialize\Serializer\Json $json,
        ?string $name = null
    ) {
        parent::__construct($name);
    }

    protected function configure(): void
    {
        $this->setName('mage-os:opensearch-hybrid:validate')
            ->setDescription('Validate one exact generation without activating it')
            ->addOption('generation', null, \Symfony\Component\Console\Input\InputOption::VALUE_REQUIRED, 'Generation ID')
            ->addOption(
                'accept-result-contract',
                null,
                \Symfony\Component\Console\Input\InputOption::VALUE_OPTIONAL,
                'Exact reviewed result-contract SHA-256'
            );
    }

    protected function execute(
        \Symfony\Component\Console\Input\InputInterface $input,
        \Symfony\Component\Console\Output\OutputInterface $output
    ): int {
        $generationId = filter_var(
            $input->getOption('generation'),
            FILTER_VALIDATE_INT,
            ['options' => ['min_range' => 1]]
        );
        if ($generationId === false) {
            $output->writeln('<error>Provide one explicit positive --generation ID.</error>');
            return self::INVALID;
        }
        $acceptedDigest = $input->getOption('accept-result-contract');
        $report = $this->validationService->validate(
            (int)$generationId,
            is_string($acceptedDigest) && $acceptedDigest !== '' ? $acceptedDigest : null
        );
        $output->writeln($this->json->serialize($report));

        return $report['ready'] ? self::SUCCESS : self::FAILURE;
    }
}
