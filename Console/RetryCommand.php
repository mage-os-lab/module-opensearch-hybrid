<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Console;

class RetryCommand extends \Symfony\Component\Console\Command\Command
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Outbox\OutboxRepository $outboxRepository,
        private readonly \MageOS\OpenSearchHybrid\Model\AsyncEvent\LifecyclePublisher $publisher,
        private readonly \Magento\Framework\Serialize\Serializer\Json $json,
        ?string $name = null
    ) {
        parent::__construct($name);
    }

    protected function configure(): void
    {
        $this->setName('mage-os:opensearch-hybrid:retry')
            ->setDescription('Explicitly retry one exact outbox job')
            ->addOption('job', null, \Symfony\Component\Console\Input\InputOption::VALUE_REQUIRED, 'Outbox job UUID')
            ->addOption(
                'confirm',
                null,
                \Symfony\Component\Console\Input\InputOption::VALUE_REQUIRED,
                'Repeat the exact outbox job UUID'
            );
    }

    protected function execute(
        \Symfony\Component\Console\Input\InputInterface $input,
        \Symfony\Component\Console\Output\OutputInterface $output
    ): int {
        $jobId = $input->getOption('job');
        if (!is_string($jobId)
            || preg_match('/^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/iD', $jobId) !== 1
            || $input->getOption('confirm') !== $jobId
        ) {
            $output->writeln('<error>Provide one UUID and repeat it exactly with --confirm.</error>');
            return self::INVALID;
        }
        $job = $this->outboxRepository->prepareManualRetry($jobId);
        $this->publisher->publish($jobId);
        $output->writeln($this->json->serialize([
            'schema_version' => 1,
            'status' => 'republished',
            'job_id' => $jobId,
            'store_id' => (int)$job['store_id'],
            'generation_id' => $job['generation_id'] === null ? null : (int)$job['generation_id'],
        ]));

        return self::SUCCESS;
    }
}
