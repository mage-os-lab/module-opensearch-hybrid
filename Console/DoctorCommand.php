<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Console;

class DoctorCommand extends \Symfony\Component\Console\Command\Command
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\DoctorService $doctorService,
        private readonly \Magento\Framework\Serialize\Serializer\Json $json,
        ?string $name = null
    ) {
        parent::__construct($name);
    }

    protected function configure(): void
    {
        $this->setName('mage-os:opensearch-hybrid:doctor')
            ->setDescription('Diagnose wrapper, RabbitMQ, encoder, and subscription readiness');
    }

    protected function execute(
        \Symfony\Component\Console\Input\InputInterface $input,
        \Symfony\Component\Console\Output\OutputInterface $output
    ): int {
        $report = $this->doctorService->diagnose();
        $output->writeln($this->json->serialize($report));

        return $report['healthy'] ? self::SUCCESS : self::FAILURE;
    }
}
