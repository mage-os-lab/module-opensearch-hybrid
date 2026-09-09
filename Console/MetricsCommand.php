<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Console;

use Symfony\Component\Console\Input\InputInterface;
use Symfony\Component\Console\Input\InputOption;
use Symfony\Component\Console\Output\OutputInterface;

class MetricsCommand extends \Symfony\Component\Console\Command\Command
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Operations\OperationalMetricsService $metricsService,
        private readonly \Magento\Framework\Serialize\Serializer\Json $json,
        private readonly \MageOS\OpenSearchHybrid\Model\Operations\OpenMetricsFormatter $openMetricsFormatter,
        private readonly \MageOS\OpenSearchHybrid\Model\Operations\QueryTelemetryMetricsService $queryMetricsService,
        private readonly \Magento\Framework\App\Filesystem\DirectoryList $directoryList,
        ?string $name = null
    ) {
        parent::__construct($name);
    }

    protected function configure(): void
    {
        $this->setName('mage-os:opensearch-hybrid:metrics')
            ->setDescription('Emit durable-state OpenSearch Hybrid metrics and alert evaluations')
            ->addOption('store', null, InputOption::VALUE_OPTIONAL, 'Store ID')
            ->addOption(
                'format',
                null,
                InputOption::VALUE_OPTIONAL,
                'Output format: json or openmetrics',
                'json'
            )
            ->addOption(
                'query-window',
                null,
                InputOption::VALUE_OPTIONAL,
                'Include query telemetry from a 60 to 3600 second window'
            )
            ->addOption(
                'query-max-bytes',
                null,
                InputOption::VALUE_OPTIONAL,
                'Maximum query telemetry log bytes to scan',
                '5242880'
            );
    }

    protected function execute(InputInterface $input, OutputInterface $output): int
    {
        $rawStore = $input->getOption('store');
        $storeId = $rawStore === null
            ? null
            : filter_var($rawStore, FILTER_VALIDATE_INT, ['options' => ['min_range' => 1]]);
        if ($rawStore !== null && $storeId === false) {
            $output->writeln('<error>--store must be a positive integer.</error>');
            return self::INVALID;
        }
        $format = $input->getOption('format');
        if (!is_string($format) || !in_array($format, ['json', 'openmetrics'], true)) {
            $output->writeln('<error>--format must be json or openmetrics.</error>');
            return self::INVALID;
        }
        $rawQueryWindow = $input->getOption('query-window');
        $queryWindow = $rawQueryWindow === null
            ? null
            : filter_var(
                $rawQueryWindow,
                FILTER_VALIDATE_INT,
                ['options' => ['min_range' => 60, 'max_range' => 3600]]
            );
        if ($rawQueryWindow !== null && $queryWindow === false) {
            $output->writeln('<error>--query-window must be an integer from 60 through 3600.</error>');
            return self::INVALID;
        }
        $rawQueryMaximumBytes = $input->getOption('query-max-bytes');
        $queryMaximumBytes = filter_var(
            $rawQueryMaximumBytes,
            FILTER_VALIDATE_INT,
            ['options' => ['min_range' => 65536, 'max_range' => 52428800]]
        );
        if ($queryWindow !== null && $queryMaximumBytes === false) {
            $output->writeln('<error>--query-max-bytes must be an integer from 65536 through 52428800.</error>');
            return self::INVALID;
        }
        if ($queryWindow !== null && $format !== 'openmetrics') {
            $output->writeln('<error>--query-window requires --format=openmetrics.</error>');
            return self::INVALID;
        }
        $snapshot = $this->metricsService->get($storeId === null ? null : (int)$storeId);
        if ($format === 'openmetrics') {
            if ($queryWindow === null) {
                $output->write($this->openMetricsFormatter->format($snapshot));
            } else {
                $querySnapshot = $this->queryMetricsService->summarize(
                    $this->directoryList->getPath(\Magento\Framework\App\Filesystem\DirectoryList::LOG)
                        . '/opensearch-hybrid-query.log',
                    (int)$queryWindow,
                    (int)$queryMaximumBytes,
                    time()
                );
                $output->write($this->openMetricsFormatter->format($snapshot, $querySnapshot));
            }
        } else {
            $output->writeln($this->json->serialize($snapshot));
        }

        return self::SUCCESS;
    }
}
