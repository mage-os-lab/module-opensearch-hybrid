<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Console;

class EncoderIdentityCommand extends \Symfony\Component\Console\Command\Command
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Encoder\EncoderIdentityClient $identityClient,
        ?string $name = null
    ) {
        parent::__construct($name);
    }

    protected function configure(): void
    {
        $this->setName('mage-os:opensearch-hybrid:encoder:identity')
            ->setDescription('Verify the configured production encoder against an operator-pinned identity')
            ->addOption(
                'expect',
                null,
                \Symfony\Component\Console\Input\InputOption::VALUE_REQUIRED,
                'Expected 64-character encoder identity digest'
            );
    }

    protected function execute(
        \Symfony\Component\Console\Input\InputInterface $input,
        \Symfony\Component\Console\Output\OutputInterface $output
    ): int {
        $expectedDigest = (string)$input->getOption('expect');
        if (!preg_match('/^[0-9a-f]{64}$/D', $expectedDigest)) {
            $output->writeln('<error>Provide the exact lowercase 64-character digest with --expect.</error>');
            return self::INVALID;
        }
        $identity = $this->identityClient->fetch($expectedDigest);
        $deployment = $identity['deployment'];
        $output->writeln(sprintf(
            '<info>Verified encoder %s at %s on %s.</info>',
            $identity['model_id'],
            $identity['model_revision'],
            $identity['architecture']
        ));
        $output->writeln('Identity: ' . $identity['encoder_identity_digest']);
        $output->writeln(sprintf(
            'Deployment: %s@%s',
            $deployment['kind'],
            $deployment['artifact_digest']
        ));
        $output->writeln(sprintf(
            'Bound artifacts: %d; qualification evidence: %d.',
            count($identity['artifacts']),
            count($identity['qualification']['evidence_artifact_roles'])
        ));

        return self::SUCCESS;
    }
}
