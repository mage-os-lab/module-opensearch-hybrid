<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Generation;

class BuildRefreshService
{
    private const PROGRESS_TABLE = 'mageos_opensearch_hybrid_generation_progress';
    private const BUILDING_STATES = ['BUILDING', 'CATCHING_UP'];

    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resourceConnection,
        private readonly \MageOS\OpenSearchHybrid\Model\OpenSearch\IndexManager $indexManager
    ) {
    }

    public function suppress(array $generation): void
    {
        $this->assertInactiveBuild($generation);
        $progress = $this->progress((int)$generation['generation_id']);
        if ((string)$progress['seeding_state'] === 'COMPLETED') {
            return;
        }
        $this->setState((int)$generation['generation_id'], 'SUPPRESSING');
        try {
            $this->indexManager->setRefreshInterval(
                $generation,
                (string)$progress['build_refresh_interval']
            );
            $this->setState((int)$generation['generation_id'], 'SUPPRESSED');
        } catch (\Throwable $throwable) {
            $this->setState((int)$generation['generation_id'], 'SUPPRESS_FAILED');
            throw $throwable;
        }
    }

    public function restore(array $generation): void
    {
        $this->assertInactiveBuild($generation);
        $progress = $this->progress((int)$generation['generation_id']);
        if ((string)$progress['seeding_state'] !== 'COMPLETED') {
            throw new \RuntimeException(
                'Full-build seeding must complete before restoring the normal refresh interval.'
            );
        }
        $this->setState((int)$generation['generation_id'], 'RESTORING');
        try {
            $this->indexManager->setRefreshInterval(
                $generation,
                (string)$progress['normal_refresh_interval']
            );
            $this->setState((int)$generation['generation_id'], 'RESTORED');
        } catch (\Throwable $throwable) {
            $this->setState((int)$generation['generation_id'], 'RESTORE_FAILED');
            throw $throwable;
        }
    }

    public function isRestored(int $generationId): bool
    {
        return (string)$this->progress($generationId)['refresh_state'] === 'RESTORED';
    }

    private function assertInactiveBuild(array $generation): void
    {
        if (!in_array((string)($generation['state'] ?? ''), self::BUILDING_STATES, true)) {
            throw new \RuntimeException(
                'Build-only refresh settings are limited to inactive building generations.'
            );
        }
    }

    private function progress(int $generationId): array
    {
        $connection = $this->resourceConnection->getConnection();
        $progress = $connection->fetchRow(
            $connection->select()
                ->from($this->resourceConnection->getTableName(self::PROGRESS_TABLE))
                ->where('generation_id = ?', $generationId)
        );
        if (!is_array($progress)) {
            throw new \RuntimeException('The generation has no build refresh state.');
        }

        return $progress;
    }

    private function setState(int $generationId, string $state): void
    {
        $affected = $this->resourceConnection->getConnection()->update(
            $this->resourceConnection->getTableName(self::PROGRESS_TABLE),
            ['refresh_state' => $state],
            ['generation_id = ?' => $generationId]
        );
        if ($affected !== 1) {
            $progress = $this->progress($generationId);
            if ((string)$progress['refresh_state'] !== $state) {
                throw new \RuntimeException('The build refresh state could not be persisted.');
            }
        }
    }
}
