<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Generation;

class AutomaticReplacementService
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Generation\ReplacementCandidateRepository $repository,
        private readonly \MageOS\OpenSearchHybrid\Model\Generation\BuildService $buildService,
        private readonly \MageOS\OpenSearchHybrid\Model\Config $config,
        private readonly \Psr\Log\LoggerInterface $logger
    ) {
    }

    public function createPending(?int $storeId = null): array
    {
        $reports = [];
        foreach ($this->repository->storeIds($storeId) as $targetStoreId) {
            try {
                $reports[] = $this->repository->withinStoreLock(
                    $targetStoreId,
                    fn (): array => $this->createForStore($targetStoreId)
                );
            } catch (\Throwable $throwable) {
                $reports[] = $this->failure($targetStoreId, $throwable);
            }
        }

        return $reports;
    }

    private function createForStore(int $storeId): array
    {
        $assessment = $this->repository->assessment($storeId);
        if (!(bool)$assessment['eligible']) {
            return [
                'store_id' => $storeId,
                'status' => 'SKIPPED',
                'reason' => (string)$assessment['reason'],
                'invalidation_boundary' => (int)$assessment['invalidation_boundary'],
            ];
        }
        if (!$this->config->isEnabled($storeId)) {
            return [
                'store_id' => $storeId,
                'status' => 'SKIPPED',
                'reason' => 'module_disabled',
                'invalidation_boundary' => (int)$assessment['invalidation_boundary'],
            ];
        }
        if (!$this->config->isStoreIncluded($storeId)) {
            return [
                'store_id' => $storeId,
                'status' => 'SKIPPED',
                'reason' => 'store_excluded',
                'invalidation_boundary' => (int)$assessment['invalidation_boundary'],
            ];
        }
        try {
            $generationId = $this->buildService->buildProduction(
                $storeId,
                (string)$assessment['encoder_identity_digest']
            );
        } catch (\Throwable $throwable) {
            return $this->failure(
                $storeId,
                $throwable,
                (int)$assessment['invalidation_boundary']
            );
        }

        return [
            'store_id' => $storeId,
            'status' => 'CREATED',
            'reason' => 'pending_configuration_invalidation',
            'invalidation_boundary' => (int)$assessment['invalidation_boundary'],
            'generation_id' => $generationId,
        ];
    }

    private function failure(int $storeId, \Throwable $throwable, int $invalidationBoundary = 0): array
    {
        $this->logger->warning('OpenSearch Hybrid replacement generation creation failed.', [
            'store_id' => $storeId,
            'invalidation_boundary' => $invalidationBoundary,
            'error_class' => $throwable::class,
        ]);

        return [
            'store_id' => $storeId,
            'status' => 'FAILED_TO_CREATE',
            'reason' => 'replacement_creation_failed',
            'invalidation_boundary' => $invalidationBoundary,
            'error_class' => $throwable::class,
        ];
    }
}
