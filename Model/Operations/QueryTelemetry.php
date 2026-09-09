<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Operations;

class QueryTelemetry
{
    public function __construct(
        private readonly \Psr\Log\LoggerInterface $logger
    ) {
    }

    public function start(int $storeId, string $requestName): QueryTelemetrySpan
    {
        return new QueryTelemetrySpan($this->logger, $storeId, $requestName);
    }
}
