<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Logger;

use Magento\Framework\Logger\Monolog;

class QueryTelemetry extends Monolog
{
    public function __construct(
        \MageOS\OpenSearchHybrid\Logger\Handler\QueryTelemetry $handler
    ) {
        parent::__construct('mageos_opensearch_hybrid_query', [$handler]);
    }
}
