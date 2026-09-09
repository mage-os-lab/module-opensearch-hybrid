<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Logger\Handler;

use Magento\Framework\Logger\Handler\Base;
use Monolog\Logger;

class QueryTelemetry extends Base
{
    protected $fileName = '/var/log/opensearch-hybrid-query.log';
    protected $loggerType = Logger::INFO;
}
