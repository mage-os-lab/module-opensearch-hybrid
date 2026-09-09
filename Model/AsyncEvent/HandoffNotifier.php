<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\AsyncEvent;

class HandoffNotifier implements \MageOS\AsyncEvents\Service\AsyncEvent\NotifierInterface
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Queue\DataPlanePublisher $dataPlanePublisher,
        private readonly \Magento\Framework\Serialize\Serializer\Json $json
    ) {
    }

    public function notify(
        \MageOS\AsyncEvents\Api\Data\AsyncEventInterface $asyncEvent,
        \CloudEvents\V1\CloudEventImmutable $event
    ): \MageOS\AsyncEvents\Helper\NotifierResult {
        $result = new \MageOS\AsyncEvents\Helper\NotifierResult();
        $result->setSubscriptionId($asyncEvent->getSubscriptionId());
        try {
            $data = $event->getData();
            if (!is_array($data) || !isset($data['job_id'], $data['lane'])) {
                throw new \InvalidArgumentException('The Async Events job payload is invalid.');
            }
            $jobId = (string)$data['job_id'];
            $lane = (string)$data['lane'];
            $this->dataPlanePublisher->publish($jobId, $lane);
            $result->setSuccess(true);
            $result->setIsRetryable(false);
            $result->setResponseData($this->json->serialize(['status' => 'handed_off', 'job_id' => $jobId]));
            $result->setAsyncEventData(['job_id' => $jobId, 'lane' => $lane]);
        } catch (\Throwable $throwable) {
            $result->setSuccess(false);
            $result->setIsRetryable(true);
            $result->setRetryAfter(5);
            $result->setResponseData($this->json->serialize([
                'status' => 'handoff_failed',
                'error_class' => $throwable::class,
            ]));
            $result->setAsyncEventData(['status' => 'retryable']);
        }

        return $result;
    }
}
