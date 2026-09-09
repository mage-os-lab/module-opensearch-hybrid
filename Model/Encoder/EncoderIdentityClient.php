<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Encoder;

class EncoderIdentityClient
{
    public function __construct(
        private readonly \MageOS\OpenSearchHybrid\Model\Config $config,
        private readonly \MageOS\OpenSearchHybrid\Model\Http\CurlJsonClient $client,
        private readonly \MageOS\OpenSearchHybrid\Model\Encoder\EncoderIdentityVerifier $verifier
    ) {
    }

    public function fetch(string $expectedDigest): array
    {
        return $this->fetchAt($this->config->encoderEndpoint(), $expectedDigest);
    }

    public function fetchAt(string $endpoint, string $expectedDigest): array
    {
        $headers = [];
        $token = $this->config->encoderToken();
        if ($token !== '') {
            $headers['Authorization'] = 'Bearer ' . $token;
        }
        $identity = $this->client->get(
            $this->config->validateEncoderEndpoint($endpoint) . '/v1/identity',
            $this->config->encoderConnectTimeoutMs(),
            $this->config->encoderQueryTimeoutMs(),
            $headers
        );

        return $this->verifier->verify($identity, $expectedDigest);
    }
}
