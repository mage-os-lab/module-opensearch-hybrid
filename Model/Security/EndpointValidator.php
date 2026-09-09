<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Security;

class EndpointValidator
{
    public function validate(string $endpoint, array $allowedHosts): string
    {
        $parts = parse_url($endpoint);
        if (!is_array($parts)
            || !isset($parts['scheme'], $parts['host'])
            || !in_array(strtolower((string)$parts['scheme']), ['http', 'https'], true)
            || isset($parts['user'])
            || isset($parts['pass'])
            || isset($parts['query'])
            || isset($parts['fragment'])
        ) {
            throw new \InvalidArgumentException('The encoder endpoint is not a valid HTTP service base URL.');
        }
        $host = strtolower(rtrim((string)$parts['host'], '.'));
        $address = trim($host, '[]');
        $normalizedAllowedHosts = array_map(
            static fn (string $value): string => trim(strtolower(rtrim(trim($value), '.')), '[]'),
            $allowedHosts
        );
        $explicitlyAllowed = in_array($address, $normalizedAllowedHosts, true);
        if ($this->isForbiddenAddress($address)
            && !($this->isLoopbackAddress($address) && $explicitlyAllowed)
        ) {
            throw new \InvalidArgumentException(
                'Link-local, metadata, unspecified, multicast, and unlisted loopback endpoints are forbidden.'
            );
        }
        if (!$this->isPrivateAddress($address) && !$explicitlyAllowed) {
            throw new \InvalidArgumentException('The encoder endpoint host is not private or explicitly allowed.');
        }
        $path = (string)($parts['path'] ?? '');
        if ($path !== '' && $path !== '/') {
            throw new \InvalidArgumentException('The encoder endpoint must not contain a path.');
        }

        return rtrim($endpoint, '/');
    }

    private function isPrivateAddress(string $host): bool
    {
        if (filter_var($host, FILTER_VALIDATE_IP, FILTER_FLAG_IPV4) !== false) {
            $address = ip2long($host);
            if ($address === false) {
                return false;
            }
            $unsigned = (int)sprintf('%u', $address);

            return ($unsigned >= 167772160 && $unsigned <= 184549375)
                || ($unsigned >= 2886729728 && $unsigned <= 2887778303)
                || ($unsigned >= 3232235520 && $unsigned <= 3232301055);
        }
        $packed = inet_pton($host);

        return is_string($packed)
            && strlen($packed) === 16
            && (ord($packed[0]) & 0xfe) === 0xfc;
    }

    private function isForbiddenAddress(string $host): bool
    {
        if ($host === 'localhost' || str_ends_with($host, '.localhost')) {
            return true;
        }
        if (filter_var($host, FILTER_VALIDATE_IP, FILTER_FLAG_IPV4) !== false) {
            return $host === '0.0.0.0'
                || str_starts_with($host, '127.')
                || str_starts_with($host, '169.254.');
        }
        $packed = inet_pton($host);
        if (!is_string($packed) || strlen($packed) !== 16) {
            return false;
        }

        return $host === '::'
            || $host === '::1'
            || ord($packed[0]) === 0xff
            || (ord($packed[0]) === 0xfe && (ord($packed[1]) & 0xc0) === 0x80);
    }

    private function isLoopbackAddress(string $host): bool
    {
        if ($host === 'localhost' || str_ends_with($host, '.localhost')) {
            return true;
        }
        if (filter_var($host, FILTER_VALIDATE_IP, FILTER_FLAG_IPV4) !== false) {
            return str_starts_with($host, '127.');
        }

        return $host === '::1';
    }
}
