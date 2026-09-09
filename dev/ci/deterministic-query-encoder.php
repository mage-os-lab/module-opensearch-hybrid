<?php
declare(strict_types=1);

$moduleRoot = getenv('MAGEOS_MODULE_ROOT');
$evidencePath = getenv('MAGEOS_ENCODER_EVIDENCE');
if (!is_string($moduleRoot) || $moduleRoot === '' || !is_file($moduleRoot . '/Model/Vector/DeterministicVector.php')) {
    http_response_code(500);
    exit;
}
if (!is_string($evidencePath) || $evidencePath === '') {
    http_response_code(500);
    exit;
}
if ($_SERVER['REQUEST_METHOD'] !== 'POST' || ($_SERVER['REQUEST_URI'] ?? '') !== '/v1/embed/query') {
    http_response_code(404);
    exit;
}

require_once $moduleRoot . '/Model/Vector/DeterministicVector.php';
$body = file_get_contents('php://input');
$payload = is_string($body) ? json_decode($body, true) : null;
if (!is_array($payload)
    || !isset(
        $payload['request_id'],
        $payload['model_id'],
        $payload['model_revision'],
        $payload['dimension'],
        $payload['recipe_version'],
        $payload['encoder_identity_digest'],
        $payload['query']
    )
    || !is_string($payload['query'])
    || (int)$payload['dimension'] !== \MageOS\OpenSearchHybrid\Model\Vector\DeterministicVector::DIMENSION
) {
    http_response_code(400);
    exit;
}

$vector = (new \MageOS\OpenSearchHybrid\Model\Vector\DeterministicVector())->encode($payload['query']);
$evidence = json_encode([
    'request_id' => (string)$payload['request_id'],
    'query_digest' => hash('sha256', $payload['query']),
], JSON_THROW_ON_ERROR | JSON_UNESCAPED_SLASHES) . "\n";
if (file_put_contents($evidencePath, $evidence, FILE_APPEND | LOCK_EX) === false) {
    http_response_code(500);
    exit;
}

header('Content-Type: application/json');
echo json_encode([
    'request_id' => (string)$payload['request_id'],
    'model_id' => (string)$payload['model_id'],
    'model_revision' => (string)$payload['model_revision'],
    'dimension' => (int)$payload['dimension'],
    'recipe_version' => (string)$payload['recipe_version'],
    'encoder_identity_digest' => (string)$payload['encoder_identity_digest'],
    'production_eligible' => true,
    'vectors' => [$vector],
], JSON_THROW_ON_ERROR | JSON_UNESCAPED_SLASHES);
