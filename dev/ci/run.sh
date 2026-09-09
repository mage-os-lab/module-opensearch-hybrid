#!/usr/bin/env bash

set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
module_root="$(CDPATH= cd -- "${script_dir}/../.." && pwd)"
module_install_root="${MAGEOS_MODULE_INSTALL_ROOT:-${module_root}}"
fixture_parent="${RUNNER_TEMP:-/tmp}"
sbom_tool_root="${MAGEOS_SBOM_TOOL_ROOT:-${fixture_parent}/mageos-opensearch-hybrid-sbom-tool}"
fixture_root="${MAGEOS_FIXTURE_ROOT:-${fixture_parent}/mageos-opensearch-hybrid-ci}"
sbom_path="${MAGEOS_SBOM_PATH:-${fixture_root}/var/mage-os-module-opensearch-hybrid.cdx.json}"

if [[ ! -f "${module_install_root}/composer.json" ]]; then
    echo "Module install root does not contain composer.json: ${module_install_root}" >&2
    exit 2
fi
if [[ -e "${fixture_root}" ]]; then
    echo "Refusing to overwrite existing fixture path: ${fixture_root}" >&2
    exit 2
fi
if [[ ! -f "${sbom_tool_root}/vendor/autoload.php" ]]; then
    echo "Install the locked SBOM tooling into the runner-owned tool path: ${sbom_tool_root}" >&2
    exit 2
fi

composer create-project \
    --no-interaction \
    --no-progress \
    --repository-url=https://repo.mage-os.org/ \
    mage-os/project-community-edition \
    "${fixture_root}" \
    "3.4.0"

composer --working-dir="${fixture_root}" config \
    repositories.mageos-opensearch-hybrid \
    path \
    "${module_install_root}"
COMPOSER_MIRROR_PATH_REPOS=1 composer --working-dir="${fixture_root}" require \
    --no-interaction \
    --no-progress \
    mage-os/module-opensearch-hybrid:@dev \
    mage-os/mageos-async-events:^4.0.8
composer --working-dir="${fixture_root}" audit \
    --locked \
    --no-interaction \
    --no-ansi \
    --abandoned=ignore
"${fixture_root}/vendor/bin/phpcs" \
    --standard="${module_install_root}/phpcs.xml.dist" \
    "${module_install_root}"
composer --working-dir="${sbom_tool_root}" CycloneDX:make-sbom \
    --output-format=JSON \
    --output-file="${sbom_path}" \
    --spec-version=1.6 \
    --output-reproducible \
    --validate \
    --omit=dev \
    --omit=plugin \
    "${fixture_root}/composer.json"
PYTHONPATH="${module_root}" python "${module_root}/dev/sbom/verify.py" \
    --normalize \
    --sbom="${sbom_path}"

"${fixture_root}/bin/magento" setup:install \
    --base-url=http://mageos-hybrid.test/ \
    --db-host=127.0.0.1:13306 \
    --db-name=mageos_hybrid_ci \
    --db-user=mageos \
    --db-password=mageos_ci_only \
    --admin-firstname=Fixture \
    --admin-lastname=Operator \
    --admin-email=fixture@example.invalid \
    --admin-user=fixture \
    --admin-password='MageosCi1!' \
    --language=en_US \
    --currency=USD \
    --timezone=UTC \
    --use-rewrites=1 \
    --search-engine=opensearch \
    --opensearch-host=127.0.0.1 \
    --opensearch-port=19201 \
    --opensearch-enable-auth=0 \
    --amqp-host=127.0.0.1 \
    --amqp-port=35672 \
    --amqp-user=mageos \
    --amqp-password=mageos_ci_only \
    --amqp-virtualhost=mageos_hybrid_ci

"${fixture_root}/bin/magento" module:status MageOS_OpenSearchHybrid
php "${script_dir}/assert-defaults.php" "${fixture_root}"
"${fixture_root}/bin/magento" setup:upgrade --keep-generated
"${fixture_root}/bin/magento" setup:db:status
"${fixture_root}/bin/magento" setup:di:compile
php "${script_dir}/assert-admin-status-render.php" "${fixture_root}"
php "${script_dir}/assert-doctor.php" "${fixture_root}"
"${fixture_root}/bin/magento" queue:consumers:list | grep -Fx mageos.opensearch_hybrid.embedding.consumer
"${fixture_root}/bin/magento" queue:consumers:list | grep -Fx mageos.opensearch_hybrid.correctness.consumer

queue_isolation_manifest="${fixture_root}/var/mageos-opensearch-hybrid-queue-isolation.json"
php "${script_dir}/prepare-queue-isolation.php" "${fixture_root}" "${queue_isolation_manifest}"
"${fixture_root}/bin/magento" indexer:reindex catalogsearch_fulltext
correctness_started_at="$(date +%s)"
"${fixture_root}/bin/magento" \
    queue:consumers:start mageos.opensearch_hybrid.correctness.consumer \
    --max-messages=1
correctness_finished_at="$(date +%s)"
correctness_elapsed="$((correctness_finished_at - correctness_started_at))"
php "${script_dir}/assert-queue-isolation.php" \
    "${fixture_root}" \
    "${queue_isolation_manifest}" \
    "${correctness_elapsed}"

php "${script_dir}/prepare-product-delete.php" "${fixture_root}" "${queue_isolation_manifest}"
"${fixture_root}/bin/magento" indexer:reindex catalogsearch_fulltext
delete_started_at="$(date +%s)"
"${fixture_root}/bin/magento" \
    queue:consumers:start mageos.opensearch_hybrid.correctness.consumer \
    --max-messages=1
delete_finished_at="$(date +%s)"
delete_elapsed="$((delete_finished_at - delete_started_at))"
php "${script_dir}/assert-product-delete.php" \
    "${fixture_root}" \
    "${queue_isolation_manifest}" \
    "${delete_elapsed}"

php "${script_dir}/assert-dependent-invalidation.php" "${fixture_root}"
php "${script_dir}/assert-atomic-change-rollback.php" "${fixture_root}"
php "${script_dir}/assert-build-resume.php" "${fixture_root}"
php "${script_dir}/assert-dual-generation-catch-up.php" "${fixture_root}"
php "${script_dir}/assert-source-invalidation.php" "${fixture_root}"
php "${script_dir}/assert-inventory-invalidation.php" "${fixture_root}"
reservation_manifest="${fixture_root}/var/mageos-opensearch-hybrid-reservation.json"
php "${script_dir}/prepare-reservation-invalidation.php" "${fixture_root}" "${reservation_manifest}"
"${fixture_root}/bin/magento" \
    queue:consumers:start inventory.reservations.updateSalabilityStatus \
    --max-messages=1
php "${script_dir}/assert-reservation-invalidation.php" "${fixture_root}" "${reservation_manifest}"
php "${script_dir}/assert-price-invalidation.php" "${fixture_root}"
php "${script_dir}/assert-full-price-and-catalog-rule-invalidation.php" "${fixture_root}"
for price_dimension_mode in website customer_group website_and_customer_group none; do
    "${fixture_root}/bin/magento" indexer:set-dimensions-mode catalog_product_price "${price_dimension_mode}"
    "${fixture_root}/bin/magento" indexer:reindex catalog_product_price
    php "${script_dir}/assert-price-dimensions.php" "${fixture_root}" "${price_dimension_mode}"
done
php "${script_dir}/assert-attribute-option-invalidation.php" "${fixture_root}"
php "${script_dir}/assert-scope-invalidation.php" "${fixture_root}"
php "${script_dir}/assert-inventory-topology-invalidation.php" "${fixture_root}"
active_readiness_manifest="${fixture_root}/var/mageos-opensearch-hybrid-active-readiness.json"
php "${script_dir}/prepare-active-readiness.php" "${fixture_root}" "${active_readiness_manifest}"
"${fixture_root}/bin/magento" \
    queue:consumers:start mageos.opensearch_hybrid.correctness.consumer \
    --max-messages=1
php "${script_dir}/assert-active-readiness.php" "${fixture_root}" "${active_readiness_manifest}"

storefront_port="${MAGEOS_STOREFRONT_PORT:-18080}"
encoder_port="${MAGEOS_ENCODER_PORT:-18081}"
storefront_log="${fixture_root}/var/log/mageos-opensearch-hybrid-storefront.log"
encoder_log="${fixture_root}/var/log/mageos-opensearch-hybrid-encoder.log"
encoder_evidence="${fixture_root}/var/mageos-opensearch-hybrid-encoder-evidence.jsonl"
storefront_pid=""
encoder_pid=""
cleanup_storefront() {
    if [[ -n "${storefront_pid}" ]] && kill -0 "${storefront_pid}" 2>/dev/null; then
        kill "${storefront_pid}"
        wait "${storefront_pid}" 2>/dev/null || true
    fi
    storefront_pid=""
    if [[ -n "${encoder_pid}" ]] && kill -0 "${encoder_pid}" 2>/dev/null; then
        kill "${encoder_pid}"
        wait "${encoder_pid}" 2>/dev/null || true
    fi
    encoder_pid=""
}
trap cleanup_storefront EXIT
MAGEOS_MODULE_ROOT="${module_install_root}" \
MAGEOS_ENCODER_EVIDENCE="${encoder_evidence}" \
php -S "127.0.0.1:${encoder_port}" \
    "${script_dir}/deterministic-query-encoder.php" \
    >"${encoder_log}" 2>&1 &
encoder_pid="$!"
php -S "127.0.0.1:${storefront_port}" \
    -t "${fixture_root}/pub" \
    "${fixture_root}/pub/index.php" \
    >"${storefront_log}" 2>&1 &
storefront_pid="$!"
if ! php "${script_dir}/assert-storefront-acceptance.php" \
    "${fixture_root}" \
    "http://127.0.0.1:${storefront_port}" \
    "http://127.0.0.1:${encoder_port}" \
    "${encoder_evidence}"; then
    tail -100 "${storefront_log}" >&2
    tail -100 "${encoder_log}" >&2
    exit 1
fi
query_metrics="${fixture_root}/var/mageos-opensearch-hybrid-query.prom"
query_metrics_only="${fixture_root}/var/mageos-opensearch-hybrid-query-only.prom"
"${fixture_root}/bin/magento" mage-os:opensearch-hybrid:metrics \
    --format=openmetrics \
    --query-window=300 \
    --query-max-bytes=1048576 \
    >"${query_metrics}"
grep '^mageos_opensearch_hybrid_query_' "${query_metrics}" >"${query_metrics_only}"
grep -Fx 'mageos_opensearch_hybrid_query_log_available 1' "${query_metrics_only}"
grep -Fx 'mageos_opensearch_hybrid_query_window_complete 1' "${query_metrics_only}"
grep -F 'request_name="quick_search_container"' "${query_metrics_only}"
grep -F 'request_name="graphql_product_search"' "${query_metrics_only}"
grep -F 'route="hybrid"' "${query_metrics_only}"
if grep -Eq 'generation_id|raw_query|error_class' "${query_metrics_only}"; then
    echo "Query OpenMetrics exposed a forbidden identity." >&2
    exit 1
fi
tail -n 1 "${query_metrics}" | grep -Fx '# EOF'
cleanup_storefront
trap - EXIT

php "${script_dir}/assert-retry-dead-letter.php" "${fixture_root}"
php "${script_dir}/assert-production-build-gate.php" "${fixture_root}"
php "${script_dir}/assert-admin-activation-controls.php" "${fixture_root}"
php "${script_dir}/assert-cli-activation-controls.php" "${fixture_root}"
php "${script_dir}/assert-subscription-cli-controls.php" "${fixture_root}"
php "${script_dir}/assert-cleanup-gate.php" "${fixture_root}"
php "${script_dir}/assert-operational-metrics.php" "${fixture_root}"
php "${script_dir}/assert-async-event-trace-health.php" "${fixture_root}" 1
php "${script_dir}/assert-admin-config-save-confirmation.php" "${fixture_root}"

MAGEOS_ROOT="${fixture_root}" \
    "${fixture_root}/vendor/bin/phpunit" \
    -c "${module_install_root}/phpunit.xml.dist"
