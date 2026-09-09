#!/usr/bin/env bash

set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
module_root="$(CDPATH= cd -- "${script_dir}/../.." && pwd)"
module_install_root="${MAGEOS_MODULE_INSTALL_ROOT:-${module_root}}"
fixture_parent="${RUNNER_TEMP:-/tmp}"
fixture_root="${MAGEOS_FIXTURE_ROOT:-${fixture_parent}/mageos-opensearch-hybrid-scale-ci}"
target_products="${MAGEOS_LARGE_CATALOG_PRODUCTS:-}"
scale_confirmation="${MAGEOS_SCALE_CONFIRMATION:-}"

if [[ "${target_products}" != "100000" && "${target_products}" != "1000000" ]]; then
    echo "MAGEOS_LARGE_CATALOG_PRODUCTS must be exactly 100000 or 1000000." >&2
    exit 2
fi
if [[ "${scale_confirmation}" != "scale-${target_products}" ]]; then
    echo "Set MAGEOS_SCALE_CONFIRMATION=scale-${target_products}." >&2
    exit 2
fi
if [[ ! -f "${module_install_root}/composer.json" ]]; then
    echo "Module install root does not contain composer.json: ${module_install_root}" >&2
    exit 2
fi
if [[ -e "${fixture_root}" ]]; then
    echo "Refusing to overwrite existing scale fixture: ${fixture_root}" >&2
    exit 2
fi
if ! command -v php >/dev/null 2>&1; then
    echo "PHP is required to install the scale fixture." >&2
    exit 2
fi
if ! command -v composer >/dev/null 2>&1; then
    echo "Composer is required to install the scale fixture." >&2
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
php "${script_dir}/assert-doctor.php" "${fixture_root}"

printf 'Scale fixture installed for the exact %s-product qualification target at %s\n' \
    "${target_products}" \
    "${fixture_root}"
