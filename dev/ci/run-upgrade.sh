#!/usr/bin/env bash

set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
module_root="$(CDPATH= cd -- "${script_dir}/../.." && pwd)"
module_install_root="${MAGEOS_MODULE_INSTALL_ROOT:-${module_root}}"
fixture_parent="${RUNNER_TEMP:-/tmp}"
sbom_tool_root="${MAGEOS_SBOM_TOOL_ROOT:-${fixture_parent}/mageos-opensearch-hybrid-sbom-tool}"
fixture_root="${MAGEOS_FIXTURE_ROOT:-${fixture_parent}/mageos-opensearch-hybrid-upgrade-ci}"
sbom_path="${MAGEOS_SBOM_PATH:-${fixture_root}/var/mage-os-module-opensearch-hybrid-upgrade.cdx.json}"
upgrade_manifest="${fixture_root}/var/mageos-opensearch-hybrid-platform-upgrade.json"

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
    "3.3.0"

"${fixture_root}/bin/magento" setup:install \
    --base-url=http://mageos-hybrid-upgrade.test/ \
    --db-host=127.0.0.1:13306 \
    --db-name=mageos_hybrid_ci \
    --db-user=mageos \
    --db-password=mageos_ci_only \
    --admin-firstname=Fixture \
    --admin-lastname=Upgrade \
    --admin-email=fixture-upgrade@example.invalid \
    --admin-user=fixture_upgrade \
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

php "${script_dir}/assert-package-version.php" "${fixture_root}" "3.3.0"
php "${script_dir}/assert-platform-upgrade.php" "${fixture_root}" prepare "${upgrade_manifest}"

"${fixture_root}/bin/magento" maintenance:enable
composer --working-dir="${fixture_root}" config \
    repositories.mageos-opensearch-hybrid \
    path \
    "${module_install_root}"
composer --working-dir="${fixture_root}" require-commerce \
    mage-os/product-community-edition=3.4.0 \
    --no-update \
    --force-root-updates \
    --no-interaction \
    --no-progress
COMPOSER_MIRROR_PATH_REPOS=1 composer --working-dir="${fixture_root}" require \
    mage-os/module-opensearch-hybrid:@dev \
    mage-os/mageos-async-events:^4.0.8 \
    --no-update \
    --no-interaction \
    --no-progress
COMPOSER_MIRROR_PATH_REPOS=1 composer --working-dir="${fixture_root}" update \
    --with-all-dependencies \
    --no-interaction \
    --no-progress
php "${script_dir}/assert-package-version.php" "${fixture_root}" "3.4.0"
"${fixture_root}/vendor/bin/phpcs" \
    --standard="${module_install_root}/phpcs.xml.dist" \
    "${module_install_root}"

"${fixture_root}/bin/magento" module:enable MageOS_AsyncEvents MageOS_OpenSearchHybrid
"${fixture_root}/bin/magento" setup:upgrade --keep-generated
"${fixture_root}/bin/magento" setup:db:status
"${fixture_root}/bin/magento" setup:di:compile
"${fixture_root}/bin/magento" cache:clean
"${fixture_root}/bin/magento" maintenance:disable
"${fixture_root}/bin/magento" module:status MageOS_OpenSearchHybrid
php "${script_dir}/assert-defaults.php" "${fixture_root}"
php "${script_dir}/assert-platform-upgrade.php" "${fixture_root}" verify "${upgrade_manifest}"
php "${script_dir}/assert-doctor.php" "${fixture_root}"
"${fixture_root}/bin/magento" queue:consumers:list | grep -Fx mageos.opensearch_hybrid.embedding.consumer
"${fixture_root}/bin/magento" queue:consumers:list | grep -Fx mageos.opensearch_hybrid.correctness.consumer

composer --working-dir="${fixture_root}" audit \
    --locked \
    --no-interaction \
    --no-ansi \
    --abandoned=ignore
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

MAGEOS_ROOT="${fixture_root}" \
    "${fixture_root}/vendor/bin/phpunit" \
    -c "${module_install_root}/phpunit.xml.dist"
