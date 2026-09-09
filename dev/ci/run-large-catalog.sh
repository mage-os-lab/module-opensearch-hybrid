#!/usr/bin/env bash

set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fixture_parent="${RUNNER_TEMP:-/tmp}"
fixture_root="${MAGEOS_FIXTURE_ROOT:-${fixture_parent}/mageos-opensearch-hybrid-ci}"
target_products="${MAGEOS_LARGE_CATALOG_PRODUCTS:-10000}"
scale_confirmation="${MAGEOS_SCALE_CONFIRMATION:-}"
scale_evidence="${MAGEOS_SCALE_EVIDENCE_PATH:-}"
package_manifest="${MAGEOS_MODULE_PACKAGE_MANIFEST:-}"
build_memory_limit="${MAGEOS_LARGE_CATALOG_PHP_MEMORY_LIMIT:-}"
storefront_port="${MAGEOS_LARGE_STOREFRONT_PORT:-18080}"
storefront_log="${fixture_root}/var/log/mageos-opensearch-hybrid-large-storefront.log"
storefront_pid=""

cleanup_storefront() {
    if [[ -n "${storefront_pid}" ]] && kill -0 "${storefront_pid}" 2>/dev/null; then
        kill "${storefront_pid}"
        wait "${storefront_pid}" 2>/dev/null || true
    fi
    storefront_pid=""
}
trap cleanup_storefront EXIT

if [[ ! "${target_products}" =~ ^[0-9]+$ ]]; then
    echo "MAGEOS_LARGE_CATALOG_PRODUCTS must be an integer." >&2
    exit 2
fi
if [[ -z "${build_memory_limit}" ]]; then
    if (( target_products == 1000000 )); then
        build_memory_limit="12G"
    else
        build_memory_limit="2G"
    fi
fi
if [[ ! "${build_memory_limit}" =~ ^[1-9][0-9]*[GM]$ ]]; then
    echo "MAGEOS_LARGE_CATALOG_PHP_MEMORY_LIMIT must be a positive whole number of M or G." >&2
    exit 2
fi
if (( target_products > 10000 )); then
    if [[ "${target_products}" != "100000" && "${target_products}" != "1000000" ]]; then
        echo "Phase 5 scale target must be 100000 or 1000000." >&2
        exit 2
    fi
    if [[
        "${scale_confirmation}" != "scale-${target_products}"
        || -z "${scale_evidence}"
        || -z "${package_manifest}"
    ]]; then
        echo "Set the exact scale confirmation, evidence path, and package manifest." >&2
        exit 2
    fi
fi

build_arguments=("${fixture_root}" "${target_products}")
if [[ -n "${scale_evidence}" ]]; then
    build_arguments+=("${scale_confirmation}" "${scale_evidence}")
fi
php -d "memory_limit=${build_memory_limit}" \
    "${script_dir}/assert-large-catalog-build.php" \
    "${build_arguments[@]}"
php -S "127.0.0.1:${storefront_port}" \
    -t "${fixture_root}/pub" \
    "${fixture_root}/pub/index.php" \
    >"${storefront_log}" 2>&1 &
storefront_pid="$!"
graphql_arguments=(
    "${fixture_root}"
    "http://127.0.0.1:${storefront_port}"
    "${target_products}"
)
if [[ -n "${scale_evidence}" ]]; then
    graphql_arguments+=("${scale_evidence}")
fi
if ! php "${script_dir}/assert-large-catalog-graphql.php" "${graphql_arguments[@]}"; then
    tail -100 "${storefront_log}" >&2
    exit 1
fi
if (( target_products > 10000 )); then
    python "${script_dir}/../qualification/scale_evidence.py" \
        --evidence "${scale_evidence}" \
        --expected-target "${target_products}"
fi
