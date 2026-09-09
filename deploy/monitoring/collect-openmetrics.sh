#!/usr/bin/env bash

set -euo pipefail

textfile_directory="${OPENMETRICS_TEXTFILE_DIRECTORY:?Set OPENMETRICS_TEXTFILE_DIRECTORY}"
store_id="${OPENMETRICS_STORE_ID:-}"
query_window="${OPENMETRICS_QUERY_WINDOW_SECONDS:-}"
query_maximum_bytes="${OPENMETRICS_QUERY_MAX_BYTES:-5242880}"
target="${textfile_directory}/mageos_opensearch_hybrid.prom"
magento_cli="${MAGEOS_CLI:-}"
magento_invocation=()

if [[ -n "${magento_cli}" ]]; then
    if [[ "${magento_cli}" != /* || ! -x "${magento_cli}" ]]; then
        echo "MAGEOS_CLI must be an absolute executable path." >&2
        exit 2
    fi
    magento_invocation=("${magento_cli}")
else
    magento_root="${MAGEOS_ROOT:?Set MAGEOS_ROOT or MAGEOS_CLI}"
    php_binary="${PHP_BINARY:-php}"
    magento_command="${magento_root}/bin/magento"
    if [[ ! -f "${magento_command}" ]]; then
        echo "Magento command not found: ${magento_command}" >&2
        exit 2
    fi
    magento_invocation=("${php_binary}" "${magento_command}")
fi
if [[ ! -d "${textfile_directory}" || ! -w "${textfile_directory}" ]]; then
    echo "OpenMetrics textfile directory is unavailable or not writable: ${textfile_directory}" >&2
    exit 2
fi
if [[ -n "${store_id}" && ! "${store_id}" =~ ^[1-9][0-9]*$ ]]; then
    echo "OPENMETRICS_STORE_ID must be one positive integer when set." >&2
    exit 2
fi
if [[ -n "${query_window}" ]]; then
    if [[ ! "${query_window}" =~ ^[1-9][0-9]{1,3}$ ]] \
        || (( query_window < 60 || query_window > 3600 )); then
        echo "OPENMETRICS_QUERY_WINDOW_SECONDS must be an integer from 60 through 3600." >&2
        exit 2
    fi
    if [[ ! "${query_maximum_bytes}" =~ ^[1-9][0-9]{4,7}$ ]] \
        || (( query_maximum_bytes < 65536 || query_maximum_bytes > 52428800 )); then
        echo "OPENMETRICS_QUERY_MAX_BYTES must be an integer from 65536 through 52428800." >&2
        exit 2
    fi
fi

temporary="$(mktemp "${textfile_directory}/.mageos_opensearch_hybrid.prom.XXXXXX")"
cleanup() {
    rm -f -- "${temporary}"
}
trap cleanup EXIT

arguments=(mage-os:opensearch-hybrid:metrics --format=openmetrics)
if [[ -n "${store_id}" ]]; then
    arguments+=("--store=${store_id}")
fi
if [[ -n "${query_window}" ]]; then
    arguments+=("--query-window=${query_window}" "--query-max-bytes=${query_maximum_bytes}")
fi

"${magento_invocation[@]}" "${arguments[@]}" >"${temporary}"
if ! tail -n 1 "${temporary}" | grep -Fxq '# EOF'; then
    echo "OpenMetrics output is incomplete; retaining the previous collector file." >&2
    exit 1
fi

chmod 0644 "${temporary}"
mv -f -- "${temporary}" "${target}"
trap - EXIT
