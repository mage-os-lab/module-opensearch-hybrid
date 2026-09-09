#!/usr/bin/env bash

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
promtool="${PROMTOOL:?Set PROMTOOL to the absolute path of the approved promtool binary}"
expected_digest="${PROMTOOL_SHA256:?Set PROMTOOL_SHA256 to the approved promtool SHA-256 digest}"
rules_path="${1:-${script_directory}/prometheus-alert-rules.yml}"
config_path="${2:-${script_directory}/prometheus.yml.example}"
tests_path="${3:-${script_directory}/prometheus-alert-rules.test.yml}"

if [[ "${promtool}" != /* || ! -x "${promtool}" ]]; then
    echo "PROMTOOL must be an absolute executable path." >&2
    exit 2
fi
if [[ ! "${expected_digest}" =~ ^[a-f0-9]{64}$ ]]; then
    echo "PROMTOOL_SHA256 must be a lowercase SHA-256 digest." >&2
    exit 2
fi
if [[ ! -f "${rules_path}" || ! -r "${rules_path}" ]]; then
    echo "Prometheus rules are unavailable or unreadable: ${rules_path}" >&2
    exit 2
fi
if [[ ! -f "${config_path}" || ! -r "${config_path}" ]]; then
    echo "Prometheus configuration is unavailable or unreadable: ${config_path}" >&2
    exit 2
fi
if [[ ! -f "${tests_path}" || ! -r "${tests_path}" ]]; then
    echo "Prometheus rule tests are unavailable or unreadable: ${tests_path}" >&2
    exit 2
fi

if command -v sha256sum >/dev/null 2>&1; then
    actual_digest="$(sha256sum -- "${promtool}" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
    actual_digest="$(shasum -a 256 -- "${promtool}" | awk '{print $1}')"
else
    echo "A SHA-256 utility is required to verify promtool." >&2
    exit 2
fi
if [[ "${actual_digest}" != "${expected_digest}" ]]; then
    echo "promtool digest does not match the approved SHA-256." >&2
    exit 2
fi

"${promtool}" --version
"${promtool}" check rules "${rules_path}"
"${promtool}" check config "${config_path}"
(
    cd -- "$(dirname -- "${tests_path}")"
    "${promtool}" test rules "$(basename -- "${tests_path}")"
)
