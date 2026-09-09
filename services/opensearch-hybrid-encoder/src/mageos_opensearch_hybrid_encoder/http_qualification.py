"""Authenticated concurrency-four HTTP evidence for one sealed encoder deployment."""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import statistics
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Protocol, cast
from urllib.parse import urlsplit

from .identity import EXPECTED_MODEL, EXPECTED_RECIPE
from .qualification import LATENCY_QUERIES, QualificationError, QualificationFailure
from .server import load_api_token

IDENTITY_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DEPLOYMENT_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
ROUTE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
IDENTITY_REQUEST_FIELDS = (
    "model_id",
    "model_revision",
    "dimension",
    "recipe_version",
    "encoder_identity_digest",
)


class HttpQualificationError(QualificationError):
    """The HTTP deployment or response does not match the qualification contract."""


class EncoderHttpClient(Protocol):
    def get_identity(self) -> dict[str, object]: ...

    def encode_query(
        self,
        query: str,
        request_id: str,
        identity: Mapping[str, object],
    ) -> dict[str, object]: ...


class UrllibEncoderHttpClient:
    """Small standard-library client that keeps the bearer token out of arguments and output."""

    def __init__(self, endpoint: str, token_file: Path) -> None:
        self._endpoint = _validate_endpoint(endpoint)
        self._token = load_api_token({"ENCODER_API_TOKEN_FILE": str(token_file)})
        if not self._token:
            raise HttpQualificationError("HTTP qualification requires an API token file")

    def get_identity(self) -> dict[str, object]:
        return self._request("/v1/identity")

    def encode_query(
        self,
        query: str,
        request_id: str,
        identity: Mapping[str, object],
    ) -> dict[str, object]:
        return self._request(
            "/v1/embed/query",
            {"request_id": request_id, "query": query, **identity},
        )

    def _request(
        self,
        path: str,
        payload: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = urllib.request.Request(
            self._endpoint + path,
            data=body,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                value = json.load(response)
        except (OSError, ValueError, urllib.error.HTTPError) as error:
            raise HttpQualificationError(f"encoder HTTP request failed: {error}") from error
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise HttpQualificationError("encoder HTTP response must be a JSON object")
        return cast(dict[str, object], value)


def measure_http_concurrency_four(
    client: EncoderHttpClient,
    *,
    expected_identity_digest: str,
    expected_deployment_digest: str,
    route_label: str,
    queries: Sequence[str] = LATENCY_QUERIES,
    warmup_samples: int = 20,
    measured_samples: int = 100,
    p95_budget_ms: float = 200.0,
    system: str | None = None,
    machine: str | None = None,
) -> dict[str, object]:
    """Measure the authenticated query API with four simultaneous batch-1 callers."""
    platform_facts = _platform_facts(system, machine)
    if platform_facts != {"system": "Linux", "architecture": "amd64"}:
        raise HttpQualificationError("HTTP latency qualification requires Linux amd64")
    if IDENTITY_PATTERN.fullmatch(expected_identity_digest) is None:
        raise HttpQualificationError("expected encoder identity digest is invalid")
    if DEPLOYMENT_PATTERN.fullmatch(expected_deployment_digest) is None:
        raise HttpQualificationError("expected deployment digest is invalid")
    if ROUTE_PATTERN.fullmatch(route_label) is None:
        raise HttpQualificationError("route label is invalid")
    if (
        len(queries) < 4
        or len(set(queries)) < 4
        or warmup_samples < 4
        or measured_samples < 4
        or warmup_samples % 4
        or measured_samples % 4
        or not math.isfinite(p95_budget_ms)
        or p95_budget_ms <= 0.0
    ):
        raise HttpQualificationError("HTTP latency protocol inputs are invalid")

    identity = client.get_identity()
    _validate_identity(identity, expected_identity_digest, expected_deployment_digest)
    request_identity = {key: identity[key] for key in IDENTITY_REQUEST_FIELDS}
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="http-qualification") as pool:
        _concurrent_samples(
            pool,
            client,
            request_identity,
            expected_identity_digest,
            queries,
            warmup_samples,
            "warmup",
        )
        samples = _concurrent_samples(
            pool,
            client,
            request_identity,
            expected_identity_digest,
            queries,
            measured_samples,
            "measured",
        )
    summary = _latency_summary(samples)
    passed = summary["p95"] <= p95_budget_ms
    evidence: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "http_latency_concurrency_four",
        "status": "passed" if passed else "failed",
        "decision_eligible": passed,
        "platform": platform_facts,
        "deployment_digest": expected_deployment_digest,
        "encoder_identity_digest": expected_identity_digest,
        "route": route_label,
        "concurrency": 4,
        "batch_size": 1,
        "warmup_samples": warmup_samples,
        "measured_samples": measured_samples,
        "distinct_queries": len(set(queries)),
        "p95_budget_ms": p95_budget_ms,
        "raw_samples_ms": samples,
        "summary_ms": summary,
    }
    if not passed:
        raise QualificationFailure(
            f"HTTP concurrency-four p95 {summary['p95']:.6f} ms exceeds "
            f"{p95_budget_ms:.6f} ms",
            evidence,
        )
    return evidence


def _validate_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise HttpQualificationError("encoder endpoint must be an HTTP service base URL")
    return endpoint.rstrip("/")


def _platform_facts(system: str | None, machine: str | None) -> dict[str, str]:
    architecture = (machine if machine is not None else platform.machine()).lower()
    if architecture in {"x86_64", "amd64"}:
        architecture = "amd64"
    return {
        "system": system if system is not None else platform.system(),
        "architecture": architecture,
    }


def _validate_identity(
    identity: Mapping[str, object],
    expected_identity_digest: str,
    expected_deployment_digest: str,
) -> None:
    deployment = identity.get("deployment")
    if (
        identity.get("model_id") != EXPECTED_MODEL["id"]
        or identity.get("model_revision") != EXPECTED_MODEL["revision"]
        or identity.get("dimension") != EXPECTED_MODEL["dimension"]
        or identity.get("recipe_version") != EXPECTED_RECIPE["version"]
        or identity.get("encoder_identity_digest") != expected_identity_digest
        or identity.get("production_eligible") is not True
        or not isinstance(deployment, dict)
        or deployment.get("artifact_digest") != expected_deployment_digest
    ):
        raise HttpQualificationError("encoder HTTP identity does not match the sealed deployment")


def _concurrent_samples(
    pool: ThreadPoolExecutor,
    client: EncoderHttpClient,
    identity: Mapping[str, object],
    expected_identity_digest: str,
    queries: Sequence[str],
    count: int,
    phase: str,
) -> list[float]:
    samples: list[float] = []
    for offset in range(0, count, 4):
        barrier = Barrier(4)
        futures = [
            pool.submit(
                _measure_request,
                client,
                queries[(offset + index) % len(queries)],
                f"http-qualification-{phase}-{offset + index}",
                identity,
                expected_identity_digest,
                barrier,
            )
            for index in range(4)
        ]
        samples.extend(future.result() for future in futures)
    return samples


def _measure_request(
    client: EncoderHttpClient,
    query: str,
    request_id: str,
    identity: Mapping[str, object],
    expected_identity_digest: str,
    barrier: Barrier,
) -> float:
    barrier.wait()
    started = time.perf_counter()
    response = client.encode_query(query, request_id, identity)
    elapsed = (time.perf_counter() - started) * 1000.0
    vectors = response.get("vectors")
    if (
        response.get("request_id") != request_id
        or response.get("encoder_identity_digest") != expected_identity_digest
        or not isinstance(vectors, list)
        or len(vectors) != 1
        or not isinstance(vectors[0], list)
        or len(vectors[0]) != EXPECTED_MODEL["dimension"]
        or not all(isinstance(value, int | float) and math.isfinite(value) for value in vectors[0])
    ):
        raise HttpQualificationError("encoder HTTP response identity or vector is invalid")
    return elapsed


def _latency_summary(samples: Sequence[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "minimum": min(ordered),
        "mean": statistics.fmean(ordered),
        "p50": ordered[math.ceil(len(ordered) * 0.50) - 1],
        "p95": ordered[math.ceil(len(ordered) * 0.95) - 1],
        "maximum": max(ordered),
    }


def _write_evidence(path: Path, evidence: Mapping[str, object]) -> None:
    if not path.is_absolute() or not path.parent.is_dir() or path.is_symlink():
        raise HttpQualificationError("HTTP evidence output must be a new absolute path")
    try:
        with path.open("x") as handle:
            json.dump(evidence, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise HttpQualificationError("HTTP evidence output already exists") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qualify one sealed encoder HTTP route")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--expected-identity-digest", required=True)
    parser.add_argument("--expected-deployment-digest", required=True)
    parser.add_argument("--route-label", required=True)
    parser.add_argument("--warmup-samples", type=int, default=20)
    parser.add_argument("--measured-samples", type=int, default=100)
    parser.add_argument("--p95-budget-ms", type=float, default=200.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = UrllibEncoderHttpClient(args.endpoint, args.token_file)
    try:
        evidence = measure_http_concurrency_four(
            client,
            expected_identity_digest=args.expected_identity_digest,
            expected_deployment_digest=args.expected_deployment_digest,
            route_label=args.route_label,
            warmup_samples=args.warmup_samples,
            measured_samples=args.measured_samples,
            p95_budget_ms=args.p95_budget_ms,
        )
    except QualificationFailure as error:
        _write_evidence(args.output, error.evidence)
        raise SystemExit(f"qualification failed: {error}") from error
    _write_evidence(args.output, evidence)
    print(f"wrote HTTP latency evidence to {args.output}")


if __name__ == "__main__":
    main()
