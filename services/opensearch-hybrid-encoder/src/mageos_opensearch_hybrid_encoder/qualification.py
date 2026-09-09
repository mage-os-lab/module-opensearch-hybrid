"""Reproducible reference, amd64 parity, and latency evidence for encoder releases."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Protocol, cast

from .identity import EXPECTED_MODEL, EXPECTED_RECIPE

REFERENCE_CONTRACT = "mageos-opensearch-hybrid-encoder-reference-v1"
PARITY_MAXIMUM_ABSOLUTE_DELTA = 1e-5
PARITY_MINIMUM_COSINE_SIMILARITY = 0.999999
ARTIFACT_ROLES = ("query_model", "document_model", "tokenizer")
REFERENCE_CASES = (
    ("query", "trail running shoe"),
    ("query", "solid walnut writing desk with drawers"),
    ("query", "sku-ABC-123"),
    ("query", "waterproof winter coat for women"),
    ("document", "Trail Shoe. Waterproof red running shoe. color: red; material: mesh"),
    ("document", "Writing Desk. Solid walnut desk with drawers. material: walnut"),
    ("document", "ABC-123. Replacement filter cartridge. material: carbon"),
    ("document", "Winter Coat. Insulated waterproof coat. color: navy; material: wool"),
)
LATENCY_QUERIES = tuple(value for kind, value in REFERENCE_CASES if kind == "query")


class QualificationError(RuntimeError):
    """Qualification inputs or observed results do not satisfy the frozen gate."""


class QualificationFailure(QualificationError):
    """A completed measurement failed its gate and carries retainable evidence."""

    def __init__(self, message: str, evidence: dict[str, object]) -> None:
        super().__init__(message)
        self.evidence = evidence


class EncoderRuntime(Protocol):
    def encode_query(self, query: str) -> list[float]: ...

    def encode_documents(self, documents: list[str]) -> list[list[float]]: ...


def artifact_facts(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise QualificationError("qualification artifact must be a regular non-symlink file")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    if size <= 0:
        raise QualificationError("qualification artifact must not be empty")
    return {"sha256": digest.hexdigest(), "size": size}


def capture_reference_vectors(
    runtime: EncoderRuntime,
    artifacts: Mapping[str, Path],
    *,
    system: str | None = None,
    machine: str | None = None,
    python_version: str | None = None,
    package_versions: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Capture deterministic cross-platform reference cases from exact candidate bytes."""
    facts = _artifact_set_facts(artifacts)
    cases = _encode_cases(runtime)
    return {
        "schema_version": 1,
        "artifact_type": "reference_vectors",
        "reference_contract": REFERENCE_CONTRACT,
        "status": "captured",
        "decision_eligible": False,
        "model": _model_identity(),
        "recipe": _recipe_identity(),
        "platform": _platform_facts(system, machine),
        "runtime_versions": _runtime_versions(python_version, package_versions),
        "artifacts": facts,
        "cases": cases,
    }


def verify_amd64_parity(
    runtime: EncoderRuntime,
    artifacts: Mapping[str, Path],
    reference: Mapping[str, object],
    *,
    system: str | None = None,
    machine: str | None = None,
    python_version: str | None = None,
    package_versions: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Verify exact candidate bytes reproduce reference vectors on Linux amd64."""
    platform_facts = _platform_facts(system, machine)
    if platform_facts != {"system": "Linux", "architecture": "amd64"}:
        raise QualificationError("parity qualification requires Linux amd64")
    _validate_reference(reference)
    facts = _artifact_set_facts(artifacts)
    if reference.get("artifacts") != facts:
        raise QualificationError("qualification artifact bytes differ from the reference")
    versions = _runtime_versions(python_version, package_versions)
    if reference.get("runtime_versions") != versions:
        raise QualificationError("qualification runtime versions differ from the reference")

    expected_cases = cast(list[dict[str, object]], reference["cases"])
    actual_cases = _encode_cases(runtime)
    if (
        [case["kind"] for case in actual_cases]
        != [case["kind"] for case in expected_cases]
        or [case["input"] for case in actual_cases]
        != [case["input"] for case in expected_cases]
    ):
        raise QualificationError("reference case identity differs")
    comparisons = [
        _compare_vectors(
            cast(list[float], expected["vector"]),
            cast(list[float], actual["vector"]),
        )
        for expected, actual in zip(expected_cases, actual_cases, strict=True)
    ]
    maximum_delta = max(float(value["maximum_absolute_delta"]) for value in comparisons)
    minimum_cosine = min(float(value["cosine_similarity"]) for value in comparisons)
    passed = (
        maximum_delta <= PARITY_MAXIMUM_ABSOLUTE_DELTA
        and minimum_cosine >= PARITY_MINIMUM_COSINE_SIMILARITY
    )
    evidence: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "amd64_parity",
        "status": "passed" if passed else "failed",
        "decision_eligible": passed,
        "model": _model_identity(),
        "recipe": _recipe_identity(),
        "platform": platform_facts,
        "runtime_versions": versions,
        "artifacts": facts,
        "reference_sha256": _canonical_digest(dict(reference)),
        "case_count": len(comparisons),
        "thresholds": {
            "maximum_absolute_delta": PARITY_MAXIMUM_ABSOLUTE_DELTA,
            "minimum_cosine_similarity": PARITY_MINIMUM_COSINE_SIMILARITY,
        },
        "comparison": {
            "maximum_absolute_delta": maximum_delta,
            "minimum_cosine_similarity": minimum_cosine,
            "cases": comparisons,
        },
    }
    if not passed:
        raise QualificationFailure(
            "amd64 parity failed: "
            f"maximum delta {maximum_delta:.9g}, minimum cosine {minimum_cosine:.9g}",
            evidence,
        )
    return evidence


def measure_concurrency_four(
    runtime: EncoderRuntime,
    artifacts: Mapping[str, Path],
    *,
    queries: Sequence[str] = LATENCY_QUERIES,
    warmup_samples: int = 20,
    measured_samples: int = 100,
    p95_budget_ms: float = 200.0,
    system: str | None = None,
    machine: str | None = None,
    python_version: str | None = None,
    package_versions: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Measure batch-1 query encoding with four simultaneous callers."""
    platform_facts = _platform_facts(system, machine)
    if platform_facts != {"system": "Linux", "architecture": "amd64"}:
        raise QualificationError("latency qualification requires Linux amd64")
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
        raise QualificationError("latency protocol inputs are invalid")
    facts = _artifact_set_facts(artifacts)
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="encoder-qualification") as pool:
        _concurrent_samples(pool, runtime, queries, warmup_samples)
        samples = _concurrent_samples(pool, runtime, queries, measured_samples)
    summary = _latency_summary(samples)
    passed = float(summary["p95"]) <= p95_budget_ms
    evidence: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "latency_concurrency_four",
        "status": "passed" if passed else "failed",
        "decision_eligible": passed,
        "model": _model_identity(),
        "recipe": _recipe_identity(),
        "platform": platform_facts,
        "runtime_versions": _runtime_versions(python_version, package_versions),
        "artifacts": facts,
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
            f"concurrency-four p95 {summary['p95']:.6f} ms exceeds {p95_budget_ms:.6f} ms",
            evidence,
        )
    return evidence


def _artifact_set_facts(artifacts: Mapping[str, Path]) -> dict[str, dict[str, object]]:
    if set(artifacts) != set(ARTIFACT_ROLES):
        raise QualificationError("qualification artifact roles are incomplete")
    return {role: artifact_facts(artifacts[role]) for role in ARTIFACT_ROLES}


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _model_identity() -> dict[str, object]:
    return {
        "id": EXPECTED_MODEL["id"],
        "revision": EXPECTED_MODEL["revision"],
        "dimension": EXPECTED_MODEL["dimension"],
    }


def _recipe_identity() -> dict[str, object]:
    return {
        "version": EXPECTED_RECIPE["version"],
        "query_prefix": EXPECTED_RECIPE["query_prefix"],
        "document_template": EXPECTED_RECIPE["document_template"],
    }


def _platform_facts(system: str | None, machine: str | None) -> dict[str, str]:
    actual_machine = machine if machine is not None else platform.machine()
    normalized_machine = actual_machine.lower()
    if normalized_machine in {"x86_64", "amd64"}:
        normalized_machine = "amd64"
    return {
        "system": system if system is not None else platform.system(),
        "architecture": normalized_machine,
    }


def _runtime_versions(
    python_version: str | None,
    package_versions: Mapping[str, str] | None,
) -> dict[str, str]:
    versions = package_versions
    if versions is None:
        try:
            versions = {
                package: importlib.metadata.version(package)
                for package in ("numpy", "onnxruntime", "tokenizers")
            }
        except importlib.metadata.PackageNotFoundError as error:
            raise QualificationError("qualification runtime package is missing") from error
    if set(versions) != {"numpy", "onnxruntime", "tokenizers"}:
        raise QualificationError("qualification runtime versions are incomplete")
    return {
        "python": python_version if python_version is not None else platform.python_version(),
        "numpy": versions["numpy"],
        "onnxruntime": versions["onnxruntime"],
        "tokenizers": versions["tokenizers"],
    }


def _encode_cases(runtime: EncoderRuntime) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for kind, value in REFERENCE_CASES:
        vector = (
            runtime.encode_query(value)
            if kind == "query"
            else runtime.encode_documents([value])[0]
        )
        if len(vector) != EXPECTED_MODEL["dimension"] or not all(
            math.isfinite(item) for item in vector
        ):
            raise QualificationError("qualification runtime returned an invalid vector")
        cases.append({"kind": kind, "input": value, "vector": vector})
    return cases


def _validate_reference(reference: Mapping[str, object]) -> None:
    if (
        reference.get("schema_version") != 1
        or reference.get("artifact_type") != "reference_vectors"
        or reference.get("reference_contract") != REFERENCE_CONTRACT
        or reference.get("status") != "captured"
        or reference.get("decision_eligible") is not False
        or reference.get("model") != _model_identity()
        or reference.get("recipe") != _recipe_identity()
        or not isinstance(reference.get("artifacts"), dict)
        or not isinstance(reference.get("runtime_versions"), dict)
        or not isinstance(reference.get("cases"), list)
    ):
        raise QualificationError("reference vector artifact is invalid")
    cases = cast(list[object], reference["cases"])
    if len(cases) != len(REFERENCE_CASES):
        raise QualificationError("reference vector cases are incomplete")
    for expected, value in zip(REFERENCE_CASES, cases, strict=True):
        if not isinstance(value, dict) or set(value) != {"kind", "input", "vector"}:
            raise QualificationError("reference vector case is invalid")
        vector = value["vector"]
        if (
            (value["kind"], value["input"]) != expected
            or not isinstance(vector, list)
            or len(vector) != EXPECTED_MODEL["dimension"]
            or not all(isinstance(item, int | float) and math.isfinite(item) for item in vector)
        ):
            raise QualificationError("reference vector case is invalid")


def _compare_vectors(expected: list[float], actual: list[float]) -> dict[str, float]:
    if len(expected) != len(actual) or not expected:
        raise QualificationError("parity vector dimensions differ")
    deltas = [abs(left - right) for left, right in zip(expected, actual, strict=True)]
    expected_norm = math.sqrt(sum(item * item for item in expected))
    actual_norm = math.sqrt(sum(item * item for item in actual))
    if expected_norm <= 0.0 or actual_norm <= 0.0:
        raise QualificationError("parity vector has zero magnitude")
    cosine = sum(left * right for left, right in zip(expected, actual, strict=True)) / (
        expected_norm * actual_norm
    )
    return {
        "maximum_absolute_delta": max(deltas),
        "cosine_similarity": cosine,
    }


def _concurrent_samples(
    pool: ThreadPoolExecutor,
    runtime: EncoderRuntime,
    queries: Sequence[str],
    count: int,
) -> list[float]:
    samples: list[float] = []
    for offset in range(0, count, 4):
        barrier = Barrier(4)
        futures = [
            pool.submit(
                _measure_query,
                runtime,
                queries[(offset + index) % len(queries)],
                barrier,
            )
            for index in range(4)
        ]
        samples.extend(future.result() for future in futures)
    return samples


def _measure_query(runtime: EncoderRuntime, query: str, barrier: Barrier) -> float:
    barrier.wait()
    started = time.perf_counter()
    vector = runtime.encode_query(query)
    elapsed = (time.perf_counter() - started) * 1000.0
    if len(vector) != EXPECTED_MODEL["dimension"]:
        raise QualificationError("latency runtime returned an invalid vector")
    return elapsed


def _latency_summary(samples: Sequence[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p50 = ordered[math.ceil(len(ordered) * 0.50) - 1]
    p95 = ordered[math.ceil(len(ordered) * 0.95) - 1]
    return {
        "minimum": min(ordered),
        "mean": statistics.fmean(ordered),
        "p50": p50,
        "p95": p95,
        "maximum": max(ordered),
    }


def _artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--query-model", type=Path, required=True)
    parser.add_argument("--document-model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def _artifacts(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "query_model": args.query_model,
        "document_model": args.document_model,
        "tokenizer": args.tokenizer,
    }


def _write_evidence(path: Path, evidence: Mapping[str, object]) -> None:
    if not path.is_absolute() or not path.parent.is_dir() or path.is_symlink():
        raise QualificationError("evidence output must be a new absolute path under a directory")
    try:
        with path.open("x") as handle:
            json.dump(evidence, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise QualificationError("evidence output already exists") from error


def _load_runtime(artifacts: Mapping[str, Path]) -> EncoderRuntime:
    from .runtime import ProductionEncoderRuntime

    return ProductionEncoderRuntime.load_qualification_artifacts(
        tokenizer_path=artifacts["tokenizer"],
        query_model_path=artifacts["query_model"],
        document_model_path=artifacts["document_model"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qualify exact encoder artifact bytes")
    commands = parser.add_subparsers(dest="command", required=True)
    reference = commands.add_parser("capture-reference")
    _artifact_arguments(reference)
    parity = commands.add_parser("verify-amd64")
    _artifact_arguments(parity)
    parity.add_argument("--reference", type=Path, required=True)
    latency = commands.add_parser("measure-latency")
    _artifact_arguments(latency)
    latency.add_argument("--warmup-samples", type=int, default=20)
    latency.add_argument("--measured-samples", type=int, default=100)
    latency.add_argument("--p95-budget-ms", type=float, default=200.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = _artifacts(args)
    runtime = _load_runtime(artifacts)
    try:
        if args.command == "capture-reference":
            evidence = capture_reference_vectors(runtime, artifacts)
        elif args.command == "verify-amd64":
            try:
                reference = json.loads(args.reference.read_text())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise QualificationError(
                    f"unable to read reference vectors: {error}"
                ) from error
            if not isinstance(reference, dict):
                raise QualificationError("reference vectors must be a JSON object")
            evidence = verify_amd64_parity(runtime, artifacts, reference)
        else:
            evidence = measure_concurrency_four(
                runtime,
                artifacts,
                warmup_samples=args.warmup_samples,
                measured_samples=args.measured_samples,
                p95_budget_ms=args.p95_budget_ms,
            )
    except QualificationFailure as error:
        _write_evidence(args.output, error.evidence)
        raise SystemExit(f"qualification failed: {error}") from error
    _write_evidence(args.output, evidence)
    print(f"wrote {evidence['artifact_type']} evidence to {args.output}")


if __name__ == "__main__":
    main()
