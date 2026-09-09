from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest

from mageos_opensearch_hybrid_encoder.qualification import (
    QualificationError,
    QualificationFailure,
    artifact_facts,
    capture_reference_vectors,
    measure_concurrency_four,
    verify_amd64_parity,
)


class FakeRuntime:
    def __init__(self, *, drift: float = 0.0, delay: float = 0.0) -> None:
        self.drift = drift
        self.delay = delay
        self._lock = threading.Lock()
        self.active = 0
        self.maximum_active = 0

    def encode_query(self, query: str) -> list[float]:
        with self._lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            return self._vector(len(query))
        finally:
            with self._lock:
                self.active -= 1

    def encode_documents(self, documents: list[str]) -> list[list[float]]:
        return [self._vector(len(document) + 1) for document in documents]

    def _vector(self, seed: int) -> list[float]:
        vector = [0.0] * 256
        vector[seed % 256] = 1.0
        vector[0] += self.drift
        return vector


def model_artifacts(tmp_path: Path) -> dict[str, Path]:
    artifacts = {
        "query_model": tmp_path / "query.onnx",
        "document_model": tmp_path / "document.onnx",
        "tokenizer": tmp_path / "tokenizer.json",
    }
    for role, path in artifacts.items():
        path.write_bytes(role.encode())
    return artifacts


def test_reference_vectors_bind_exact_artifacts_runtime_and_cases(tmp_path: Path) -> None:
    artifacts = model_artifacts(tmp_path)

    evidence = capture_reference_vectors(
        FakeRuntime(),
        artifacts,
        system="Darwin",
        machine="arm64",
        python_version="3.13.15",
        package_versions={
            "numpy": "2.5.2",
            "onnxruntime": "1.28.0",
            "tokenizers": "0.22.2",
        },
    )

    assert evidence["artifact_type"] == "reference_vectors"
    assert evidence["status"] == "captured"
    assert evidence["decision_eligible"] is False
    platform_facts = cast(dict[str, str], evidence["platform"])
    artifact_records = cast(dict[str, dict[str, object]], evidence["artifacts"])
    cases = cast(list[dict[str, Any]], evidence["cases"])
    assert platform_facts["architecture"] == "arm64"
    assert artifact_records["query_model"] == artifact_facts(artifacts["query_model"])
    assert len(cases) >= 8
    assert all(len(case["vector"]) == 256 for case in cases)


def test_amd64_parity_requires_linux_amd64_exact_artifacts_and_bounded_drift(
    tmp_path: Path,
) -> None:
    artifacts = model_artifacts(tmp_path)
    reference = capture_reference_vectors(
        FakeRuntime(),
        artifacts,
        system="Darwin",
        machine="arm64",
        python_version="3.13.15",
        package_versions={
            "numpy": "2.5.2",
            "onnxruntime": "1.28.0",
            "tokenizers": "0.22.2",
        },
    )

    evidence = verify_amd64_parity(
        FakeRuntime(drift=1e-7),
        artifacts,
        reference,
        system="Linux",
        machine="x86_64",
        python_version="3.13.15",
        package_versions={
            "numpy": "2.5.2",
            "onnxruntime": "1.28.0",
            "tokenizers": "0.22.2",
        },
    )

    assert evidence["artifact_type"] == "amd64_parity"
    assert evidence["status"] == "passed"
    assert evidence["decision_eligible"] is True
    comparison = cast(dict[str, float], evidence["comparison"])
    assert comparison["maximum_absolute_delta"] <= 1e-5

    with pytest.raises(QualificationError, match="Linux amd64"):
        verify_amd64_parity(
            FakeRuntime(),
            artifacts,
            reference,
            system="Darwin",
            machine="arm64",
        )

    changed = dict(artifacts)
    changed["query_model"].write_bytes(b"changed")
    with pytest.raises(QualificationError, match="artifact"):
        verify_amd64_parity(
            FakeRuntime(),
            changed,
            reference,
            system="Linux",
            machine="x86_64",
        )


def test_amd64_parity_rejects_vector_drift(tmp_path: Path) -> None:
    artifacts = model_artifacts(tmp_path)
    reference = capture_reference_vectors(
        FakeRuntime(),
        artifacts,
        system="Darwin",
        machine="arm64",
    )

    with pytest.raises(QualificationFailure, match="parity") as captured:
        verify_amd64_parity(
            FakeRuntime(drift=0.01),
            artifacts,
            reference,
            system="Linux",
            machine="amd64",
        )
    assert captured.value.evidence["status"] == "failed"
    assert captured.value.evidence["decision_eligible"] is False


def test_concurrency_four_latency_records_raw_samples_and_enforces_budget(
    tmp_path: Path,
) -> None:
    artifacts = model_artifacts(tmp_path)
    runtime = FakeRuntime(delay=0.002)

    evidence = measure_concurrency_four(
        runtime,
        artifacts,
        queries=["first", "second", "third", "fourth"],
        warmup_samples=4,
        measured_samples=8,
        p95_budget_ms=200.0,
        system="Linux",
        machine="x86_64",
    )

    assert evidence["artifact_type"] == "latency_concurrency_four"
    assert evidence["status"] == "passed"
    assert evidence["decision_eligible"] is True
    assert evidence["concurrency"] == 4
    samples = cast(list[float], evidence["raw_samples_ms"])
    summary = cast(dict[str, float], evidence["summary_ms"])
    assert len(samples) == 8
    assert summary["p95"] <= 200.0
    assert runtime.maximum_active == 4

    with pytest.raises(QualificationFailure, match="p95") as captured:
        measure_concurrency_four(
            FakeRuntime(delay=0.002),
            artifacts,
            queries=["first", "second", "third", "fourth"],
            warmup_samples=4,
            measured_samples=4,
            p95_budget_ms=0.1,
            system="Linux",
            machine="x86_64",
        )
    assert captured.value.evidence["status"] == "failed"
    assert captured.value.evidence["decision_eligible"] is False
