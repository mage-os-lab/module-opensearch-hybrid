from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from mageos_opensearch_hybrid_encoder.http_qualification import (
    HttpQualificationError,
    UrllibEncoderHttpClient,
    measure_http_concurrency_four,
)
from mageos_opensearch_hybrid_encoder.qualification import QualificationFailure

IDENTITY_DIGEST = "a" * 64
DEPLOYMENT_DIGEST = "sha256:" + ("b" * 64)


def test_http_client_rejects_credentialed_or_pathful_endpoints(tmp_path: Path) -> None:
    token = tmp_path / "token"
    token.write_text("t" * 32)

    for endpoint in ("http://user:pass@encoder.test", "http://encoder.test/api"):
        with pytest.raises(HttpQualificationError, match="service base URL"):
            UrllibEncoderHttpClient(endpoint, token)


class FakeClient:
    def __init__(self, *, delay: float = 0.0, drift: bool = False) -> None:
        self.delay = delay
        self.drift = drift
        self._lock = threading.Lock()
        self.active = 0
        self.maximum_active = 0

    def get_identity(self) -> dict[str, object]:
        return {
            "model_id": "Snowflake/snowflake-arctic-embed-m-v2.0",
            "model_revision": "95c2741480856aa9666782eb4afe11959938017f",
            "dimension": 256,
            "recipe_version": "mageos-v1",
            "encoder_identity_digest": IDENTITY_DIGEST,
            "production_eligible": True,
            "deployment": {"artifact_digest": DEPLOYMENT_DIGEST},
        }

    def encode_query(
        self,
        query: str,
        request_id: str,
        identity: Mapping[str, object],
    ) -> dict[str, object]:
        with self._lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            return {
                "request_id": request_id,
                "encoder_identity_digest": "c" * 64 if self.drift else IDENTITY_DIGEST,
                "vectors": [[1.0] + ([0.0] * 255)],
            }
        finally:
            with self._lock:
                self.active -= 1


def test_http_latency_records_the_exact_identity_route_and_four_callers() -> None:
    client = FakeClient(delay=0.002)

    evidence = measure_http_concurrency_four(
        client,
        expected_identity_digest=IDENTITY_DIGEST,
        expected_deployment_digest=DEPLOYMENT_DIGEST,
        route_label="docker_network_identity_alias",
        warmup_samples=4,
        measured_samples=8,
        p95_budget_ms=200.0,
        system="Linux",
        machine="x86_64",
    )

    assert evidence["artifact_type"] == "http_latency_concurrency_four"
    assert evidence["status"] == "passed"
    assert evidence["decision_eligible"] is True
    assert evidence["encoder_identity_digest"] == IDENTITY_DIGEST
    assert evidence["deployment_digest"] == DEPLOYMENT_DIGEST
    assert evidence["route"] == "docker_network_identity_alias"
    samples = cast(list[float], evidence["raw_samples_ms"])
    assert len(samples) == 8
    assert client.maximum_active == 4


def test_http_latency_rejects_identity_drift_and_budget_failure() -> None:
    with pytest.raises(HttpQualificationError, match="response identity"):
        measure_http_concurrency_four(
            FakeClient(drift=True),
            expected_identity_digest=IDENTITY_DIGEST,
            expected_deployment_digest=DEPLOYMENT_DIGEST,
            route_label="loopback",
            warmup_samples=4,
            measured_samples=4,
            system="Linux",
            machine="amd64",
        )

    with pytest.raises(QualificationFailure, match="p95") as captured:
        measure_http_concurrency_four(
            FakeClient(delay=0.002),
            expected_identity_digest=IDENTITY_DIGEST,
            expected_deployment_digest=DEPLOYMENT_DIGEST,
            route_label="loopback",
            warmup_samples=4,
            measured_samples=4,
            p95_budget_ms=0.1,
            system="Linux",
            machine="amd64",
        )
    assert captured.value.evidence["status"] == "failed"
    assert captured.value.evidence["decision_eligible"] is False
