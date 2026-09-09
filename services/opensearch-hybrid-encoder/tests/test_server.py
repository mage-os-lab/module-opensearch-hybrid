import http.client
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from mageos_opensearch_hybrid_encoder.server import (
    FAKE_RUNTIME,
    OPENMETRICS_CONTENT_TYPE,
    EncoderConfigurationError,
    EncoderHTTPServer,
    EncoderMetrics,
    load_api_token,
    load_metrics_token,
    load_runtime,
    serve_until_stopped,
    validate_service_tokens,
)


class FakeStoppableServer:
    def __init__(self, handlers: dict[int, Any]) -> None:
        self.handlers = handlers
        self.shutdown_called = threading.Event()
        self.closed = False

    def serve_forever(self) -> None:
        handler = self.handlers[signal.SIGTERM]
        assert callable(handler)
        handler(signal.SIGTERM, None)
        assert self.shutdown_called.wait(timeout=1)

    def shutdown(self) -> None:
        self.shutdown_called.set()

    def server_close(self) -> None:
        self.closed = True


class FailingRuntime:
    def __init__(self) -> None:
        self.identity: dict[str, object] = dict(FAKE_RUNTIME.identity)

    def encode_query(self, query: str) -> list[float]:
        raise RuntimeError("private runtime detail")

    def encode_documents(self, documents: list[str]) -> list[list[float]]:
        raise RuntimeError("private runtime detail")


def test_metrics_are_fixed_cardinality_cumulative_and_identity_bound() -> None:
    metrics = EncoderMetrics(maximum_concurrency=4)
    metrics.request_started()
    metrics.request_finished(route="query", outcome="ok", elapsed_seconds=0.04, text_count=1)
    metrics.request_started()
    metrics.request_finished(
        route="documents",
        outcome="runtime_error",
        elapsed_seconds=0.6,
        text_count=0,
    )

    rendered = metrics.render(FAKE_RUNTIME.identity).decode()

    assert rendered.endswith("# EOF\n")
    assert (
        'mageos_opensearch_hybrid_encoder_requests_total{route="query",outcome="ok"} 1' in rendered
    )
    assert (
        "mageos_opensearch_hybrid_encoder_request_duration_seconds_bucket"
        '{route="query",le="0.025"} 0' in rendered
    )
    assert (
        "mageos_opensearch_hybrid_encoder_request_duration_seconds_bucket"
        '{route="query",le="0.05"} 1' in rendered
    )
    assert (
        "mageos_opensearch_hybrid_encoder_request_duration_seconds_bucket"
        '{route="query",le="+Inf"} 1' in rendered
    )
    assert "mageos_opensearch_hybrid_encoder_inflight_requests 0" in rendered
    assert "mageos_opensearch_hybrid_encoder_concurrency_limit 4" in rendered
    assert str(FAKE_RUNTIME.identity["encoder_identity_digest"]) in rendered
    assert "request_id" not in rendered


def test_metrics_endpoint_requires_token_and_records_success_without_text() -> None:
    token = "t" * 32
    metrics_token = "m" * 32
    server = EncoderHTTPServer(("127.0.0.1", 0), FAKE_RUNTIME, token, 4, metrics_token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    try:
        connection.request("GET", "/metrics")
        unauthorized = connection.getresponse()
        assert unauthorized.status == 401
        unauthorized.read()

        connection.request("GET", "/metrics", headers={"Authorization": f"Bearer {token}"})
        wrong_scope = connection.getresponse()
        assert wrong_scope.status == 401
        wrong_scope.read()

        payload = {
            "request_id": "request-1",
            "model_id": FAKE_RUNTIME.identity["model_id"],
            "model_revision": FAKE_RUNTIME.identity["model_revision"],
            "dimension": FAKE_RUNTIME.identity["dimension"],
            "recipe_version": FAKE_RUNTIME.identity["recipe_version"],
            "encoder_identity_digest": FAKE_RUNTIME.identity["encoder_identity_digest"],
            "query": "private query must not enter metrics",
        }
        connection.request(
            "POST",
            "/v1/embed/query",
            body=json.dumps(payload),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        encoded = connection.getresponse()
        assert encoded.status == 200
        encoded.read()

        connection.request(
            "GET",
            "/metrics",
            headers={"Authorization": f"Bearer {metrics_token}"},
        )
        metrics = connection.getresponse()
        body = metrics.read().decode()
        assert metrics.status == 200
        assert metrics.getheader("Content-Type") == OPENMETRICS_CONTENT_TYPE
        assert 'requests_total{route="query",outcome="ok"} 1' in body
        assert 'texts_total{route="query"} 1' in body
        assert "private query" not in body
        assert body.endswith("# EOF\n")
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_runtime_failure_is_bounded_and_counted_without_diagnostic_leakage() -> None:
    runtime = FailingRuntime()
    server = EncoderHTTPServer(("127.0.0.1", 0), runtime, "", 4)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    try:
        payload = {
            "request_id": "request-1",
            "model_id": runtime.identity["model_id"],
            "model_revision": runtime.identity["model_revision"],
            "dimension": runtime.identity["dimension"],
            "recipe_version": runtime.identity["recipe_version"],
            "encoder_identity_digest": runtime.identity["encoder_identity_digest"],
            "query": "private query",
        }
        connection.request("POST", "/v1/embed/query", body=json.dumps(payload))
        response = connection.getresponse()
        body = response.read().decode()
        assert response.status == 503
        assert json.loads(body) == {"error": "runtime_unavailable"}
        assert "private runtime detail" not in body

        connection.request("GET", "/metrics")
        metrics = connection.getresponse().read().decode()
        assert 'requests_total{route="query",outcome="runtime_error"} 1' in metrics
        assert "private query" not in metrics
        assert "private runtime detail" not in metrics
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_server_handles_sigterm_with_graceful_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers: dict[int, Any] = {}

    def install(signum: int, handler: Any) -> Any:
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    monkeypatch.setattr(signal, "signal", install)
    server = FakeStoppableServer(handlers)

    serve_until_stopped(server)

    assert server.shutdown_called.is_set()
    assert server.closed is True
    assert handlers[signal.SIGTERM] is signal.SIG_DFL


def test_fake_server_import_does_not_require_production_dependencies() -> None:
    source = Path(__file__).resolve().parents[1] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source)

    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "from mageos_opensearch_hybrid_encoder.server import identity; "
            "assert identity()['production_eligible'] is False",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr


def test_api_token_can_be_loaded_from_a_read_only_secret_file(tmp_path: Path) -> None:
    token_file = tmp_path / "encoder-token"
    token_file.write_text(("s" * 32) + "\n")

    assert load_api_token({"ENCODER_API_TOKEN_FILE": str(token_file)}) == "s" * 32

    metrics_file = tmp_path / "metrics-token"
    metrics_file.write_text(("m" * 32) + "\n")
    assert load_metrics_token({"ENCODER_METRICS_TOKEN_FILE": str(metrics_file)}) == "m" * 32


def test_api_token_rejects_ambiguous_or_unsafe_configuration(tmp_path: Path) -> None:
    token_file = tmp_path / "encoder-token"
    token_file.write_text("s" * 32)

    with pytest.raises(EncoderConfigurationError, match="only one"):
        load_api_token(
            {
                "ENCODER_API_TOKEN": "i" * 32,
                "ENCODER_API_TOKEN_FILE": str(token_file),
            }
        )

    symlink = tmp_path / "token-link"
    symlink.symlink_to(token_file)
    with pytest.raises(EncoderConfigurationError, match="regular file"):
        load_api_token({"ENCODER_API_TOKEN_FILE": str(symlink)})


def test_api_and_metrics_tokens_are_independently_scoped() -> None:
    validate_service_tokens(False, "", "")
    validate_service_tokens(False, "a" * 32, "m" * 32)
    validate_service_tokens(True, "a" * 32, "m" * 32)

    with pytest.raises(EncoderConfigurationError, match="both API and metrics"):
        validate_service_tokens(False, "a" * 32, "")
    with pytest.raises(EncoderConfigurationError, match="must be different"):
        validate_service_tokens(True, "a" * 32, "a" * 32)


def test_production_runtime_receives_registry_neutral_deployment_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mageos_opensearch_hybrid_encoder import runtime as runtime_module

    loaded: list[tuple[Path, Path, str]] = []
    sentinel = object()

    def fake_load(manifest_path: Path, artifact_root: Path, digest: str) -> object:
        loaded.append((manifest_path, artifact_root, digest))
        return sentinel

    monkeypatch.setattr(
        runtime_module.ProductionEncoderRuntime,
        "load",
        staticmethod(fake_load),
    )
    environment = {
        "ENCODER_MODE": "production",
        "ENCODER_IDENTITY_MANIFEST": "/run/encoder/identity.json",
        "ENCODER_ARTIFACT_ROOT": "/srv/encoder/artifacts",
        "ENCODER_EXPECTED_DEPLOYMENT_DIGEST": "sha256:" + ("a" * 64),
    }

    assert load_runtime(environment) is sentinel
    assert loaded == [
        (
            Path("/run/encoder/identity.json"),
            Path("/srv/encoder/artifacts"),
            "sha256:" + ("a" * 64),
        )
    ]


def test_production_runtime_does_not_accept_legacy_image_digest_setting() -> None:
    with pytest.raises(EncoderConfigurationError, match="deployment digest"):
        load_runtime(
            {
                "ENCODER_MODE": "production",
                "ENCODER_IDENTITY_MANIFEST": "/run/encoder/identity.json",
                "ENCODER_ARTIFACT_ROOT": "/srv/encoder/artifacts",
                "ENCODER_EXPECTED_IMAGE_DIGEST": "sha256:" + ("a" * 64),
            }
        )
