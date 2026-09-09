"""Bounded HTTP surface for fake and sealed production encoder runtimes."""

from __future__ import annotations

import hmac
import json
import os
import signal
import socket
import stat
import threading
import time
from collections.abc import Mapping
from contextlib import suppress
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import FrameType
from typing import Any, Protocol, cast

from .encoder import DIMENSION, MODEL_ID, MODEL_REVISION, RECIPE_VERSION, RUNTIME, encode

MAX_BODY_BYTES = 1_048_576
MAX_TEXT_BYTES = 32_768
MAX_BATCH_SIZE = 64
METRIC_DURATION_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
OPENMETRICS_CONTENT_TYPE = "application/openmetrics-text; version=1.0.0; charset=utf-8"


class EncoderRequestError(ValueError):
    """A client request violates the encoder contract."""


class EncoderConfigurationError(RuntimeError):
    """The service startup configuration is unsafe or incomplete."""


class EncoderRuntime(Protocol):
    identity: dict[str, object]

    def encode_query(self, query: str) -> list[float]: ...

    def encode_documents(self, documents: list[str]) -> list[list[float]]: ...


class StoppableHttpServer(Protocol):
    def serve_forever(self) -> None: ...

    def shutdown(self) -> None: ...

    def server_close(self) -> None: ...


class FakeEncoderRuntime:
    """Deterministic development runtime that can never pass production gates."""

    def __init__(self) -> None:
        self.identity: dict[str, object] = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dimension": DIMENSION,
            "recipe_version": RECIPE_VERSION,
            "encoder_identity_digest": MODEL_REVISION,
            "runtime": RUNTIME,
            "production_eligible": False,
        }

    def encode_query(self, query: str) -> list[float]:
        return encode(query)

    def encode_documents(self, documents: list[str]) -> list[list[float]]:
        return [encode(document) for document in documents]


FAKE_RUNTIME = FakeEncoderRuntime()


class EncoderMetrics:
    """Fixed-cardinality, dependency-free OpenMetrics counters and histograms."""

    def __init__(self, maximum_concurrency: int) -> None:
        self.maximum_concurrency = maximum_concurrency
        self._lock = threading.Lock()
        self._inflight = 0
        self._requests: dict[tuple[str, str], int] = {}
        self._texts: dict[str, int] = {}
        self._duration_count: dict[str, int] = {}
        self._duration_sum: dict[str, float] = {}
        self._duration_buckets: dict[str, list[int]] = {}

    def request_started(self) -> None:
        with self._lock:
            self._inflight += 1

    def request_finished(
        self,
        *,
        route: str,
        outcome: str,
        elapsed_seconds: float,
        text_count: int,
    ) -> None:
        if route not in ("query", "documents") or outcome not in (
            "ok",
            "unauthorized",
            "client_error",
            "runtime_error",
            "overloaded",
        ):
            raise ValueError("encoder metric labels are invalid")
        if elapsed_seconds < 0.0 or text_count < 0:
            raise ValueError("encoder metric values are invalid")
        with self._lock:
            if self._inflight <= 0:
                raise RuntimeError("encoder metric inflight count is inconsistent")
            self._inflight -= 1
            key = (route, outcome)
            self._requests[key] = self._requests.get(key, 0) + 1
            self._texts[route] = self._texts.get(route, 0) + text_count
            self._duration_count[route] = self._duration_count.get(route, 0) + 1
            self._duration_sum[route] = self._duration_sum.get(route, 0.0) + elapsed_seconds
            buckets = self._duration_buckets.setdefault(
                route, [0 for _bucket in METRIC_DURATION_BUCKETS]
            )
            for position, upper_bound in enumerate(METRIC_DURATION_BUCKETS):
                if elapsed_seconds <= upper_bound:
                    buckets[position] += 1

    def render(self, runtime_identity: Mapping[str, object]) -> bytes:
        with self._lock:
            inflight = self._inflight
            requests = dict(self._requests)
            texts = dict(self._texts)
            duration_count = dict(self._duration_count)
            duration_sum = dict(self._duration_sum)
            duration_buckets = {
                route: list(buckets) for route, buckets in self._duration_buckets.items()
            }
        lines = [
            "# HELP mageos_opensearch_hybrid_encoder_requests_total Completed encoder requests.",
            "# TYPE mageos_opensearch_hybrid_encoder_requests_total counter",
        ]
        for (route, outcome), count in sorted(requests.items()):
            lines.append(
                "mageos_opensearch_hybrid_encoder_requests_total"
                f'{{route="{route}",outcome="{outcome}"}} {count}'
            )
        lines.extend(
            [
                "# HELP mageos_opensearch_hybrid_encoder_texts_total "
                "Text items encoded successfully.",
                "# TYPE mageos_opensearch_hybrid_encoder_texts_total counter",
            ]
        )
        for route, count in sorted(texts.items()):
            lines.append(f'mageos_opensearch_hybrid_encoder_texts_total{{route="{route}"}} {count}')
        lines.extend(
            [
                "# HELP mageos_opensearch_hybrid_encoder_request_duration_seconds "
                "Encoder request processing time.",
                "# TYPE mageos_opensearch_hybrid_encoder_request_duration_seconds histogram",
            ]
        )
        for route in sorted(duration_count):
            for upper_bound, count in zip(
                METRIC_DURATION_BUCKETS, duration_buckets[route], strict=True
            ):
                lines.append(
                    "mageos_opensearch_hybrid_encoder_request_duration_seconds_bucket"
                    f'{{route="{route}",le="{upper_bound:g}"}} {count}'
                )
            lines.append(
                "mageos_opensearch_hybrid_encoder_request_duration_seconds_bucket"
                f'{{route="{route}",le="+Inf"}} {duration_count[route]}'
            )
            lines.append(
                "mageos_opensearch_hybrid_encoder_request_duration_seconds_sum"
                f'{{route="{route}"}} {duration_sum[route]:.12g}'
            )
            lines.append(
                "mageos_opensearch_hybrid_encoder_request_duration_seconds_count"
                f'{{route="{route}"}} {duration_count[route]}'
            )
        lines.extend(
            [
                "# HELP mageos_opensearch_hybrid_encoder_inflight_requests "
                "Current encoder requests.",
                "# TYPE mageos_opensearch_hybrid_encoder_inflight_requests gauge",
                f"mageos_opensearch_hybrid_encoder_inflight_requests {inflight}",
                "# HELP mageos_opensearch_hybrid_encoder_concurrency_limit "
                "Configured request concurrency limit.",
                "# TYPE mageos_opensearch_hybrid_encoder_concurrency_limit gauge",
                f"mageos_opensearch_hybrid_encoder_concurrency_limit {self.maximum_concurrency}",
                "# HELP mageos_opensearch_hybrid_encoder_identity_info Loaded encoder identity.",
                "# TYPE mageos_opensearch_hybrid_encoder_identity_info gauge",
                "mageos_opensearch_hybrid_encoder_identity_info"
                f'{{model_id="{_metric_label(runtime_identity.get("model_id"))}",'
                f'model_revision="{_metric_label(runtime_identity.get("model_revision"))}",'
                f'recipe_version="{_metric_label(runtime_identity.get("recipe_version"))}",'
                f'encoder_identity_digest="{_metric_label(runtime_identity.get("encoder_identity_digest"))}",'
                'production_eligible="'
                f'{str(runtime_identity.get("production_eligible") is True).lower()}"}} 1',
                "# EOF",
            ]
        )
        return ("\n".join(lines) + "\n").encode()


def _metric_label(value: object) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("\r", "\\n")
        .replace("\n", "\\n")
        .replace('"', '\\"')
    )


def identity(runtime: EncoderRuntime = FAKE_RUNTIME) -> dict[str, object]:
    return dict(runtime.identity)


def embed_request(
    payload: object,
    *,
    documents: bool,
    runtime: EncoderRuntime = FAKE_RUNTIME,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise EncoderRequestError("request body must be a JSON object")
    request_id = _required_text(payload, "request_id", maximum_bytes=128)
    _require_identity(payload, runtime.identity)
    key = "documents" if documents else "query"
    raw_texts = payload.get(key)
    if documents:
        if not isinstance(raw_texts, list) or not raw_texts or len(raw_texts) > MAX_BATCH_SIZE:
            raise EncoderRequestError(f"documents must contain 1 to {MAX_BATCH_SIZE} strings")
        texts = [_validated_text(value, key) for value in raw_texts]
    else:
        texts = [_validated_text(raw_texts, key)]
    vectors = runtime.encode_documents(texts) if documents else [runtime.encode_query(texts[0])]
    return {
        "request_id": request_id,
        **identity(runtime),
        "vectors": vectors,
    }


def _require_identity(payload: dict[str, object], loaded: dict[str, object]) -> None:
    if (
        payload.get("model_id") != loaded.get("model_id")
        or payload.get("model_revision") != loaded.get("model_revision")
        or payload.get("dimension") != loaded.get("dimension")
        or payload.get("recipe_version") != loaded.get("recipe_version")
        or payload.get("encoder_identity_digest") != loaded.get("encoder_identity_digest")
    ):
        raise EncoderRequestError("requested model identity is not loaded")


def _required_text(payload: dict[str, object], key: str, *, maximum_bytes: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value or len(value.encode()) > maximum_bytes:
        raise EncoderRequestError(f"{key} must be a non-empty bounded string")
    return value


def _validated_text(value: object, key: str) -> str:
    if not isinstance(value, str) or not value or len(value.encode()) > MAX_TEXT_BYTES:
        raise EncoderRequestError(f"{key} contains an empty or oversized value")
    return value


class EncoderHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_timeout_seconds = 5.0

    def __init__(
        self,
        server_address: tuple[str, int],
        runtime: EncoderRuntime,
        api_token: str,
        maximum_concurrency: int,
        metrics_token: str | None = None,
    ) -> None:
        self.runtime = runtime
        self.api_token = api_token
        self.metrics_token = api_token if metrics_token is None else metrics_token
        self.metrics = EncoderMetrics(maximum_concurrency)
        # Keep HTTP admission independent of expensive model execution so health
        # and metrics requests remain usable while every inference slot is busy.
        self.encoding_slots = threading.BoundedSemaphore(maximum_concurrency)
        self._connection_slots = threading.BoundedSemaphore(maximum_concurrency + 8)
        self._connections: dict[socket.socket, float | None] = {}
        self._connections_lock = threading.Lock()
        super().__init__(server_address, EncoderHandler)

    def process_request(
        self, request: socket.socket | tuple[bytes, socket.socket], client_address: Any
    ) -> None:
        assert isinstance(request, socket.socket)
        if not self._connection_slots.acquire(blocking=False):
            # The accept loop never waits for a worker or for an unresponsive peer.
            request.settimeout(0.05)
            try:
                request.sendall(
                    b"HTTP/1.0 503 Service Unavailable\r\n"
                    b"Content-Type: application/json\r\nContent-Length: 22\r\n"
                    b"Connection: close\r\nRetry-After: 1\r\n\r\n"
                    b'{"error":"overloaded"}'
                )
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        request.settimeout(self.request_timeout_seconds)
        with self._connections_lock:
            self._connections[request] = time.monotonic() + self.request_timeout_seconds
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._finish_connection(request)
            raise

    def process_request_thread(
        self, request: socket.socket | tuple[bytes, socket.socket], client_address: Any
    ) -> None:
        assert isinstance(request, socket.socket)
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._finish_connection(request)

    def _finish_connection(self, request: socket.socket) -> None:
        with self._connections_lock:
            self._connections.pop(request, None)
        self._connection_slots.release()

    def finish_reading(self, request: socket.socket) -> None:
        with self._connections_lock:
            deadline = self._connections.get(request)
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("request read deadline exceeded")
            self._connections[request] = None

    def service_actions(self) -> None:
        # An absolute deadline also bounds peers that trickle bytes frequently
        # enough to avoid the socket's inactivity timeout.
        with self._connections_lock:
            expired = [
                request
                for request, deadline in self._connections.items()
                if deadline is not None and time.monotonic() >= deadline
            ]
            for request in expired:
                with suppress(OSError):
                    request.shutdown(socket.SHUT_RDWR)

    def server_close(self) -> None:
        with self._connections_lock:
            for request in self._connections:
                with suppress(OSError):
                    request.shutdown(socket.SHUT_RDWR)
        super().server_close()


class EncoderHandler(BaseHTTPRequestHandler):
    server_version = "MageOSOpenSearchHybridEncoder/0.2"

    def handle(self) -> None:
        try:
            super().handle()
        except (ConnectionError, TimeoutError):
            # Disconnects and read deadlines are ordinary client outcomes.
            return

    def do_GET(self) -> None:
        cast(EncoderHTTPServer, self.server).finish_reading(self.connection)
        if self.path == "/health/live":
            self._send_json(HTTPStatus.OK, {"status": "live"})
            return
        if self.path == "/health/ready":
            self._send_json(HTTPStatus.OK, {"status": "ready", **identity(self._runtime())})
            return
        if self.path == "/v1/identity":
            if not self._is_authorized():
                self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            self._send_json(HTTPStatus.OK, identity(self._runtime()))
            return
        if self.path == "/metrics":
            if not self._is_metrics_authorized():
                self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            self._send_text(
                HTTPStatus.OK,
                cast(EncoderHTTPServer, self.server).metrics.render(identity(self._runtime())),
                OPENMETRICS_CONTENT_TYPE,
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path not in {"/v1/embed/query", "/v1/embed/documents"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        route = "documents" if self.path.endswith("/documents") else "query"
        metrics = cast(EncoderHTTPServer, self.server).metrics
        metrics.request_started()
        started = time.perf_counter()
        outcome = "runtime_error"
        text_count = 0
        try:
            if not self._is_authorized():
                outcome = "unauthorized"
                status = HTTPStatus.UNAUTHORIZED
                response: dict[str, Any] = {"error": "unauthorized"}
            else:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > MAX_BODY_BYTES:
                        raise EncoderRequestError("request body size is invalid")
                    raw = self.rfile.read(length)
                    server = cast(EncoderHTTPServer, self.server)
                    server.finish_reading(self.connection)
                    payload = json.loads(raw)
                    if not server.encoding_slots.acquire(blocking=False):
                        outcome = "overloaded"
                        status = HTTPStatus.SERVICE_UNAVAILABLE
                        response = {"error": "overloaded"}
                    else:
                        try:
                            response = embed_request(
                                payload,
                                documents=route == "documents",
                                runtime=self._runtime(),
                            )
                            vectors = response.get("vectors")
                            text_count = len(vectors) if isinstance(vectors, list) else 0
                            outcome = "ok"
                            status = HTTPStatus.OK
                        finally:
                            server.encoding_slots.release()
                except (
                    EncoderRequestError,
                    json.JSONDecodeError,
                    UnicodeDecodeError,
                    ValueError,
                ) as error:
                    outcome = "client_error"
                    status = HTTPStatus.BAD_REQUEST
                    response = {"error": str(error)}
                except Exception:
                    outcome = "runtime_error"
                    status = HTTPStatus.SERVICE_UNAVAILABLE
                    response = {"error": "runtime_unavailable"}
        finally:
            metrics.request_finished(
                route=route,
                outcome=outcome,
                elapsed_seconds=time.perf_counter() - started,
                text_count=text_count,
            )
        self._send_json(status, response)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _is_authorized(self) -> bool:
        expected = cast(EncoderHTTPServer, self.server).api_token
        if expected == "":
            return True
        supplied = self.headers.get("Authorization", "")
        return hmac.compare_digest(supplied, f"Bearer {expected}")

    def _runtime(self) -> EncoderRuntime:
        return cast(EncoderHTTPServer, self.server).runtime

    def _is_metrics_authorized(self) -> bool:
        expected = cast(EncoderHTTPServer, self.server).metrics_token
        if expected == "":
            return True
        supplied = self.headers.get("Authorization", "")
        return hmac.compare_digest(supplied, f"Bearer {expected}")

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        self._send_text(status, encoded, "application/json")

    def _send_text(self, status: HTTPStatus, encoded: bytes, content_type: str) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(encoded)


def load_api_token(environ: Mapping[str, str]) -> str:
    """Load one bounded bearer token without following a secret-file symlink."""
    return _load_token(
        environ,
        inline_key="ENCODER_API_TOKEN",
        file_key="ENCODER_API_TOKEN_FILE",
        label="encoder API token",
    )


def load_metrics_token(environ: Mapping[str, str]) -> str:
    """Load the independently scoped metrics token."""
    return _load_token(
        environ,
        inline_key="ENCODER_METRICS_TOKEN",
        file_key="ENCODER_METRICS_TOKEN_FILE",
        label="encoder metrics token",
    )


def _load_token(
    environ: Mapping[str, str],
    *,
    inline_key: str,
    file_key: str,
    label: str,
) -> str:
    inline = environ.get(inline_key, "")
    filename = environ.get(file_key, "")
    if inline and filename:
        raise EncoderConfigurationError(f"configure only one of {inline_key} and {file_key}")
    if inline:
        return _validate_token(inline)
    if not filename:
        return ""
    path = Path(filename)
    if not path.is_absolute():
        raise EncoderConfigurationError(f"{file_key} must be an absolute path")
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_size <= 0 or details.st_size > 4096:
            raise EncoderConfigurationError(f"{label} must be a bounded regular file")
        raw = os.read(descriptor, 4097)
    except OSError as error:
        raise EncoderConfigurationError(
            f"{label} must be a readable bounded regular file"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        token = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EncoderConfigurationError(f"{label} must be UTF-8 text") from error
    token = token.removesuffix("\n").removesuffix("\r")
    return _validate_token(token, label)


def _validate_token(token: str, label: str = "encoder API token") -> str:
    size = len(token.encode())
    if (
        size < 32
        or size > 4096
        or any(ord(character) < 33 or ord(character) > 126 for character in token)
    ):
        raise EncoderConfigurationError(f"{label} must be 32 to 4096 visible ASCII bytes")
    return token


def validate_service_tokens(
    production_eligible: bool,
    api_token: str,
    metrics_token: str,
) -> None:
    if production_eligible and (not api_token or not metrics_token):
        raise EncoderConfigurationError("production mode requires separate API and metrics tokens")
    if bool(api_token) != bool(metrics_token):
        raise EncoderConfigurationError("configure both API and metrics tokens or neither")
    if api_token and hmac.compare_digest(api_token, metrics_token):
        raise EncoderConfigurationError("encoder API and metrics tokens must be different")


def load_runtime(environ: Mapping[str, str]) -> EncoderRuntime:
    mode = environ.get("ENCODER_MODE", "fake")
    if mode == "fake":
        return FAKE_RUNTIME
    if mode != "production":
        raise EncoderConfigurationError("ENCODER_MODE must be fake or production")
    manifest_path = environ.get("ENCODER_IDENTITY_MANIFEST", "")
    artifact_root = environ.get("ENCODER_ARTIFACT_ROOT", "")
    expected_deployment_digest = environ.get("ENCODER_EXPECTED_DEPLOYMENT_DIGEST", "")
    if not manifest_path or not artifact_root or not expected_deployment_digest:
        raise EncoderConfigurationError(
            "production mode requires identity manifest, artifact root, and deployment digest"
        )
    from .runtime import ProductionEncoderRuntime

    return ProductionEncoderRuntime.load(
        Path(manifest_path),
        Path(artifact_root),
        expected_deployment_digest,
    )


def serve_until_stopped(server: StoppableHttpServer) -> None:
    """Translate container stop signals into a bounded ThreadingHTTPServer shutdown."""
    shutdown_thread: threading.Thread | None = None

    def stop(_signum: int, _frame: FrameType | None) -> None:
        nonlocal shutdown_thread
        if shutdown_thread is None:
            shutdown_thread = threading.Thread(
                target=server.shutdown,
                name="encoder-graceful-shutdown",
                daemon=True,
            )
            shutdown_thread.start()

    previous_handlers: dict[signal.Signals, Any] = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, stop)
    try:
        server.serve_forever()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if shutdown_thread is not None:
            shutdown_thread.join(timeout=5)
            if shutdown_thread.is_alive():
                raise EncoderConfigurationError("encoder graceful shutdown did not complete")
        server.server_close()


def main() -> None:
    host = os.environ.get("ENCODER_HOST", "127.0.0.1")
    port = int(os.environ.get("ENCODER_PORT", "8080"))
    maximum_concurrency = int(os.environ.get("ENCODER_MAX_CONCURRENCY", "4"))
    if port < 1 or port > 65535 or maximum_concurrency < 1 or maximum_concurrency > 64:
        raise EncoderConfigurationError("encoder port or maximum concurrency is out of range")
    runtime = load_runtime(os.environ)
    api_token = load_api_token(os.environ)
    metrics_token = load_metrics_token(os.environ)
    validate_service_tokens(
        runtime.identity.get("production_eligible") is True,
        api_token,
        metrics_token,
    )
    server = EncoderHTTPServer(
        (host, port),
        runtime,
        api_token,
        maximum_concurrency,
        metrics_token,
    )
    serve_until_stopped(server)


if __name__ == "__main__":
    main()
