from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from mageos_opensearch_hybrid_encoder.server import (
    FAKE_RUNTIME,
    EncoderHTTPServer,
    EncoderRuntime,
)


@contextmanager
def running_server(runtime: EncoderRuntime = FAKE_RUNTIME) -> Iterator[EncoderHTTPServer]:
    server = EncoderHTTPServer(("127.0.0.1", 0), runtime, "t" * 32, 1, "m" * 32)
    server.request_timeout_seconds = 0.2
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


def query_payload() -> str:
    return json.dumps(
        {
            **FAKE_RUNTIME.identity,
            "request_id": "capacity-test",
            "query": "a chair",
        }
    )


def connection(server: EncoderHTTPServer) -> http.client.HTTPConnection:
    return http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=0.5)


def test_health_responds_while_an_unauthenticated_connection_is_idle() -> None:
    with running_server() as server:
        idle = socket.create_connection(("127.0.0.1", server.server_port), timeout=1)
        health = connection(server)
        try:
            health.request("GET", "/health/live")
            response = health.getresponse()
            assert response.status == 200
            response.read()
        finally:
            idle.close()
            health.close()


@pytest.mark.parametrize(
    "partial_request",
    [
        b"GET /health/live HTTP/1.1\r\nHost: localhost\r\nX-Slow: ",
        b"POST /v1/embed/query HTTP/1.1\r\nHost: localhost\r\n"
        b"Authorization: Bearer " + b"t" * 32 + b"\r\nContent-Length: 500\r\n\r\n{",
    ],
)
def test_partial_requests_have_an_absolute_read_deadline(partial_request: bytes) -> None:
    with running_server() as server:
        client = socket.create_connection(("127.0.0.1", server.server_port), timeout=0.7)
        client.sendall(partial_request)
        stopped = threading.Event()

        def trickle() -> None:
            while not stopped.wait(0.04):
                try:
                    client.sendall(b" ")
                except OSError:
                    return

        sender = threading.Thread(target=trickle)
        sender.start()
        started = time.monotonic()
        try:
            # A byte trickle must not extend the total header/body deadline.
            assert client.recv(4096) == b""
            assert time.monotonic() - started < 0.6
        finally:
            stopped.set()
            sender.join(timeout=1)
            client.close()


class BlockingRuntime:
    identity = FAKE_RUNTIME.identity

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def encode_query(self, query: str) -> list[float]:
        self.entered.set()
        assert self.release.wait(timeout=3)
        return FAKE_RUNTIME.encode_query(query)

    def encode_documents(self, documents: list[str]) -> list[list[float]]:
        return [self.encode_query(document) for document in documents]


def test_inference_saturation_preserves_health_and_rejects_excess_work() -> None:
    runtime = BlockingRuntime()
    with running_server(runtime) as server:
        first = connection(server)
        extra = connection(server)
        health = connection(server)
        try:
            first.request(
                "POST",
                "/v1/embed/query",
                body=query_payload(),
                headers={"Authorization": "Bearer " + "t" * 32},
            )
            assert runtime.entered.wait(timeout=1)
            health.request("GET", "/health/ready")
            response = health.getresponse()
            assert response.status == 200
            response.read()
            extra.request(
                "POST",
                "/v1/embed/query",
                body=query_payload(),
                headers={"Authorization": "Bearer " + "t" * 32},
            )
            overloaded = extra.getresponse()
            assert overloaded.status == 503
            assert json.loads(overloaded.read()) == {"error": "overloaded"}
            runtime.release.set()
            completed = first.getresponse()
            assert completed.status == 200
            completed.read()
            extra.request(
                "POST",
                "/v1/embed/query",
                body=query_payload(),
                headers={"Authorization": "Bearer " + "t" * 32},
            )
            recovered = extra.getresponse()
            assert recovered.status == 200
            recovered.read()
        finally:
            runtime.release.set()
            first.close()
            extra.close()
            health.close()


def test_connection_overload_is_bounded_and_recovers_after_idle_clients_expire() -> None:
    with running_server() as server:
        server.request_timeout_seconds = 1.0
        clients = []
        try:
            for expected in range(1, 10):
                clients.append(
                    socket.create_connection(("127.0.0.1", server.server_port), timeout=2)
                )
                # Wait for each accept before opening another connection. Otherwise
                # Linux's listen backlog can cause a retransmit long enough for the
                # previously admitted idle clients to expire during fixture setup.
                deadline = time.monotonic() + 0.5
                while time.monotonic() < deadline:
                    with server._connections_lock:
                        admitted = len(server._connections)
                    if admitted == expected:
                        break
                    time.sleep(0.005)
                assert admitted == expected
            with socket.create_connection(("127.0.0.1", server.server_port), timeout=0.5) as extra:
                assert extra.recv(4096).startswith(b"HTTP/1.0 503 Service Unavailable")
            for client in clients:
                assert client.recv(4096) == b""
            health = connection(server)
            try:
                health.request("GET", "/health/ready")
                assert health.getresponse().status == 200
            finally:
                health.close()
        finally:
            for client in clients:
                client.close()
