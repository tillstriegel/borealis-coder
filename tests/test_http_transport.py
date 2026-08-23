from __future__ import annotations

import asyncio
import io
import threading
import unittest
import urllib.error
import urllib.request
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast
from unittest.mock import patch

from borealis_coder.errors import ProviderError, ProviderRateLimitError, ProviderUnavailableError
from borealis_coder.network import SameOriginRedirectHandler
from borealis_coder.providers import http as http_module
from borealis_coder.providers.http import HttpClient


class FakeResponse:
    def __init__(
        self,
        body: bytes = b"{}",
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        lines: list[bytes] | None = None,
    ) -> None:
        self.body = body
        self.status = status
        self.headers = headers or {"Content-Type": "application/json"}
        self.lines = iter(lines or [])

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, *_args: object) -> bytes:
        return self.body

    def readline(self) -> bytes:
        return next(self.lines, b"")


class HttpTransportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.urlopen_only = patch.object(HttpClient, "_pooled_target", return_value=None)
        self.urlopen_only.start()

    def tearDown(self) -> None:
        self.urlopen_only.stop()

    async def test_post_json_success_text_http_and_network_errors(self) -> None:
        client = HttpClient(timeout_seconds=1)
        with patch(
            "borealis_coder.providers.http.open_same_origin",
            return_value=FakeResponse(b'{"ok":true}', headers={"Content-Type": "application/json"}),
        ):
            response = await client.post_json("https://example.test", payload={"x": 1})
        self.assertEqual(response.data, {"ok": True})
        self.assertEqual(response.status, 200)

        with patch(
            "borealis_coder.providers.http.open_same_origin",
            return_value=FakeResponse(b"plain", headers={"Content-Type": "text/plain"}),
        ):
            response = await client.post_json("https://example.test", payload={})
        self.assertEqual(response.data, "plain")

        headers = Message()
        headers["Content-Type"] = "application/json"
        http_error_stream = io.BytesIO(b'{"error":{"message":"slow down"}}')
        http_error = urllib.error.HTTPError(
            "https://example.test",
            429,
            "rate",
            headers,
            http_error_stream,
        )
        with (
            patch(
                "borealis_coder.providers.http.open_same_origin",
                side_effect=http_error,
            ),
            self.assertRaisesRegex(ProviderRateLimitError, "slow down"),
        ):
            await client.post_json("https://example.test", payload={})
        self.assertTrue(http_error_stream.closed)

        text_error_stream = io.BytesIO(b"not json")
        text_error = urllib.error.HTTPError(
            "https://example.test", 400, "bad", headers, text_error_stream
        )
        with (
            patch(
                "borealis_coder.providers.http.open_same_origin",
                side_effect=text_error,
            ),
            self.assertRaises(ProviderError),
        ):
            await client.post_json("https://example.test", payload={})
        self.assertTrue(text_error_stream.closed)

        with (
            patch(
                "borealis_coder.providers.http.open_same_origin",
                side_effect=urllib.error.URLError("offline"),
            ),
            self.assertRaisesRegex(ProviderError, "offline"),
        ):
            await client.post_json("https://example.test", payload={})

    async def test_stream_sse_parsing_and_errors(self) -> None:
        lines = [
            b": heartbeat\n",
            b"event: delta\n",
            b"id: 7\n",
            b"data: first\n",
            b"data: second\n",
            b"\n",
            b"data: final\n",
        ]
        with patch(
            "borealis_coder.providers.http.open_same_origin",
            return_value=FakeResponse(lines=lines, headers={"Content-Type": "text/event-stream"}),
        ):
            events = [
                item
                async for item in HttpClient(timeout_seconds=1).stream_sse("https://x", payload={})
            ]
        self.assertEqual(events[0].event, "delta")
        self.assertEqual(events[0].data, "first\nsecond")
        self.assertEqual(events[0].id, "7")
        self.assertEqual(events[1].data, "final")

        headers = Message()
        http_error_stream = io.BytesIO(b'{"error":"unavailable"}')
        http_error = urllib.error.HTTPError(
            "https://x", 503, "down", headers, http_error_stream
        )
        with (
            patch(
                "borealis_coder.providers.http.open_same_origin",
                side_effect=http_error,
            ),
            self.assertRaises(ProviderUnavailableError),
        ):
            _ = [item async for item in HttpClient().stream_sse("https://x", payload={})]
        self.assertTrue(http_error_stream.closed)

        with (
            patch(
                "borealis_coder.providers.http.open_same_origin",
                side_effect=RuntimeError("thread failed"),
            ),
            self.assertRaisesRegex(ProviderError, "thread failed"),
        ):
            _ = [item async for item in HttpClient().stream_sse("https://x", payload={})]

    async def test_stream_sse_applies_backpressure_to_a_fast_producer(self) -> None:
        class TrackingQueue(asyncio.Queue):
            peak = 0

            def put_nowait(self, item: object) -> None:
                super().put_nowait(item)
                type(self).peak = max(type(self).peak, self.qsize())

        lines = [part for _ in range(1_000) for part in (b"data: x\n", b"\n")]
        with (
            patch.object(http_module.asyncio, "Queue", TrackingQueue),
            patch.object(
                http_module,
                "open_same_origin",
                return_value=FakeResponse(
                    lines=lines,
                    headers={"Content-Type": "text/event-stream"},
                ),
            ),
        ):
            stream = HttpClient().stream_sse("https://x", payload={})
            first = await anext(stream)
            await asyncio.sleep(0.1)
            self.assertEqual(first.data, "x")
            self.assertLessEqual(TrackingQueue.peak, 64)
            await cast(Any, stream).aclose()

    async def test_stream_sse_close_interrupts_a_blocked_response(self) -> None:
        class BlockingResponse:
            status = 200

            def __init__(self) -> None:
                self.headers = {"Content-Type": "text/event-stream"}
                self.calls = 0
                self.release = threading.Event()
                self.closed = threading.Event()

            def __enter__(self):
                return self

            def __exit__(self, *_args: object) -> None:
                self.close()

            def close(self) -> None:
                self.closed.set()
                self.release.set()

            def readline(self) -> bytes:
                self.calls += 1
                if self.calls == 1:
                    return b"data: first\n"
                if self.calls == 2:
                    return b"\n"
                self.release.wait(5)
                return b""

        response = BlockingResponse()
        existing = {
            thread.ident for thread in threading.enumerate() if thread.name == "borealis-sse"
        }
        with patch.object(
            http_module,
            "open_same_origin",
            return_value=response,
        ):
            stream = HttpClient().stream_sse("https://x", payload={})
            first = await anext(stream)
            await cast(Any, stream).aclose()
        self.assertEqual(first.data, "first")
        self.assertTrue(response.closed.is_set())
        remaining = [
            thread
            for thread in threading.enumerate()
            if thread.name == "borealis-sse" and thread.ident not in existing
        ]
        self.assertEqual(remaining, [])

    def test_decode_and_error_message_helpers(self) -> None:
        self.assertEqual(HttpClient._decode(b"[1,2]", {}), [1, 2])
        self.assertEqual(HttpClient._decode(b"hello", {}), "hello")
        with self.assertRaises(ProviderError):
            HttpClient._decode(b"{bad", {"content-type": "application/json"})
        self.assertEqual(HttpClient._error_message({"error": {"status": "BAD"}}), "BAD")
        self.assertEqual(HttpClient._error_message({"error": "oops"}), "oops")
        self.assertEqual(HttpClient._error_message({"message": "m"}), "m")
        self.assertEqual(HttpClient._error_message("raw"), "raw")
        self.assertEqual(HttpClient._error_message(None), "")

    def test_redirects_must_stay_on_the_same_origin(self) -> None:
        request = urllib.request.Request(
            "https://provider.example/v1/messages",
            data=b"{}",
            headers={"Authorization": "Bearer synthetic-secret"},
            method="POST",
        )
        handler = SameOriginRedirectHandler()
        with self.assertRaisesRegex(urllib.error.HTTPError, "Cross-origin") as captured:
            handler.redirect_request(
                request,
                io.BytesIO(),
                302,
                "Found",
                Message(),
                "https://attacker.example/collect",
            )
        captured.exception.close()

        redirected = handler.redirect_request(
            request,
            io.BytesIO(),
            302,
            "Found",
            Message(),
            "https://provider.example/v1/next",
        )
        assert redirected is not None
        self.assertEqual(redirected.full_url, "https://provider.example/v1/next")
        self.assertEqual(redirected.get_header("Authorization"), "Bearer synthetic-secret")

        with self.assertRaisesRegex(ProviderError, "unsafe HTTP redirect"):
            HttpClient._redirect_request(
                "https://provider.example/v1/messages",
                302,
                {"location": "https://attacker.example/collect"},
                {"Authorization": "Bearer synthetic-secret"},
            )


class HttpConnectionReuseTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_json_errors_and_sse_reuse_one_connection(self) -> None:
        client_ports: list[int] = []
        requests: list[tuple[str, str]] = []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                client_ports.append(self.client_address[1])
                requests.append(("POST", self.path))
                if self.path == "/redirect":
                    body = b""
                    status = 302
                    content_type = "text/plain"
                elif self.path == "/rate":
                    body = b'{"error":{"message":"slow down"}}'
                    status = 429
                    content_type = "application/json"
                elif self.path == "/stream":
                    body = b"data: streamed\n\n"
                    status = 200
                    content_type = "text/event-stream"
                else:
                    body = b'{"ok":true}'
                    status = 200
                    content_type = "application/json"
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                if self.path == "/redirect":
                    self.send_header("Location", "/json")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                client_ports.append(self.client_address[1])
                requests.append(("GET", self.path))
                body = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return None

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = HttpClient(timeout_seconds=2)
        base_url = f"http://127.0.0.1:{server.server_port}"
        try:
            first = await client.post_json(base_url + "/json", payload={"n": 1})
            second = await client.post_json(base_url + "/json", payload={"n": 2})
            self.assertEqual(first.data, {"ok": True})
            self.assertEqual(second.data, {"ok": True})
            with self.assertRaisesRegex(ProviderRateLimitError, "slow down"):
                await client.post_json(base_url + "/rate", payload={})
            for _ in range(2):
                events = [
                    event async for event in client.stream_sse(base_url + "/stream", payload={})
                ]
                self.assertEqual([event.data for event in events], ["streamed"])
            self.assertEqual(len(client_ports), 5)
            self.assertEqual(len(set(client_ports)), 1)
            redirected = await client.post_json(base_url + "/redirect", payload={})
            self.assertEqual(redirected.data, {"ok": True})
            self.assertEqual(requests[-2:], [("POST", "/redirect"), ("GET", "/json")])
        finally:
            client.close()
            server.shutdown()
            server.server_close()
            thread.join()

    async def test_direct_sse_cancellation_closes_the_leased_connection(self) -> None:
        release_server = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"data: first\n\n")
                self.wfile.flush()
                release_server.wait(5)
                self.close_connection = True

            def log_message(self, format: str, *args: object) -> None:
                return None

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        existing = {
            thread.ident for thread in threading.enumerate() if thread.name == "borealis-sse"
        }
        client = HttpClient(timeout_seconds=2)
        try:
            stream = client.stream_sse(
                f"http://127.0.0.1:{server.server_port}/stream",
                payload={},
            )
            first = await anext(stream)
            await cast(Any, stream).aclose()
            self.assertEqual(first.data, "first")
            remaining = [
                thread
                for thread in threading.enumerate()
                if thread.name == "borealis-sse" and thread.ident not in existing
            ]
            self.assertEqual(remaining, [])
        finally:
            client.close()
            release_server.set()
            server.shutdown()
            server.server_close()
            server_thread.join()


if __name__ == "__main__":
    unittest.main()
