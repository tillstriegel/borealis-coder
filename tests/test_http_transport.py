from __future__ import annotations

import asyncio
import io
import subprocess
import sys
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

    async def test_stream_sse_reports_payload_encoding_errors(self) -> None:
        client = HttpClient()
        try:
            with (
                patch.object(threading, "excepthook") as thread_errors,
                patch.object(http_module, "open_same_origin") as send,
                self.assertRaisesRegex(ProviderError, "not JSON serializable"),
            ):
                async with asyncio.timeout(1):
                    _ = [item async for item in client.stream_sse(
                        "https://unused.test", payload={"unsupported": object()},
                    )]
            thread_errors.assert_not_called()
            send.assert_not_called()
        finally:
            client.close()

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

        preserved = handler.redirect_request(
            request,
            io.BytesIO(),
            307,
            "Temporary Redirect",
            Message(),
            "https://provider.example/v1/preserved",
        )
        assert preserved is not None
        self.assertEqual(preserved.get_method(), "POST")
        self.assertEqual(preserved.data, b"{}")
        self.assertEqual(preserved.get_header("Authorization"), "Bearer synthetic-secret")

        with self.assertRaisesRegex(ProviderError, "unsafe HTTP redirect"):
            HttpClient._redirect_request(
                "https://provider.example/v1/messages",
                302,
                {"location": "https://attacker.example/collect"},
                {"Authorization": "Bearer synthetic-secret"},
                b"{}",
            )

        redirected_post = HttpClient._redirect_request(
            "https://provider.example/v1/messages",
            308,
            {"location": "/v1/canonical"},
            {
                "Authorization": "Bearer synthetic-secret",
                "Content-Type": "application/json",
                "Content-Length": "2",
            },
            b"{}",
        )
        assert redirected_post is not None
        self.assertEqual(redirected_post.get_method(), "POST")
        self.assertEqual(redirected_post.data, b"{}")
        self.assertEqual(redirected_post.get_header("Content-type"), "application/json")
        self.assertIsNone(redirected_post.get_header("Content-length"))


class HttpConnectionReuseTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_redirect_cancellation_does_not_abort_a_released_connection(self) -> None:
        released = threading.Event()
        finish_release = threading.Event()
        shutdown = threading.Event()

        class Socket:
            def shutdown(self, how):
                shutdown.set()

            def settimeout(self, timeout):
                pass

        class Response(FakeResponse):
            will_close = False

            def getheaders(self):
                return [("Location", "/next")]

            def close(self):
                pass

        class Connection:
            sock = Socket()

            def request(self, *args, **kwargs):
                pass

            def getresponse(self):
                return Response(b"", status=307)

            def close(self):
                pass

        client = HttpClient(timeout_seconds=3)
        origin = ("http", "unused.test", 80)
        connection = Connection()
        release = client._pool.release
        existing = set(threading.enumerate())

        def hold_after_release(*args, **kwargs):
            release(*args, **kwargs)
            released.set()
            finish_release.wait(3)

        with (
            patch.object(client._pool, "_new_connection", return_value=connection),
            patch.object(client._pool, "_is_usable", return_value=True),
            patch.object(client._pool, "release", side_effect=hold_after_release),
            patch.object(http_module, "_HTTP_JOIN_TIMEOUT_SECONDS", 0.01),
            patch.object(http_module, "open_same_origin", return_value=FakeResponse()) as redirected,
        ):
            stream = client.stream_sse("http://unused.test", payload={})
            request = asyncio.ensure_future(anext(stream))
            borrowed = None
            try:
                self.assertTrue(await asyncio.to_thread(released.wait, 1))
                borrowed = client._pool.acquire(origin, 3)
                self.assertIs(borrowed, connection)
                request.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(request, timeout=1)
                self.assertFalse(shutdown.is_set(), "Cancellation aborted the next pool borrower")
                finish_release.set()
                for thread in threading.enumerate():
                    if thread.name == "borealis-sse" and thread not in existing:
                        await asyncio.to_thread(thread.join, 1)
                        self.assertFalse(thread.is_alive())
                redirected.assert_not_called()
            finally:
                finish_release.set()
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)
                if borrowed is not None:
                    release(origin, borrowed, reusable=False)
                client.close()

    async def test_stream_cancellation_during_connect_sends_no_late_request(self) -> None:
        started = threading.Event()
        release = threading.Event()
        closed = threading.Event()
        sent = []

        class Connecting:
            sock = None

            def connect(self):
                started.set()
                release.wait(3)

            def request(self, *args, **kwargs):
                if self.sock is None:
                    self.connect()
                sent.append(args)
                raise OSError("Fixture connection has no response")

            def close(self):
                closed.set()

        client = HttpClient(timeout_seconds=3)
        with (
            patch.object(client._pool, "_new_connection", return_value=Connecting()),
            patch.object(http_module, "_HTTP_JOIN_TIMEOUT_SECONDS", 0.01),
        ):
            stream = client.stream_sse("http://unused.test", payload={})
            request = asyncio.ensure_future(anext(stream))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 1))
                request.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(request, timeout=1)
                self.assertFalse(release.is_set())
                release.set()
                self.assertTrue(await asyncio.to_thread(closed.wait, 1))
                self.assertEqual(sent, [])
                self.assertEqual(client._pool._counts, {})
            finally:
                release.set()
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)
                client.close()

    async def test_json_cancellation_releases_stalled_responses_and_allows_reuse(self) -> None:
        for pooled, headers_first, status in (
            (True, False, 200), (True, True, 200), (False, True, 200), (False, True, 429),
        ):
            with self.subTest(pooled=pooled, headers_first=headers_first, status=status):
                await self._cancel_stalled_response(pooled, headers_first, status)

    async def test_stream_cancellation_releases_stalled_responses_and_allows_reuse(self) -> None:
        for pooled, headers_first, status in (
            (True, False, 200), (True, True, 200), (True, True, 429),
            (False, True, 200), (False, True, 429),
        ):
            with self.subTest(pooled=pooled, headers_first=headers_first, status=status):
                await self._cancel_stalled_response(pooled, headers_first, status, streamed=True)

    async def _cancel_stalled_response(
        self, pooled: bool, headers_first: bool, status: int, *, streamed: bool = False,
    ) -> None:
        ready = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        existing = set(threading.enumerate())

        class ObservedClient(HttpClient):
            def _post_json_sync(self, *args, **kwargs):
                try:
                    return super()._post_json_sync(*args, **kwargs)
                finally:
                    if args[0].endswith("/stalled"):
                        finished.set()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                stalled = self.path == "/stalled"
                if stalled:
                    self.close_connection = True
                if stalled and not headers_first:
                    ready.set()
                    release.wait(5)
                try:
                    self.send_response(status if stalled else 200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", "2")
                    if stalled:
                        self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.flush()
                    if stalled and headers_first:
                        ready.set()
                        release.wait(5)
                    self.wfile.write(b"{}")
                except OSError:
                    self.close_connection = True

            def log_message(self, format: str, *args: object) -> None:
                return None

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        client = ObservedClient(timeout_seconds=5)
        base_url = f"http://127.0.0.1:{server.server_port}"
        request = None
        try:
            with patch.object(client, "_pooled_target", wraps=client._pooled_target) as target:
                if not pooled:
                    target.return_value = None
                if streamed:
                    stream = client.stream_sse(base_url + "/stalled", payload={})
                    request = asyncio.ensure_future(anext(stream))
                else:
                    request = asyncio.create_task(client.post_json(base_url + "/stalled", payload={}))
                self.assertTrue(await asyncio.to_thread(ready.wait, 2))
                request.cancel()
                await asyncio.sleep(0)
                request.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    async with asyncio.timeout(2):
                        await request
                self.assertFalse(release.is_set())
                if streamed:
                    self.assertFalse(any(
                        thread.name == "borealis-sse" and thread not in existing
                        for thread in threading.enumerate()
                    ), "Cancelled stream worker is still running")
                else:
                    self.assertTrue(finished.is_set(), "Cancelled response is still being read")
                response = await client.post_json(base_url + "/ready", payload={})
                self.assertEqual(response.data, {})
        finally:
            release.set()
            if request is not None:
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)
            client.close()
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            server_thread.join()

    async def test_cancelled_json_worker_without_a_socket_does_not_hold_shutdown(self) -> None:
        script = """
import asyncio, threading
from borealis_coder.providers.http import HttpClient
started = threading.Event()
class ConnectingClient(HttpClient):
    def _post_json_sync(self, *args, **kwargs):
        started.set()
        threading.Event().wait(60)
async def main():
    client = ConnectingClient()
    task = asyncio.create_task(client.post_json('https://unused.test', payload={}))
    while not started.is_set():
        await asyncio.sleep(.01)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    client.close()
asyncio.run(main())
print('shutdown complete', flush=True)
"""
        result = await asyncio.to_thread(
            subprocess.run, [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=4,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("shutdown complete", result.stdout)

    async def test_json_cancellation_during_connect_sends_no_late_request(self) -> None:
        started = threading.Event()
        release = threading.Event()
        closed = threading.Event()
        sent = []

        class Connecting:
            sock = None

            def connect(self):
                started.set()
                release.wait(3)

            def request(self, *args, **kwargs):
                sent.append(args)

            def close(self):
                closed.set()

        client = HttpClient(timeout_seconds=3)
        with (
            patch.object(client._pool, "_new_connection", return_value=Connecting()),
            patch.object(http_module, "_HTTP_JOIN_TIMEOUT_SECONDS", 0.01),
        ):
            request = asyncio.create_task(client.post_json("http://unused.test", payload={}))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 1))
                request.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(request, timeout=1)
                self.assertFalse(release.is_set())
                release.set()
                self.assertTrue(await asyncio.to_thread(closed.wait, 1))
                self.assertEqual(sent, [])
                self.assertEqual(client._pool._counts, {})
            finally:
                release.set()
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)
                client.close()

    async def test_json_cancellation_releases_a_connection_pool_wait(self) -> None:
        client = HttpClient(timeout_seconds=3)
        origin = ("http", "unused.test", 80)
        acquire = client._pool.acquire
        leased = [acquire(origin, 3) for _ in range(http_module._MAX_CONNECTIONS_PER_ORIGIN)]
        waiting = threading.Event()
        finished = threading.Event()

        def wait_for_connection(*args, **kwargs):
            waiting.set()
            try:
                return acquire(*args, **kwargs)
            finally:
                finished.set()

        with patch.object(client._pool, "acquire", side_effect=wait_for_connection):
            request = asyncio.create_task(client.post_json("http://unused.test", payload={}))
            try:
                self.assertTrue(await asyncio.to_thread(waiting.wait, 1))
                request.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(request, timeout=2)
                self.assertTrue(finished.is_set(), "Cancelled request still waits for a pool slot")
                self.assertEqual(client._pool._counts, {origin: len(leased)})
            finally:
                client.close()
                for connection in leased:
                    client._pool.release(origin, connection, reusable=False)
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)

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
                elif self.path in {"/redirect-preserve", "/stream-redirect"}:
                    body = b""
                    status = 307
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
                if self.path in {"/redirect", "/redirect-preserve"}:
                    self.send_header("Location", "/json")
                elif self.path == "/stream-redirect":
                    self.send_header("Location", "/stream")
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
            preserved = await client.post_json(base_url + "/redirect-preserve", payload={})
            self.assertEqual(preserved.data, {"ok": True})
            self.assertEqual(
                requests[-2:],
                [("POST", "/redirect-preserve"), ("POST", "/json")],
            )
            redirected_events = [
                event
                async for event in client.stream_sse(
                    base_url + "/stream-redirect",
                    payload={},
                )
            ]
            self.assertEqual([event.data for event in redirected_events], ["streamed"])
            self.assertEqual(
                requests[-2:],
                [("POST", "/stream-redirect"), ("POST", "/stream")],
            )
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
