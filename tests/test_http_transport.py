from __future__ import annotations

import io
import unittest
import urllib.error
from email.message import Message
from unittest.mock import patch

from borealis_coder.errors import ProviderError, ProviderRateLimitError, ProviderUnavailableError
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

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, *_args: object) -> bytes:
        return self.body

    def readline(self) -> bytes:
        return next(self.lines, b"")


class HttpTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_post_json_success_text_http_and_network_errors(self) -> None:
        client = HttpClient(timeout_seconds=1)
        with patch(
            "borealis_coder.providers.http.urllib.request.urlopen",
            return_value=FakeResponse(b'{"ok":true}', headers={"Content-Type": "application/json"}),
        ):
            response = await client.post_json("https://example.test", payload={"x": 1})
        self.assertEqual(response.data, {"ok": True})
        self.assertEqual(response.status, 200)

        with patch(
            "borealis_coder.providers.http.urllib.request.urlopen",
            return_value=FakeResponse(b"plain", headers={"Content-Type": "text/plain"}),
        ):
            response = await client.post_json("https://example.test", payload={})
        self.assertEqual(response.data, "plain")

        headers = Message()
        headers["Content-Type"] = "application/json"
        http_error = urllib.error.HTTPError(
            "https://example.test", 429, "rate", headers, io.BytesIO(b'{"error":{"message":"slow down"}}')
        )
        with patch("borealis_coder.providers.http.urllib.request.urlopen", side_effect=http_error):
            with self.assertRaisesRegex(ProviderRateLimitError, "slow down"):
                await client.post_json("https://example.test", payload={})

        text_error = urllib.error.HTTPError(
            "https://example.test", 400, "bad", headers, io.BytesIO(b"not json")
        )
        with patch("borealis_coder.providers.http.urllib.request.urlopen", side_effect=text_error):
            with self.assertRaises(ProviderError):
                await client.post_json("https://example.test", payload={})

        with patch(
            "borealis_coder.providers.http.urllib.request.urlopen",
            side_effect=urllib.error.URLError("offline"),
        ):
            with self.assertRaisesRegex(ProviderError, "offline"):
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
            "borealis_coder.providers.http.urllib.request.urlopen",
            return_value=FakeResponse(lines=lines, headers={"Content-Type": "text/event-stream"}),
        ):
            events = [item async for item in HttpClient(timeout_seconds=1).stream_sse("https://x", payload={})]
        self.assertEqual(events[0].event, "delta")
        self.assertEqual(events[0].data, "first\nsecond")
        self.assertEqual(events[0].id, "7")
        self.assertEqual(events[1].data, "final")

        headers = Message()
        http_error = urllib.error.HTTPError(
            "https://x", 503, "down", headers, io.BytesIO(b'{"error":"unavailable"}')
        )
        with patch("borealis_coder.providers.http.urllib.request.urlopen", side_effect=http_error):
            with self.assertRaises(ProviderUnavailableError):
                _ = [item async for item in HttpClient().stream_sse("https://x", payload={})]

        with patch(
            "borealis_coder.providers.http.urllib.request.urlopen",
            side_effect=RuntimeError("thread failed"),
        ):
            with self.assertRaisesRegex(ProviderError, "thread failed"):
                _ = [item async for item in HttpClient().stream_sse("https://x", payload={})]

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


if __name__ == "__main__":
    unittest.main()
