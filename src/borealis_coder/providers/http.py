"""Minimal async JSON/SSE HTTP transport built on urllib."""

from __future__ import annotations

import asyncio
import json
import ssl
import threading
import urllib.error
import urllib.request
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from .. import __version__
from ..errors import ProviderError
from ..safety.redaction import Redactor
from ..util import json_dumps
from .base import classify_provider_error


@dataclass(slots=True)
class HttpResponse:
    status: int
    headers: dict[str, str]
    data: Any
    raw: bytes


@dataclass(slots=True)
class SSEEvent:
    event: str
    data: str
    id: str | None = None


class HttpClient:
    def __init__(self, *, timeout_seconds: int = 180, redactor: Redactor | None = None) -> None:
        self.timeout_seconds = timeout_seconds
        self.redactor = redactor or Redactor()
        self.ssl_context = ssl.create_default_context()

    async def post_json(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        payload: Any,
        timeout_seconds: int | None = None,
    ) -> HttpResponse:
        return await asyncio.to_thread(
            self._post_json_sync,
            url,
            headers or {},
            payload,
            timeout_seconds or self.timeout_seconds,
        )

    def _post_json_sync(
        self,
        url: str,
        headers: dict[str, str],
        payload: Any,
        timeout_seconds: int,
    ) -> HttpResponse:
        request_headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": f"borealis-coder/{__version__}",
            **headers,
        }
        body = json_dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
        try:
            with urllib.request.urlopen(
                request, timeout=timeout_seconds, context=self.ssl_context
            ) as response:
                raw = response.read()
                response_headers = {key.lower(): value for key, value in response.headers.items()}
                data = self._decode(raw, response_headers)
                return HttpResponse(response.status, response_headers, data, raw)
        except urllib.error.HTTPError as error:
            raw = error.read()
            try:
                details = json.loads(raw.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                details = raw.decode("utf-8", errors="replace")
            message = self._error_message(details) or f"HTTP {error.code} from provider"
            raise classify_provider_error(error.code, self.redactor.text(message), details) from error
        except urllib.error.URLError as error:
            raise classify_provider_error(None, f"Provider connection failed: {error.reason}") from error

    async def stream_sse(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        payload: Any,
        timeout_seconds: int | None = None,
    ) -> AsyncIterator[SSEEvent]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[SSEEvent | BaseException | None] = asyncio.Queue()
        stop = threading.Event()

        def worker() -> None:
            request_headers = {
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
                "User-Agent": f"borealis-coder/{__version__}",
                **(headers or {}),
            }
            request = urllib.request.Request(
                url,
                data=json_dumps(payload).encode("utf-8"),
                headers=request_headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=timeout_seconds or self.timeout_seconds,
                    context=self.ssl_context,
                ) as response:
                    event_name = "message"
                    event_id: str | None = None
                    data_lines: list[str] = []
                    while not stop.is_set():
                        raw_line = response.readline()
                        if not raw_line:
                            break
                        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                        if not line:
                            if data_lines:
                                item = SSEEvent(event_name, "\n".join(data_lines), event_id)
                                loop.call_soon_threadsafe(queue.put_nowait, item)
                            event_name = "message"
                            event_id = None
                            data_lines = []
                            continue
                        if line.startswith(":"):
                            continue
                        field, _, value = line.partition(":")
                        value = value[1:] if value.startswith(" ") else value
                        if field == "event":
                            event_name = value
                        elif field == "data":
                            data_lines.append(value)
                        elif field == "id":
                            event_id = value
                    if data_lines:
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            SSEEvent(event_name, "\n".join(data_lines), event_id),
                        )
            except urllib.error.HTTPError as error:
                raw = error.read()
                try:
                    details = json.loads(raw.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    details = raw.decode("utf-8", errors="replace")
                message = self._error_message(details) or f"HTTP {error.code} from provider"
                failure: BaseException = classify_provider_error(
                    error.code, self.redactor.text(message), details
                )
                loop.call_soon_threadsafe(queue.put_nowait, failure)
            except BaseException as error:
                loop.call_soon_threadsafe(queue.put_nowait, error)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        thread = threading.Thread(target=worker, name="borealis-sse", daemon=True)
        thread.start()
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    if isinstance(item, ProviderError):
                        raise item
                    raise classify_provider_error(None, str(item)) from item
                yield item
        finally:
            stop.set()

    @staticmethod
    def _decode(raw: bytes, headers: dict[str, str]) -> Any:
        content_type = headers.get("content-type", "")
        if "json" in content_type or raw[:1] in {b"{", b"["}:
            try:
                return json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as error:
                raise classify_provider_error(None, f"Invalid JSON from provider: {error}") from error
        return raw.decode("utf-8", errors="replace")

    @staticmethod
    def _error_message(details: Any) -> str:
        if isinstance(details, dict):
            error = details.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or error.get("status") or error)
            if error:
                return str(error)
            return str(details.get("message") or "")
        return str(details or "")
