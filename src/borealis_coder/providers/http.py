"""Minimal async JSON/SSE HTTP transport built on the Python standard library."""

from __future__ import annotations

import asyncio
import contextlib
import http.client
import json
import select
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from .. import __version__
from ..errors import ProviderError
from ..network import open_same_origin, same_origin_redirect_url
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


_SSE_BUFFER_SIZE = 64
_SSE_JOIN_TIMEOUT_SECONDS = 1.0
_MAX_CONNECTIONS_PER_ORIGIN = 4
_Origin = tuple[str, str, int]


class _ConnectionPool:
    def __init__(self, ssl_context: ssl.SSLContext) -> None:
        self.ssl_context = ssl_context
        self._condition = threading.Condition()
        self._idle: dict[_Origin, list[http.client.HTTPConnection]] = {}
        self._counts: dict[_Origin, int] = {}
        self._closed = False

    def acquire(
        self,
        origin: _Origin,
        timeout_seconds: int,
        *,
        cancel: threading.Event | None = None,
    ) -> http.client.HTTPConnection:
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while True:
                if self._closed:
                    raise RuntimeError("HTTP connection pool is closed")
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("HTTP request was cancelled")
                idle = self._idle.get(origin)
                if idle:
                    connection = idle.pop()
                    if not self._is_usable(connection):
                        connection.close()
                        remaining = self._counts.get(origin, 1) - 1
                        if remaining:
                            self._counts[origin] = remaining
                        else:
                            self._counts.pop(origin, None)
                            self._idle.pop(origin, None)
                        continue
                    self._set_timeout(connection, timeout_seconds)
                    return connection
                count = self._counts.get(origin, 0)
                if count < _MAX_CONNECTIONS_PER_ORIGIN:
                    self._counts[origin] = count + 1
                    return self._new_connection(origin, timeout_seconds)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for an HTTP connection to {origin[1]}")
                self._condition.wait(min(remaining, 0.05 if cancel is not None else remaining))

    def release(
        self,
        origin: _Origin,
        connection: http.client.HTTPConnection,
        *,
        reusable: bool,
    ) -> None:
        with self._condition:
            if reusable and not self._closed and connection.sock is not None:
                self._idle.setdefault(origin, []).append(connection)
            else:
                connection.close()
                remaining = self._counts.get(origin, 1) - 1
                if remaining:
                    self._counts[origin] = remaining
                else:
                    self._counts.pop(origin, None)
                    self._idle.pop(origin, None)
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            idle = [connection for connections in self._idle.values() for connection in connections]
            self._idle.clear()
            self._counts.clear()
            for connection in idle:
                connection.close()
            self._condition.notify_all()

    def _new_connection(
        self,
        origin: _Origin,
        timeout_seconds: int,
    ) -> http.client.HTTPConnection:
        scheme, host, port = origin
        if scheme == "https":
            return http.client.HTTPSConnection(
                host,
                port,
                timeout=timeout_seconds,
                context=self.ssl_context,
            )
        return http.client.HTTPConnection(host, port, timeout=timeout_seconds)

    @staticmethod
    def _set_timeout(
        connection: http.client.HTTPConnection,
        timeout_seconds: int,
    ) -> None:
        connection.timeout = timeout_seconds
        if connection.sock is not None:
            connection.sock.settimeout(timeout_seconds)

    @staticmethod
    def _is_usable(connection: http.client.HTTPConnection) -> bool:
        socket = connection.sock
        if socket is None:
            return False
        try:
            return not select.select([socket], [], [], 0)[0]
        except (OSError, ValueError):
            return False


class HttpClient:
    def __init__(self, *, timeout_seconds: int = 180, redactor: Redactor | None = None) -> None:
        self.timeout_seconds = timeout_seconds
        self.redactor = redactor or Redactor()
        self.ssl_context = ssl.create_default_context()
        self._pool = _ConnectionPool(self.ssl_context)
        self._proxies = urllib.request.getproxies()

    def close(self) -> None:
        self._pool.close()

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
        pooled_target = self._pooled_target(url)
        if pooled_target is None:
            return self._post_json_urlopen(
                url,
                request_headers,
                body,
                timeout_seconds,
            )
        origin, target = pooled_target
        connection = self._pool.acquire(origin, timeout_seconds)
        reusable = False
        redirect_request: urllib.request.Request | None = None
        try:
            connection.request("POST", target, body=body, headers=request_headers)
            response = connection.getresponse()
            raw = response.read()
            response_headers = {key.lower(): value for key, value in response.getheaders()}
            reusable = not response.will_close and not _connection_close_requested(request_headers)
            if 300 <= response.status < 400:
                redirect_request = self._redirect_request(
                    url,
                    response.status,
                    response_headers,
                    request_headers,
                )
                if redirect_request is None:
                    raise self._http_error(response.status, raw)
            else:
                if response.status >= 400:
                    raise self._http_error(response.status, raw)
                data = self._decode(raw, response_headers)
                return HttpResponse(response.status, response_headers, data, raw)
        except ProviderError:
            raise
        except (http.client.HTTPException, OSError) as error:
            raise classify_provider_error(
                None,
                self.redactor.text(f"Provider connection failed: {error}"),
            ) from error
        finally:
            self._pool.release(origin, connection, reusable=reusable)
        if redirect_request is None:
            raise ProviderError("Provider returned an unusable HTTP redirect")
        return self._json_urlopen_request(redirect_request, timeout_seconds)

    def _post_json_urlopen(
        self,
        url: str,
        headers: dict[str, str],
        body: bytes,
        timeout_seconds: int,
    ) -> HttpResponse:
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        return self._json_urlopen_request(request, timeout_seconds)

    def _json_urlopen_request(
        self,
        request: urllib.request.Request,
        timeout_seconds: int,
    ) -> HttpResponse:
        try:
            with open_same_origin(
                request, timeout=timeout_seconds, context=self.ssl_context
            ) as response:
                raw = response.read()
                response_headers = {key.lower(): value for key, value in response.headers.items()}
                data = self._decode(raw, response_headers)
                return HttpResponse(response.status, response_headers, data, raw)
        except urllib.error.HTTPError as error:
            try:
                raw = error.read()
            finally:
                error.close()
            raise self._http_error(error.code, raw) from error
        except urllib.error.URLError as error:
            raise classify_provider_error(
                None,
                self.redactor.text(f"Provider connection failed: {error.reason}"),
            ) from error

    async def stream_sse(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        payload: Any,
        timeout_seconds: int | None = None,
    ) -> AsyncIterator[SSEEvent]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[SSEEvent | BaseException | None] = asyncio.Queue(
            maxsize=_SSE_BUFFER_SIZE
        )
        slots = threading.BoundedSemaphore(_SSE_BUFFER_SIZE)
        stop = threading.Event()
        response_lock = threading.Lock()
        active_response: Any = None
        active_connection: http.client.HTTPConnection | None = None
        active_socket: socket.socket | None = None

        def enqueue(item: SSEEvent | BaseException | None) -> bool:
            while not stop.is_set():
                if not slots.acquire(timeout=0.05):
                    continue

                def deliver() -> None:
                    try:
                        queue.put_nowait(item)
                    except asyncio.QueueFull:
                        slots.release()

                try:
                    loop.call_soon_threadsafe(deliver)
                except RuntimeError:
                    slots.release()
                    return False
                return True
            return False

        def worker() -> None:
            nonlocal active_connection, active_response, active_socket
            request_headers = {
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
                "User-Agent": f"borealis-coder/{__version__}",
                **(headers or {}),
            }
            body = json_dumps(payload).encode("utf-8")
            origin: _Origin | None = None
            connection: http.client.HTTPConnection | None = None
            response: Any = None
            completed = False
            reusable = False
            try:
                request_timeout = timeout_seconds or self.timeout_seconds
                pooled_target = self._pooled_target(url)
                if pooled_target is None:
                    request = urllib.request.Request(
                        url,
                        data=body,
                        headers=request_headers,
                        method="POST",
                    )
                    response = open_same_origin(
                        request,
                        timeout=request_timeout,
                        context=self.ssl_context,
                    )
                else:
                    origin, target = pooled_target
                    connection = self._pool.acquire(
                        origin,
                        request_timeout,
                        cancel=stop,
                    )
                    with response_lock:
                        active_connection = connection
                    connection.request("POST", target, body=body, headers=request_headers)
                    with response_lock:
                        active_socket = connection.sock
                    response = connection.getresponse()
                    if 300 <= response.status < 400:
                        raw = response.read()
                        response_headers = {
                            key.lower(): value for key, value in response.getheaders()
                        }
                        redirect_request = self._redirect_request(
                            url,
                            response.status,
                            response_headers,
                            request_headers,
                        )
                        if redirect_request is None:
                            completed = True
                            reusable = not response.will_close and not _connection_close_requested(
                                request_headers
                            )
                            raise self._http_error(response.status, raw)
                        reusable = not response.will_close and not _connection_close_requested(
                            request_headers
                        )
                        response.close()
                        self._pool.release(origin, connection, reusable=reusable)
                        connection = None
                        origin = None
                        with response_lock:
                            active_connection = None
                            active_socket = None
                        response = open_same_origin(
                            redirect_request,
                            timeout=request_timeout,
                            context=self.ssl_context,
                        )
                    elif response.status >= 400:
                        raw = response.read()
                        completed = True
                        reusable = not response.will_close and not _connection_close_requested(
                            request_headers
                        )
                        raise self._http_error(response.status, raw)

                with response_lock:
                    active_response = response
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
                            if not enqueue(item):
                                return
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
                if data_lines and not enqueue(
                    SSEEvent(event_name, "\n".join(data_lines), event_id)
                ):
                    return
                completed = not stop.is_set()
                if connection is not None:
                    reusable = (
                        completed
                        and not response.will_close
                        and not _connection_close_requested(request_headers)
                    )
            except urllib.error.HTTPError as error:
                try:
                    raw = error.read()
                finally:
                    error.close()
                enqueue(self._http_error(error.code, raw))
            except urllib.error.URLError as error:
                enqueue(
                    classify_provider_error(
                        None,
                        self.redactor.text(f"Provider connection failed: {error.reason}"),
                    )
                )
            except BaseException as error:
                enqueue(error)
            finally:
                with response_lock:
                    if active_response is response:
                        active_response = None
                    if active_connection is connection:
                        active_connection = None
                    if active_socket is not None:
                        active_socket = None
                close = getattr(response, "close", None)
                if close is not None:
                    with contextlib.suppress(Exception):
                        close()
                if connection is not None and origin is not None:
                    self._pool.release(
                        origin,
                        connection,
                        reusable=reusable and completed and not stop.is_set(),
                    )
                enqueue(None)

        thread = threading.Thread(target=worker, name="borealis-sse", daemon=True)
        thread.start()
        try:
            while True:
                item = await queue.get()
                slots.release()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    if isinstance(item, ProviderError):
                        raise item
                    raise classify_provider_error(None, str(item)) from item
                yield item
        finally:
            stop.set()
            with response_lock:
                response = active_response
                connection = active_connection
                socket_handle = active_socket or (
                    connection.sock if connection is not None else None
                )
            if socket_handle is not None:
                with contextlib.suppress(Exception):
                    socket_handle.shutdown(socket.SHUT_RDWR)
            else:
                close = getattr(response, "close", None)
                if close is not None:
                    with contextlib.suppress(Exception):
                        close()
            await asyncio.to_thread(thread.join, _SSE_JOIN_TIMEOUT_SECONDS)

    def _pooled_target(self, url: str) -> tuple[_Origin, str] | None:
        try:
            parsed = urllib.parse.urlsplit(url)
            port = parsed.port
        except ValueError:
            return None
        scheme = parsed.scheme.lower()
        host = parsed.hostname
        if (
            scheme not in {"http", "https"}
            or not host
            or parsed.username is not None
            or parsed.password is not None
        ):
            return None
        proxy = self._proxies.get(scheme)
        if proxy and not urllib.request.proxy_bypass(host):
            return None
        origin = (scheme, host, port or (443 if scheme == "https" else 80))
        target = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        return origin, target

    @staticmethod
    def _redirect_request(
        url: str,
        status: int,
        response_headers: dict[str, str],
        request_headers: dict[str, str],
    ) -> urllib.request.Request | None:
        location = response_headers.get("location")
        if status not in {301, 302, 303} or not location:
            return None
        try:
            redirect_url = same_origin_redirect_url(url, location)
        except ValueError as error:
            raise ProviderError("Provider refused an unsafe HTTP redirect") from error
        redirect_headers = {
            name: value
            for name, value in request_headers.items()
            if name.lower() not in {"content-length", "content-type"}
        }
        return urllib.request.Request(redirect_url, headers=redirect_headers, method="GET")

    def _http_error(self, status: int, raw: bytes) -> ProviderError:
        try:
            details: Any = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            details = raw.decode("utf-8", errors="replace")
        message = self._error_message(details) or f"HTTP {status} from provider"
        return classify_provider_error(status, self.redactor.text(message), details)

    @staticmethod
    def _decode(raw: bytes, headers: dict[str, str]) -> Any:
        content_type = headers.get("content-type", "")
        if "json" in content_type or raw[:1] in {b"{", b"["}:
            try:
                return json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as error:
                raise classify_provider_error(
                    None, f"Invalid JSON from provider: {error}"
                ) from error
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


def _connection_close_requested(headers: dict[str, str]) -> bool:
    return any(
        name.lower() == "connection" and value.lower() == "close" for name, value in headers.items()
    )
