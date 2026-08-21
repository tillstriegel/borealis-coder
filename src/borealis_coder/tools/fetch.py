"""Small policy-gated HTTP text fetcher for public documentation and APIs."""

from __future__ import annotations

import asyncio
import http.client
import ipaddress
import socket
import ssl
import urllib.parse
from typing import Any

from ..errors import ToolError
from ..models import Effect, ToolResult
from ..util import truncate_text
from .base import Tool, ToolContext, object_schema

_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class FetchUrlTool(Tool):
    name = "fetch_url"
    description = "Fetch a public HTTP(S) URL as text. Disabled unless workspace network access is enabled."
    effect = Effect.NETWORK
    default_risk = "high"
    parameters = object_schema({
        "url": {"type": "string", "minLength": 8, "maxLength": 4000},
        "max_chars": {"type": "integer", "minimum": 100, "maximum": 200000},
    })

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        max_chars = int(arguments["max_chars"])
        body, status, content_type, final_url = await asyncio.to_thread(
            _fetch_public_url,
            str(arguments["url"]),
            max_chars * 4,
        )
        return ToolResult(
            truncate_text(body, max_chars),
            metadata={"status": status, "content_type": content_type, "url": final_url},
        )


def _fetch_public_url(
    url: str,
    max_bytes: int,
    *,
    timeout: int = 30,
    max_redirects: int = 5,
) -> tuple[str, int, str, str]:
    current = url
    for redirect_count in range(max_redirects + 1):
        normalized, parsed, addresses = _resolve_public_url(current)
        response = None
        last_error: OSError | http.client.HTTPException | None = None
        for address in addresses:
            try:
                response = _request_once(parsed, address, max_bytes, timeout)
                break
            except (OSError, http.client.HTTPException) as error:
                last_error = error
        if response is None:
            raise ToolError(f"Network error: {last_error}") from last_error
        data, status, content_type, charset, location = response
        if status in _REDIRECT_STATUSES and location:
            if redirect_count == max_redirects:
                raise ToolError(f"Too many redirects (maximum {max_redirects})")
            current = urllib.parse.urljoin(normalized, location)
            continue
        if status >= 400:
            body = data[:4096].decode(charset, errors="replace")
            raise ToolError(f"HTTP {status}: {body}")
        return data.decode(charset, errors="replace"), status, content_type, normalized
    raise ToolError(f"Too many redirects (maximum {max_redirects})")


def _validate_public_url(url: str) -> str:
    normalized, _, _ = _resolve_public_url(url)
    return normalized


def _resolve_public_url(
    url: str,
) -> tuple[str, urllib.parse.SplitResult, tuple[str, ...]]:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise ToolError("Only HTTP(S) URLs are supported")
    if not parsed.hostname:
        raise ToolError("URL must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ToolError("Credentials in URLs are not allowed")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as error:
        raise ToolError(f"Invalid URL port: {error}") from error
    try:
        records = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise ToolError(f"Could not resolve URL host: {error}") from error
    addresses = tuple(dict.fromkeys(str(record[4][0]) for record in records))
    if not addresses:
        raise ToolError("URL hostname resolved to no addresses")
    for value in addresses:
        address = ipaddress.ip_address(value)
        if not address.is_global:
            raise ToolError(f"URL resolves to a non-public address: {address}")
    return urllib.parse.urlunsplit(parsed), parsed, addresses


def _request_once(
    parsed: urllib.parse.SplitResult,
    address: str,
    max_bytes: int,
    timeout: int,
) -> tuple[bytes, int, str, str, str | None]:
    hostname = parsed.hostname
    if hostname is None:  # Defensive: _resolve_public_url already requires it.
        raise ToolError("URL must include a hostname")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if parsed.scheme == "https":
        connection: http.client.HTTPConnection = _PinnedHTTPSConnection(
            hostname,
            address,
            port=port,
            timeout=timeout,
        )
    else:
        connection = http.client.HTTPConnection(address, port=port, timeout=timeout)
    default_port = 443 if parsed.scheme == "https" else 80
    host_header = hostname if port == default_port else f"{hostname}:{port}"
    target = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    try:
        connection.request(
            "GET",
            target,
            headers={"Host": host_header, "User-Agent": "Borealis-Coder/0.1"},
        )
        response = connection.getresponse()
        data = response.read(max_bytes)
        content_type = response.headers.get("Content-Type", "")
        charset = response.headers.get_content_charset() or "utf-8"
        return data, response.status, content_type, charset, response.headers.get("Location")
    finally:
        connection.close()


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connect to a vetted IP while preserving hostname verification and SNI."""

    def __init__(
        self,
        host: str,
        address: str,
        *,
        port: int,
        timeout: int,
    ) -> None:
        self._ssl_context = ssl.create_default_context()
        super().__init__(host, port=port, timeout=timeout, context=self._ssl_context)
        self._pinned_address = address

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._pinned_address, self.port),
            self.timeout,
        )
        self.sock = self._ssl_context.wrap_socket(self.sock, server_hostname=self.host)
