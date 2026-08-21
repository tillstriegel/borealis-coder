"""Minimal MCP client supporting current stdio and Streamable HTTP transports."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import __version__
from ..config import MCPServerConfig
from ..errors import ProtocolError

MCP_PROTOCOL_VERSION = "2025-11-25"


@dataclass(slots=True)
class MCPToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, Any]


class MCPClient(ABC):
    def __init__(self, name: str, config: MCPServerConfig, workspace: Path) -> None:
        self.name = name
        self.config = config
        self.workspace = workspace
        self.server_info: dict[str, Any] = {}
        self.capabilities: dict[str, Any] = {}

    @abstractmethod
    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        raise NotImplementedError

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        # Transports may override for true notifications. Default request is valid
        # enough for tolerant servers but stdio implements notification semantics.
        await self.request(method, params)

    async def initialize(self) -> None:
        result = await self.request("initialize", {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "borealis-coder", "version": __version__},
        })
        if not isinstance(result, dict):
            raise ProtocolError(f"MCP server {self.name} returned invalid initialize result")
        self.server_info = dict(result.get("serverInfo") or {})
        self.capabilities = dict(result.get("capabilities") or {})
        await self.notify("notifications/initialized", {})

    async def list_tools(self) -> list[MCPToolDefinition]:
        tools: list[MCPToolDefinition] = []
        cursor: str | None = None
        while True:
            params = {"cursor": cursor} if cursor else {}
            result = await self.request("tools/list", params)
            if not isinstance(result, dict):
                raise ProtocolError(f"MCP server {self.name} returned invalid tools/list result")
            for item in result.get("tools", []) or []:
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                tools.append(MCPToolDefinition(
                    name=str(item["name"]), description=str(item.get("description") or ""),
                    input_schema=dict(item.get("inputSchema") or {"type": "object", "properties": {}}),
                    annotations=dict(item.get("annotations") or {}),
                ))
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self.request("tools/call", {"name": name, "arguments": arguments})
        if not isinstance(result, dict):
            raise ProtocolError(f"MCP server {self.name} returned invalid tools/call result")
        return result

    async def close(self) -> None:
        return None


class StdioMCPClient(MCPClient):
    def __init__(self, name: str, config: MCPServerConfig, workspace: Path, *, env_allowlist: list[str]) -> None:
        super().__init__(name, config, workspace)
        self.env_allowlist = env_allowlist
        self.process: asyncio.subprocess.Process | None = None
        self._pending: dict[str | int, asyncio.Future[Any]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()
        self._counter = 0

    async def start(self) -> None:
        if not self.config.command:
            raise ProtocolError(f"MCP stdio server {self.name} has no command")
        env = {key: value for key, value in os.environ.items() if key in self.env_allowlist}
        env.update(self.config.env)
        self.process = await asyncio.create_subprocess_exec(
            self.config.command, *self.config.args, cwd=str(self.workspace), env=env,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, start_new_session=os.name == "posix",
        )
        self._reader_task = asyncio.create_task(self._reader_loop(), name=f"mcp:{self.name}:reader")
        self._stderr_task = asyncio.create_task(self._drain_stderr(), name=f"mcp:{self.name}:stderr")
        await asyncio.wait_for(self.initialize(), timeout=self.config.timeout_seconds)

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        if self.process is None or self.process.stdin is None:
            raise ProtocolError(f"MCP server {self.name} is not running")
        self._counter += 1
        request_id = self._counter
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
        try:
            return await asyncio.wait_for(future, timeout=self.config.timeout_seconds)
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    async def _send(self, value: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise ProtocolError(f"MCP server {self.name} is not running")
        data = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
        async with self._write_lock:
            self.process.stdin.write(data)
            await self.process.stdin.drain()

    async def _reader_loop(self) -> None:
        assert self.process and self.process.stdout
        try:
            while line := await self.process.stdout.readline():
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "id" not in message:
                    continue
                future = self._pending.get(message["id"])
                if future is None or future.done():
                    continue
                if "error" in message:
                    error = message["error"]
                    future.set_exception(ProtocolError(f"MCP {self.name} error {error.get('code')}: {error.get('message')}"))
                else:
                    future.set_result(message.get("result"))
        finally:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(ProtocolError(f"MCP server {self.name} closed its stdout"))

    async def _drain_stderr(self) -> None:
        if self.process is None or self.process.stderr is None:
            return
        while await self.process.stderr.readline():
            pass

    async def close(self) -> None:
        if self.process is None:
            return
        if self.process.stdin:
            self.process.stdin.close()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=2)
        except TimeoutError:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=2)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        for task in (self._reader_task, self._stderr_task):
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self.process = None


class HttpMCPClient(MCPClient):
    def __init__(self, name: str, config: MCPServerConfig, workspace: Path) -> None:
        super().__init__(name, config, workspace)
        self.session_id: str | None = None
        self._counter = 0

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self._counter += 1
        request_id = self._counter
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
        response, headers = await asyncio.to_thread(self._post, payload)
        session_id = headers.get("Mcp-Session-Id") or headers.get("mcp-session-id")
        if session_id:
            self.session_id = session_id
        if response.get("id") != request_id:
            raise ProtocolError(f"MCP HTTP {self.name} response id mismatch")
        if "error" in response:
            error = response["error"]
            raise ProtocolError(f"MCP {self.name} error {error.get('code')}: {error.get('message')}")
        return response.get("result")

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        await asyncio.to_thread(self._post, payload)

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        if not self.config.url:
            raise ProtocolError(f"MCP HTTP server {self.name} has no URL")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
            **self.config.headers,
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        request = urllib.request.Request(
            self.config.url, data=json.dumps(payload).encode(), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                body = response.read().decode("utf-8", errors="replace")
                content_type = response.headers.get("Content-Type", "")
                parsed = _parse_http_body(body, content_type)
                return parsed, response.headers
        except urllib.error.HTTPError as error:
            body = error.read(4096).decode("utf-8", errors="replace")
            raise ProtocolError(f"MCP HTTP {self.name} returned {error.code}: {body}") from error
        except urllib.error.URLError as error:
            raise ProtocolError(f"MCP HTTP {self.name} network error: {error.reason}") from error


def _parse_http_body(body: str, content_type: str) -> dict[str, Any]:
    if "text/event-stream" in content_type:
        events: list[dict[str, Any]] = []
        for line in body.splitlines():
            if line.startswith("data:"):
                try:
                    value = json.loads(line[5:].strip())
                    if isinstance(value, dict):
                        events.append(value)
                except json.JSONDecodeError:
                    continue
        if not events:
            raise ProtocolError("MCP HTTP response contained no JSON SSE data")
        return events[-1]
    try:
        value = json.loads(body or "{}")
    except json.JSONDecodeError as error:
        raise ProtocolError(f"MCP HTTP response was not JSON: {body[:500]}") from error
    if not isinstance(value, dict):
        raise ProtocolError("MCP HTTP response root must be an object")
    return value
