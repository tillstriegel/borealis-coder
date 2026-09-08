"""Bidirectional newline-delimited JSON-RPC 2.0 transport."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import stat
import sys
from collections.abc import Awaitable, Callable
from typing import Any

from ..errors import ProtocolError

JsonHandler = Callable[[str, dict[str, Any]], Awaitable[Any]]
MAX_STDIO_MESSAGE_BYTES = 16 * 1024 * 1024


class JsonRpcConnection:
    def __init__(self, handler: JsonHandler) -> None:
        self.handler = handler
        self._write_lock = asyncio.Lock()
        self._writer: asyncio.StreamWriter | None = None
        self._pending: dict[str | int, asyncio.Future[Any]] = {}
        self._dispatch_tasks: set[asyncio.Task[None]] = set()
        self._counter = 10_000
        self.closed = False

    async def serve_stdio(self) -> None:
        reader = asyncio.StreamReader(limit=MAX_STDIO_MESSAGE_BYTES)
        protocol = asyncio.StreamReaderProtocol(reader)
        loop = asyncio.get_running_loop()
        transport: asyncio.BaseTransport | None = None
        try:
            transport, _ = await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)
            while not self.closed:
                try:
                    line = await reader.readline()
                except ValueError as error:
                    raise ProtocolError(
                        f"JSON-RPC message exceeds the {MAX_STDIO_MESSAGE_BYTES}-byte limit"
                    ) from error
                if not line:
                    break
                try:
                    message = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    await self.send_error(None, -32700, f"Parse error: {error}")
                    continue
                task = asyncio.create_task(self._dispatch(message))
                self._dispatch_tasks.add(task)
                task.add_done_callback(self._finish_dispatch)
        finally:
            self.closed = True
            if transport is not None:
                transport.close()
            if self._writer is not None:
                self._writer.transport.abort()
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(ProtocolError("JSON-RPC connection closed"))
            for task in self._dispatch_tasks:
                task.cancel()
            if self._dispatch_tasks:
                await asyncio.gather(*self._dispatch_tasks, return_exceptions=True)

    async def _dispatch(self, message: Any) -> None:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            await self.send_error(message.get("id") if isinstance(message, dict) else None, -32600, "Invalid Request")
            return
        if "method" not in message:
            request_id = message.get("id")
            future = (
                self._pending.get(request_id)
                if isinstance(request_id, str | int)
                else None
            )
            if future and not future.done():
                if "error" in message:
                    error = message["error"]
                    if isinstance(error, dict):
                        future.set_exception(ProtocolError(f"Peer error {error.get('code')}: {error.get('message')}"))
                    else:
                        future.set_exception(ProtocolError("Peer returned an invalid error response"))
                else:
                    future.set_result(message.get("result"))
            return
        method = str(message["method"])
        params = message.get("params") or {}
        if not isinstance(params, dict):
            if "id" in message:
                await self.send_error(message["id"], -32602, "Params must be an object")
            return
        try:
            result = await self.handler(method, params)
            if "id" in message:
                await self.send_result(message["id"], result if result is not None else {})
        except ProtocolError as error:
            if "id" in message:
                await self.send_error(message["id"], -32602, str(error))
        except Exception as error:
            if "id" in message:
                await self.send_error(message["id"], -32603, f"{type(error).__name__}: {error}")

    def _finish_dispatch(self, task: asyncio.Task[None]) -> None:
        self._dispatch_tasks.discard(task)
        with contextlib.suppress(asyncio.CancelledError):
            task.exception()

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        await self._write({"jsonrpc": "2.0", "method": method, "params": params})

    async def request(self, method: str, params: dict[str, Any], *, timeout: float | None = None) -> Any:
        if self.closed:
            raise ProtocolError("JSON-RPC connection closed")
        self._counter += 1
        request_id = self._counter
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            async with asyncio.timeout(timeout or None):
                await self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
                return await future
        finally:
            self._pending.pop(request_id, None)
            future.cancel()
            if not future.cancelled():
                # Connection shutdown can race with the request write.
                future.exception()

    async def send_result(self, request_id: Any, result: Any) -> None:
        await self._write({"jsonrpc": "2.0", "id": request_id, "result": result})

    async def send_error(self, request_id: Any, code: int, message: str, data: Any = None) -> None:
        error: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        await self._write({"jsonrpc": "2.0", "id": request_id, "error": error})

    async def _write(self, value: dict[str, Any]) -> None:
        data = json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        async with self._write_lock:
            if self.closed:
                raise ProtocolError("JSON-RPC connection closed")
            if self._writer is None:
                # Keep Windows stdio and regular-file redirection on their
                # existing path. POSIX pipe transports support async flow control.
                if os.name == "nt" or stat.S_ISREG(os.fstat(sys.stdout.fileno()).st_mode):
                    await asyncio.to_thread(_write_stdout, data)
                    return
                loop = asyncio.get_running_loop()
                transport, protocol = await loop.connect_write_pipe(
                    lambda: asyncio.streams.FlowControlMixin(loop=loop), sys.stdout.buffer
                )
                # drain() must flush the whole frame, including small replies.
                transport.set_write_buffer_limits(high=0)
                self._writer = asyncio.StreamWriter(transport, protocol, None, loop)
                if self.closed:
                    transport.abort()
                    raise ProtocolError("JSON-RPC connection closed")
            # A previous timed-out request may still have a complete frame
            # queued. Drain it before adding more bytes to the same stream.
            await self._writer.drain()
            if self.closed:
                raise ProtocolError("JSON-RPC connection closed")
            self._writer.write(data.encode("utf-8"))
            await self._writer.drain()


def _write_stdout(data: str) -> None:
    sys.stdout.write(data)
    sys.stdout.flush()
