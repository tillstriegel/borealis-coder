"""Bidirectional newline-delimited JSON-RPC 2.0 transport."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from collections.abc import Awaitable, Callable
from typing import Any

from ..errors import ProtocolError

JsonHandler = Callable[[str, dict[str, Any]], Awaitable[Any]]


class JsonRpcConnection:
    def __init__(self, handler: JsonHandler) -> None:
        self.handler = handler
        self._write_lock = asyncio.Lock()
        self._pending: dict[str | int, asyncio.Future[Any]] = {}
        self._dispatch_tasks: set[asyncio.Task[None]] = set()
        self._counter = 10_000
        self.closed = False

    async def serve_stdio(self) -> None:
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        loop = asyncio.get_running_loop()
        await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)
        while not self.closed:
            line = await reader.readline()
            if not line:
                break
            try:
                message = json.loads(line)
            except json.JSONDecodeError as error:
                await self.send_error(None, -32700, f"Parse error: {error}")
                continue
            task = asyncio.create_task(self._dispatch(message))
            self._dispatch_tasks.add(task)
            task.add_done_callback(self._finish_dispatch)
        self.closed = True
        for task in self._dispatch_tasks:
            task.cancel()
        if self._dispatch_tasks:
            await asyncio.gather(*self._dispatch_tasks, return_exceptions=True)
        for future in self._pending.values():
            if not future.done():
                future.set_exception(ProtocolError("JSON-RPC connection closed"))

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
                    future.set_exception(ProtocolError(f"Peer error {error.get('code')}: {error.get('message')}"))
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
        self._counter += 1
        request_id = self._counter
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        try:
            return await asyncio.wait_for(future, timeout=timeout) if timeout else await future
        finally:
            self._pending.pop(request_id, None)

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
            await asyncio.to_thread(_write_stdout, data)


def _write_stdout(data: str) -> None:
    sys.stdout.write(data)
    sys.stdout.flush()
