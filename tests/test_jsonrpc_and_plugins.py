from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from borealis_coder.errors import ProtocolError
from borealis_coder.models import Effect, ToolResult
from borealis_coder.plugins import load_entrypoint_tools, load_workspace_plugins
from borealis_coder.protocol.jsonrpc import JsonRpcConnection
from borealis_coder.tools.base import FunctionTool, Tool, ToolRegistry, object_schema


def make_tool(name: str = "plugin_tool") -> Tool:
    return FunctionTool(
        name=name,
        description="plugin",
        parameters=object_schema({}),
        function=lambda _args, _ctx: ToolResult("ok"),
        effect=Effect.READ,
    )


class JsonRpcTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        async def handler(method: str, params: dict[str, object]) -> object:
            if method == "protocol":
                raise ProtocolError("bad params")
            if method == "explode":
                raise RuntimeError("boom")
            if method == "none":
                return None
            return {"method": method, "params": params}

        self.connection = JsonRpcConnection(handler)
        self.writes: list[dict[str, object]] = []

        async def capture(value: dict[str, object]) -> None:
            self.writes.append(value)

        self.connection._write = capture  # type: ignore[method-assign]

    async def test_failed_request_write_releases_pending_request(self) -> None:
        with (
            patch.object(self.connection, "_write", side_effect=BrokenPipeError("closed")),
            self.assertRaises(BrokenPipeError),
        ):
            await self.connection.request("question", {})
        self.assertFalse(self.connection._pending)

    async def test_request_timeout_covers_blocked_write(self) -> None:
        async def blocked_write(value):
            await asyncio.Event().wait()

        with patch.object(self.connection, "_write", side_effect=blocked_write):
            request = asyncio.create_task(self.connection.request("question", {}, timeout=0.01))
            try:
                done, _ = await asyncio.wait({request}, timeout=1)
                self.assertIn(request, done, "Request timeout did not cover the write")
                with self.assertRaises(TimeoutError):
                    await request
                self.assertFalse(self.connection._pending)
            finally:
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)

    @unittest.skipUnless(os.name == "posix", "Uses POSIX stdio pipe transports")
    async def test_stdio_accepts_large_requests(self) -> None:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "borealis_coder", "acp",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert process.stdin and process.stdout
        try:
            payload = {
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": 2, "padding": "x" * 100_000},
            }
            process.stdin.write(json.dumps(payload).encode() + b"\n")
            await process.stdin.drain()
            line = await asyncio.wait_for(process.stdout.readline(), timeout=3)
            self.assertTrue(line, "ACP closed before replying to a large request")
            self.assertEqual(json.loads(line)["result"]["protocolVersion"], 2)
        finally:
            process.stdin.close()
            try:
                await asyncio.wait_for(process.communicate(), timeout=3)
            except TimeoutError:
                process.kill()
                await process.communicate()

    @unittest.skipUnless(os.name == "posix", "Uses POSIX stdout flow control")
    async def test_blocked_stdout_request_timeout_does_not_hold_process_open(self) -> None:
        script = """
import asyncio, sys
from borealis_coder.protocol.jsonrpc import JsonRpcConnection
async def main():
    async def handler(method, params): return {}
    connection = JsonRpcConnection(handler)
    try:
        await connection.request("blocked", {"text": "x" * 1_000_000}, timeout=0.1)
    except TimeoutError:
        print("timed out", file=sys.stderr, flush=True)
asyncio.run(main())
print("exited", file=sys.stderr, flush=True)
"""
        with subprocess.Popen(
            [sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ) as process:
            try:
                # Do not drain stdout until the process exits. A blocking
                # executor writer would prevent asyncio.run from finishing.
                async with asyncio.timeout(3):
                    while process.poll() is None:
                        await asyncio.sleep(0.01)
                _, stderr = process.communicate(timeout=1)
                self.assertEqual(process.returncode, 0, stderr.decode())
                self.assertIn(b"timed out\nexited\n", stderr)
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=2)

    @unittest.skipUnless(os.name == "posix", "Uses POSIX stdio pipe transports")
    async def test_stdio_disconnect_aborts_a_blocked_output_pipe(self) -> None:
        script = """
import asyncio
from borealis_coder.protocol.jsonrpc import JsonRpcConnection
async def handler(method, params): return {"text": "x" * 1_000_000}
asyncio.run(JsonRpcConnection(handler).serve_stdio())
"""
        with subprocess.Popen(
            [sys.executable, "-c", script], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ) as process:
            assert process.stdin and process.stdout
            watchdog = threading.Timer(5, process.kill)
            watchdog.start()
            try:
                process.stdin.write(b'{"jsonrpc":"2.0","id":1,"method":"large"}\n')
                process.stdin.flush()
                self.assertEqual(process.stdout.read(1), b"{", "Server did not start its reply")
                process.stdin.close()
                process.stdin = None
                async with asyncio.timeout(3):
                    while process.poll() is None:
                        await asyncio.sleep(0.01)
                _, stderr = process.communicate(timeout=1)
                self.assertEqual(process.returncode, 0, stderr.decode())
            finally:
                watchdog.cancel()
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=2)
                watchdog.join(timeout=1)

    @unittest.skipUnless(os.name == "posix", "Uses POSIX stdout flow control")
    async def test_timed_out_output_preserves_frames_when_reading_resumes(self) -> None:
        script = """
import asyncio, sys
from borealis_coder.protocol.jsonrpc import JsonRpcConnection
async def main():
    async def handler(method, params): return {}
    connection = JsonRpcConnection(handler)
    try:
        await connection.request("first", {"text": "✓" * 300_000}, timeout=0.1)
    except TimeoutError:
        print("timed out", file=sys.stderr, flush=True)
    await connection.notify("second", {"done": True})
asyncio.run(main())
"""
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", script,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        assert process.stderr
        try:
            self.assertEqual(await asyncio.wait_for(process.stderr.readline(), timeout=3), b"timed out\n")
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=3)
            self.assertEqual(process.returncode, 0, stderr.decode())
            frames = [json.loads(line) for line in stdout.splitlines()]
            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[0]["method"], "first")
            self.assertEqual(frames[0]["params"]["text"], "✓" * 300_000)
            self.assertEqual(frames[1]["method"], "second")
            self.assertTrue(frames[1]["params"]["done"])
        finally:
            if process.returncode is None:
                process.kill()
            await process.communicate()

    async def test_stdio_cancellation_cleans_up_connection_work(self) -> None:
        started = asyncio.Event()
        finished = asyncio.Event()

        async def handler(method, params):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                finished.set()

        self.connection.handler = handler
        transport = Mock(spec=asyncio.ReadTransport)
        transport.get_extra_info.return_value = None

        async def connect(factory, pipe):
            protocol = factory()
            protocol.connection_made(transport)
            protocol.data_received(b'{"jsonrpc":"2.0","id":1,"method":"block"}\n')
            return transport, protocol

        with patch.object(asyncio.get_running_loop(), "connect_read_pipe", side_effect=connect):
            serve = asyncio.create_task(self.connection.serve_stdio())
            request = None
            try:
                await asyncio.wait_for(started.wait(), timeout=1)
                request = asyncio.create_task(self.connection.request("question", {}))
                await asyncio.sleep(0)
                serve.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await serve
                self.assertTrue(self.connection.closed)
                self.assertTrue(finished.is_set())
                transport.close.assert_called_once()
                with self.assertRaisesRegex(ProtocolError, "connection closed"):
                    await asyncio.wait_for(request, timeout=1)
                with self.assertRaisesRegex(ProtocolError, "connection closed"):
                    await self.connection.request("later", {})
            finally:
                tasks = [serve, *self.connection._dispatch_tasks]
                if request is not None:
                    tasks.append(request)
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    async def test_acp_serve_closes_runners_after_failure_or_cancellation(self) -> None:
        from borealis_coder.protocol.acp import ACPServer

        for error in (RuntimeError("read failed"), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__):
                server = ACPServer()
                with (
                    patch.object(server.connection, "serve_stdio", side_effect=error),
                    patch.object(server, "close", new_callable=AsyncMock) as close,
                    self.assertRaises(type(error)),
                ):
                    await server.serve()
                close.assert_awaited_once()

    async def test_oversized_stdio_request_closes_input_transport(self) -> None:
        transport = Mock(spec=asyncio.ReadTransport)
        transport.get_extra_info.return_value = None

        async def connect(factory, pipe):
            protocol = factory()
            protocol.connection_made(transport)
            protocol.data_received(b"x" * 2048 + b"\n")
            return transport, protocol

        with (
            patch("borealis_coder.protocol.jsonrpc.MAX_STDIO_MESSAGE_BYTES", 1024),
            patch.object(asyncio.get_running_loop(), "connect_read_pipe", side_effect=connect),
            self.assertRaisesRegex(ProtocolError, "1024-byte limit"),
        ):
            await self.connection.serve_stdio()
        self.assertTrue(self.connection.closed)
        transport.close.assert_called_once()

    async def test_invalid_peer_error_fails_the_pending_request(self) -> None:
        request = asyncio.create_task(self.connection.request("question", {}))
        await asyncio.sleep(0)
        request_id = self.writes[-1]["id"]
        try:
            await self.connection._dispatch({"jsonrpc": "2.0", "id": request_id, "error": []})
            with self.assertRaisesRegex(ProtocolError, "invalid error response"):
                await request
            self.assertFalse(self.connection._pending)
        finally:
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)

    async def test_dispatch_requests_notifications_responses_and_errors(self) -> None:
        await self.connection._dispatch("bad")
        await self.connection._dispatch({"jsonrpc": "1.0", "id": 1})
        self.assertEqual(self.writes[-1]["error"]["code"], -32600)  # type: ignore[index]

        await self.connection._dispatch(
            {"jsonrpc": "2.0", "id": 2, "method": "echo", "params": [1]}
        )
        self.assertEqual(self.writes[-1]["error"]["code"], -32602)  # type: ignore[index]
        before = len(self.writes)
        await self.connection._dispatch({"jsonrpc": "2.0", "method": "echo", "params": [1]})
        self.assertEqual(len(self.writes), before)

        await self.connection._dispatch(
            {"jsonrpc": "2.0", "id": 3, "method": "echo", "params": {"x": 1}}
        )
        self.assertEqual(self.writes[-1]["result"]["method"], "echo")  # type: ignore[index]
        await self.connection._dispatch(
            {"jsonrpc": "2.0", "id": 4, "method": "none", "params": {}}
        )
        self.assertEqual(self.writes[-1]["result"], {})
        await self.connection._dispatch(
            {"jsonrpc": "2.0", "id": 5, "method": "protocol", "params": {}}
        )
        self.assertEqual(self.writes[-1]["error"]["code"], -32602)  # type: ignore[index]
        await self.connection._dispatch(
            {"jsonrpc": "2.0", "id": 6, "method": "explode", "params": {}}
        )
        self.assertEqual(self.writes[-1]["error"]["code"], -32603)  # type: ignore[index]

        future = asyncio.get_running_loop().create_future()
        self.connection._pending[9] = future
        await self.connection._dispatch({"jsonrpc": "2.0", "id": 9, "result": {"ok": True}})
        self.assertEqual(await future, {"ok": True})
        error_future = asyncio.get_running_loop().create_future()
        self.connection._pending[10] = error_future
        await self.connection._dispatch(
            {"jsonrpc": "2.0", "id": 10, "error": {"code": -1, "message": "peer"}}
        )
        with self.assertRaisesRegex(ProtocolError, "peer"):
            await error_future
        await self.connection._dispatch({"jsonrpc": "2.0", "id": 999, "result": {}})

    async def test_request_notify_send_and_timeout(self) -> None:
        request_task = asyncio.create_task(
            self.connection.request("question", {"x": 1}, timeout=1)
        )
        await asyncio.sleep(0)
        outbound = self.writes[-1]
        request_id = outbound["id"]
        await self.connection._dispatch(
            {"jsonrpc": "2.0", "id": request_id, "result": {"answer": 42}}
        )
        self.assertEqual(await request_task, {"answer": 42})
        self.assertNotIn(request_id, self.connection._pending)

        await self.connection.notify("event", {"value": 1})
        await self.connection.send_result("id", {"ok": True})
        await self.connection.send_error("id", -1, "bad", {"detail": 1})
        self.assertEqual(self.writes[-3]["method"], "event")
        self.assertEqual(self.writes[-1]["error"]["data"], {"detail": 1})  # type: ignore[index]

        with self.assertRaises(TimeoutError):
            await self.connection.request("never", {}, timeout=0.001)
        self.assertFalse(self.connection._pending)

    async def test_real_write_serializes_json(self) -> None:
        connection = JsonRpcConnection(self.connection.handler)
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as output:
            with patch("borealis_coder.protocol.jsonrpc.sys.stdout", output):
                await connection.notify("event", {"unicode": "✓"})
            output.seek(0)
            data = output.read()
        self.assertTrue(data.endswith("\n"))
        self.assertIn("✓", data)
        self.assertEqual(json.loads(data)["params"], {"unicode": "✓"})

    async def test_windows_output_keeps_the_existing_stdio_writer(self) -> None:
        connection = JsonRpcConnection(self.connection.handler)
        output: list[str] = []
        with (
            patch("borealis_coder.protocol.jsonrpc.os.name", "nt"),
            patch("borealis_coder.protocol.jsonrpc._write_stdout", side_effect=output.append),
            patch.object(asyncio.get_running_loop(), "connect_write_pipe", side_effect=AssertionError("pipe transport")),
        ):
            await connection.notify("event", {"unicode": "✓"})
        self.assertEqual(len(output), 1)
        self.assertEqual(json.loads(output[0])["params"], {"unicode": "✓"})


class PluginTests(unittest.TestCase):
    def test_entrypoint_tools_new_and_legacy_api(self) -> None:
        registry = ToolRegistry()
        direct = SimpleNamespace(name="direct", load=lambda: make_tool("entry_direct"))
        factory = SimpleNamespace(name="factory", load=lambda: lambda: [make_tool("entry_factory")])
        with patch("borealis_coder.plugins.metadata.entry_points", return_value=[direct, factory]):
            self.assertEqual(
                load_entrypoint_tools(registry), ["entry_direct", "entry_factory"]
            )

        legacy_registry = ToolRegistry()
        legacy = Mock()
        legacy.side_effect = [TypeError("old API"), {"borealis.tools": [direct]}]
        with patch("borealis_coder.plugins.metadata.entry_points", legacy):
            self.assertEqual(load_entrypoint_tools(legacy_registry), ["entry_direct"])

        bad = SimpleNamespace(name="bad", load=lambda: [object()])
        with (
            patch("borealis_coder.plugins.metadata.entry_points", return_value=[bad]),
            self.assertRaisesRegex(TypeError, "non-Tool"),
        ):
            load_entrypoint_tools(ToolRegistry())

    def test_workspace_plugins_opt_in_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plugins = root / ".borealis" / "plugins"
            plugins.mkdir(parents=True)
            (plugins / "good.py").write_text(
                textwrap.dedent(
                    """
                    from borealis_coder.models import Effect, ToolResult
                    from borealis_coder.tools.base import FunctionTool, object_schema
                    def register(registry):
                        registry.register(FunctionTool(
                            name='workspace_echo', description='echo', parameters=object_schema({}),
                            function=lambda args, ctx: ToolResult('ok'), effect=Effect.READ,
                        ))
                    """
                ),
                encoding="utf-8",
            )
            registry = ToolRegistry()
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(load_workspace_plugins(root, registry), [])
            with patch.dict(os.environ, {"BOREALIS_ENABLE_WORKSPACE_PLUGINS": "yes"}, clear=False):
                self.assertEqual(load_workspace_plugins(root, registry), ["workspace_echo"])

            missing = root / "other"
            with patch.dict(os.environ, {"BOREALIS_ENABLE_WORKSPACE_PLUGINS": "1"}, clear=False):
                self.assertEqual(load_workspace_plugins(missing, ToolRegistry()), [])

            (plugins / "bad.py").write_text("value = 1\n", encoding="utf-8")
            with (
                patch.dict(
                    os.environ,
                    {"BOREALIS_ENABLE_WORKSPACE_PLUGINS": "true"},
                    clear=False,
                ),
                self.assertRaisesRegex(TypeError, "must expose register"),
            ):
                load_workspace_plugins(root, ToolRegistry())


if __name__ == "__main__":
    unittest.main()
