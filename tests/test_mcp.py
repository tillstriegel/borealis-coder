from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import signal
import sys
import tempfile
import unittest
import urllib.error
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch

from borealis_coder.config import MCPServerConfig
from borealis_coder.errors import ProtocolError
from borealis_coder.mcp import MCPManager, MCPTool
from borealis_coder.mcp.client import HttpMCPClient, MCPClient, MCPToolDefinition, StdioMCPClient
from borealis_coder.models import ToolCall
from borealis_coder.tools import MutationScope, build_builtin_registry
from tests.helpers import make_config, make_context

SERVER = r'''
import json, sys
for line in sys.stdin:
    msg=json.loads(line)
    if "id" not in msg:
        continue
    method=msg.get("method")
    if method=="initialize":
        result={"protocolVersion":"2025-11-25","capabilities":{"tools":{}},"serverInfo":{"name":"fake","version":"1"}}
    elif method=="tools/list":
        result={"tools":[{"name":"echo","description":"Echo text","inputSchema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"],"additionalProperties":False},"annotations":{"readOnlyHint":True}}]}
    elif method=="tools/call":
        text=msg.get("params",{}).get("arguments",{}).get("text","")
        result={"content":[{"type":"text","text":"echo:"+text}],"isError":False}
    else:
        result={}
    print(json.dumps({"jsonrpc":"2.0","id":msg["id"],"result":result}), flush=True)
'''


class MCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_pagination_preserves_opaque_cursors(self):
        client = HttpMCPClient("pages", MCPServerConfig(), Path.cwd())
        responses = [
            {"tools": [{"name": "first"}], "nextCursor": ""},
            {"tools": [{"name": "second"}], "nextCursor": "  opaque/token==  "},
            {"tools": [{"name": "third"}]},
        ]
        with patch.object(client, "request", side_effect=responses) as request:
            definitions = await client.list_tools()
        self.assertEqual([item.name for item in definitions], ["first", "second", "third"])
        self.assertEqual(
            [call.args for call in request.call_args_list],
            [("tools/list", {}), ("tools/list", {"cursor": ""}),
             ("tools/list", {"cursor": "  opaque/token==  "})],
        )

    async def test_tool_pagination_rejects_cycles_and_invalid_cursors(self):
        for cursors in (("same", "same"), ("a", "b", "a"), ([],), (False,), (12,)):
            with self.subTest(cursors=cursors):
                client = HttpMCPClient("pages", MCPServerConfig(), Path.cwd())
                responses = [{"tools": [{"name": "echo"}], "nextCursor": cursor} for cursor in cursors]
                with (
                    patch.object(client, "request", side_effect=responses) as request,
                    self.assertRaisesRegex(ProtocolError, "pagination cursor"),
                ):
                    await client.list_tools()
                self.assertEqual(request.call_count, len(cursors))

    @unittest.skipUnless(os.name == "posix", "Process groups require POSIX")
    async def test_stdio_close_stops_children_on_exit_and_cancellation(self):
        import fcntl

        child_code = (
            "import fcntl, os, time\nfrom pathlib import Path\n"
            "handle=open('child.lock', 'w')\nfcntl.flock(handle, fcntl.LOCK_EX)\n"
            "Path('child.ready').write_text(str(os.getpid()))\ntime.sleep(60)\n"
        )
        for cancel_close in (False, True):
            with self.subTest(cancel_close=cancel_close), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                script = root / "server.py"
                script.write_text(
                    "import subprocess, sys, time\nfrom pathlib import Path\n"
                    f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
                    "while not Path('child.ready').exists(): time.sleep(0.01)\n"
                    + SERVER + ("\ntime.sleep(60)\n" if cancel_close else "")
                )
                client = StdioMCPClient(
                    "tree", MCPServerConfig(command=sys.executable, args=[str(script)]),
                    root, env_allowlist=[],
                )
                await client.start()
                process = client.process
                assert process is not None
                closing = asyncio.create_task(client.close())
                child_stopped = False
                try:
                    if cancel_close:
                        await asyncio.sleep(0.02)
                        self.assertFalse(closing.done())
                        closing.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await closing
                    else:
                        await asyncio.wait_for(closing, timeout=5)
                    self.assertIsNone(client.process)
                    self.assertIsNotNone(process.returncode)
                    self.assertTrue(client._reader_task and client._reader_task.done())
                    self.assertTrue(client._stderr_task and client._stderr_task.done())
                    # A dead child releases this kernel lock even if the system
                    # has not yet reaped its process-table entry.
                    with (root / "child.lock").open("rb") as handle:
                        async with asyncio.timeout(1):
                            while True:
                                try:
                                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                    child_stopped = True
                                    break
                                except BlockingIOError:
                                    await asyncio.sleep(0.01)
                    await client.close()
                finally:
                    if not child_stopped:
                        with contextlib.suppress(ProcessLookupError, PermissionError):
                            os.killpg(process.pid, signal.SIGKILL)
                    closing.cancel()
                    await asyncio.gather(closing, return_exceptions=True)
                    await asyncio.wait_for(process.wait(), timeout=2)
                    tasks = [task for task in (client._reader_task, client._stderr_task) if task]
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

    async def test_failed_stdio_send_removes_pending_request(self):
        client = StdioMCPClient("fake", MCPServerConfig(), Path.cwd(), env_allowlist=[])
        client.process = cast(asyncio.subprocess.Process, SimpleNamespace(stdin=object()))
        with (
            patch.object(client, "_send", side_effect=BrokenPipeError("closed")),
            self.assertRaises(BrokenPipeError),
        ):
            await client.request("tools/list")
        self.assertFalse(client._pending)

    async def test_stdio_request_timeout_includes_blocked_send(self):
        client = StdioMCPClient(
            "fake", MCPServerConfig(timeout_seconds=1), Path.cwd(), env_allowlist=[]
        )
        client.process = cast(asyncio.subprocess.Process, SimpleNamespace(stdin=object()))
        sending = asyncio.Event()

        async def blocked_send(value):
            sending.set()
            await asyncio.Event().wait()

        with patch.object(client, "_send", side_effect=blocked_send):
            request = asyncio.create_task(client.request("tools/list"))
            try:
                await sending.wait()
                done, _ = await asyncio.wait({request}, timeout=2)
                self.assertIn(request, done, "Request timeout did not cover the write")
                with self.assertRaises(TimeoutError):
                    await request
                self.assertFalse(client._pending)
            finally:
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)

    async def test_stdio_stderr_drains_long_lines(self):
        client = StdioMCPClient("fake", MCPServerConfig(), Path.cwd(), env_allowlist=[])
        stderr = asyncio.StreamReader()
        stderr.feed_data(b"x" * 200_000 + b"\n")
        stderr.feed_eof()
        client.process = cast(asyncio.subprocess.Process, SimpleNamespace(stderr=stderr))
        await client._drain_stderr()
        self.assertTrue(stderr.at_eof())

    async def test_cancelled_stdio_send_removes_pending_request(self):
        client = StdioMCPClient("fake", MCPServerConfig(), Path.cwd(), env_allowlist=[])
        client.process = cast(asyncio.subprocess.Process, SimpleNamespace(stdin=object()))
        sending = asyncio.Event()

        async def blocked_send(value):
            sending.set()
            await asyncio.Event().wait()

        with patch.object(client, "_send", side_effect=blocked_send):
            request = asyncio.create_task(client.request("tools/list"))
            await sending.wait()
            request.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await request
        self.assertFalse(client._pending)

    async def test_stdio_accepts_large_tool_results(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script = root / "server.py"
            script.write_text(SERVER)
            client = StdioMCPClient(
                "large",
                MCPServerConfig(command=sys.executable, args=[str(script)]),
                root,
                env_allowlist=[],
            )
            await client.start()
            try:
                text = "x" * 100_000
                result = await client.call_tool("echo", {"text": text})
                self.assertEqual(result["content"][0]["text"], "echo:" + text)
            finally:
                await client.close()

    async def test_oversized_stdio_response_fails_and_closes_cleanly(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script = root / "server.py"
            script.write_text(SERVER)
            client = StdioMCPClient(
                "oversized",
                MCPServerConfig(command=sys.executable, args=[str(script)]),
                root,
                env_allowlist=[],
            )
            with patch("borealis_coder.mcp.client.MAX_STDIO_MESSAGE_BYTES", 1024):
                await client.start()
            try:
                with self.assertRaisesRegex(ProtocolError, "failed to read a response"):
                    await client.call_tool("echo", {"text": "x" * 200_000})
                with self.assertRaisesRegex(ProtocolError, "failed to read a response"):
                    await asyncio.wait_for(client.list_tools(), timeout=1)
            finally:
                await asyncio.wait_for(client.close(), timeout=5)
            self.assertIsNone(client.process)
            self.assertFalse(client._pending)

    async def test_stdio_ignores_unrelated_messages_and_rejects_invalid_errors(self):
        client = StdioMCPClient("fake", MCPServerConfig(), Path.cwd(), env_allowlist=[])
        stdout = asyncio.StreamReader()
        messages = [
            None,
            {"id": []},
            {"id": True, "result": "wrong"},
            {"id": 1, "method": "notifications/progress"},
            {"id": 1, "error": "invalid"},
            {"id": 2, "result": {"ok": True}},
        ]
        stdout.feed_data(b"\n".join(json.dumps(item).encode() for item in messages) + b"\n")
        stdout.feed_eof()
        client.process = cast(asyncio.subprocess.Process, SimpleNamespace(stdout=stdout))
        first = asyncio.get_running_loop().create_future()
        second = asyncio.get_running_loop().create_future()
        client._pending = {1: first, 2: second}
        await client._reader_loop()
        with self.assertRaisesRegex(ProtocolError, "invalid error response"):
            await first
        self.assertEqual(await second, {"ok": True})

    def test_mutating_mcp_tools_declare_external_mutation_scope(self):
        definition = MCPToolDefinition("mutate", "Mutate state", {}, {})
        client = cast(MCPClient, SimpleNamespace())
        tool = MCPTool("server", definition, client, read_only=False)

        self.assertEqual(tool.effective_mutation_scope, MutationScope.EXTERNAL)

    def test_http_error_response_is_closed(self):
        with tempfile.TemporaryDirectory() as td:
            stream = io.BytesIO(b'{"error":"down"}')
            error = urllib.error.HTTPError(
                "https://mcp.example/rpc",
                503,
                "down",
                Message(),
                stream,
            )
            client = HttpMCPClient(
                "remote",
                MCPServerConfig(type="http", url="https://mcp.example/rpc"),
                Path(td),
            )
            with (
                patch("borealis_coder.mcp.client.open_same_origin", side_effect=error),
                self.assertRaisesRegex(ProtocolError, "MCP HTTP remote returned 503"),
            ):
                client._post({"jsonrpc": "2.0"})
            self.assertTrue(stream.closed)

    async def test_stdio_discovery_and_call(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script = root / "server.py"
            script.write_text(SERVER)
            config = make_config(root)
            config.mcp_servers["fake"] = MCPServerConfig(
                type="stdio",
                command=sys.executable,
                args=[str(script)],
                read_only_tools=["echo"],
            )
            registry = build_builtin_registry()
            manager = MCPManager(root, config)
            await manager.connect_all(registry)
            try:
                self.assertFalse(manager.errors)
                tool = registry.get("mcp__fake__echo")
                self.assertIsNotNone(tool)
                context = make_context(root, config)
                result = await registry.execute(ToolCall(name="mcp__fake__echo", arguments={"text":"hello"}), context)
                self.assertFalse(result.is_error, result.output)
                self.assertEqual(result.output, "echo:hello")
            finally:
                await manager.close()

    async def test_server_annotation_cannot_reduce_local_risk(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script = root / "server.py"
            script.write_text(SERVER)
            config = make_config(root)
            config.mcp_servers["fake"] = MCPServerConfig(
                type="stdio", command=sys.executable, args=[str(script)]
            )
            registry = build_builtin_registry()
            manager = MCPManager(root, config)
            await manager.connect_all(registry)
            try:
                context = make_context(root, config)
                result = await registry.execute(
                    ToolCall(name="mcp__fake__echo", arguments={"text": "hello"}),
                    context,
                )
                self.assertTrue(result.is_error)
                self.assertIn("no approval channel", result.output)
            finally:
                await manager.close()

    async def test_failed_startup_closes_partial_client(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(root)
            config.mcp_servers["broken"] = MCPServerConfig(
                type="stdio", command=sys.executable
            )
            client = SimpleNamespace(
                start=AsyncMock(side_effect=TimeoutError("not ready")),
                close=AsyncMock(),
            )
            manager = MCPManager(root, config)
            with patch("borealis_coder.mcp.manager.StdioMCPClient", return_value=client):
                await manager.connect_all(build_builtin_registry())
            client.close.assert_awaited_once()
            self.assertIn("broken", manager.errors)

    async def test_cancelled_startup_closes_partial_client(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(root)
            config.mcp_servers["cancelled"] = MCPServerConfig(
                type="stdio", command=sys.executable
            )
            client = SimpleNamespace(
                start=AsyncMock(side_effect=asyncio.CancelledError()),
                close=AsyncMock(),
            )
            manager = MCPManager(root, config)
            with (
                patch("borealis_coder.mcp.manager.StdioMCPClient", return_value=client),
                self.assertRaises(asyncio.CancelledError),
            ):
                await manager.connect_all(build_builtin_registry())
            client.close.assert_awaited_once()
            self.assertNotIn("cancelled", manager.errors)

    async def test_failed_registration_removes_tools_from_closed_client(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(root)
            config.mcp_servers["collision"] = MCPServerConfig(
                type="stdio", command=sys.executable
            )
            definitions = [
                MCPToolDefinition("foo-bar", "first", {}, {}),
                MCPToolDefinition("foo_bar", "second", {}, {}),
            ]
            client = SimpleNamespace(
                start=AsyncMock(),
                list_tools=AsyncMock(return_value=definitions),
                close=AsyncMock(),
            )
            registry = build_builtin_registry()
            manager = MCPManager(root, config)
            with patch("borealis_coder.mcp.manager.StdioMCPClient", return_value=client):
                await manager.connect_all(registry)

            client.close.assert_awaited_once()
            self.assertIn("collision", manager.errors)
            self.assertIsNone(registry.get("mcp__collision__foo_bar"))
            self.assertNotIn("collision", manager.clients)


if __name__ == "__main__":
    unittest.main()
