from __future__ import annotations

import asyncio
import io
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
from borealis_coder.mcp.client import HttpMCPClient, MCPClient, MCPToolDefinition
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
