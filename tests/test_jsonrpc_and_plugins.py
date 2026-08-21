from __future__ import annotations

import asyncio
import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
        output: list[str] = []
        with patch("borealis_coder.protocol.jsonrpc._write_stdout", side_effect=output.append):
            await connection.notify("event", {"unicode": "✓"})
        self.assertEqual(len(output), 1)
        self.assertTrue(output[0].endswith("\n"))
        self.assertIn("✓", output[0])


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
