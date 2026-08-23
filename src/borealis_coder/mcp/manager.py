"""MCP server lifecycle and dynamic tool registration."""

from __future__ import annotations

import asyncio
import contextlib
import re
from pathlib import Path
from typing import Any

from ..config import Config
from ..errors import ProtocolError
from ..models import Effect, ToolResult
from ..tools.base import Tool, ToolContext, ToolRegistry
from ..util import json_dumps
from .client import HttpMCPClient, MCPClient, MCPToolDefinition, StdioMCPClient


class MCPTool(Tool):
    def __init__(
        self,
        server_name: str,
        definition: MCPToolDefinition,
        client: MCPClient,
        *,
        read_only: bool = False,
    ) -> None:
        self.server_name = server_name
        self.remote_name = definition.name
        self.name = _tool_name(server_name, definition.name)
        self.description = f"MCP server {server_name}: {definition.description or definition.name}"
        self.parameters = definition.input_schema or {"type": "object", "properties": {}, "additionalProperties": True}
        self.annotations = definition.annotations
        self.client = client
        open_world = bool(self.annotations.get("openWorldHint"))
        destructive = bool(self.annotations.get("destructiveHint"))
        if destructive:
            self.effect = Effect.CONTROL
        elif open_world:
            self.effect = Effect.NETWORK
        elif read_only:
            self.effect = Effect.READ
        else:
            self.effect = Effect.CONTROL
        self.concurrent = read_only and not (open_world or destructive)
        self.default_risk = (
            "critical" if destructive else ("high" if open_world or not read_only else "low")
        )

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        result = await self.client.call_tool(self.remote_name, arguments)
        content = result.get("content") or []
        chunks: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                chunks.append(str(item.get("text") or ""))
            elif item.get("type") in {"resource", "resource_link"}:
                chunks.append(json_dumps(item, pretty=True))
            else:
                chunks.append(json_dumps(item, pretty=True))
        if result.get("structuredContent") is not None:
            chunks.append("structuredContent:\n" + json_dumps(result["structuredContent"], pretty=True))
        output = "\n".join(item for item in chunks if item) or "(MCP tool returned no content)"
        return ToolResult(output, is_error=bool(result.get("isError")), metadata={"mcp_server": self.server_name, "remote_tool": self.remote_name})


class MCPManager:
    def __init__(self, workspace: Path, config: Config) -> None:
        self.workspace = workspace
        self.config = config
        self.clients: dict[str, MCPClient] = {}
        self.errors: dict[str, str] = {}

    async def connect_all(self, registry: ToolRegistry) -> None:
        async def connect(name: str):
            server = self.config.mcp_servers[name]
            if not server.enabled:
                return
            client: MCPClient | None = None
            registered: list[str] = []
            try:
                if server.type == "stdio":
                    client = StdioMCPClient(name, server, self.workspace, env_allowlist=self.config.safety.env_allowlist)
                    await client.start()
                elif server.type == "http":
                    client = HttpMCPClient(name, server, self.workspace)
                    await asyncio.wait_for(client.initialize(), timeout=server.timeout_seconds)
                else:
                    raise ProtocolError(f"Unsupported MCP transport: {server.type}")
                definitions = await client.list_tools()
                allowed = set(server.allowed_tools)
                read_only = set(server.read_only_tools)
                for definition in definitions:
                    if allowed and definition.name not in allowed:
                        continue
                    tool = MCPTool(
                        name,
                        definition,
                        client,
                        read_only=definition.name in read_only,
                    )
                    registry.register(tool)
                    registered.append(tool.name)
                self.clients[name] = client
            except asyncio.CancelledError:
                for tool_name in reversed(registered):
                    registry.unregister(tool_name)
                if client is not None:
                    with contextlib.suppress(Exception):
                        await client.close()
                raise
            except Exception as error:
                for tool_name in reversed(registered):
                    registry.unregister(tool_name)
                self.errors[name] = f"{type(error).__name__}: {error}"
                if client is not None:
                    with contextlib.suppress(Exception):
                        await client.close()

        await asyncio.gather(*(connect(name) for name in sorted(self.config.mcp_servers)))

    async def close(self) -> None:
        await asyncio.gather(*(client.close() for client in self.clients.values()), return_exceptions=True)
        self.clients.clear()


def _tool_name(server: str, tool: str) -> str:
    value = f"mcp__{server}__{tool}"
    return re.sub(r"[^A-Za-z0-9_]", "_", value)[:128]
