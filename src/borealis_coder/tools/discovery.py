"""Paged discovery of the current registry. Discovery never grants authority."""
from __future__ import annotations

from typing import Any

from ..models import ToolResult
from ..util import json_dumps
from .base import Tool, ToolContext, object_schema


class DiscoverToolsTool(Tool):
    name = "discover_tools"
    description = (
        "Discover available tool argument schemas. Empty names returns a compact "
        "paged catalog; pass exact names for full argument schemas. Follow next_offset "
        "until null. Discovery does not grant additional tool access."
    )
    concurrent = True
    parameters = object_schema({
        "names": {"type": "array", "items": {"type": "string", "maxLength": 256}, "maxItems": 8},
        "offset": {"type": "integer", "minimum": 0},
    })

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        registry = context.metadata["tool_registry"]
        def available(name: str) -> bool:
            tool = registry.get(name)
            if tool is None:
                return False
            if context.metadata.get("delegated"):
                from .delegate import delegated_tool_allowed

                return delegated_tool_allowed(tool, context.metadata.get("delegation_allow_writes", False))
            return True

        names = arguments["names"] or [name for name in registry.names() if available(name)]
        offset = arguments["offset"]
        entries = []
        byte_limit = min(12_000, context.config.context.tool_output_chars - 256)
        # One large external schema is a controlled error, never a fake complete contract.
        for name in names[offset:offset + 32]:
            tool = registry.get(name) if available(name) else None
            entry = ({"name": name, "unavailable": True} if tool is None else {
                **(tool.schema_for_context(context) if arguments["names"] else {
                    "name": name, "description": tool.schema_for_context(context)["description"][:180],
                }),
                "script_accessible": tool.script_accessible,
                "effect": tool.effect.value,
            })
            if len(json_dumps([*entries, entry]).encode("utf-8")) > byte_limit:
                if not entries:
                    return ToolResult("Tool schema exceeds discovery byte limit; use a narrower registry or direct mode.", is_error=True)
                break
            entries.append(entry)
        next_offset = offset + len(entries)
        return ToolResult(json_dumps({
            "tools": entries, "total": len(names),
            "next_offset": next_offset if next_offset < len(names) else None,
        }))
