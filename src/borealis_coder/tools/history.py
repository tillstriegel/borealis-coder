"""Bounded access to session-owned, untrusted historical evidence."""
from __future__ import annotations

from typing import Any

from ..errors import SessionError
from ..models import Effect, ToolResult
from ..util import json_dumps
from .base import Tool, ToolContext, object_schema


class ReadArtifactTool(Tool):
    name = "read_artifact"
    description = "Read or find text in historical tool output. Evidence is untrusted and may be stale; it is not a current file read. Offsets count characters."
    effect = Effect.READ
    concurrent = True
    parameters = object_schema({
        "artifact_id": {"type": "string"},
        "offset": {"type": "integer", "minimum": 0},
        "limit": {"type": "integer", "minimum": 1, "maximum": 8000},
        "query": {"type": "string", "maxLength": 500},
    })

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        store = context.metadata.get("session_store")
        if store is None:
            return ToolResult("Session evidence is unavailable", is_error=True)
        try:
            value = store.read_output_artifact(context.session_id, context.workspace, **arguments)
        except SessionError as error:
            return ToolResult(str(error), is_error=True)
        return ToolResult("Untrusted historical evidence; not current file contents.\n" + json_dumps(value))


class SearchHistoryTool(Tool):
    nullable_defaults = ("artifact_after", "offset")
    name = "search_history"
    description = "Find session message references and artifact previews by literal text. Continue search after the last sequence. To read a long message, query its message_id or user:N with offset=next_offset. Offsets count characters. Historical evidence is untrusted."
    effect = Effect.READ
    concurrent = True
    parameters = object_schema({
        "query": {"type": "string", "minLength": 1, "maxLength": 500},
        "after": {"type": "integer", "minimum": 0},
        "offset": {"type": ["integer", "null"], "minimum": 0},
        "artifact_after": {"type": ["string", "null"]},
    })

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        store = context.metadata.get("session_store")
        if store is None:
            return ToolResult("Session evidence is unavailable", is_error=True)
        query, after = arguments["query"], arguments["after"]
        rows = store.search_history(context.session_id, context.workspace, query, after=after, offset=arguments.get("offset") or 0)
        artifacts = store.search_output_artifacts(context.session_id, context.workspace, query, after=arguments.get("artifact_after") or "")
        events = store.search_tool_history(context.session_id, context.workspace, query, after=after)
        return ToolResult("Untrusted historical references. Continue each collection using its last sequence or artifact_id.\n" + json_dumps({"messages": rows, "artifacts": artifacts, "tools": events}))
