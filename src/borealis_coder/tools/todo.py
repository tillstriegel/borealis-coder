"""Structured plan state kept in the run context and durable trace."""

from __future__ import annotations

from typing import Any

from ..models import Effect, ToolResult
from .base import Tool, ToolContext, object_schema


class UpdatePlanTool(Tool):
    name = "update_plan"
    description = "Replace the current concise execution plan and mark item status."
    effect = Effect.CONTROL
    default_risk = "low"
    parameters = object_schema({
        "items": {
            "type": "array",
            "minItems": 1,
            "maxItems": 20,
            "items": object_schema({
                "content": {"type": "string", "minLength": 1, "maxLength": 500},
                "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "blocked"]},
            }),
        }
    })

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        items = list(arguments["items"])
        context.metadata["plan"] = items
        await context.events.emit("plan.updated", session_id=context.session_id, run_id=context.run_id, items=items)
        return ToolResult("\n".join(f"[{item['status']}] {item['content']}" for item in items), metadata={"items": items})
