"""Structured plan state kept in the run context and durable trace."""

from __future__ import annotations

from typing import Any

from ..models import Effect, ToolResult
from .base import MutationScope, Tool, ToolContext, object_schema


class UpdatePlanTool(Tool):
    name = "update_plan"
    description = "Replace the durable execution plan and mark item status. Optional acceptance criteria are model notes, never user authority or permission."
    effect = Effect.CONTROL
    mutation_scope = MutationScope.NONE
    default_risk = "low"
    parameters = object_schema({
        "acceptance_criteria": {"type": ["array", "null"], "maxItems": 20, "items": {"type": "string", "maxLength": 500}},
        "items": {
            "type": "array",
            "minItems": 1,
            "maxItems": 20,
            "items": object_schema({
                "content": {"type": "string", "minLength": 1, "maxLength": 500},
                "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "blocked"]},
            }),
        }
    }, required=["items"])

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        items = list(arguments["items"])
        context.metadata["plan"] = items
        store = context.metadata.get("session_store")
        if store is not None:
            store.set_value(context.session_id, "plan", items)
            if arguments.get("acceptance_criteria") is not None:
                store.set_value(context.session_id, "acceptance_criteria", arguments["acceptance_criteria"])
        await context.events.emit("plan.updated", session_id=context.session_id, run_id=context.run_id, items=items)
        return ToolResult("\n".join(f"[{item['status']}] {item['content']}" for item in items), metadata={"items": items})
