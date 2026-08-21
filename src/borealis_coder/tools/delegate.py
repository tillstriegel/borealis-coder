"""Read-only delegated subagents for parallel repository investigation."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from ..models import Effect, Message, ProviderRequest, Role, ToolCall, ToolResult, Usage
from ..util import json_dumps, truncate_text
from .base import Tool, ToolContext, object_schema


class DelegateTaskTool(Tool):
    name = "delegate_task"
    description = "Delegate a bounded, read-only investigation to an isolated subagent and return its evidence-backed conclusion."
    effect = Effect.READ
    concurrent = True
    parameters = object_schema({
        "task": {"type": "string", "minLength": 1, "maxLength": 8000},
        "max_turns": {"type": "integer", "minimum": 1, "maximum": 16},
    })

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        routes = context.metadata.get("provider_routes")
        registry = context.metadata.get("tool_registry")
        builder = context.metadata.get("context_builder")
        usage_sink = context.metadata.get("usage_sink")
        if not routes or registry is None or builder is None:
            return ToolResult("Delegation is unavailable in this runtime", is_error=True)
        route = routes[0]
        read_tools = [
            tool for name in registry.names()
            if (tool := registry.get(name)) is not None
            and tool.effect == Effect.READ
            and name != self.name
        ]
        schemas = [tool.schema() for tool in read_tools]
        allowed = {tool.name for tool in read_tools}
        system = builder.system_prompt(query=str(arguments["task"])) + "\n\n# Delegated role\nYou are a read-only investigator. Gather precise evidence, do not mutate files, and return a concise conclusion with relevant paths and line references."
        messages = [Message(role=Role.USER, content=str(arguments["task"]))]
        repeated: set[str] = set()
        usage = Usage()
        final = ""
        for _ in range(int(arguments["max_turns"])):
            request = ProviderRequest(
                model=route.model, system=system, messages=messages, tools=schemas,
                max_output_tokens=min(context.config.agent.max_output_tokens, 8000),
                reasoning_effort=context.config.agent.reasoning_effort or None,
                parallel_tool_calls=True,
                metadata={"parent_session_id": context.session_id, "delegated": True},
            )
            response = await route.provider.with_retries(
                lambda request=request: route.provider.complete(request)
            )
            usage.add(response.usage)
            assistant = Message(role=Role.ASSISTANT, content=response.text, tool_calls=response.tool_calls)
            messages.append(assistant)
            if response.text:
                final = response.text
            if not response.tool_calls:
                break
            calls: list[ToolCall] = []
            for call in response.tool_calls:
                if call.name not in allowed:
                    messages.append(Message(role=Role.TOOL, content=f"Tool {call.name} is unavailable to read-only subagents", tool_call_id=call.id, tool_name=call.name, is_error=True))
                    continue
                signature = hashlib.sha256((call.name + json_dumps(call.arguments)).encode()).hexdigest()
                if signature in repeated:
                    messages.append(Message(role=Role.TOOL, content="Identical delegated tool call suppressed", tool_call_id=call.id, tool_name=call.name, is_error=True))
                    continue
                repeated.add(signature)
                calls.append(call)
            results = await asyncio.gather(*(registry.execute(call, context) for call in calls))
            for call, result in zip(calls, results, strict=True):
                messages.append(Message(role=Role.TOOL, content=result.output, tool_call_id=call.id, tool_name=call.name, is_error=result.is_error, metadata=result.metadata))
        if callable(usage_sink):
            result = usage_sink(usage)
            if asyncio.iscoroutine(result):
                await result
        return ToolResult(
            truncate_text(final or "Delegated investigation completed without a textual conclusion.", context.config.context.tool_output_chars),
            metadata={"usage": usage.to_dict(), "turns": sum(1 for item in messages if item.role == Role.ASSISTANT)},
        )
