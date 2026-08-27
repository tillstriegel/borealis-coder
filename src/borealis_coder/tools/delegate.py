"""Read-only delegated subagents for parallel repository investigation."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from typing import Any

from ..models import Effect, Message, ProviderRequest, Role, ToolCall, ToolResult, Usage
from ..util import json_dumps, new_id, truncate_text
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
        before_model_request = context.metadata.get("before_model_request")
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
        delegation_id = new_id("delegate")
        delegated_run_id = f"{context.run_id}.{delegation_id}"
        delegated_context = replace(
            context,
            run_id=delegated_run_id,
            tool_call_id="",
            metadata={**context.metadata, "delegated": True, "delegation_id": delegation_id},
        )
        await context.events.emit(
            "delegate.started",
            session_id=context.session_id,
            run_id=context.run_id,
            delegation_id=delegation_id,
            delegated_run_id=delegated_run_id,
            parent_tool_call_id=context.tool_call_id,
        )
        pending_cancellation: asyncio.CancelledError | None = None
        try:
            max_turns = int(arguments["max_turns"])
            for turn_index in range(max_turns):
                synthesis_turn = turn_index == max_turns - 1
                if callable(before_model_request):
                    before_model_request()
                request = ProviderRequest(
                    model=route.model,
                    system=(
                        system
                        if not synthesis_turn
                        else system
                        + "\n\nThis is your final turn. Do not request tools. Return the concise, evidence-backed conclusion now."
                    ),
                    messages=messages,
                    tools=[] if synthesis_turn else schemas,
                    max_output_tokens=min(context.config.agent.max_output_tokens, 8000),
                    reasoning_effort=context.config.agent.reasoning_effort or None,
                    parallel_tool_calls=not synthesis_turn,
                    metadata={"parent_session_id": context.session_id, "delegated": True},
                )
                response = await route.provider.with_retries(
                    lambda request=request: route.provider.complete(request)
                )
                usage.add(response.usage)
                assistant = Message(role=Role.ASSISTANT, content=response.text, tool_calls=response.tool_calls)
                messages.append(assistant)
                if response.text and not response.tool_calls:
                    final = response.text
                if not response.tool_calls:
                    break
                if synthesis_turn:
                    final = ""
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
                results = await asyncio.gather(
                    *(
                        registry.execute(
                            call,
                            replace(delegated_context, tool_call_id=call.id),
                        )
                        for call in calls
                    )
                )
                for call, result in zip(calls, results, strict=True):
                    messages.append(Message(role=Role.TOOL, content=result.output, tool_call_id=call.id, tool_name=call.name, is_error=result.is_error, metadata=result.metadata))
        except asyncio.CancelledError as error:
            pending_cancellation = error
            await context.events.emit(
                "delegate.cancelled",
                session_id=context.session_id,
                run_id=context.run_id,
                delegation_id=delegation_id,
                delegated_run_id=delegated_run_id,
            )
        finally:
            try:
                if callable(usage_sink):
                    result = usage_sink(usage)
                    if asyncio.iscoroutine(result):
                        await result
            except BaseException as error:
                if pending_cancellation is not None:
                    raise pending_cancellation from error
                raise
        if pending_cancellation is not None:
            raise pending_cancellation
        turns = sum(1 for item in messages if item.role == Role.ASSISTANT)
        await context.events.emit(
            "delegate.completed",
            session_id=context.session_id,
            run_id=context.run_id,
            delegation_id=delegation_id,
            delegated_run_id=delegated_run_id,
            turns=turns,
            usage=usage.to_dict(),
        )
        if not final:
            return ToolResult(
                "Delegated investigation ended without the required textual conclusion.",
                is_error=True,
                metadata={"usage": usage.to_dict(), "turns": turns, "delegation_id": delegation_id},
            )
        return ToolResult(
            truncate_text(final, context.config.context.tool_output_chars),
            metadata={"usage": usage.to_dict(), "turns": turns, "delegation_id": delegation_id},
        )
