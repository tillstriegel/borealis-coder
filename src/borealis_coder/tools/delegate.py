"""Read-only delegated subagents for parallel repository investigation."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
from dataclasses import replace
from typing import Any

from ..agent.budget import prepare_route_request
from ..errors import BudgetExceeded, ProviderContextOverflowError, ProviderError
from ..models import Effect, Message, ProviderRequest, Role, ToolCall, ToolResult, Usage
from ..util import finish_on_cancellation, json_dumps, new_id
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
        system = await asyncio.to_thread(builder.system_prompt, query=str(arguments["task"]))
        system += "\n\n# Delegated role\nYou are a read-only investigator. Gather precise evidence, do not mutate files, and return a concise handoff with findings, relevant paths and line references, unresolved questions, and completion status."
        messages = [Message(role=Role.USER, content=str(arguments["task"]))]
        repeated: set[str] = set()
        usage = Usage()

        budget = context.metadata.get("budget")
        reservation = 0.0
        usage_recorded = False
        cancelled_usage = Usage()

        def observe_usage(increment: Usage) -> None:
            cancelled_usage.add(increment)
            if budget:
                budget.hold_usage(cancelled_usage)

        async def record_usage(increment: Usage) -> None:
            nonlocal reservation, usage_recorded
            usage_recorded = True
            usage.add(increment)
            async def settle() -> None:
                nonlocal reservation
                if budget:
                    budget.release_cost(reservation)
                    budget.release_usage(cancelled_usage)
                    reservation = 0.0
                if callable(usage_sink):
                    result = usage_sink(increment)
                    if asyncio.iscoroutine(result):
                        await result
            await finish_on_cancellation(settle())
            await context.events.emit("delegate.usage", session_id=context.session_id, run_id=context.run_id,
                                      provider=route.name, model=route.model, usage=increment.to_dict())

        final = ""
        sources: list[dict[str, Any]] = []
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
        try:
            overflow_retries = 0
            max_turns = int(arguments["max_turns"])
            turn_index = 0
            while turn_index < max_turns:
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
                    metadata={
                        "parent_session_id": context.session_id, "delegated": True,
                        "provider_route": route.name,
                    },
                )
                request = prepare_route_request(request, route.provider, context.config.agent, overflow_retry_count=overflow_retries)
                reservation = budget.reserve_cost(route.provider, request) if budget else 0.0
                cancelled_usage = Usage()
                usage_recorded = False
                try:
                    response = await route.provider.with_retries(
                        lambda request=request: route.provider.complete(request),
                        on_usage=observe_usage,
                        before_recovery=before_model_request if callable(before_model_request) else None,
                        request=request,
                        check_usage=(lambda _, observed=cancelled_usage: budget.check_unsettled(observed)) if budget else None,
                    )
                    await record_usage(response.usage)
                except asyncio.CancelledError:
                    if not usage_recorded and not cancelled_usage.is_empty:
                        with contextlib.suppress(BudgetExceeded):
                            await record_usage(cancelled_usage)
                    raise
                except BudgetExceeded:
                    if not usage_recorded and not cancelled_usage.is_empty:
                        with contextlib.suppress(BudgetExceeded):
                            await record_usage(cancelled_usage)
                    raise
                except ProviderError as error:
                    if error.usage is not None:
                        await record_usage(error.usage)
                    if isinstance(error, ProviderContextOverflowError) and overflow_retries < context.config.agent.compaction_max_overflow_retries:
                        overflow_retries += 1
                        continue
                    raise
                except Exception:
                    if not usage_recorded and not cancelled_usage.is_empty:
                        with contextlib.suppress(BudgetExceeded):
                            await record_usage(cancelled_usage)
                    raise
                finally:
                    if budget:
                        budget.release_cost(reservation)
                        budget.release_usage(cancelled_usage)
                overflow_retries = 0
                turn_index += 1
                assistant = Message(
                    role=Role.ASSISTANT,
                    content=response.text,
                    tool_calls=[] if response.incomplete else response.tool_calls,
                )
                if response.continuation_state is not None and not response.incomplete:
                    continuation = response.continuation_state.to_metadata(
                        provider=route.name, model=route.model,
                    )
                    if continuation is not None:
                        assistant.metadata["continuation_state"] = continuation
                messages.append(assistant)
                if response.incomplete:
                    break
                if response.text.strip() and not response.tool_calls:
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
                    sources.append({"tool_call_id": call.id, **{key: result.metadata[key] for key in ("path", "sha256", "output_artifact") if key in result.metadata}})
                    messages.append(Message(role=Role.TOOL, content=result.output, tool_call_id=call.id, tool_name=call.name, is_error=result.is_error, metadata=result.metadata))
        except asyncio.CancelledError:
            await context.events.emit(
                "delegate.cancelled",
                session_id=context.session_id,
                run_id=context.run_id,
                delegation_id=delegation_id,
                delegated_run_id=delegated_run_id,
            )
            raise
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
                "Delegated investigation ended without the required textual conclusion. Incomplete; evidence references: " + json_dumps(sources),
                is_error=True,
                metadata={"usage": usage.to_dict(), "turns": turns, "delegation_id": delegation_id},
            )
        return ToolResult(
            "Delegation complete. Findings and source references:\n" + final + ("\nEvidence references (search_history/read_artifact): " + json_dumps(sources) if sources else ""),
            metadata={"usage": usage.to_dict(), "turns": turns, "delegation_id": delegation_id},
        )
