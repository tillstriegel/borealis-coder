"""Bounded recursive subagents with isolated conversations and inherited authority."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any, cast

from ..agent.budget import prepare_route_request
from ..agent.swarm import SwarmState, bounded_calls
from ..errors import BudgetExceeded, ProviderContextOverflowError, ProviderError
from ..models import (
    Effect,
    Message,
    ModelResponse,
    ProviderRequest,
    Role,
    ToolCall,
    ToolResult,
    Usage,
)
from ..providers.base import Provider
from ..util import finish_on_cancellation, json_dumps, new_id, truncate_text
from .base import MutationScope, Tool, ToolContext, object_schema


def delegated_tool_allowed(tool: Tool, allow_writes: bool) -> bool:
    """Keep this execution guard identical to the advertised child palette."""
    from .filesystem import (
        DeleteFileTool,
        MakeDirectoryTool,
        ReplaceInFileTool,
        WriteFileTool,
    )
    from .patch import ApplyPatchTool
    from .script import ScriptJobTool, ScriptTool

    if isinstance(tool, ScriptTool | ScriptJobTool):
        return False
    if isinstance(tool, DelegateTaskTool):
        return tool.effect == Effect.READ and tool.effective_mutation_scope == MutationScope.NONE
    if allow_writes and type(tool) in (
        WriteFileTool, ReplaceInFileTool, DeleteFileTool, MakeDirectoryTool, ApplyPatchTool,
    ):
        return True
    return (tool.effect == Effect.READ and tool.script_accessible
            and tool.effective_mutation_scope == MutationScope.NONE)


class DelegateTaskTool(Tool):
    name = "delegate_task"
    script_accessible = False
    description = "Delegate a bounded task to an isolated subagent with inherited operator permissions and return its evidence-backed conclusion."
    effect = Effect.READ
    concurrent = True
    parameters = object_schema({
        "task": {"type": "string", "minLength": 1, "maxLength": 8000},
        "max_turns": {"type": "integer", "minimum": 1, "maximum": 16},
    })

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        state = context.metadata.get("swarm_state")
        if not isinstance(state, SwarmState):
            # Direct registry users also get one context-local tree, never global state.
            state = SwarmState.start(context.config.agent)
            context.metadata["swarm_state"] = state
        depth = context.metadata.get("delegation_depth", 0) + 1
        delegation_id = new_id("delegate")
        delegated_run_id = f"{context.run_id}.{delegation_id}"
        allow_writes = state.allow_writes and context.metadata.get("delegation_allow_writes", True)
        lineage = {
            "delegation_id": delegation_id, "delegated_run_id": delegated_run_id,
            "parent_run_id": context.run_id,
            "parent_delegation_id": context.metadata.get("delegation_id"),
            "parent_tool_call_id": context.tool_call_id, "depth": depth,
            "task": truncate_text(str(arguments["task"]), 160),
            "capability": "write" if allow_writes else "read-only",
        }

        async def emit(kind: str, **data: Any) -> None:
            budget = context.metadata.get("budget")
            await context.events.emit(kind, session_id=context.session_id, run_id=context.run_id,
                                      **lineage, limits=state.limits(),
                                      model_requests=budget.model_requests if budget else 0, **data)

        refusal = state.admit(depth)
        if refusal:
            await emit("delegate.refused", error=refusal)
            return ToolResult(refusal, is_error=True, metadata={**lineage, "refused": True})
        # A started child owns its entire subtree until its terminal event.
        try:
            await emit("delegate.started", state="waiting")
            delegated_context = replace(
                context, run_id=delegated_run_id, tool_call_id="",
                metadata={**context.metadata, "delegated": True,
                          "delegation_id": delegation_id, "delegation_depth": depth,
                          "delegation_allow_writes": allow_writes},
            )
            result = await self._investigate(arguments, delegated_context, state, emit)
            await emit("delegate.failed" if result.is_error else "delegate.completed",
                       result=truncate_text(result.output, context.config.context.tool_output_chars),
                       turns=result.metadata.get("turns", 0), usage=result.metadata.get("usage", {}),
                       evidence=result.metadata.get("evidence", []))
            result.metadata.update(lineage)
            return result
        except asyncio.CancelledError:
            await finish_on_cancellation(emit("delegate.cancelled"))
            raise
        except BaseException as error:
            await finish_on_cancellation(emit("delegate.failed", error=str(error)))
            raise

    async def _investigate(
        self, arguments: dict[str, Any], context: ToolContext, state: SwarmState, emit: Any,
    ) -> ToolResult:
        routes = context.metadata.get("provider_routes")
        registry = context.metadata.get("tool_registry")
        builder = context.metadata.get("context_builder")
        usage_sink = context.metadata.get("usage_sink")
        before_model_request = cast(Callable[[], None] | None, context.metadata.get("before_model_request"))
        execute_nested = cast(Callable[[ToolCall, ToolContext], Awaitable[ToolResult]] | None,
                              context.metadata.get("delegate_execute"))
        if not routes or registry is None or builder is None:
            return ToolResult("Delegation is unavailable in this runtime", is_error=True)
        route = routes[0]
        provider = cast(Provider, route.provider)
        tools = [tool for name in registry.names()
                 if (tool := registry.get(name)) is not None
                 and delegated_tool_allowed(tool, context.metadata["delegation_allow_writes"])]
        schemas = [tool.schema() for tool in tools]
        system = await asyncio.to_thread(builder.system_prompt, query=str(arguments["task"]))
        capability = "built-in workspace file edits" if context.metadata["delegation_allow_writes"] else "read-only investigation"
        system += ("\n\n# Delegated role\nYour maximum capability is " + capability
                   + ". Use only the available tools. You may delegate within shared operator limits. "
                   "Return a concise handoff with findings, relevant paths and line references, unresolved questions, and completion status.")
        messages = [Message(role=Role.USER, content=str(arguments["task"]))]
        repeated: set[str] = set()
        usage = Usage()  # This agent's provider usage only. The root sink owns all billing.
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
            await emit("delegate.usage", provider=route.name, model=route.model, usage=increment.to_dict())

        final = ""
        sources: list[dict[str, Any]] = []
        overflow_retries = 0
        max_turns = min(int(arguments["max_turns"]), state.max_turns)
        turn_index = 0
        while turn_index < max_turns:
            synthesis_turn = turn_index == max_turns - 1
            request = ProviderRequest(
                model=route.model,
                system=system + ("\n\nThis is your final turn. Do not request tools. Return the concise, evidence-backed conclusion now." if synthesis_turn else ""),
                messages=messages, tools=[] if synthesis_turn else schemas,
                max_output_tokens=min(context.config.agent.max_output_tokens, 8000),
                reasoning_effort=context.config.agent.reasoning_effort or None,
                parallel_tool_calls=not synthesis_turn,
                metadata={"parent_session_id": context.session_id, "delegated": True,
                          "provider_route": route.name, "delegation_id": context.metadata["delegation_id"],
                          "delegation_depth": context.metadata["delegation_depth"]},
            )
            request = prepare_route_request(request, route.provider, context.config.agent, overflow_retry_count=overflow_retries)
            await emit("delegate.state", state="waiting")
            async with state.provider_slots:
                state.active_requests += 1
                try:
                    reservation = budget.reserve_cost(route.provider, request) if budget else 0.0
                    cancelled_usage = Usage()
                    usage_recorded = False
                    try:
                        async def send_request(request: ProviderRequest = request) -> ModelResponse:
                            if callable(before_model_request):
                                before_model_request()
                            await emit("delegate.state", state="provider")
                            return await provider.complete(request)

                        response: ModelResponse = await provider.with_retries(
                            send_request,
                            on_usage=observe_usage,
                            before_recovery=before_model_request if callable(before_model_request) else None,
                            request=request,
                            check_usage=(lambda _, observed=cancelled_usage: budget.check_unsettled(observed)) if budget else None,
                        )
                        await record_usage(response.usage)
                    except ProviderError as error:
                        if error.usage is not None:
                            await record_usage(error.usage)
                        if isinstance(error, ProviderContextOverflowError) and overflow_retries < context.config.agent.compaction_max_overflow_retries:
                            overflow_retries += 1
                            continue
                        raise
                    except BaseException:
                        if not usage_recorded and not cancelled_usage.is_empty:
                            with contextlib.suppress(BudgetExceeded):
                                await record_usage(cancelled_usage)
                        raise
                    finally:
                        if budget:
                            budget.release_cost(reservation)
                            budget.release_usage(cancelled_usage)
                        reservation = 0.0
                finally:
                    state.active_requests -= 1
            # Never retain a provider slot while waiting for descendant tools.
            overflow_retries = 0
            turn_index += 1
            assistant = Message(role=Role.ASSISTANT, content=response.text,
                                tool_calls=[] if response.incomplete else response.tool_calls)
            if response.continuation_state is not None and not response.incomplete:
                continuation = response.continuation_state.to_metadata(provider=route.name, model=route.model)
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
                break
            calls: list[ToolCall] = []
            originals: list[ToolCall] = []
            for call in response.tool_calls:
                signature = hashlib.sha256((call.name + json_dumps(call.arguments)).encode()).hexdigest()
                if signature in repeated:
                    messages.append(Message(role=Role.TOOL, content="Identical delegated tool call suppressed", tool_call_id=call.id, tool_name=call.name, is_error=True))
                    continue
                repeated.add(signature)
                # Namespace model-supplied IDs so independent conversations cannot collide in storage.
                originals.append(call)
                calls.append(replace(call, id=context.metadata["delegation_id"] + "." + call.id))
            await emit("delegate.state", state="tool")

            async def execute(call: ToolCall) -> ToolResult:
                nested_context = replace(context, tool_call_id=call.id)
                if callable(execute_nested):
                    return await execute_nested(call, nested_context)
                result = await registry.execute(call, nested_context)
                assert isinstance(result, ToolResult)
                return result

            results = await bounded_calls(calls, context.config.agent.parallel_reads, execute)
            for original, call, result in zip(originals, calls, results, strict=True):
                sources.append({"tool_call_id": call.id, **{key: result.metadata[key] for key in
                                ("path", "sha256", "output_artifact", "delegation_id") if key in result.metadata}})
                messages.append(Message(role=Role.TOOL, content=result.output, tool_call_id=original.id,
                                        tool_name=call.name, is_error=result.is_error, metadata=result.metadata))
            # Assistant call IDs stay provider-native for continuation replay.
        metadata = {"usage": usage.to_dict(), "turns": turn_index, "evidence": sources}
        if not final:
            return ToolResult("Delegated task ended without the required textual conclusion. Incomplete; evidence references: " + json_dumps(sources), is_error=True, metadata=metadata)
        return ToolResult("Delegation complete. Findings and source references:\n" + final
                          + ("\nEvidence references (search_history/read_artifact): " + json_dumps(sources) if sources else ""), metadata=metadata)
