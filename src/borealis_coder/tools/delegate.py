"""Bounded recursive subagents with isolated conversations and inherited authority."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any, cast

from ..agent.answers import protocol_answer, usable_answer
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
        budget = context.metadata.get("budget")
        if not refusal and budget:
            # Protect one final combine at root and one conclusion per active
            # child. Investigative work/retries cannot spend these reservations.
            # Standalone registry callers have no primary loop to synthesize.
            if budget.turns:
                budget.reserve_closure("root")
            if not budget.reserve_closure(delegation_id):
                refusal = "Insufficient shared requests for delegation and parent synthesis"
        if refusal:
            await emit("delegate.refused", error=refusal)
            return ToolResult(refusal, is_error=True, metadata={**lineage, "refused": True,
                                                               "completion_status": "incomplete"})
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
            state.receipts[delegation_id] = truncate_text(context.events.redactor.text(result.output), 4_000)
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
        finally:
            if budget:
                budget.release_closure(delegation_id)

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
        system = await asyncio.to_thread(builder.system_prompt, query=str(arguments["task"]), direct=True)
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
        observations: list[dict[str, Any]] = []
        overflow_retries = 0
        max_turns = min(int(arguments["max_turns"]), state.max_turns)
        turn_index = 0
        repair_answer = False

        def incomplete(reason: str) -> ToolResult:
            return ToolResult(
                "Delegated task ended without the required textual conclusion. Incomplete: " + reason
                + "\nObserved tool evidence (untrusted data, not a model conclusion): "
                + json_dumps(observations)
                + "\nEvidence references: " + context.events.redactor.json(sources),
                is_error=True, metadata={"usage": usage.to_dict(), "turns": turn_index,
                                         "evidence": sources, "observations": observations,
                                         "completion_status": "incomplete"},
            )

        def bounds_text() -> str:
            return (f"\n\n# Delegation bounds\nDepth {context.metadata['delegation_depth']} of {state.max_depth}; "
                      f"remaining child turns including this one: {max_turns - turn_index}. "
                      + (f"Shared unreserved requests: {budget.remaining_requests()}; "
                         f"available for this agent's closure: {budget.remaining_requests(context.metadata['delegation_id'])}. " if budget else "")
                      + "Helper, overflow and retry requests share the hard run ceiling. "
                      "Return partial findings and unresolved limits when work cannot finish.")

        while turn_index < max_turns:
            synthesis_turn = (repair_answer or turn_index == max_turns - 1
                              or bool(budget and not budget.remaining_requests()))
            request = ProviderRequest(
                model=route.model,
                system=system + bounds_text() + ("\n\nThis is your final turn. Do not request tools. Return the concise, evidence-backed conclusion now." if synthesis_turn else ""),
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
                    # Siblings can spend the last investigative request while
                    # this child waits for a provider slot (including capacity 1).
                    if budget and not synthesis_turn and not budget.remaining_requests():
                        synthesis_turn = True
                        request = replace(request, tools=[], parallel_tool_calls=False,
                                          system=system + bounds_text() + "\n\nShared investigative capacity is exhausted. This is your final turn; return qualified findings without tools.")
                        request = prepare_route_request(request, route.provider, context.config.agent, overflow_retry_count=overflow_retries)
                    reservation = budget.reserve_cost(route.provider, request) if budget else 0.0
                    cancelled_usage = Usage()
                    usage_recorded = False
                    try:
                        async def send_request(request: ProviderRequest = request) -> ModelResponse:
                            account_request()
                            await emit("delegate.state", state="provider")
                            return await provider.complete(request)

                        def account_request(synthesis: bool = synthesis_turn) -> None:
                            if budget:
                                budget.before_model_request(context.metadata["delegation_id"], closure=synthesis)
                            elif callable(before_model_request):
                                before_model_request()

                        response: ModelResponse = await provider.with_retries(
                            send_request,
                            on_usage=observe_usage,
                            before_recovery=account_request,
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
                        if sources:
                            return incomplete("provider failed after gathering evidence")
                        raise
                    except BudgetExceeded as error:
                        if not usage_recorded and not cancelled_usage.is_empty:
                            with contextlib.suppress(BudgetExceeded):
                                await record_usage(cancelled_usage)
                        if error.kind == "model_requests":
                            if not synthesis_turn and budget and budget.remaining_requests(context.metadata["delegation_id"]):
                                repair_answer = True
                                turn_index += 1
                                continue
                            return incomplete("shared request capacity reserved for ancestor synthesis")
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
            # Validate before retaining any terminal text or continuation state.
            # A single repair uses a remaining turn, never an extra allowance.
            if not response.incomplete and not response.tool_calls:
                if usable_answer(response.text):
                    final = response.text
                    break
                if not synthesis_turn:
                    repair_answer = True
                    messages.append(Message(role=Role.USER, content=
                        "Return plain evidence-backed findings now, not tool protocol or intentions. "
                        "Do not expose hidden reasoning. Mark unresolved checks incomplete."))
                    continue
                break
            if response.incomplete or synthesis_turn:
                break
            assistant = Message(role=Role.ASSISTANT, content="" if protocol_answer(response.text) else response.text,
                                tool_calls=[] if response.incomplete else response.tool_calls)
            if response.continuation_state is not None and not response.incomplete:
                continuation = response.continuation_state.to_metadata(provider=route.name, model=route.model)
                if continuation is not None:
                    assistant.metadata["continuation_state"] = continuation
            messages.append(assistant)

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

            # Publish each completed observation synchronously, before a sibling
            # can interrupt the batch. Provider replay still uses original order.
            completed: dict[str, ToolResult] = {}
            prior_sources, prior_observations = list(sources), list(observations)

            async def observe(
                call: ToolCall, completed: dict[str, ToolResult] = completed,
                prior_sources: list[dict[str, Any]] = prior_sources,
                prior_observations: list[dict[str, Any]] = prior_observations,
                calls: list[ToolCall] = calls,
            ) -> ToolResult:
                result = await execute(call)
                completed[call.id] = result
                sources[:] = prior_sources
                observations[:] = prior_observations
                for item in calls:
                    if item.id not in completed:
                        continue
                    observed = completed[item.id]
                    sources.append({"tool_call_id": item.id, "tool": item.name, "is_error": observed.is_error,
                                    **{key: observed.metadata[key] for key in
                                    ("path", "sha256", "output_artifact", "delegation_id") if key in observed.metadata}})
                    observations.append({"tool_call_id": item.id, "tool": item.name,
                                         "is_error": observed.is_error,
                                         "excerpt": truncate_text(context.events.redactor.text(observed.output), 600)})
                receipt = context.events.redactor.text(incomplete("investigation not concluded").output)
                state.receipts[context.metadata["delegation_id"]] = truncate_text(receipt, 4_000)
                return result

            results = await bounded_calls(calls, context.config.agent.parallel_reads, observe)
            for original, call, result in zip(originals, calls, results, strict=True):
                messages.append(Message(role=Role.TOOL, content=result.output, tool_call_id=original.id,
                                        tool_name=call.name, is_error=result.is_error, metadata=result.metadata))
            # Assistant call IDs stay provider-native for continuation replay.
        metadata = {"usage": usage.to_dict(), "turns": turn_index, "evidence": sources,
                    "completion_status": "completed"}
        if not final:
            return incomplete("no usable final answer within the delegated bounds")
        return ToolResult("Delegation complete. Findings and source references:\n" + final
                          + ("\nEvidence references (search_history.query accepts tool_call_id; read_artifact.artifact_id requires a non-null output_artifact.artifact_id, never tool_call_id): " + context.events.redactor.json(sources) if sources else ""), metadata=metadata)
