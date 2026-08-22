"""Resumable async coding-agent loop with parallel reads and durable state."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import Config
from ..context import ContextBuilder
from ..errors import (
    BudgetExceeded,
    Cancelled,
    ProviderContextOverflowError,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    SessionError,
)
from ..events import EventBus
from ..models import (
    AgentResult,
    Message,
    ModelResponse,
    ProviderRequest,
    Role,
    StopReason,
    ToolCall,
    Usage,
)
from ..providers.base import Provider
from ..safety import ApprovalManager
from ..sessions import SessionStore
from ..tools import ToolContext, ToolRegistry, VerificationPlanner
from ..util import json_dumps, new_id, truncate_text
from .budget import Budget, estimate_request_tokens
from .compaction import Summarizer, compact_messages_with_summary

if TYPE_CHECKING:
    from ..mcp import MCPManager


@dataclass(slots=True)
class ProviderRoute:
    name: str
    model: str
    provider: Provider


class AgentRunner:
    def __init__(
        self,
        *,
        workspace: Path,
        config: Config,
        providers: list[ProviderRoute],
        tools: ToolRegistry,
        tool_context: ToolContext,
        context_builder: ContextBuilder,
        sessions: SessionStore,
        events: EventBus,
    ) -> None:
        if not providers:
            raise ValueError("At least one provider route is required")
        self.workspace = workspace.resolve()
        self.config = config
        self.providers = providers
        self.tools = tools
        self.tool_context = tool_context
        self.context_builder = context_builder
        self.sessions = sessions
        self.events = events
        self._cancel: dict[str, asyncio.Event] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._steering: dict[
            str,
            asyncio.Queue[tuple[str, str | None, dict[str, Any]]],
        ] = {}
        self._approval_managers: dict[str, ApprovalManager] = {}
        self.mcp_manager: MCPManager | None = None


    async def close(self) -> None:
        if self.mcp_manager is not None:
            await self.mcp_manager.close()
        await asyncio.to_thread(self.sessions.close)

    def cancel(self, session_id: str) -> bool:
        event = self._cancel.get(session_id)
        if event is None:
            return False
        event.set()
        return True

    def is_busy(self, session_id: str) -> bool:
        lock = self._locks.get(session_id)
        return bool(lock and lock.locked())

    def steer(
        self,
        session_id: str,
        prompt: str,
        *,
        message_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Steering prompt cannot be empty")
        if not self.is_busy(session_id):
            raise SessionError(f"Session {session_id} has no active run to steer")
        self._steering.setdefault(session_id, asyncio.Queue()).put_nowait(
            (prompt, message_id, dict(metadata or {}))
        )

    def queued_prompts(self, session_id: str) -> int:
        queue = self._steering.get(session_id)
        return queue.qsize() if queue else 0

    async def run(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        user_message_id: str | None = None,
        user_metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        route = self.providers[0]
        if session_id is None:
            session = await asyncio.to_thread(
                self.sessions.create_session,
                workspace=self.workspace,
                provider=route.name,
                model=route.model,
                title=_title(prompt),
                metadata={"safety_mode": self.config.safety.mode},
            )
            session_id = session.id
        else:
            session = await asyncio.to_thread(self.sessions.get_session, session_id)
            if Path(session.workspace).resolve() != self.workspace:
                raise SessionError(f"Session workspace is {session.workspace}, not {self.workspace}")
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        if lock.locked():
            raise SessionError(f"Session {session_id} is already running")
        async with lock:
            run_id = new_id("run")
            deadline_expired = asyncio.Event()
            worker = asyncio.create_task(
                self._run_locked(
                    prompt,
                    session_id,
                    run_id=run_id,
                    deadline_expired=deadline_expired,
                    user_message_id=user_message_id,
                    user_metadata=user_metadata,
                )
            )
            try:
                return await asyncio.wait_for(
                    asyncio.shield(worker),
                    timeout=self.config.agent.max_time_seconds,
                )
            except TimeoutError:
                deadline_expired.set()
            except asyncio.CancelledError:
                pass
            self.cancel(session_id)
            worker.cancel()
            try:
                return await worker
            except asyncio.CancelledError:
                await asyncio.to_thread(self.sessions.update_session, session_id, status="idle")
                stop_reason = (
                    StopReason.BUDGET if deadline_expired.is_set() else StopReason.CANCELLED
                )
                error = (
                    f"Maximum {self.config.agent.max_time_seconds}s run time reached"
                    if deadline_expired.is_set()
                    else "Run cancelled"
                )
                return AgentResult(
                    session_id=session_id,
                    run_id=run_id,
                    text="",
                    stop_reason=stop_reason,
                    usage=Usage(),
                    turns=0,
                    error=error,
                )

    async def _run_locked(
        self,
        prompt: str,
        session_id: str,
        *,
        run_id: str,
        deadline_expired: asyncio.Event,
        user_message_id: str | None,
        user_metadata: dict[str, Any] | None,
    ) -> AgentResult:
        cancel = asyncio.Event()
        self._cancel[session_id] = cancel
        approvals = self._approval_managers.get(session_id)
        if approvals is None:
            approvals = ApprovalManager(
                self.tool_context.approvals.callback,
                cache=self.tool_context.approvals.cache,
            )
            self._approval_managers[session_id] = approvals
        else:
            approvals.callback = self.tool_context.approvals.callback
        context = ToolContext(
            workspace=self.tool_context.workspace,
            roots=self.tool_context.roots,
            config=self.tool_context.config,
            events=self.tool_context.events,
            policy=self.tool_context.policy,
            approvals=approvals,
            process=self.tool_context.process,
            checkpoints=self.tool_context.checkpoints,
            session_id=session_id,
            run_id=run_id,
            changed_files=set(),
            metadata=dict(self.tool_context.metadata),
        )
        context.metadata["context_builder"] = self.context_builder
        context.metadata["provider_routes"] = self.providers
        context.metadata["tool_registry"] = self.tools
        budget = Budget.start(self.config.agent)

        async def usage_sink(usage: Usage) -> None:
            await asyncio.to_thread(self.sessions.add_usage, session_id, usage)
            budget.add_usage(usage)

        context.metadata["usage_sink"] = usage_sink
        messages = await asyncio.to_thread(self.sessions.messages, session_id)
        user = Message(
            id=user_message_id or new_id("msg"),
            role=Role.USER,
            content=prompt,
            metadata=dict(user_metadata or {}),
        )
        messages.append(user)
        await asyncio.to_thread(self.sessions.append_message, session_id, user)
        await asyncio.to_thread(self.sessions.update_session, session_id, status="running")
        await self.events.emit("run.started", session_id=session_id, run_id=run_id, prompt=prompt)
        last_batch_signature: str | None = None
        repeated_batch_count = 0
        final_text = ""
        stop_reason = StopReason.END_TURN
        error_message: str | None = None
        verification: dict[str, Any] | None = None
        compacted = False
        try:
            prompt_context = await asyncio.to_thread(self.context_builder.build, query=prompt)
            system = prompt_context.text
            session_usage = await asyncio.to_thread(self.sessions.usage, session_id)
            adaptive_cache = self._low_cache_effectiveness(session_usage)
            conversation_cache = (
                self.config.cache.prompt_cache_enabled
                and self.config.cache.conversation_cache_enabled
                and not adaptive_cache
            )
            if adaptive_cache:
                await self.events.emit(
                    "cache.adaptive",
                    session_id=session_id,
                    run_id=run_id,
                    hit_rate=session_usage.provider_cache_hit_rate,
                    action="stable-prefix-only; shorter compaction window",
                )
            while True:
                self._check_cancel(cancel)
                budget.before_turn()
                schemas = self.tools.schemas()
                estimated = estimate_request_tokens(system, messages, schemas)
                threshold = int(self.config.agent.max_input_tokens * self.config.agent.compact_at_ratio)
                if estimated >= threshold:
                    compacted_messages = await compact_messages_with_summary(
                        messages,
                        self._summarizer(usage_sink, cancel),
                        keep_recent=12 if adaptive_cache else 18,
                    )
                    if compacted_messages == messages:
                        if estimated > self.config.agent.max_input_tokens:
                            raise BudgetExceeded(
                                "context",
                                f"Estimated request size {estimated} exceeds context budget",
                            )
                    else:
                        messages = compacted_messages
                        compacted = True
                        await self.events.emit(
                            "context.compacted",
                            session_id=session_id,
                            run_id=run_id,
                            estimated_tokens=estimated,
                            messages=len(messages),
                        )
                        estimated = estimate_request_tokens(system, messages, schemas)
                if estimated > self.config.agent.max_input_tokens:
                    raise BudgetExceeded(
                        "context",
                        f"Estimated request size {estimated} exceeds "
                        f"{self.config.agent.max_input_tokens} token context budget",
                    )
                request = ProviderRequest(
                    model=self.providers[0].model,
                    system=system,
                    messages=messages,
                    tools=schemas,
                    max_output_tokens=self.config.agent.max_output_tokens,
                    reasoning_effort=self.config.agent.reasoning_effort or None,
                    parallel_tool_calls=True,
                    metadata={
                        "session_id": session_id,
                        "run_id": run_id,
                        "prompt_cache_key": prompt_context.stable_fingerprint,
                        "prompt_cache_enabled": self.config.cache.prompt_cache_enabled,
                        "prompt_cache_ttl": self.config.cache.anthropic_ttl,
                        "anthropic_conversation_cache": conversation_cache,
                        "system_blocks": [
                            {"text": prompt_context.stable, "cacheable": True},
                            {"text": prompt_context.dynamic, "cacheable": False},
                        ],
                    },
                )
                await self.events.emit(
                    "model.started",
                    session_id=session_id,
                    run_id=run_id,
                    turn=budget.turns,
                    estimated_input_tokens=estimated,
                    provider=self.providers[0].name,
                    model=self.providers[0].model,
                )
                assistant_message_id = new_id("msg")
                try:
                    response, used_route = await self._complete_with_fallback(
                        request,
                        session_id,
                        run_id,
                        cancel,
                        assistant_message_id,
                    )
                except ProviderContextOverflowError:
                    if compacted:
                        raise
                    messages = await compact_messages_with_summary(
                        messages, self._summarizer(usage_sink, cancel), keep_recent=12
                    )
                    compacted = True
                    await self.events.emit("context.compacted", session_id=session_id, run_id=run_id, provider_overflow=True, messages=len(messages))
                    continue
                await asyncio.to_thread(self.sessions.add_usage, session_id, response.usage)
                usage_budget_error: BudgetExceeded | None = None
                try:
                    budget.add_usage(response.usage)
                except BudgetExceeded as error:
                    usage_budget_error = error
                cumulative_usage = await asyncio.to_thread(self.sessions.usage, session_id)
                if not adaptive_cache and self._low_cache_effectiveness(cumulative_usage):
                    adaptive_cache = True
                    conversation_cache = False
                    await self.events.emit(
                        "cache.adaptive",
                        session_id=session_id,
                        run_id=run_id,
                        hit_rate=cumulative_usage.provider_cache_hit_rate,
                        action="stable-prefix-only; shorter compaction window",
                    )
                await asyncio.to_thread(self.sessions.update_session, session_id, provider=used_route.name, model=used_route.model)
                assistant_metadata = {
                    "model": response.model or used_route.model,
                    "response_id": response.response_id,
                }
                if isinstance(response.raw, dict):
                    responses_state = response.raw.get("responses_state")
                    if isinstance(responses_state, list) and responses_state:
                        # Keep only the encrypted continuation state selected by the
                        # provider. Full raw responses are never persisted here.
                        assistant_metadata["responses_state"] = responses_state
                assistant = Message(
                    id=assistant_message_id,
                    role=Role.ASSISTANT,
                    content=response.text,
                    tool_calls=response.tool_calls,
                    metadata=assistant_metadata,
                )
                messages.append(assistant)
                await asyncio.to_thread(self.sessions.append_message, session_id, assistant)
                await self.events.emit(
                    "model.completed", session_id=session_id, run_id=run_id, turn=budget.turns,
                    message_id=assistant.id,
                    text=response.text, tool_calls=[call.to_dict() for call in response.tool_calls],
                    usage=response.usage.to_dict(), stop_reason=response.stop_reason,
                )
                if response.text:
                    final_text = response.text
                if usage_budget_error is not None:
                    raise usage_budget_error
                if not response.tool_calls:
                    if await self._drain_steering(session_id, run_id, messages):
                        continue
                    break
                batch_signature = hashlib.sha256(
                    json_dumps(
                        [
                            {"name": call.name, "arguments": call.arguments}
                            for call in response.tool_calls
                        ]
                    ).encode()
                ).hexdigest()
                if batch_signature == last_batch_signature:
                    repeated_batch_count += 1
                else:
                    last_batch_signature = batch_signature
                    repeated_batch_count = 1
                if repeated_batch_count > self.config.agent.max_repeated_calls:
                    stop_reason = StopReason.STUCK
                    raise BudgetExceeded(
                        "stuck",
                        f"Repeated identical tool-call batch {repeated_batch_count} times",
                    )
                results = await self._execute_calls(response.tool_calls, cancel, context)
                for call, result in zip(response.tool_calls, results, strict=True):
                    tool_message = Message(
                        role=Role.TOOL, content=result.output, tool_call_id=call.id,
                        tool_name=call.name, is_error=result.is_error, metadata=result.metadata,
                    )
                    messages.append(tool_message)
                    await asyncio.to_thread(self.sessions.append_message, session_id, tool_message)
                await self._drain_steering(session_id, run_id, messages)
            if context.changed_files and self.config.agent.auto_verify:
                planner = VerificationPlanner(self.workspace)
                await self.events.emit(
                    "verification.started",
                    session_id=session_id,
                    run_id=run_id,
                )
                report = await planner.run(context)
                verification = report.to_dict()
                await self.events.emit("verification.completed", session_id=session_id, run_id=run_id, **verification)
        except Cancelled as error:
            stop_reason = StopReason.CANCELLED
            error_message = str(error)
        except BudgetExceeded as error:
            if error.kind == "stuck":
                stop_reason = StopReason.STUCK
            elif error.kind == "turns":
                stop_reason = StopReason.MAX_TURNS
            else:
                stop_reason = StopReason.BUDGET
            error_message = str(error)
        except asyncio.CancelledError:
            if deadline_expired.is_set():
                stop_reason = StopReason.BUDGET
                error_message = (
                    f"Maximum {self.config.agent.max_time_seconds}s run time reached"
                )
            else:
                stop_reason = StopReason.CANCELLED
                error_message = "Run cancelled"
        except Exception as error:
            stop_reason = StopReason.ERROR
            error_message = f"{type(error).__name__}: {error}"
            await self.events.emit("run.error", session_id=session_id, run_id=run_id, error=error_message)
        finally:
            self._cancel.pop(session_id, None)
            queue = self._steering.get(session_id)
            if queue is not None and queue.empty():
                self._steering.pop(session_id, None)
            await asyncio.to_thread(self.sessions.update_session, session_id, status="idle")
        usage = budget.usage or Usage()
        result = AgentResult(
            session_id=session_id, run_id=run_id, text=final_text,
            stop_reason=stop_reason, usage=usage, turns=budget.turns,
            changed_files=sorted(context.changed_files), verification=verification,
            error=error_message,
        )
        await self.events.emit("run.completed", session_id=session_id, run_id=run_id, result=result.to_dict())
        return result

    async def _drain_steering(
        self, session_id: str, run_id: str, messages: list[Message]
    ) -> bool:
        queue = self._steering.get(session_id)
        if queue is None:
            return False
        added = False
        while not queue.empty():
            prompt, message_id, metadata = queue.get_nowait()
            message = Message(
                id=message_id or new_id("msg"),
                role=Role.USER,
                content=prompt,
                metadata={"steering": True, **metadata},
            )
            messages.append(message)
            await asyncio.to_thread(self.sessions.append_message, session_id, message)
            await self.events.emit(
                "user.steered", session_id=session_id, run_id=run_id,
                message_id=message.id, prompt=prompt,
            )
            added = True
        return added

    async def _complete_with_fallback(
        self,
        request: ProviderRequest,
        session_id: str,
        run_id: str,
        cancel: asyncio.Event,
        assistant_message_id: str,
    ) -> tuple[ModelResponse, ProviderRoute]:
        errors: list[str] = []
        cache_misses = 0
        for index, route in enumerate(self.providers):
            self._check_cancel(cancel)
            routed = ProviderRequest(
                model=route.model, system=request.system, messages=request.messages,
                tools=request.tools, max_output_tokens=request.max_output_tokens,
                temperature=request.temperature, reasoning_effort=request.reasoning_effort,
                parallel_tool_calls=request.parallel_tool_calls,
                response_schema=request.response_schema, metadata=request.metadata,
            )
            try:
                cache_key = self._response_cache_key(route, routed)
                cached = None
                if self.config.cache.response_cache_enabled:
                    cached = await asyncio.to_thread(
                        self.sessions.get_cached_response,
                        cache_key,
                    )
                if cached is not None:
                    original_usage = Usage.from_dict(cached.get("usage"))
                    payload = cached.get("response") or {}
                    usage = Usage(
                        application_cache_hits=1,
                        application_cache_misses=cache_misses,
                        application_cache_saved_tokens=original_usage.total_tokens,
                        application_cache_saved_cost_usd=original_usage.cost_usd,
                    )
                    response = ModelResponse(
                        text=str(payload.get("text") or ""),
                        usage=usage,
                        stop_reason=payload.get("stop_reason"),
                        model=str(payload.get("model") or route.model),
                        raw={"application_cache": True},
                    )
                    await self.events.emit(
                        "model.cache_hit",
                        session_id=session_id,
                        run_id=run_id,
                        provider=route.name,
                        model=route.model,
                        saved_tokens=usage.application_cache_saved_tokens,
                        saved_cost_usd=usage.application_cache_saved_cost_usd,
                    )
                    if response.text:
                        await self.events.emit(
                            "model.text_delta",
                            session_id=session_id,
                            run_id=run_id,
                            message_id=assistant_message_id,
                            text=response.text,
                            provider=route.name,
                            model=route.model,
                        )
                    return response, route
                if self.config.cache.response_cache_enabled:
                    cache_misses += 1
                    await self.events.emit(
                        "model.cache_miss",
                        session_id=session_id,
                        run_id=run_id,
                        provider=route.name,
                        model=route.model,
                    )
                response = await self._stream_route(
                    route,
                    routed,
                    session_id,
                    run_id,
                    cancel,
                    assistant_message_id,
                )
                if self.config.cache.response_cache_enabled:
                    response.usage.application_cache_misses += cache_misses
                    if not response.tool_calls and response.text and response.stop_reason not in {
                        "error",
                        "cancelled",
                    }:
                        await asyncio.to_thread(
                            self.sessions.put_cached_response,
                            cache_key,
                            provider=route.name,
                            model=route.model,
                            response={
                                "text": response.text,
                                "stop_reason": response.stop_reason,
                                "model": response.model or route.model,
                            },
                            usage=response.usage,
                            ttl_seconds=self.config.cache.response_cache_ttl_seconds,
                            max_entries=self.config.cache.response_cache_max_entries,
                        )
                if index:
                    await self.events.emit("model.fallback_succeeded", session_id=session_id, run_id=run_id, provider=route.name, model=route.model)
                return response, route
            except (ProviderUnavailableError, ProviderRateLimitError) as error:
                errors.append(f"{route.name}/{route.model}: {error}")
                await self.events.emit("model.route_failed", session_id=session_id, run_id=run_id, provider=route.name, model=route.model, error=str(error), retryable=True)
                continue
            except ProviderError:
                raise
        raise ProviderUnavailableError("All provider routes failed: " + "; ".join(errors), retryable=False)

    def _response_cache_key(
        self,
        route: ProviderRoute,
        request: ProviderRequest,
    ) -> str:
        messages = [
            {
                "role": item.role.value,
                "content": item.content,
                "tool_calls": [call.to_dict() for call in item.tool_calls],
                "tool_call_id": item.tool_call_id,
                "tool_name": item.tool_name,
                "is_error": item.is_error,
                "metadata": item.metadata,
            }
            for item in request.messages
        ]
        provider_config = asdict(route.provider.config)
        provider_config.pop("auth_file", None)
        provider_config.pop("codex_home", None)
        value = {
            "version": 1,
            "provider": route.name,
            "model": route.model,
            "provider_config": provider_config,
            "request": {
                "system": request.system,
                "messages": messages,
                "tools": request.tools,
                "max_output_tokens": request.max_output_tokens,
                "temperature": request.temperature,
                "reasoning_effort": request.reasoning_effort,
                "parallel_tool_calls": request.parallel_tool_calls,
                "response_schema": request.response_schema,
            },
        }
        return hashlib.sha256(json_dumps(value).encode("utf-8")).hexdigest()

    def _summarizer(
        self,
        usage_sink: Callable[[Usage], Awaitable[None]],
        cancel: asyncio.Event,
    ) -> Summarizer | None:
        """Build an LLM-backed compaction summarizer from the primary route.

        Uses the small model when configured (cheap summarization), falling back
        to the primary model. Returns None when compaction should stay purely
        deterministic (as configured, or when no provider route is available).
        """
        if self.config.agent.deterministic_compaction or not self.providers:
            return None
        route = self.providers[0]

        async def summarize(transcript: str) -> str:
            request = ProviderRequest(
                model=self.config.agent.small_model or route.model,
                system=(
                    "You summarize coding-agent conversations. Output only the "
                    "summary: factual, dense, and complete with respect to tool "
                    "outputs such as test failures and stack traces."
                ),
                messages=[Message(role=Role.USER, content=transcript)],
                max_output_tokens=min(4_000, self.config.agent.max_output_tokens),
                metadata={"purpose": "compaction_summary"},
            )
            request_task = asyncio.create_task(route.provider.complete(request))
            cancel_task = asyncio.create_task(cancel.wait())
            try:
                done, _ = await asyncio.wait(
                    {request_task, cancel_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if cancel_task in done and cancel.is_set():
                    request_task.cancel()
                    raise Cancelled("Run cancelled")
                response = await request_task
            finally:
                cancel_task.cancel()
                if not request_task.done():
                    request_task.cancel()
                await asyncio.gather(
                    request_task,
                    cancel_task,
                    return_exceptions=True,
                )
            await usage_sink(response.usage)
            return response.text

        return summarize

    def _low_cache_effectiveness(self, usage: Usage) -> bool:
        return bool(
            self.config.cache.adaptive
            and usage.requests >= self.config.cache.adaptive_min_requests
            and usage.input_tokens >= self.config.cache.adaptive_min_input_tokens
            and usage.provider_cache_hit_rate < self.config.cache.low_hit_rate_threshold
            and usage.cache_write_tokens > usage.cached_input_tokens
        )

    async def _stream_route(
        self, route: ProviderRoute, request: ProviderRequest,
        session_id: str, run_id: str, cancel: asyncio.Event,
        assistant_message_id: str,
    ) -> ModelResponse:
        attempts = max(0, route.provider.config.max_retries) + 1
        delay = max(0.0, route.provider.config.initial_backoff_seconds)
        for attempt in range(attempts):
            emitted = False
            completed: ModelResponse | None = None
            try:
                stream = route.provider.stream(request).__aiter__()
                while True:
                    next_item = asyncio.ensure_future(anext(stream))
                    cancelled = asyncio.create_task(cancel.wait())
                    done, _ = await asyncio.wait(
                        {next_item, cancelled},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if cancelled in done and cancel.is_set():
                        next_item.cancel()
                        with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                            await next_item
                        close = getattr(stream, "aclose", None)
                        if close is not None:
                            with contextlib.suppress(Exception):
                                await close()
                        raise Cancelled("Run cancelled")
                    cancelled.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await cancelled
                    try:
                        item = await next_item
                    except StopAsyncIteration:
                        break
                    self._check_cancel(cancel)
                    if item.type == "text_delta" and item.text:
                        emitted = True
                        await self.events.emit(
                            "model.text_delta", session_id=session_id, run_id=run_id,
                            message_id=assistant_message_id,
                            text=item.text, provider=route.name, model=route.model,
                        )
                    elif item.type == "tool_call_delta":
                        emitted = True
                        await self.events.emit(
                            "model.tool_call_delta", session_id=session_id, run_id=run_id,
                            provider=route.name, model=route.model, **item.data,
                        )
                    elif item.type == "completed" and item.response is not None:
                        completed = item.response
                if completed is None:
                    raise ProviderUnavailableError(
                        f"Provider {route.name} stream ended without a completed response", retryable=True
                    )
                return completed
            except (ProviderUnavailableError, ProviderRateLimitError) as error:
                if emitted or attempt + 1 >= attempts:
                    raise
                retry_delay = min(
                    route.provider.config.max_backoff_seconds,
                    max(0.25, delay),
                )
                await self.events.emit(
                    "model.retrying",
                    session_id=session_id,
                    run_id=run_id,
                    provider=route.name,
                    model=route.model,
                    attempt=attempt + 2,
                    max_attempts=attempts,
                    delay_seconds=retry_delay,
                    error=str(error),
                )
                await asyncio.sleep(retry_delay)
                delay = max(0.25, delay * 2)
        raise ProviderUnavailableError(f"Provider {route.name} exhausted retries", retryable=False)

    async def _execute_calls(
        self,
        calls: list[ToolCall],
        cancel: asyncio.Event,
        context: ToolContext,
    ):
        results: dict[str, Any] = {}
        reads: list[ToolCall] = []
        writes: list[ToolCall] = []
        for call in calls:
            tool = self.tools.get(call.name)
            (reads if tool and tool.concurrent else writes).append(call)
        semaphore = asyncio.Semaphore(max(1, self.config.agent.parallel_reads))

        async def run_read(call: ToolCall):
            async with semaphore:
                self._check_cancel(cancel)
                return await self._execute_one(call, context, cancel)

        if reads:
            read_results = await asyncio.gather(*(run_read(call) for call in reads))
            results.update({call.id: result for call, result in zip(reads, read_results, strict=True)})
        for call in writes:
            self._check_cancel(cancel)
            results[call.id] = await self._execute_one(call, context, cancel)
        return [results[call.id] for call in calls]

    async def _execute_one(
        self,
        call: ToolCall,
        context: ToolContext,
        cancel: asyncio.Event,
    ):
        await asyncio.to_thread(
            self.sessions.start_tool_call,
            context.session_id,
            context.run_id,
            call.id,
            call.name,
            call.arguments,
        )
        tool_task = asyncio.create_task(self.tools.execute(call, context))
        cancel_task = asyncio.create_task(cancel.wait())
        try:
            done, _ = await asyncio.wait(
                {tool_task, cancel_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_task in done and cancel.is_set():
                tool_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await tool_task
                raise Cancelled("Run cancelled")
            cancel_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cancel_task
            result = await tool_task
        except (asyncio.CancelledError, Cancelled):
            tool_task.cancel()
            cancel_task.cancel()
            await asyncio.to_thread(
                self.sessions.cancel_tool_call,
                context.session_id,
                call.id,
            )
            raise
        await asyncio.to_thread(
            self.sessions.complete_tool_call,
            context.session_id,
            call.id,
            output=result.output,
            is_error=result.is_error,
            metadata=result.metadata,
        )
        return result

    @staticmethod
    def _check_cancel(cancel: asyncio.Event) -> None:
        if cancel.is_set():
            raise Cancelled("Run cancelled")


def _title(prompt: str) -> str:
    words = prompt.replace("\n", " ").split()
    value = " ".join(words[:10])
    return truncate_text(value, 80, marker="…") or "New coding session"
