from __future__ import annotations

import asyncio
import json
import os
import stat
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from borealis_coder.agent import build_runner
from borealis_coder.config import ProviderConfig
from borealis_coder.errors import SessionError
from borealis_coder.models import (
    ContinuationState,
    Effect,
    ModelResponse,
    ProviderRequest,
    Role,
    ToolCall,
    ToolResult,
    Usage,
)
from borealis_coder.providers.base import Provider, ProviderStreamEvent
from borealis_coder.providers.http import SSEEvent
from borealis_coder.providers.mock import MockProvider
from borealis_coder.providers.openrouter import OpenRouterProvider
from borealis_coder.providers.registry import ProviderRegistry
from borealis_coder.safety import ProcessResult
from borealis_coder.tools import FunctionTool, object_schema
from borealis_coder.tools.verification import (
    VerificationReport,
    VerificationStep,
)
from tests.helpers import make_config


class SlowProvider(Provider):
    name = "slow"

    async def complete(self, request):
        await asyncio.sleep(10)
        return ModelResponse(text="late", usage=Usage(requests=1))


class CostProvider(Provider):
    name = "cost"

    async def complete(self, request):
        return ModelResponse(
            text="expensive but durable",
            usage=Usage(input_tokens=10, output_tokens=5, requests=1, cost_usd=2.0),
        )


class CountingProvider(Provider):
    name = "counting"

    def __init__(self, config, api_key=""):
        super().__init__(config, api_key)
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        return ModelResponse(
            text="cacheable answer",
            reasoning_summary="Checked cacheability.",
            stop_reason="end_turn",
            response_id=f"response-{self.calls}",
            usage=Usage(input_tokens=20, output_tokens=3, requests=1, cost_usd=0.2),
            continuation_state=ContinuationState(
                kind="counting.state",
                items=[{"type": "opaque", "value": "replay-me"}],
            ),
        )


class AliasContinuationProvider(Provider):
    name = "alias_implementation"

    def __init__(self, config, api_key=""):
        super().__init__(config, api_key)
        self.calls = 0
        self.replayed: list[bool] = []
        self.route_names: list[str] = []

    async def complete(self, request):
        self.calls += 1
        route_name = self._continuation_provider(request)
        self.route_names.append(route_name)
        for message in reversed(request.messages):
            if "continuation_state" not in message.metadata:
                continue
            state = ContinuationState.from_metadata(
                message.metadata["continuation_state"],
                provider=route_name,
                model=request.model,
                kind="alias.state",
            )
            self.replayed.append(state is not None)
            break
        return ModelResponse(
            text=f"answer-{len(request.messages)}",
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=2, requests=1),
            continuation_state=ContinuationState(
                kind="alias.state",
                items=[{"type": "opaque", "value": f"state-{self.calls}"}],
            ),
        )


class IncompleteProvider(Provider):
    name = "incomplete"

    def __init__(self, config, api_key=""):
        super().__init__(config, api_key)
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        return ModelResponse(
            text="partial answer",
            usage=Usage(input_tokens=20, output_tokens=3, requests=1),
        )


class OneToolProvider(Provider):
    name = "one_tool"

    async def complete(self, request):
        return ModelResponse(
            tool_calls=[
                ToolCall(
                    id="same_call",
                    name="read_file",
                    arguments={
                        "path": "a.txt",
                        "start_line": None,
                        "end_line": None,
                        "max_chars": None,
                    },
                )
            ],
            usage=Usage(requests=1),
        )


class FinalTurnProvider(Provider):
    name = "final_turn"

    def __init__(self, config, api_key=""):
        super().__init__(config, api_key)
        self.tool_counts: list[int] = []
        self.systems: list[str] = []
        self.system_blocks: list[list[dict[str, object]]] = []
        self.final_stop_reason: str | None = None

    async def complete(self, request):
        self.tool_counts.append(len(request.tools))
        self.systems.append(request.system)
        self.system_blocks.append(list(request.metadata["system_blocks"]))
        if len(self.tool_counts) == 1:
            return ModelResponse(
                tool_calls=[
                    ToolCall(
                        name="write_file",
                        arguments={
                            "path": "completed.txt",
                            "content": "done",
                            "expected_sha256": None,
                        },
                    )
                ],
                usage=Usage(requests=1),
            )
        return ModelResponse(
            text="Completed cleanly.",
            stop_reason=self.final_stop_reason,
            usage=Usage(requests=1),
        )


class DefiantFinalTurnProvider(Provider):
    name = "defiant_final_turn"

    def __init__(self, config, api_key=""):
        super().__init__(config, api_key)
        self.calls = 0
        self.tool_counts: list[int] = []
        self.pause_final = False
        self.final_cost_usd = 0.0
        self.final_started = asyncio.Event()
        self.release_final = asyncio.Event()

    async def complete(self, request):
        self.calls += 1
        self.tool_counts.append(len(request.tools))
        if self.calls == 1:
            return ModelResponse(
                tool_calls=[
                    ToolCall(
                        name="write_file",
                        arguments={
                            "path": "durable.txt",
                            "content": "durable",
                            "expected_sha256": None,
                        },
                    )
                ],
                usage=Usage(requests=1),
            )
        if self.calls == 2:
            if self.pause_final:
                self.final_started.set()
                await self.release_final.wait()
            return ModelResponse(
                tool_calls=[
                    ToolCall(
                        name="write_file",
                        arguments={
                            "path": "must-not-run.txt",
                            "content": "unexpected",
                            "expected_sha256": None,
                        },
                    )
                ],
                usage=Usage(requests=1, cost_usd=self.final_cost_usd),
                continuation_state=ContinuationState(
                    kind="gemini.interactions.steps",
                    items=[
                        {
                            "type": "function_call",
                            "id": "rejected-call",
                            "name": "write_file",
                            "arguments": {"path": "must-not-run.txt"},
                        }
                    ],
                ),
            )
        return ModelResponse(text="Resumed to completion.", usage=Usage(requests=1))


class ShellMutationFinalTurnProvider(Provider):
    name = "shell_mutation_final_turn"

    def __init__(self, config, api_key=""):
        super().__init__(config, api_key)
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                tool_calls=[ToolCall(name="shell", arguments={"command": "true"})],
                usage=Usage(requests=1),
            )
        return ModelResponse(
            tool_calls=[ToolCall(name="write_file", arguments={})],
            usage=Usage(requests=1),
        )


class LateSteeringProvider(Provider):
    name = "late_steering"

    def __init__(self, config, api_key=""):
        super().__init__(config, api_key)
        self.calls = 0
        self.final_started = asyncio.Event()
        self.release_final = asyncio.Event()

    async def complete(self, request):
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                tool_calls=[
                    ToolCall(
                        name="read_file",
                        arguments={
                            "path": "a.txt",
                            "start_line": None,
                            "end_line": None,
                            "max_chars": None,
                        },
                    )
                ],
                usage=Usage(requests=1),
            )
        if self.calls == 2:
            self.final_started.set()
            await self.release_final.wait()
            return ModelResponse(text="Initial request complete.", usage=Usage(requests=1))
        saw_steering = any(
            message.role == Role.USER and message.content == "late direction"
            for message in request.messages
        )
        return ModelResponse(
            text="Late steering handled." if saw_steering else "Late steering missing.",
            usage=Usage(requests=1),
        )


class ConcurrentProvider(Provider):
    name = "concurrent"

    async def complete(self, request):
        last_user = next(
            message.content for message in reversed(request.messages) if message.role == Role.USER
        )
        if request.messages[-1].role == Role.TOOL:
            return ModelResponse(text=f"done {last_user}", usage=Usage(requests=1))
        return ModelResponse(
            tool_calls=[
                ToolCall(
                    id="same_call",
                    name="write_file",
                    arguments={
                        "path": f"{last_user}.txt",
                        "content": last_user,
                        "expected_sha256": None,
                    },
                )
            ],
            usage=Usage(requests=1),
        )


class BurstProvider(Provider):
    name = "burst"

    async def complete(self, request):
        return ModelResponse(text="abc", stop_reason="end_turn")

    async def stream(self, request):
        yield ProviderStreamEvent(
            type="reasoning_summary_delta",
            text="Checked the stream.",
        )
        for text in "abc":
            yield ProviderStreamEvent(type="text_delta", text=text)
        yield ProviderStreamEvent(
            type="completed",
            response=ModelResponse(
                text="abc",
                reasoning_summary="Checked the stream.",
                stop_reason="end_turn",
            ),
        )


class SplitSecretReasoningProvider(Provider):
    name = "split_secret_reasoning"

    async def complete(self, request):
        return ModelResponse(text="done", stop_reason="end_turn")

    async def stream(self, request):
        yield ProviderStreamEvent(
            type="reasoning_summary_delta",
            text="Checked sk-abc",
        )
        yield ProviderStreamEvent(
            type="reasoning_summary_delta",
            text="defghijklmnop safely.",
        )
        yield ProviderStreamEvent(type="text_delta", text="done")
        yield ProviderStreamEvent(
            type="completed",
            response=ModelResponse(
                text="done",
                reasoning_summary="Checked sk-abcdefghijklmnop safely.",
                stop_reason="end_turn",
            ),
        )


class SummaryOnlyRetryProvider(Provider):
    name = "summary_only_retry"

    def __init__(self, config, api_key=""):
        super().__init__(config, api_key)
        self.calls = 0

    async def complete(self, request):
        return ModelResponse(text="unused")

    async def stream(self, request):
        self.calls += 1
        if self.calls == 1:
            yield ProviderStreamEvent(
                type="reasoning_summary_delta",
                text="Abandoned summary.",
            )
            yield ProviderStreamEvent(
                type="completed",
                response=ModelResponse(
                    reasoning_summary="Abandoned summary.",
                    usage=Usage(input_tokens=3, requests=1),
                ),
            )
            return
        yield ProviderStreamEvent(type="reasoning_summary_delta", text="Checked ")
        yield ProviderStreamEvent(type="reasoning_summary_delta", text="the retry.")
        yield ProviderStreamEvent(type="text_delta", text="answer")
        yield ProviderStreamEvent(
            type="completed",
            response=ModelResponse(
                text="answer",
                reasoning_summary="Checked the retry.",
                stop_reason="end_turn",
                usage=Usage(input_tokens=4, requests=1),
            ),
        )


class UnsafeSummaryProvider(Provider):
    name = "unsafe_summary"

    def __init__(self, config, api_key=""):
        super().__init__(config, api_key)
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        return ModelResponse(
            text="done",
            reasoning_summary="Unsafe sk-",
            stop_reason="end_turn",
            usage=Usage(input_tokens=2, output_tokens=1, requests=1, cost_usd=0.1),
        )


class FakeOpenAIStreamHttp:
    def __init__(self, event_batches):
        self.event_batches = event_batches
        self.calls = []

    async def stream_sse(self, url, **kwargs):
        self.calls.append((url, dict(kwargs)))
        for event in self.event_batches.pop(0):
            yield event

    def close(self):
        return None


class EmptyProvider(Provider):
    name = "empty"

    async def complete(self, request):
        return ModelResponse(stop_reason="stop")


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_event_persistence_failure_cannot_return_a_successful_run(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runner = await build_runner(root, config=make_config(root), interactive=False)

            def fail_persistence(_events):
                raise OSError("disk full")

            runner.events._persist = fail_persistence
            try:
                with self.assertRaisesRegex(SessionError, "disk full"):
                    await runner.run("must be durable")
                session = runner.sessions.list_sessions(workspace=root)[0]
                self.assertEqual(session.status, "idle")
                self.assertEqual(runner.sessions.events(session.id), [])
            finally:
                runner.events._persist = runner.sessions.append_events
                await runner.close()

    async def test_stream_route_uses_constant_tasks_for_all_fragments(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(root, agent={"provider": "burst"})
            config.providers["burst"] = ProviderConfig(
                type="burst",
                model="burst",
                max_retries=0,
            )
            registry = ProviderRegistry()
            registry.register("burst", lambda cfg, key: BurstProvider(cfg, key))
            runner = await build_runner(
                root,
                config=config,
                interactive=False,
                provider_registry=registry,
            )
            original_create_task = asyncio.create_task
            try:
                session = runner.sessions.create_session(
                    workspace=root,
                    provider="burst",
                    model="burst",
                )
                with patch(
                    "borealis_coder.agent.runner.asyncio.create_task",
                    wraps=original_create_task,
                ) as create_task:
                    response = await runner._stream_route(
                        runner.providers[0],
                        ProviderRequest(model="burst", system="", messages=[]),
                        session.id,
                        "run_1",
                        asyncio.Event(),
                        "msg_1",
                    )
                self.assertEqual(response.text, "abc")
                self.assertEqual(response.reasoning_summary, "Checked the stream.")
                await runner.events.flush()
                reasoning = [
                    event.data["text"]
                    for _, event in runner.sessions.events(session.id)
                    if event.type == "model.reasoning_delta"
                ]
                self.assertEqual(reasoning, ["Checked the stream."])
                self.assertEqual(create_task.call_count, 2)
            finally:
                await runner.close()

    async def test_reasoning_stream_redacts_secrets_across_chunks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(root, agent={"provider": "split_secret_reasoning"})
            config.providers["split_secret_reasoning"] = ProviderConfig(
                type="split_secret_reasoning",
                model="split-secret-reasoning",
                max_retries=0,
            )
            registry = ProviderRegistry()
            registry.register(
                "split_secret_reasoning",
                lambda cfg, key: SplitSecretReasoningProvider(cfg, key),
            )
            with patch.dict(
                os.environ,
                {"BOREALIS_TEST_SECRET": "sk-abcdefghijklmnop"},
                clear=False,
            ):
                runner = await build_runner(
                    root,
                    config=config,
                    interactive=False,
                    provider_registry=registry,
                )
            try:
                session = runner.sessions.create_session(
                    workspace=root,
                    provider="split_secret_reasoning",
                    model="split-secret-reasoning",
                )
                await runner._stream_route(
                    runner.providers[0],
                    ProviderRequest(
                        model="split-secret-reasoning",
                        system="",
                        messages=[],
                    ),
                    session.id,
                    "run_1",
                    asyncio.Event(),
                    "msg_1",
                )
                await runner.events.flush()
                reasoning = [
                    str(event.data.get("text") or "")
                    for _, event in runner.sessions.events(session.id)
                    if event.type == "model.reasoning_delta"
                ]
                self.assertEqual("".join(reasoning), "Checked [REDACTED] safely.")
            finally:
                await runner.close()

    async def test_summary_only_response_can_retry_same_route(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(root, agent={"provider": "summary_only_retry"})
            config.providers["summary_only_retry"] = ProviderConfig(
                type="summary_only_retry",
                model="summary-only-retry",
                max_retries=1,
                initial_backoff_seconds=0,
                max_backoff_seconds=0,
            )
            provider: SummaryOnlyRetryProvider | None = None

            def factory(cfg, key):
                nonlocal provider
                provider = SummaryOnlyRetryProvider(cfg, key)
                return provider

            registry = ProviderRegistry()
            registry.register("summary_only_retry", factory)
            runner = await build_runner(
                root,
                config=config,
                interactive=False,
                provider_registry=registry,
            )
            try:
                session = runner.sessions.create_session(
                    workspace=root,
                    provider="summary_only_retry",
                    model="summary-only-retry",
                )
                response = await runner._stream_route(
                    runner.providers[0],
                    ProviderRequest(model="summary-only-retry", system="", messages=[]),
                    session.id,
                    "run_1",
                    asyncio.Event(),
                    "msg_1",
                )
                await runner.events.flush()
                assert provider is not None
                self.assertEqual(provider.calls, 2)
                self.assertEqual(response.text, "answer")
                self.assertEqual(response.usage.input_tokens, 7)
                self.assertEqual(response.usage.requests, 2)
                events = [event for _, event in runner.sessions.events(session.id)]
                self.assertTrue(any(event.type == "model.retrying" for event in events))
                reasoning = [
                    str(event.data.get("text") or "")
                    for event in events
                    if event.type == "model.reasoning_delta"
                ]
                self.assertEqual(reasoning, ["Checked ", "the retry."])
            finally:
                await runner.close()

    async def test_streaming_chat_retry_preserves_usage_from_failed_internal_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(
                root,
                agent={"provider": "openrouter_fake", "reasoning_effort": "high"},
            )
            config.providers["openrouter_fake"] = ProviderConfig(
                type="openrouter_fake",
                model="openrouter-fake",
                base_url="https://openrouter.test/api/v1",
                api_style="chat",
                max_retries=1,
                initial_backoff_seconds=0,
                max_backoff_seconds=0,
            )
            first = [
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "id": "empty",
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "delta": {
                                        "reasoning_details": [
                                            {
                                                "type": "reasoning.summary",
                                                "summary": "Billed summary.",
                                                "index": 0,
                                            }
                                        ]
                                    },
                                }
                            ],
                            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                        }
                    ),
                ),
                SSEEvent("message", "[DONE]"),
            ]
            failed_fallback = [
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "error": {
                                "code": 503,
                                "message": "Fallback provider unavailable.",
                            }
                        }
                    ),
                )
            ]
            recovered = [
                SSEEvent(
                    "message",
                    json.dumps(
                        {
                            "id": "recovered",
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "delta": {"content": "Recovered answer."},
                                }
                            ],
                            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
                        }
                    ),
                ),
                SSEEvent("message", "[DONE]"),
            ]
            fake_http = FakeOpenAIStreamHttp([first, failed_fallback, recovered])

            def factory(cfg, key):
                provider = OpenRouterProvider(cfg, key)
                provider.http = fake_http  # type: ignore[assignment]
                return provider

            registry = ProviderRegistry()
            registry.register("openrouter_fake", factory)
            runner = await build_runner(
                root,
                config=config,
                interactive=False,
                provider_registry=registry,
            )
            try:
                result = await runner.run("recover with paid failed attempt")
                self.assertEqual(result.text, "Recovered answer.")
                self.assertEqual(result.usage.input_tokens, 9)
                self.assertEqual(result.usage.output_tokens, 5)
                self.assertEqual(result.usage.requests, 2)
                self.assertEqual(len(fake_http.calls), 3)
                events = [event for _, event in runner.sessions.events(result.session_id)]
                self.assertTrue(any(event.type == "model.retrying" for event in events))
            finally:
                await runner.close()

    async def test_completed_and_cached_reasoning_summaries_use_stream_redaction(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(root, agent={"provider": "unsafe_summary"})
            config.providers["unsafe_summary"] = ProviderConfig(
                type="unsafe_summary",
                model="unsafe-summary",
                max_retries=0,
            )
            provider: UnsafeSummaryProvider | None = None

            def factory(cfg, key):
                nonlocal provider
                provider = UnsafeSummaryProvider(cfg, key)
                return provider

            registry = ProviderRegistry()
            registry.register("unsafe_summary", factory)
            runner = await build_runner(
                root,
                config=config,
                interactive=False,
                provider_registry=registry,
            )
            try:
                first = await runner.run("same exact unsafe request")
                second = await runner.run("same exact unsafe request")
                assert provider is not None
                self.assertEqual(provider.calls, 1)

                for result in (first, second):
                    events = [event for _, event in runner.sessions.events(result.session_id)]
                    completed = next(event for event in events if event.type == "model.completed")
                    self.assertEqual(
                        completed.data["reasoning_summary"],
                        "Unsafe [REDACTED]",
                    )
                    all_event_text = " ".join(str(event.data) for event in events)
                    self.assertNotIn("sk-", all_event_text)

                cached_events = [
                    event for _, event in runner.sessions.events(second.session_id)
                ]
                cached_reasoning = [
                    str(event.data.get("text") or "")
                    for event in cached_events
                    if event.type == "model.reasoning_delta"
                ]
                self.assertEqual(cached_reasoning, ["Unsafe [REDACTED]"])
            finally:
                await runner.close()

    async def test_exact_response_cache_avoids_duplicate_provider_request(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(root, agent={"provider": "counting"})
            config.providers["counting"] = ProviderConfig(
                type="counting", model="counting", max_retries=0
            )
            provider: CountingProvider | None = None

            def factory(cfg, key):
                nonlocal provider
                provider = CountingProvider(cfg, key)
                return provider

            registry = ProviderRegistry()
            registry.register("counting", factory)
            runner = await build_runner(
                root,
                config=config,
                interactive=False,
                provider_registry=registry,
            )
            try:
                first = await runner.run("same exact request")
                second = await runner.run("same exact request")
                assert provider is not None
                self.assertEqual(provider.calls, 1)
                self.assertEqual(first.usage.application_cache_misses, 1)
                self.assertEqual(second.usage.application_cache_hits, 1)
                self.assertEqual(second.usage.application_cache_saved_tokens, 23)
                self.assertEqual(second.usage.cost_usd, 0.0)
                for result in (first, second):
                    reasoning = "".join(
                        str(event.data.get("text") or "")
                        for _, event in runner.sessions.events(result.session_id)
                        if event.type == "model.reasoning_delta"
                    )
                    self.assertEqual(reasoning, "Checked cacheability.")
                first_messages = runner.sessions.messages(first.session_id)
                second_messages = runner.sessions.messages(second.session_id)
                first_state = first_messages[-1].metadata.get("continuation_state")
                second_state = second_messages[-1].metadata.get("continuation_state")
                self.assertEqual(second_state, first_state)
                assert isinstance(second_state, dict)
                self.assertEqual(second_state["provider"], "counting")
                self.assertEqual(second_state["model"], "counting")

                first_follow_up = await runner.run(
                    "same follow-up",
                    session_id=first.session_id,
                )
                cached_follow_up = await runner.run(
                    "same follow-up",
                    session_id=second.session_id,
                )
                self.assertEqual(provider.calls, 2)
                self.assertEqual(first_follow_up.usage.application_cache_misses, 1)
                self.assertEqual(cached_follow_up.usage.application_cache_hits, 1)
            finally:
                await runner.close()

    async def test_provider_alias_replays_live_and_cached_continuation_state(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(root, agent={"provider": "corp_openai"})
            config.providers["corp_openai"] = ProviderConfig(
                type="alias_implementation",
                model="alias-model",
                max_retries=0,
            )
            provider: AliasContinuationProvider | None = None

            def factory(cfg, key):
                nonlocal provider
                provider = AliasContinuationProvider(cfg, key)
                return provider

            registry = ProviderRegistry()
            registry.register("alias_implementation", factory)
            runner = await build_runner(
                root,
                config=config,
                interactive=False,
                provider_registry=registry,
            )
            try:
                first = await runner.run("same request")
                second = await runner.run("same request")
                await runner.run("same follow-up", session_id=first.session_id)
                cached_follow_up = await runner.run(
                    "same follow-up",
                    session_id=second.session_id,
                )
                self.assertEqual(cached_follow_up.usage.application_cache_hits, 1)
                await runner.run("unique final prompt", session_id=second.session_id)

                assert provider is not None
                self.assertEqual(provider.calls, 3)
                self.assertEqual(provider.route_names, ["corp_openai"] * 3)
                self.assertEqual(provider.replayed, [True, True])
            finally:
                await runner.close()

    async def test_incomplete_response_is_not_cached(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(root, agent={"provider": "incomplete"})
            config.providers["incomplete"] = ProviderConfig(
                type="incomplete", model="incomplete", max_retries=0
            )
            provider: IncompleteProvider | None = None

            def factory(cfg, key):
                nonlocal provider
                provider = IncompleteProvider(cfg, key)
                return provider

            registry = ProviderRegistry()
            registry.register("incomplete", factory)
            runner = await build_runner(
                root,
                config=config,
                interactive=False,
                provider_registry=registry,
            )
            try:
                first = await runner.run("same incomplete request")
                second = await runner.run("same incomplete request")
                assert provider is not None
                self.assertEqual(provider.calls, 2)
                self.assertEqual(first.text, "partial answer")
                self.assertEqual(second.text, "partial answer")
                self.assertEqual(first.usage.application_cache_misses, 1)
                self.assertEqual(second.usage.application_cache_misses, 1)
                self.assertEqual(second.usage.application_cache_hits, 0)
            finally:
                await runner.close()

    async def test_offline_tool_cycle_resume_and_full_history(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(root)
            runner = await build_runner(root, config=config, interactive=False)
            try:
                first = await runner.run("OFFLINE_WRITE_DEMO")
                second = await runner.run("continue", session_id=first.session_id)
                messages = runner.sessions.messages(first.session_id)
                self.assertEqual(first.stop_reason.value, "end_turn")
                self.assertEqual(second.session_id, first.session_id)
                self.assertTrue((root / "borealis-demo.txt").is_file())
                self.assertGreaterEqual(len(messages), 5)
                self.assertTrue(
                    any(
                        event.type == "model.text_delta"
                        for _, event in runner.sessions.events(first.session_id)
                    )
                )
            finally:
                await runner.close()

    async def test_provider_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(root, agent={"provider": "bad", "provider_fallbacks": ["mock"]})
            config.providers["bad"] = ProviderConfig(
                type="slow",
                model="slow",
                max_retries=1,
                initial_backoff_seconds=0,
                max_backoff_seconds=0,
            )
            registry = ProviderRegistry()
            registry.register("slow", lambda cfg, key: FailingProvider(cfg, key))
            runner = await build_runner(
                root, config=config, interactive=False, provider_registry=registry
            )
            try:
                result = await runner.run("hello")
                events = [event for _, event in runner.sessions.events(result.session_id)]
                started = next(event for event in events if event.type == "model.started")
                retry = next(event for event in events if event.type == "model.retrying")
                self.assertEqual(result.stop_reason.value, "end_turn")
                self.assertIn("Offline mock", result.text)
                self.assertEqual(started.data["provider"], "bad")
                self.assertEqual(started.data["model"], "slow")
                self.assertEqual(retry.data["attempt"], 2)
                self.assertEqual(retry.data["max_attempts"], 2)
            finally:
                await runner.close()

    async def test_empty_provider_response_uses_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(
                root,
                agent={"provider": "empty", "provider_fallbacks": ["mock"]},
            )
            config.providers["empty"] = ProviderConfig(
                type="empty",
                model="empty",
                max_retries=0,
            )
            registry = ProviderRegistry()
            registry.register("empty", lambda cfg, key: EmptyProvider(cfg, key))
            runner = await build_runner(
                root,
                config=config,
                interactive=False,
                provider_registry=registry,
            )
            try:
                result = await runner.run("hello")
                events = [event for _, event in runner.sessions.events(result.session_id)]
                route_failure = next(
                    event for event in events if event.type == "model.route_failed"
                )
                self.assertIn("empty response", route_failure.data["error"])
                self.assertIn("Offline mock", result.text)
                self.assertEqual(result.stop_reason.value, "end_turn")
            finally:
                await runner.close()

    async def test_cancel(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(root, agent={"provider": "slow"})
            config.providers["slow"] = ProviderConfig(type="slow", model="slow", max_retries=0)
            registry = ProviderRegistry()
            registry.register("slow", lambda cfg, key: SlowProvider(cfg, key))
            runner = await build_runner(
                root, config=config, interactive=False, provider_registry=registry
            )
            task = asyncio.create_task(runner.run("hello"))
            while not runner._cancel:
                await asyncio.sleep(0.01)
            session_id = next(iter(runner._cancel))
            started = asyncio.get_running_loop().time()
            runner.cancel(session_id)
            result = await asyncio.wait_for(task, timeout=1)
            self.assertLess(asyncio.get_running_loop().time() - started, 0.75)
            self.assertEqual(result.stop_reason.value, "cancelled")
            await runner.close()

    async def test_max_time_is_an_end_to_end_deadline(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(
                root,
                agent={"provider": "slow", "max_time_seconds": 1},
            )
            config.providers["slow"] = ProviderConfig(type="slow", model="slow", max_retries=0)
            registry = ProviderRegistry()
            registry.register("slow", lambda cfg, key: SlowProvider(cfg, key))
            runner = await build_runner(
                root,
                config=config,
                interactive=False,
                provider_registry=registry,
            )
            try:
                started = asyncio.get_running_loop().time()
                result = await runner.run("hello")
                elapsed = asyncio.get_running_loop().time() - started
                self.assertLess(elapsed, 2.0)
                self.assertEqual(result.stop_reason.value, "budget")
                self.assertIn("Maximum 1s", result.error or "")
            finally:
                await runner.close()

    async def test_usage_and_response_are_persisted_before_cost_stop(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(root, agent={"provider": "cost", "max_cost_usd": 1.0})
            config.providers["cost"] = ProviderConfig(type="cost", model="cost", max_retries=0)
            registry = ProviderRegistry()
            registry.register("cost", lambda cfg, key: CostProvider(cfg, key))
            runner = await build_runner(
                root, config=config, interactive=False, provider_registry=registry
            )
            try:
                result = await runner.run("hello")
                self.assertEqual(result.stop_reason.value, "budget")
                self.assertEqual(runner.sessions.usage(result.session_id).cost_usd, 2.0)
                self.assertTrue(
                    any(
                        message.content == "expensive but durable"
                        for message in runner.sessions.messages(result.session_id)
                    )
                )
            finally:
                await runner.close()

    async def test_max_turns_has_specific_stop_reason(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.txt").write_text("a")
            config = make_config(root, agent={"provider": "one", "max_turns": 1})
            config.providers["one"] = ProviderConfig(type="one_tool", model="one", max_retries=0)
            registry = ProviderRegistry()
            registry.register("one_tool", lambda cfg, key: OneToolProvider(cfg, key))
            runner = await build_runner(
                root, config=config, interactive=False, provider_registry=registry
            )
            try:
                result = await runner.run("hello")
                self.assertEqual(result.stop_reason.value, "max_turns")
                self.assertTrue(result.incomplete)
                self.assertIn("send 'continue' to resume", result.error or "")
                assistant = runner.sessions.messages(result.session_id)[-1]
                self.assertEqual(assistant.role, Role.ASSISTANT)
                self.assertEqual(assistant.tool_calls, [])
            finally:
                await runner.close()

    async def test_last_allowed_turn_disables_tools_and_can_finish_cleanly(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(root, agent={"provider": "final", "max_turns": 2})
            config.providers["final"] = ProviderConfig(
                type="final_turn",
                model="final",
                max_retries=0,
            )
            provider = FinalTurnProvider(config.providers["final"])
            registry = ProviderRegistry()
            registry.register("final_turn", lambda cfg, key: provider)
            runner = await build_runner(
                root,
                config=config,
                interactive=False,
                provider_registry=registry,
            )
            try:
                result = await runner.run("finish within the limit")
                self.assertEqual(result.stop_reason.value, "end_turn")
                self.assertFalse(result.incomplete)
                self.assertEqual(result.turns, 2)
                self.assertGreater(provider.tool_counts[0], 0)
                self.assertEqual(provider.tool_counts[1], 0)
                self.assertEqual(
                    "\n\n".join(str(block["text"]) for block in provider.system_blocks[1]),
                    provider.systems[1],
                )
                self.assertFalse(provider.system_blocks[1][-1]["cacheable"])
                self.assertEqual((root / "completed.txt").read_text(), "done")
            finally:
                await runner.close()

    async def test_truncated_final_turn_preserves_recovery_state(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(root, agent={"provider": "final", "max_turns": 2})
            config.providers["final"] = ProviderConfig(
                type="final_turn",
                model="final",
                max_retries=0,
            )
            provider = FinalTurnProvider(config.providers["final"])
            provider.final_stop_reason = "max_tokens"
            registry = ProviderRegistry()
            registry.register("final_turn", lambda cfg, key: provider)
            runner = await build_runner(
                root,
                config=config,
                interactive=False,
                provider_registry=registry,
            )
            try:
                result = await runner.run("finish within the limit")
                self.assertEqual(result.stop_reason.value, "max_turns")
                self.assertTrue(result.incomplete)
                self.assertEqual(result.text, "Completed cleanly.")
                self.assertIn("final response was incomplete", result.error or "")
                self.assertIn("send 'continue' to resume", result.error or "")
            finally:
                await runner.close()

    async def test_max_turns_verifies_mutations_and_resumes_without_dangling_tools(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(
                root,
                agent={"provider": "defiant", "max_turns": 2, "auto_verify": True},
            )
            config.providers["defiant"] = ProviderConfig(
                type="defiant_final_turn",
                model="defiant",
                max_retries=0,
            )
            provider = DefiantFinalTurnProvider(config.providers["defiant"])
            provider.pause_final = True
            registry = ProviderRegistry()
            registry.register("defiant_final_turn", lambda cfg, key: provider)
            runner = await build_runner(
                root,
                config=config,
                interactive=False,
                provider_registry=registry,
            )
            try:
                with patch(
                    "borealis_coder.agent.runner.VerificationPlanner.run",
                    new_callable=AsyncMock,
                    return_value=VerificationReport(ok=True),
                ) as verify:
                    task = asyncio.create_task(runner.run("make a durable change"))
                    await asyncio.wait_for(provider.final_started.wait(), timeout=1)
                    session_id = next(iter(runner._cancel))
                    runner.steer(session_id, "late direction", message_id="late-message")
                    provider.release_final.set()
                    result = await task
                self.assertEqual(result.stop_reason.value, "max_turns")
                self.assertTrue(result.incomplete)
                self.assertEqual(result.verification, {"ok": True, "steps": []})
                verify.assert_awaited_once()
                self.assertEqual((root / "durable.txt").read_text(), "durable")
                self.assertFalse((root / "must-not-run.txt").exists())
                self.assertEqual(provider.tool_counts, [len(runner.tools.schemas()), 0])
                persisted = runner.sessions.messages(result.session_id)
                final_assistant = persisted[-2]
                self.assertEqual(final_assistant.role, Role.ASSISTANT)
                self.assertEqual(final_assistant.tool_calls, [])
                self.assertNotIn("continuation_state", final_assistant.metadata)
                self.assertEqual(persisted[-1].id, "late-message")
                self.assertTrue(persisted[-1].metadata["steering"])
                self.assertEqual(runner.queued_prompts(result.session_id), 0)
                self.assertEqual(
                    runner.sessions.get_session(result.session_id).status,
                    "idle",
                )

                resumed = await runner.run("continue", session_id=result.session_id)
                self.assertEqual(resumed.stop_reason.value, "end_turn")
                self.assertFalse(resumed.incomplete)
                self.assertEqual(resumed.text, "Resumed to completion.")
            finally:
                await runner.close()

    async def test_max_turns_persists_steering_queued_during_verification(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(
                root,
                agent={"provider": "defiant", "max_turns": 2, "auto_verify": True},
            )
            config.providers["defiant"] = ProviderConfig(
                type="defiant_final_turn",
                model="defiant",
                max_retries=0,
            )
            provider = DefiantFinalTurnProvider(config.providers["defiant"])
            registry = ProviderRegistry()
            registry.register("defiant_final_turn", lambda cfg, key: provider)
            runner = await build_runner(
                root,
                config=config,
                interactive=False,
                provider_registry=registry,
            )
            verification_started = asyncio.Event()
            release_verification = asyncio.Event()

            async def pause_verification(*_args):
                verification_started.set()
                await release_verification.wait()
                return VerificationReport(ok=True)

            try:
                with patch(
                    "borealis_coder.agent.runner.VerificationPlanner.run",
                    new=AsyncMock(side_effect=pause_verification),
                ):
                    task = asyncio.create_task(runner.run("make a durable change"))
                    await asyncio.wait_for(verification_started.wait(), timeout=1)
                    session_id = next(iter(runner._cancel))
                    runner.steer(
                        session_id,
                        "verification direction",
                        message_id="verification-message",
                    )
                    release_verification.set()
                    result = await task

                self.assertEqual(result.stop_reason.value, "max_turns")
                persisted = runner.sessions.messages(session_id)
                steering = [
                    message for message in persisted if message.id == "verification-message"
                ]
                self.assertEqual(len(steering), 1)
                self.assertTrue(steering[0].metadata["steering"])
                self.assertEqual(runner.queued_prompts(session_id), 0)
            finally:
                release_verification.set()
                await runner.close()

    async def test_max_turns_survives_a_deadline_during_verification(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(
                root,
                agent={
                    "provider": "defiant",
                    "max_turns": 2,
                    "max_time_seconds": 1,
                    "auto_verify": True,
                },
            )
            config.providers["defiant"] = ProviderConfig(
                type="defiant_final_turn",
                model="defiant",
                max_retries=0,
            )
            provider = DefiantFinalTurnProvider(config.providers["defiant"])
            registry = ProviderRegistry()
            registry.register("defiant_final_turn", lambda cfg, key: provider)
            runner = await build_runner(
                root,
                config=config,
                interactive=False,
                provider_registry=registry,
            )
            verification_started = asyncio.Event()

            async def block_verification(*_args):
                verification_started.set()
                await asyncio.Event().wait()

            try:
                with patch(
                    "borealis_coder.agent.runner.VerificationPlanner.run",
                    new=AsyncMock(side_effect=block_verification),
                ):
                    task = asyncio.create_task(runner.run("make a durable change"))
                    await asyncio.wait_for(verification_started.wait(), timeout=1)
                    session_id = next(iter(runner._cancel))
                    runner.steer(
                        session_id,
                        "verification direction",
                        message_id="deadline-message",
                    )
                    result = await task

                self.assertEqual(result.stop_reason.value, "max_turns")
                self.assertTrue(result.incomplete)
                self.assertIn("maximum 2 model turns reached", result.error or "")
                self.assertIn("Maximum 1s run time reached", result.error or "")
                self.assertEqual(
                    result.verification,
                    {
                        "ok": False,
                        "steps": [],
                        "error": "Maximum 1s run time reached",
                    },
                )
                persisted = runner.sessions.messages(session_id)
                self.assertEqual(
                    [message.id for message in persisted].count("deadline-message"),
                    1,
                )
                self.assertEqual(runner.queued_prompts(session_id), 0)
            finally:
                await runner.close()

    async def test_max_turns_closes_steering_before_completion_event(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(root, agent={"provider": "defiant", "max_turns": 2})
            config.providers["defiant"] = ProviderConfig(
                type="defiant_final_turn",
                model="defiant",
                max_retries=0,
            )
            provider = DefiantFinalTurnProvider(config.providers["defiant"])
            registry = ProviderRegistry()
            registry.register("defiant_final_turn", lambda cfg, key: provider)
            runner = await build_runner(
                root,
                config=config,
                interactive=False,
                provider_registry=registry,
            )
            completion_seen = asyncio.Event()

            async def reject_cleanup_steering(event):
                if event.type != "run.completed":
                    return
                self.assertFalse(runner.accepts_steering(event.session_id or ""))
                with self.assertRaises(SessionError):
                    runner.steer(event.session_id or "", "too late")
                completion_seen.set()

            runner.events.subscribe(reject_cleanup_steering)
            try:
                result = await runner.run("make a durable change")
                self.assertEqual(result.stop_reason.value, "max_turns")
                self.assertTrue(completion_seen.is_set())
            finally:
                await runner.close()

    async def test_final_turn_cost_overrun_preserves_max_turn_recovery(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(
                root,
                agent={
                    "provider": "defiant",
                    "max_turns": 2,
                    "max_cost_usd": 1.0,
                    "auto_verify": True,
                },
            )
            config.providers["defiant"] = ProviderConfig(
                type="defiant_final_turn",
                model="defiant",
                max_retries=0,
            )
            provider = DefiantFinalTurnProvider(config.providers["defiant"])
            provider.final_cost_usd = 2.0
            registry = ProviderRegistry()
            registry.register("defiant_final_turn", lambda cfg, key: provider)
            runner = await build_runner(
                root,
                config=config,
                interactive=False,
                provider_registry=registry,
            )
            try:
                with patch(
                    "borealis_coder.agent.runner.VerificationPlanner.run",
                    new_callable=AsyncMock,
                    return_value=VerificationReport(ok=True),
                ) as verify:
                    result = await runner.run("make a durable change")

                self.assertEqual(result.stop_reason.value, "max_turns")
                self.assertTrue(result.incomplete)
                self.assertIn("maximum 2 model turns reached", result.error or "")
                self.assertIn("exceeded $1.00", result.error or "")
                self.assertEqual(result.verification, {"ok": True, "steps": []})
                verify.assert_awaited_once()
            finally:
                await runner.close()

    async def test_max_turns_verifies_files_mutated_by_shell_in_an_additional_root(self):
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as extra:
            root = Path(td)
            extra_root = Path(extra)
            mutated_path = extra_root / "shell-mutated.txt"
            mutated_path.write_text("before")
            (extra_root / ".gitignore").write_text("shell-mutated.txt\n")
            original_stat = mutated_path.stat()
            config = make_config(
                root,
                agent={"provider": "shell_mutation", "max_turns": 2, "auto_verify": True},
            )
            config.providers["shell_mutation"] = ProviderConfig(
                type="shell_mutation_final_turn",
                model="shell-mutation",
                max_retries=0,
            )
            provider = ShellMutationFinalTurnProvider(config.providers["shell_mutation"])
            registry = ProviderRegistry()
            registry.register("shell_mutation_final_turn", lambda cfg, key: provider)
            runner = await build_runner(
                root,
                config=config,
                interactive=False,
                provider_registry=registry,
                additional_roots=[extra_root],
            )

            def mutate_workspace(arguments, context):
                del arguments, context
                mutated_path.write_text("change")
                os.utime(
                    mutated_path,
                    ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
                )
                return ToolResult("changed through shell")

            runner.tools.register(
                FunctionTool(
                    name="shell",
                    description="Test shell mutation tracking.",
                    parameters=object_schema({"command": {"type": "string"}}),
                    function=mutate_workspace,
                    effect=Effect.EXECUTE,
                ),
                replace=True,
            )
            run_process = AsyncMock(
                return_value=ProcessResult("true", 0, "", "", 1)
            )
            runner.tool_context.process.run = run_process
            try:
                with patch(
                    "borealis_coder.agent.runner.VerificationPlanner.detect",
                    return_value=[VerificationStep("Root check", "true", 10)],
                ):
                    result = await runner.run("change a file through shell")
                self.assertEqual(result.stop_reason.value, "max_turns")
                self.assertEqual(
                    result.changed_files,
                    [f"{extra_root.name}:shell-mutated.txt"],
                )
                verification = result.verification
                self.assertIsNotNone(verification)
                assert verification is not None
                self.assertTrue(verification["ok"])
                self.assertEqual(
                    verification["steps"][0]["root"],
                    f"{extra_root.name}:.",
                )
                run_process.assert_awaited_once()
                process_call = run_process.await_args
                assert process_call is not None
                self.assertEqual(process_call.kwargs["cwd"], extra_root.resolve())
            finally:
                await runner.close()

    @unittest.skipIf(os.name == "nt", "POSIX permission bits are required")
    async def test_max_turns_verifies_shell_permission_changes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mutated_path = root / "script.sh"
            mutated_path.write_text("#!/bin/sh\n")
            original_mode = stat.S_IMODE(mutated_path.stat().st_mode)
            config = make_config(
                root,
                agent={"provider": "shell_mutation", "max_turns": 2, "auto_verify": True},
            )
            config.providers["shell_mutation"] = ProviderConfig(
                type="shell_mutation_final_turn",
                model="shell-mutation",
                max_retries=0,
            )
            provider = ShellMutationFinalTurnProvider(config.providers["shell_mutation"])
            registry = ProviderRegistry()
            registry.register("shell_mutation_final_turn", lambda cfg, key: provider)
            runner = await build_runner(
                root,
                config=config,
                interactive=False,
                provider_registry=registry,
            )

            def mutate_permissions(arguments, context):
                del arguments, context
                mutated_path.chmod(original_mode ^ stat.S_IXUSR)
                return ToolResult("changed permissions through shell")

            runner.tools.register(
                FunctionTool(
                    name="shell",
                    description="Test shell permission tracking.",
                    parameters=object_schema({"command": {"type": "string"}}),
                    function=mutate_permissions,
                    effect=Effect.EXECUTE,
                ),
                replace=True,
            )
            try:
                with patch(
                    "borealis_coder.agent.runner.VerificationPlanner.run",
                    new_callable=AsyncMock,
                    return_value=VerificationReport(ok=True),
                ) as verify:
                    result = await runner.run("change permissions through shell")

                self.assertEqual(result.stop_reason.value, "max_turns")
                self.assertEqual(result.changed_files, ["script.sh"])
                verify.assert_awaited_once()
            finally:
                await runner.close()

    async def test_shell_result_is_durable_when_cancelled_during_change_detection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mutated_path = root / "shell-mutated.txt"
            mutated_path.write_text("before")
            config = make_config(root, agent={"provider": "shell_mutation"})
            config.providers["shell_mutation"] = ProviderConfig(
                type="shell_mutation_final_turn",
                model="shell-mutation",
                max_retries=0,
            )
            provider = ShellMutationFinalTurnProvider(config.providers["shell_mutation"])
            registry = ProviderRegistry()
            registry.register("shell_mutation_final_turn", lambda cfg, key: provider)
            runner = await build_runner(
                root,
                config=config,
                interactive=False,
                provider_registry=registry,
            )

            def mutate_workspace(arguments, context):
                del arguments, context
                mutated_path.write_text("after")
                return ToolResult("changed through shell")

            runner.tools.register(
                FunctionTool(
                    name="shell",
                    description="Test durable shell results.",
                    parameters=object_schema({"command": {"type": "string"}}),
                    function=mutate_workspace,
                    effect=Effect.EXECUTE,
                ),
                replace=True,
            )
            original_file_state = runner._workspace_file_state
            after_snapshot_started = threading.Event()
            release_snapshot = threading.Event()
            snapshot_calls = 0

            def pause_after_snapshot():
                nonlocal snapshot_calls
                snapshot_calls += 1
                state = original_file_state()
                if snapshot_calls == 2:
                    after_snapshot_started.set()
                    release_snapshot.wait(timeout=2)
                return state

            try:
                with patch.object(runner, "_workspace_file_state", side_effect=pause_after_snapshot):
                    task = asyncio.create_task(runner.run("change a file through shell"))
                    started = await asyncio.to_thread(after_snapshot_started.wait, 1)
                    self.assertTrue(started)
                    task.cancel()
                    await asyncio.sleep(0)
                    release_snapshot.set()
                    result = await asyncio.wait_for(task, timeout=2)

                self.assertEqual(result.stop_reason.value, "cancelled")
                tool_call = runner.sessions.tool_calls(result.session_id)[0]
                self.assertEqual(tool_call["status"], "completed")
                tool_messages = [
                    message
                    for message in runner.sessions.messages(result.session_id)
                    if message.role == Role.TOOL
                ]
                self.assertEqual(len(tool_messages), 1)
                self.assertEqual(tool_messages[0].content, "changed through shell")
            finally:
                release_snapshot.set()
                await runner.close()

    async def test_late_steering_is_durable_when_final_turn_is_already_running(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.txt").write_text("a")
            config = make_config(root, agent={"provider": "late", "max_turns": 2})
            config.providers["late"] = ProviderConfig(
                type="late_steering",
                model="late",
                max_retries=0,
            )
            provider = LateSteeringProvider(config.providers["late"])
            registry = ProviderRegistry()
            registry.register("late_steering", lambda cfg, key: provider)
            runner = await build_runner(
                root,
                config=config,
                interactive=False,
                provider_registry=registry,
            )
            try:
                task = asyncio.create_task(runner.run("start"))
                await asyncio.wait_for(provider.final_started.wait(), timeout=1)
                session_id = next(iter(runner._cancel))
                runner.steer(session_id, "late direction", message_id="late-message")
                provider.release_final.set()
                result = await task

                self.assertEqual(result.stop_reason.value, "max_turns")
                self.assertTrue(result.incomplete)
                steering = [
                    message
                    for message in runner.sessions.messages(session_id)
                    if message.id == "late-message"
                ]
                self.assertEqual(len(steering), 1)
                self.assertTrue(steering[0].metadata["steering"])
                self.assertEqual(runner.queued_prompts(session_id), 0)

                resumed = await runner.run("continue", session_id=session_id)
                self.assertEqual(resumed.stop_reason.value, "end_turn")
                self.assertEqual(resumed.text, "Late steering handled.")
            finally:
                await runner.close()

    async def test_concurrent_sessions_keep_context_and_tool_ids_isolated(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(root, agent={"provider": "concurrent"})
            config.providers["concurrent"] = ProviderConfig(
                type="concurrent", model="concurrent", max_retries=0
            )
            registry = ProviderRegistry()
            registry.register("concurrent", lambda cfg, key: ConcurrentProvider(cfg, key))
            runner = await build_runner(
                root, config=config, interactive=False, provider_registry=registry
            )
            try:
                alpha, beta = await asyncio.gather(runner.run("alpha"), runner.run("beta"))
                self.assertEqual(alpha.changed_files, ["alpha.txt"])
                self.assertEqual(beta.changed_files, ["beta.txt"])
                self.assertEqual(
                    runner.sessions.tool_calls(alpha.session_id)[0]["output"].splitlines()[0],
                    "Wrote 5 bytes to alpha.txt",
                )
                self.assertEqual(
                    runner.sessions.tool_calls(beta.session_id)[0]["output"].splitlines()[0],
                    "Wrote 4 bytes to beta.txt",
                )
            finally:
                await runner.close()

    async def test_context_hard_limit_stops_before_provider_call(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(root, agent={"max_input_tokens": 1, "compact_at_ratio": 0.5})
            runner = await build_runner(root, config=config, interactive=False)
            try:
                result = await runner.run("hello")
                self.assertEqual(result.stop_reason.value, "budget")
                provider = runner.providers[0].provider
                assert isinstance(provider, MockProvider)
                self.assertEqual(provider.calls, 0)
            finally:
                await runner.close()


class FailingProvider(Provider):
    name = "slow"

    async def complete(self, request):
        from borealis_coder.errors import ProviderUnavailableError

        raise ProviderUnavailableError("down", retryable=True)


if __name__ == "__main__":
    unittest.main()
