from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from borealis_coder.agent import build_runner
from borealis_coder.config import ProviderConfig
from borealis_coder.errors import SessionError
from borealis_coder.models import (
    ContinuationState,
    ModelResponse,
    ProviderRequest,
    Role,
    ToolCall,
    Usage,
)
from borealis_coder.providers.base import Provider, ProviderStreamEvent
from borealis_coder.providers.mock import MockProvider
from borealis_coder.providers.registry import ProviderRegistry
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
        for text in "abc":
            yield ProviderStreamEvent(type="text_delta", text=text)
        yield ProviderStreamEvent(
            type="completed",
            response=ModelResponse(text="abc", stop_reason="end_turn"),
        )


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
                self.assertEqual(create_task.call_count, 2)
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
