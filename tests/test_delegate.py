from __future__ import annotations

import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from borealis_coder.agent import build_runner
from borealis_coder.config import ProviderConfig
from borealis_coder.errors import ProviderError, ProviderUnavailableError
from borealis_coder.models import Effect, ModelResponse, Role, ToolCall, ToolResult, Usage
from borealis_coder.providers.gemini import GeminiProvider
from borealis_coder.providers.mock import MockProvider
from borealis_coder.tools.base import FunctionTool, object_schema
from borealis_coder.tools.delegate import DelegateTaskTool
from tests.helpers import make_config, make_context


class DelegationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.runner = await build_runner(
            self.root, config=make_config(self.root), interactive=False,
        )
        self.addAsyncCleanup(self.runner.close)
        provider = self.runner.providers[0].provider
        assert isinstance(provider, MockProvider)
        self.provider = provider

    @staticmethod
    def delegation():
        return ModelResponse(tool_calls=[ToolCall(
            name="delegate_task", arguments={"task": "inspect the workspace", "max_turns": 3},
        )])

    def delegate_result(self, session_id):
        return next(
            message for message in self.runner.sessions.messages(session_id)
            if message.role == Role.TOOL and message.tool_name == "delegate_task"
        )

    async def test_parallel_delegations_report_their_own_parent_calls(self):
        ready = threading.Barrier(2)
        calls = [
            ToolCall(id=name, name="delegate_task", arguments={"task": name, "max_turns": 1})
            for name in ("first", "second")
        ]

        def prepare(*args, **kwargs):
            ready.wait(timeout=2)
            return "Inspect the project."

        def handler(request, call):
            return ModelResponse(tool_calls=calls) if call == 1 else ModelResponse(text="Finished.")

        self.provider.handler = handler
        events = []
        self.runner.events.subscribe(events.append)
        with patch.object(self.runner.context_builder, "system_prompt", side_effect=prepare):
            result = await self.runner.run("Run two investigations")

        self.assertEqual(result.text, "Finished.")
        starts = [event for event in events if event.type == "delegate.started"]
        self.assertCountEqual(
            [event.data["parent_tool_call_id"] for event in starts], [call.id for call in calls],
        )

    async def test_delegated_gemini_tools_replay_provider_continuation(self):
        provider = GeminiProvider(ProviderConfig(type="gemini", base_url="https://gemini.test"))
        self.addAsyncCleanup(provider.close)
        steps = [
            {"type": "thought", "signature": "opaque-signature", "summary": []},
            {"type": "function_call", "id": "inspect", "name": "list_directory", "arguments": {
                "path": ".", "depth": 1, "include_hidden": False, "max_entries": 10,
            }},
        ]
        request = AsyncMock(side_effect=[
            SimpleNamespace(data={"status": "requires_action", "steps": steps}),
            SimpleNamespace(data={"status": "completed", "steps": [
                {"type": "model_output", "content": [{"type": "text", "text": "Investigation complete."}]},
            ]}),
        ])
        provider.http.post_json = request
        context = make_context(self.root)
        context.metadata.update({
            "provider_routes": [SimpleNamespace(
                name="corporate_gemini", model="gemini-model", provider=provider,
            )],
            "tool_registry": self.runner.tools,
            "context_builder": self.runner.context_builder,
        })
        result = await DelegateTaskTool().execute({"task": "inspect", "max_turns": 2}, context)
        self.assertFalse(result.is_error)
        self.assertEqual(result.output, "Investigation complete.")
        self.assertEqual(request.await_count, 2)
        replay = request.await_args_list[1].kwargs["payload"]["input"]
        self.assertEqual(replay[1:3], steps)
        self.assertEqual(replay[3]["type"], "function_result")
        self.assertEqual(replay[3]["call_id"], "inspect")
        self.assertNotIn("opaque-signature", str(result))

    async def test_failed_delegated_request_keeps_reported_usage(self):
        def handler(request, call):
            if call == 1:
                return self.delegation()
            if call == 2:
                raise ProviderError(
                    "Delegated provider failed",
                    usage=Usage(input_tokens=123, output_tokens=17, cost_usd=0.25, requests=1),
                )
            return ModelResponse(text="The delegated investigation failed.")

        self.provider.handler = handler
        result = await self.runner.run("investigate")
        self.assertEqual(result.usage.requests, 3)
        self.assertEqual(result.usage.input_tokens, 123)
        self.assertAlmostEqual(result.usage.cost_usd, 0.25)
        stored = self.runner.sessions.usage(result.session_id)
        self.assertEqual(stored.to_dict(), result.usage.to_dict())
        self.assertTrue(self.delegate_result(result.session_id).is_error)

    async def test_delegated_provider_does_not_multiply_retries(self):
        provider = GeminiProvider(ProviderConfig(
            type="gemini", base_url="https://gemini.test", max_retries=1,
            initial_backoff_seconds=0, max_backoff_seconds=0,
        ))
        self.addAsyncCleanup(provider.close)

        async def failed_request(*args, **kwargs):
            raise ProviderUnavailableError(
                "Temporary failure", retryable=True,
                usage=Usage(cost_usd=0.01, requests=1),
            )

        request = AsyncMock(side_effect=failed_request)
        provider.http.post_json = request
        recorded = Usage()
        context = make_context(self.root)
        context.metadata.update({
            "provider_routes": [SimpleNamespace(name="gemini", model="model", provider=provider)],
            "tool_registry": self.runner.tools,
            "context_builder": self.runner.context_builder,
            "usage_sink": recorded.add,
        })
        with self.assertRaises(ProviderUnavailableError) as caught:
            await DelegateTaskTool().execute({"task": "inspect", "max_turns": 2}, context)
        self.assertEqual(request.await_count, 2)
        self.assertEqual(recorded.requests, 2)
        self.assertAlmostEqual(recorded.cost_usd, 0.02)
        self.assertEqual(caught.exception.usage, recorded)

    async def test_failed_delegated_request_enforces_parent_cost_limit(self):
        self.runner.config.agent.max_cost_usd = 0.1

        def handler(request, call):
            if call == 1:
                return self.delegation()
            if call == 2:
                raise ProviderError("Failed after billing", usage=Usage(cost_usd=0.2, requests=1))
            return ModelResponse(text="This request must not occur.")

        self.provider.handler = handler
        result = await self.runner.run("investigate")
        self.assertEqual(result.stop_reason.value, "budget")
        self.assertEqual(self.provider.calls, 2)
        self.assertAlmostEqual(result.usage.cost_usd, 0.2)

    async def test_incomplete_delegated_response_does_not_execute_tools(self):
        executed: list[str] = []

        def inspect(arguments, context):
            executed.append("inspect")
            return ToolResult("Evidence")

        self.runner.tools.register(FunctionTool(
            name="inspect_example", description="Inspect an example",
            parameters=object_schema({}), function=inspect, effect=Effect.READ,
        ))
        self.provider.enqueue(
            self.delegation(),
            ModelResponse(
                text="Partial findings",
                tool_calls=[ToolCall(name="inspect_example", arguments={})],
                stop_reason="length",
                usage=Usage(requests=1),
            ),
            ModelResponse(text="The investigation was incomplete."),
        )
        result = await self.runner.run("investigate")
        self.assertEqual(executed, [])
        self.assertTrue(self.delegate_result(result.session_id).is_error)
        self.assertEqual(result.usage.requests, 3)

    async def test_incomplete_text_is_not_a_successful_conclusion(self):
        self.provider.enqueue(
            self.delegation(),
            ModelResponse(text="I found the cause and it is", stop_reason="max_tokens"),
            ModelResponse(text="The investigation was incomplete."),
        )
        result = await self.runner.run("investigate")
        self.assertTrue(self.delegate_result(result.session_id).is_error)

    async def test_whitespace_is_not_a_successful_conclusion(self):
        self.provider.enqueue(
            self.delegation(),
            ModelResponse(text=" \n\t", stop_reason="stop"),
            ModelResponse(text="The investigation returned no conclusion."),
        )
        result = await self.runner.run("investigate")
        self.assertTrue(self.delegate_result(result.session_id).is_error)

    async def test_successful_delegated_request_checks_cost_before_more_work(self):
        self.runner.config.agent.max_cost_usd = 0.1
        executed: list[str] = []

        def inspect(arguments, context):
            executed.append("inspect")
            return ToolResult("Evidence")

        self.runner.tools.register(FunctionTool(
            name="inspect_example", description="Inspect an example",
            parameters=object_schema({}), function=inspect, effect=Effect.READ,
        ))
        self.provider.enqueue(
            self.delegation(),
            ModelResponse(
                tool_calls=[ToolCall(name="inspect_example", arguments={})],
                usage=Usage(cost_usd=0.2, requests=1),
            ),
            ModelResponse(text="This request must not occur."),
        )
        result = await self.runner.run("investigate")
        self.assertEqual(result.stop_reason.value, "budget")
        self.assertEqual(self.provider.calls, 2)
        self.assertEqual(executed, [])
        self.assertAlmostEqual(result.usage.cost_usd, 0.2)

    async def test_delegate_starts_no_new_request_at_exact_cost_limit(self):
        self.runner.config.agent.max_cost_usd = 0.1
        self.provider.enqueue(
            self.delegation(),
            ModelResponse(
                tool_calls=[ToolCall(name="list_directory", arguments={
                    "path": ".", "depth": 1, "include_hidden": False, "max_entries": 10,
                })],
                usage=Usage(cost_usd=0.1, requests=1),
            ),
            ModelResponse(text="This request must not occur."),
        )
        result = await self.runner.run("investigate")
        self.assertEqual(result.stop_reason.value, "budget")
        self.assertEqual(self.provider.calls, 2)
        self.assertAlmostEqual(result.usage.cost_usd, 0.1)

    async def test_cancellation_waits_for_in_progress_usage_accounting(self):
        started = asyncio.Event()
        release = asyncio.Event()
        recorded = Usage()

        async def sink(usage):
            started.set()
            await release.wait()
            recorded.add(usage)

        context = make_context(self.root)
        context.metadata.update({
            "provider_routes": self.runner.providers,
            "tool_registry": self.runner.tools,
            "context_builder": self.runner.context_builder,
            "usage_sink": sink,
        })
        self.provider.enqueue(ModelResponse(text="Conclusion", usage=Usage(cost_usd=0.05)))
        task = asyncio.create_task(DelegateTaskTool().execute(
            {"task": "inspect", "max_turns": 1}, context,
        ))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
        finally:
            release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(recorded.requests, 1)
        self.assertAlmostEqual(recorded.cost_usd, 0.05)

    async def test_cancelled_delegated_retry_preserves_parent_usage(self):
        self.provider.config.max_retries = 1
        self.provider.config.initial_backoff_seconds = 30
        self.provider.config.max_backoff_seconds = 30
        failed = asyncio.Event()
        sessions = asyncio.Queue()

        def observe(event):
            if event.type == "delegate.started":
                sessions.put_nowait(event.session_id)

        def handler(request, call):
            if call == 1:
                return self.delegation()
            failed.set()
            raise ProviderUnavailableError(
                "Billed delegated failure", retryable=True,
                usage=Usage(cost_usd=0.1, requests=1),
            )

        self.runner.events.subscribe(observe)
        self.provider.handler = handler
        for cancellation in ("task", "session"):
            for cost_limit in (0, 0.05):
                with self.subTest(cancellation=cancellation, limit=cost_limit):
                    self.provider.calls = 0
                    self.runner.config.agent.max_cost_usd = cost_limit
                    failed.clear()
                    task = asyncio.create_task(self.runner.run("Cancel delegated retry"))
                    try:
                        session_id = await asyncio.wait_for(sessions.get(), timeout=2)
                        await asyncio.wait_for(failed.wait(), timeout=2)
                        if cancellation == "task":
                            task.cancel()
                        else:
                            self.assertTrue(self.runner.cancel(session_id))
                        result = await asyncio.wait_for(task, timeout=2)
                    finally:
                        if not task.done():
                            task.cancel()
                            await asyncio.gather(task, return_exceptions=True)

                    self.assertEqual(result.stop_reason.value, "cancelled")
                    self.assertEqual(self.provider.calls, 2)
                    self.assertEqual(result.usage.requests, 2)
                    self.assertAlmostEqual(result.usage.cost_usd, 0.1)
                    self.assertEqual(
                        self.runner.sessions.usage(result.session_id).to_dict(),
                        result.usage.to_dict(),
                    )
                    self.assertNotIn("running", {
                        row["status"] for row in self.runner.sessions.tool_calls(result.session_id)
                    })
