from __future__ import annotations

import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from borealis_coder.agent import build_runner, validate_tool_call_order
from borealis_coder.errors import (
    BudgetExceeded,
    ProviderAuthenticationError,
    ProviderContextOverflowError,
    ProviderError,
    ProviderUnavailableError,
)
from borealis_coder.models import Effect, ModelResponse, Role, ToolCall, ToolResult, Usage
from borealis_coder.providers.mock import MockProvider
from borealis_coder.tools import FunctionTool, object_schema
from tests.helpers import make_config


class BudgetLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.runner = await build_runner(self.root, config=make_config(self.root), interactive=False)
        self.addAsyncCleanup(self.runner.close)
        provider = self.runner.providers[0].provider
        assert isinstance(provider, MockProvider)
        self.provider = provider

    def assert_finished_calls(self, result, expected):
        rows = self.runner.sessions.tool_calls(result.session_id)
        self.assertEqual({row["tool_call_id"] for row in rows}, set(expected))
        self.assertNotIn("running", {row["status"] for row in rows})
        messages = self.runner.sessions.messages(result.session_id)
        validate_tool_call_order(messages)
        tool_messages = {message.tool_call_id: message for message in messages if message.role == Role.TOOL}
        self.assertEqual(set(tool_messages), set(expected))
        return tool_messages

    async def test_terminal_stream_errors_keep_usage_from_prior_retries(self):
        self.provider.config.max_retries = 1
        self.provider.config.initial_backoff_seconds = 0
        self.provider.config.max_backoff_seconds = 0
        self.runner.config.agent.compaction_max_overflow_retries = 0
        for error_class in (ProviderError, ProviderAuthenticationError, ProviderContextOverflowError):
            for terminal_cost in (None, 0.2):
                for cost_limit in (0, 0.05):
                    with self.subTest(error=error_class, terminal_cost=terminal_cost, limit=cost_limit):
                        self.runner.config.agent.max_cost_usd = cost_limit
                        self.provider.calls = 0

                        def handler(request, call, error_class=error_class, terminal_cost=terminal_cost):
                            if call == 1:
                                raise ProviderUnavailableError(
                                    "First billed failure", retryable=True,
                                    usage=Usage(cost_usd=0.1, requests=1),
                                )
                            raise error_class(
                                "Terminal failure",
                                usage=(
                                    Usage(cost_usd=terminal_cost, requests=1)
                                    if terminal_cost is not None else None
                                ),
                            )

                        self.provider.handler = handler
                        result = await self.runner.run("Check streamed retry accounting")

                        self.assertEqual(self.provider.calls, 2)
                        self.assertEqual(result.stop_reason.value, "budget" if cost_limit else "error")
                        self.assertAlmostEqual(result.usage.cost_usd, 0.1 + (terminal_cost or 0))
                        self.assertEqual(result.usage.requests, 1 if terminal_cost is None else 2)
                        self.assertEqual(
                            self.runner.sessions.usage(result.session_id).to_dict(),
                            result.usage.to_dict(),
                        )

    async def test_delegated_budget_exit_is_terminal_before_run_completion(self):
        self.runner.config.agent.max_model_requests = 2
        self.provider.enqueue(
            ModelResponse(tool_calls=[ToolCall(
                id="delegate", name="delegate_task", arguments={"task": "inspect", "max_turns": 3},
            )]),
            ModelResponse(tool_calls=[ToolCall(name="list_directory", arguments={
                "path": ".", "depth": 1, "include_hidden": False, "max_entries": 10,
            })]),
        )
        statuses_at_completion = []

        def observe(event):
            if event.type == "run.completed":
                statuses_at_completion.extend(
                    row["status"] for row in self.runner.sessions.tool_calls(event.session_id)
                )

        self.runner.events.subscribe(observe)
        result = await self.runner.run("inspect")
        self.assertEqual(result.stop_reason.value, "budget")
        self.assertEqual(statuses_at_completion, ["error"])
        messages = self.assert_finished_calls(result, {"delegate"})
        self.assertEqual(messages["delegate"].metadata["budget_kind"], "model_requests")
        self.assertNotIn("unknown_outcome", messages["delegate"].metadata)

        self.provider.enqueue(ModelResponse(text="Resumed cleanly."))
        resumed = await self.runner.run("continue", session_id=result.session_id)
        self.assertEqual(resumed.text, "Resumed cleanly.")
        history = self.runner.sessions.messages(result.session_id)
        self.assertFalse(any(message.metadata.get("abandoned") for message in history))

    async def test_cancelled_stream_retries_keep_reported_usage(self):
        self.provider.config.max_retries = 1
        self.provider.config.initial_backoff_seconds = 30
        self.provider.config.max_backoff_seconds = 30
        retrying = asyncio.Queue()

        def observe(event):
            if event.type == "model.retrying":
                retrying.put_nowait(event.session_id)

        def handler(request, call):
            raise ProviderUnavailableError(
                "Billed failure", retryable=True,
                usage=Usage(cost_usd=0.1, requests=1),
            )

        self.runner.events.subscribe(observe)
        self.provider.handler = handler
        for cancellation in ("task", "session", "deadline"):
            for cost_limit in (0, 0.05):
                with self.subTest(cancellation=cancellation, limit=cost_limit):
                    self.runner.config.agent.max_time_seconds = 1 if cancellation == "deadline" else 60
                    self.runner.config.agent.max_cost_usd = cost_limit
                    self.provider.calls = 0
                    task = asyncio.create_task(self.runner.run("Cancel during retry backoff"))
                    try:
                        session_id = await asyncio.wait_for(retrying.get(), timeout=2)
                        if cancellation == "task":
                            task.cancel()
                        elif cancellation == "session":
                            self.assertTrue(self.runner.cancel(session_id))
                        result = await asyncio.wait_for(task, timeout=2)
                    finally:
                        if not task.done():
                            task.cancel()
                            await asyncio.gather(task, return_exceptions=True)

                    self.assertEqual(self.provider.calls, 1)
                    self.assertEqual(
                        result.stop_reason.value,
                        "budget" if cancellation == "deadline" else "cancelled",
                    )
                    self.assertEqual(result.usage.requests, 1)
                    self.assertAlmostEqual(result.usage.cost_usd, 0.1)
                    self.assertEqual(
                        self.runner.sessions.usage(result.session_id).to_dict(),
                        result.usage.to_dict(),
                    )

    async def test_repeated_cancellation_finishes_retry_usage_storage(self):
        self.provider.config.max_retries = 1
        self.provider.config.initial_backoff_seconds = 30
        self.provider.config.max_backoff_seconds = 30
        retrying = asyncio.Event()
        writing = threading.Event()
        release = threading.Event()
        original_add_usage = self.runner.sessions.add_usage

        def observe(event):
            if event.type == "model.retrying":
                retrying.set()

        def handler(request, call):
            raise ProviderUnavailableError(
                "Billed failure", retryable=True,
                usage=Usage(cost_usd=0.1, requests=1),
            )

        def delayed_add_usage(session_id, usage):
            writing.set()
            if not release.wait(timeout=3):
                raise AssertionError("Accounting was not released")
            original_add_usage(session_id, usage)

        self.runner.events.subscribe(observe)
        self.provider.handler = handler
        with patch.object(self.runner.sessions, "add_usage", side_effect=delayed_add_usage):
            task = asyncio.create_task(self.runner.run("Cancel while preserving reported usage"))
            try:
                await asyncio.wait_for(retrying.wait(), timeout=2)
                task.cancel()
                self.assertTrue(await asyncio.to_thread(writing.wait, 2))
                task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(task.done())
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                result = await asyncio.wait_for(task, timeout=2)

        self.assertEqual(result.stop_reason.value, "cancelled")
        self.assertEqual(result.usage.requests, 1)
        self.assertAlmostEqual(result.usage.cost_usd, 0.1)
        self.assertEqual(self.runner.sessions.usage(result.session_id).to_dict(), result.usage.to_dict())

    async def test_cancelling_completed_response_accounting_preserves_run_and_storage_totals(self):
        loop = asyncio.get_running_loop()
        errors = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: errors.append(context))
        self.addCleanup(loop.set_exception_handler, previous_handler)
        for cost_limit in (0, 0.05):
            with self.subTest(cost_limit=cost_limit):
                self.runner.config.agent.max_cost_usd = cost_limit
                self.provider.enqueue(ModelResponse(
                    text="Complete response", usage=Usage(cost_usd=0.1, requests=1),
                ))
                writing = threading.Event()
                release = threading.Event()
                original_add_usage = self.runner.sessions.add_usage

                def delayed_add_usage(session_id, usage, writing=writing, release=release, original=original_add_usage):
                    writing.set()
                    if not release.wait(timeout=3):
                        raise AssertionError("Accounting was not released")
                    original(session_id, usage)

                with patch.object(self.runner.sessions, "add_usage", side_effect=delayed_add_usage):
                    task = asyncio.create_task(self.runner.run("Cancel during completed-response accounting"))
                    try:
                        self.assertTrue(await asyncio.to_thread(writing.wait, 2))
                        task.cancel()
                        await asyncio.sleep(0)
                        task.cancel()
                        await asyncio.sleep(0)
                        self.assertFalse(task.done())
                    finally:
                        release.set()
                        result = await asyncio.wait_for(task, timeout=2)

                self.assertEqual(result.stop_reason.value, "cancelled")
                self.assertEqual(result.usage.requests, 1)
                self.assertAlmostEqual(result.usage.cost_usd, 0.1)
                self.assertEqual(
                    self.runner.sessions.usage(result.session_id).to_dict(),
                    result.usage.to_dict(),
                )
        await asyncio.sleep(0)
        self.assertEqual(errors, [])

    async def test_budget_exit_closes_unstarted_effectful_calls(self):
        def exhaust(arguments, context):
            raise BudgetExceeded("cost", "Cost budget reached")

        self.runner.tools.register(FunctionTool(
            name="exhaust", description="Exercise a budget exit", parameters=object_schema({}),
            function=exhaust, concurrent=True,
        ))
        self.provider.enqueue(ModelResponse(tool_calls=[
            ToolCall(id="exhaust", name="exhaust", arguments={}),
            ToolCall(id="write", name="write_file", arguments={
                "path": "must-not-exist.txt", "content": "No", "expected_sha256": None,
            }),
        ]))
        result = await self.runner.run("inspect then write")
        self.assertEqual(result.stop_reason.value, "budget")
        self.assertFalse((self.root / "must-not-exist.txt").exists())
        messages = self.assert_finished_calls(result, {"exhaust", "write"})
        self.assertTrue(messages["write"].metadata["not_started"])
        self.assertEqual(messages["write"].metadata["budget_kind"], "cost")

    async def test_budget_exit_cancels_and_drains_parallel_reads(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()
        never = asyncio.Event()

        async def blocked(arguments, context):
            started.set()
            try:
                await never.wait()
            finally:
                cancelled.set()
            return ToolResult("Unexpected completion")

        async def exhaust(arguments, context):
            await started.wait()
            raise BudgetExceeded("cost", "Cost budget reached")

        for name, function in (("blocked", blocked), ("exhaust", exhaust)):
            self.runner.tools.register(FunctionTool(
                name=name, description="Exercise parallel budget shutdown", parameters=object_schema({}),
                function=function, concurrent=True,
            ))
        self.provider.enqueue(ModelResponse(tool_calls=[
            ToolCall(id="blocked", name="blocked", arguments={}),
            ToolCall(id="exhaust", name="exhaust", arguments={}),
        ]))
        try:
            result = await asyncio.wait_for(self.runner.run("inspect"), timeout=2)
            self.assertEqual(result.stop_reason.value, "budget")
            self.assertTrue(cancelled.is_set())
            self.assert_finished_calls(result, {"blocked", "exhaust"})
        finally:
            never.set()

    async def test_effectful_budget_exit_preserves_observed_changes(self):
        def change_then_stop(arguments, context):
            (self.root / "changed.txt").write_text("Changed before budget exit", encoding="utf-8")
            raise BudgetExceeded("cost", "Cost budget reached")

        self.runner.tools.register(FunctionTool(
            name="change_then_stop", description="Exercise mutation accounting", parameters=object_schema({}),
            function=change_then_stop, effect=Effect.WRITE,
        ))
        self.provider.enqueue(ModelResponse(tool_calls=[ToolCall(
            id="change", name="change_then_stop", arguments={},
        )]))
        result = await self.runner.run("change")
        self.assertEqual(result.stop_reason.value, "budget")
        self.assertIn("changed.txt", result.changed_files)
        messages = self.assert_finished_calls(result, {"change"})
        self.assertIn("changed.txt", messages["change"].metadata["changed_files"])
