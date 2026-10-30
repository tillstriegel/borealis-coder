"""Regression coverage for Script settlement at runner and provider boundaries."""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from borealis_coder.agent import build_runner
from borealis_coder.errors import BudgetExceeded, Cancelled, SessionError
from borealis_coder.models import Effect, ModelResponse, StopReason, ToolCall, ToolResult, Usage
from borealis_coder.protocol.acp import ACPServer
from borealis_coder.providers.mock import MockProvider
from borealis_coder.tools.base import FunctionTool, object_schema
from tests.helpers import make_config


class ScriptRepairTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = make_config(self.root)
        self.runner = await build_runner(self.root, config=self.config, interactive=False)
        self.addAsyncCleanup(self.runner.close)
        self.provider = cast(MockProvider, self.runner.providers[0].provider)

    def register(self, name, function, *, effect=Effect.READ):
        self.runner.tools.register(FunctionTool(name=name, description="Synthetic barrier fixture",
            parameters=object_schema({}), function=function, effect=effect, concurrent=True))

    def script(self, source, call_id="fixture-script"):
        return ToolCall(id=call_id, name="script", arguments={"source": source, "background": True})

    async def fatal_siblings(self, *, completion_failure=False, healthy=False, cancelled=False):
        writer_entered, fault_entered = asyncio.Event(), asyncio.Event()
        trigger, allow_write, allow_drain = asyncio.Event(), asyncio.Event(), asyncio.Event()
        writer_cancelled, writer_drained = asyncio.Event(), asyncio.Event()
        provider_blocked, provider_release = asyncio.Event(), asyncio.Event()

        async def writer(args, context):
            writer_entered.set()
            try:
                await allow_write.wait()
                (self.root / "writer.txt").write_text("healthy write")
            except asyncio.CancelledError:
                writer_cancelled.set()
                raise
            finally:
                await allow_drain.wait()
                writer_drained.set()
            return ToolResult("writer completed")

        async def fault(args, context):
            fault_entered.set()
            await trigger.wait()
            if cancelled:
                raise Cancelled("Synthetic primary cancellation")
            if not completion_failure and not healthy:
                raise BudgetExceeded("cost", "Synthetic primary fatal")
            return ToolResult("other job completed")

        self.register("fixture_writer", writer, effect=Effect.WRITE)
        self.register("fixture_fault", fault)
        original_emit = self.runner.events.emit
        async def emit(event_type, **data):
            if completion_failure and event_type == "script.completed":
                scope = next(iter(self.runner._script_scopes))
                job = scope.jobs[data["script_job"]["job_id"]]
                if job.parent_id == "fault-script":
                    raise OSError("Synthetic completion persistence failure")
            return await original_emit(event_type, **data)
        self.runner.events.emit = emit
        self.provider.enqueue(ModelResponse(tool_calls=[
            self.script('await tools.fixture_writer({});', "writer-script"),
            self.script('await tools.fixture_fault({});', "fault-script"),
        ]), ModelResponse(text="provider candidate"), ModelResponse(text="healthy final"))
        original_stream = self.provider.stream
        async def stream(request):
            if self.provider.calls == 1:
                await asyncio.wait_for(writer_entered.wait(), 3)
                await asyncio.wait_for(fault_entered.wait(), 3)
                provider_blocked.set()
                await provider_release.wait()
            async for event in original_stream(request):
                yield event
        self.provider.stream = stream
        task = asyncio.create_task(self.runner.run("Run two fixture jobs"))
        try:
            await asyncio.wait_for(provider_blocked.wait(), 5)
            scope = next(iter(self.runner._script_scopes))
            jobs = list(scope.jobs.values())
            writer_job = next(j for j in jobs if j.parent_id == "writer-script")
            trigger.set()
            if healthy:
                allow_write.set()
                allow_drain.set()
            else:
                await asyncio.wait_for(writer_cancelled.wait(), 2)
                self.assertIn(writer_job.id, self.runner._script_job_slots)
                self.assertFalse(writer_drained.is_set())
                allow_drain.set()
            await asyncio.wait_for(asyncio.gather(*(j.task for j in jobs if j.task is not None), return_exceptions=True), 3)
            self.assertTrue(writer_drained.is_set())
            self.assertFalse(provider_release.is_set())
            self.assertEqual(self.provider.calls, 1)
            self.assertFalse(self.runner._script_job_slots)
            for job in jobs:
                self.assertNotEqual(job.state, "running")
                self.assertTrue(all(child.done() for child in job.children))
                assert job.process is not None
                self.assertIsNotNone(job.process.returncode)
            session_id = scope.context.session_id
            self.assertTrue(all(row["status"] != "running" for row in self.runner.sessions.tool_calls(session_id)))
            if not healthy:
                self.assertFalse((self.root / "writer.txt").exists())
            provider_release.set()
            result = await asyncio.wait_for(task, 5)
            if healthy:
                self.assertIsNone(result.error)
                self.assertEqual((self.root / "writer.txt").read_text(), "healthy write")
            elif completion_failure:
                self.assertEqual(result.stop_reason, StopReason.ERROR)
                self.assertIn("completion could not be recorded", result.error or "")
            elif cancelled:
                self.assertEqual(result.stop_reason, StopReason.CANCELLED)
                self.assertIn("Synthetic primary cancellation", result.error or "")
            else:
                self.assertEqual(result.stop_reason, StopReason.BUDGET)
                self.assertIn("Synthetic primary fatal", result.error or "")
        finally:
            allow_write.set()
            allow_drain.set()
            trigger.set()
            provider_release.set()
            await asyncio.wait_for(task, 5)

    async def test_fatal_cancels_and_drains_admitted_writer_before_provider_returns(self):
        await self.fatal_siblings()

    async def test_completion_failure_cancels_and_drains_siblings(self):
        await self.fatal_siblings(completion_failure=True)

    async def test_nested_cancellation_drains_siblings_before_provider_returns(self):
        await self.fatal_siblings(cancelled=True)

    async def test_healthy_jobs_finish_while_provider_is_blocked(self):
        await self.fatal_siblings(healthy=True)

    async def final_turn(self, *, fail=False, request_budget=False):
        self.config.agent.max_turns = 2
        if request_budget:
            self.config.agent.max_model_requests = 1
        entered, release = asyncio.Event(), asyncio.Event()
        async def fixture(args, context):
            entered.set()
            await release.wait()
            return ToolResult("ordinary fixture failure" if fail else "fixture success", is_error=fail)
        self.register("fixture_wait", fixture)
        requests, updates = [], []
        server = ACPServer()
        class Connection:
            async def notify(self, method, params):
                updates.append(params)
        server.connection = cast(Any, Connection())
        session = self.runner.sessions.create_session(workspace=self.root, provider="mock", model="fixture")
        server._subscribe(session.id, self.runner)
        def handler(request, count):
            requests.append(request)
            if count == 1:
                return ModelResponse(tool_calls=[self.script('await tools.fixture_wait({});text("job settled");')])
            notices = [m for m in request.messages if m.metadata.get("internal") == "script_completion"]
            return ModelResponse(text=("Failure acknowledged" if fail else "Success acknowledged") if notices else "Stale candidate")
        self.provider.handler = handler
        task = asyncio.create_task(self.runner.run("Run final-turn fixture", session_id=session.id))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            # The test driver releases the fixture independently of the final request.
            release.set()
            result = await asyncio.wait_for(task, 5)
        finally:
            release.set()
            await asyncio.wait_for(task, 5)
        if request_budget:
            self.assertEqual(self.provider.calls, 1)
            self.assertEqual(result.stop_reason, StopReason.BUDGET)
            self.assertTrue(result.incomplete)
            self.assertFalse(result.text)
            self.assertFalse(any("Stale candidate" in json.dumps(u) for u in updates))
            return
        notices = [m for m in requests[1].messages if m.metadata.get("internal") == "script_completion"]
        self.assertEqual(len(notices), 1, "last request must see settled Script result")
        self.assertIn('"tool_errors":1' if fail else '"tool_errors":0', notices[0].content)
        self.assertIn("ordinary fixture failure" if fail else "job settled", notices[0].content)
        self.assertEqual(result.text, "Failure acknowledged" if fail else "Success acknowledged")
        self.assertEqual(result.stop_reason, StopReason.END_TURN)
        self.assertIsNone(result.error)
        self.assertEqual(self.provider.calls, 2)
        self.assertFalse(requests[1].tools)
        visible = [u["update"] for u in updates if u["update"].get("sessionUpdate") == "agent_message"]
        self.assertTrue(any(result.text in json.dumps(u) for u in visible), updates)

    async def test_last_turn_sees_background_success_and_publishes_acp_answer(self):
        await self.final_turn()

    async def test_last_turn_sees_background_failure_and_publishes_acp_answer(self):
        await self.final_turn(fail=True)

    async def test_last_turn_cannot_synthesize_without_remaining_request_budget(self):
        await self.final_turn(request_budget=True)

    async def test_completed_response_usage_settles_before_background_fatal(self):
        entered, trigger, blocked, release = (asyncio.Event() for _ in range(4))
        async def fault(args, context):
            entered.set()
            await trigger.wait()
            raise BudgetExceeded("cost", "Synthetic usage fatal")
        self.register("fixture_fault", fault)
        self.provider.enqueue(
            ModelResponse(tool_calls=[self.script('await tools.fixture_fault({});')],
                          usage=Usage(input_tokens=11, output_tokens=7, cost_usd=0.01)),
            ModelResponse(text="Must not publish", tool_calls=[ToolCall(name="write_file", arguments={
                "path": "forbidden.txt", "content": "no", "expected_sha256": None})],
                          usage=Usage(input_tokens=101, output_tokens=29, cost_usd=0.125)),
        )
        original_stream = self.provider.stream
        async def stream(request):
            if self.provider.calls == 1:
                await entered.wait()
                blocked.set()
                await release.wait()
            async for event in original_stream(request):
                yield event
        self.provider.stream = stream
        events = []
        async def observe(event):
            events.append(event)
        self.runner.events.subscribe(observe)
        task = asyncio.create_task(self.runner.run("Settle usage fixture"))
        try:
            await asyncio.wait_for(blocked.wait(), 4)
            scope = next(iter(self.runner._script_scopes))
            trigger.set()
            await asyncio.wait_for(asyncio.gather(*(j.task for j in scope.jobs.values() if j.task is not None)), 3)
        finally:
            trigger.set()
            release.set()
        result = await asyncio.wait_for(task, 5)
        self.assertEqual(result.stop_reason, StopReason.BUDGET)
        self.assertIn("Synthetic usage fatal", result.error or "")
        for usage in (result.usage, self.runner.sessions.usage(result.session_id)):
            self.assertEqual((usage.input_tokens, usage.output_tokens, usage.requests), (112, 36, 2))
            self.assertAlmostEqual(usage.cost_usd, 0.135)
        self.assertEqual(self.provider.calls, 2)
        self.assertFalse((self.root / "forbidden.txt").exists())
        self.assertFalse(any(e.type in {"model.completed", "model.text_delta"} and "Must not publish" in str(e.data) for e in events))
        self.assertFalse(any(r["tool_name"] == "write_file" for r in self.runner.sessions.tool_calls(result.session_id)))

    async def test_background_preview_is_bounded_and_full_redacted_artifact_is_owned(self):
        self.config.context.tool_output_chars = 480
        entered, release = asyncio.Event(), asyncio.Event()
        async def fixture(args, context):
            entered.set()
            await release.wait()
            return ToolResult("ready")
        self.register("fixture_wait", fixture)
        secret = "sk-proj-" + "a" * 60
        payload = "x" * 6000 + "\n" + secret + "\nRETAINED_TAIL"
        requests = []
        contexts = []
        def observe(event):
            if event.type == "script.completed":
                contexts.extend(scope.context for scope in self.runner._script_scopes)
        self.runner.events.subscribe(observe)
        def handler(request, count):
            requests.append(request)
            if count == 1:
                return ModelResponse(tool_calls=[self.script('await tools.fixture_wait({});text(' + json.dumps(payload) + ');')])
            if count == 2:
                release.set()
                return ModelResponse(text="early")
            return ModelResponse(text="settled")
        original_stream = self.provider.stream
        async def stream(request):
            if self.provider.calls == 1:
                await asyncio.wait_for(entered.wait(), 3)
            async for event in original_stream(request):
                yield event
        self.provider.stream, self.provider.handler = stream, handler
        result = await asyncio.wait_for(self.runner.run("Bound background preview"), 5)
        notice = next(m for m in requests[2].messages if m.metadata.get("internal") == "script_completion")
        self.assertLessEqual(len(notice.content), 480)
        self.assertNotIn(secret, json.dumps(notice.to_dict()))
        artifact = notice.metadata["output_artifact"]
        self.assertEqual(notice.metadata["script_job"]["state"], "completed")
        stored = self.runner.sessions.read_output_artifact(result.session_id, self.root, artifact["artifact_id"], limit=16000)
        self.assertIn("RETAINED_TAIL", stored["content"])
        self.assertIn("x" * 6000, stored["content"])
        self.assertNotIn(secret, stored["content"])
        self.assertEqual(stored["source"], notice.metadata["script_job"]["job_id"])
        self.assertTrue(self.runner.sessions.search_output_artifacts(result.session_id, self.root, "RETAINED_TAIL"))
        other = self.runner.sessions.create_session(workspace=self.root, provider="mock", model="fixture")
        with self.assertRaises(SessionError):
            self.runner.sessions.read_output_artifact(other.id, self.root, artifact["artifact_id"])
        read = self.runner.tools.get("read_artifact")
        search = self.runner.tools.get("search_history")
        assert read is not None and search is not None
        found = await read.execute({"artifact_id": artifact["artifact_id"], "offset": 0,
                                   "limit": 8000, "query": "RETAINED_TAIL"}, contexts[0])
        self.assertFalse(found.is_error)
        self.assertIn("RETAINED_TAIL", found.output)
        references = await search.execute({"query": "RETAINED_TAIL", "after": 0}, contexts[0])
        self.assertIn(artifact["artifact_id"], references.output)
        refused = await read.execute({"artifact_id": artifact["artifact_id"], "offset": 0,
                                     "limit": 100, "query": ""}, replace(contexts[0], session_id=other.id))
        self.assertTrue(refused.is_error)
