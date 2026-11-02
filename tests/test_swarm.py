from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from borealis_coder.agent import build_runner, validate_tool_call_order
from borealis_coder.config import ConfigurationError, load_config
from borealis_coder.errors import (
    ProviderContextOverflowError,
    ProviderError,
    ProviderUnavailableError,
)
from borealis_coder.models import Effect, Message, ModelResponse, Role, ToolCall, ToolResult, Usage
from borealis_coder.providers.mock import MockProvider
from borealis_coder.tools import FunctionTool, object_schema
from borealis_coder.tools.delegate import DelegateTaskTool
from borealis_coder.tools.verification import VerificationReport
from tests.helpers import make_config


def delegate(task="child", turns=3, identifier="delegate"):
    return ToolCall(id=identifier, name="delegate_task", arguments={"task": task, "max_turns": turns})


def write(path="child.txt", content="child evidence\n", expected=None):
    return ToolCall(id="write", name="write_file", arguments={
        "path": path, "content": content, "expected_sha256": expected,
    })


def response(text="", calls=(), cost=0.01):
    return ModelResponse(text=text, tool_calls=list(calls), usage=Usage(
        requests=1, input_tokens=10, output_tokens=2, cost_usd=cost, cost_status="known",
    ))


class SwarmTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.runner = await build_runner(self.root, config=make_config(self.root), interactive=False)
        self.addAsyncCleanup(self.runner.close)
        provider = self.runner.providers[0].provider
        assert isinstance(provider, MockProvider)
        self.provider = provider
        self.events = []
        self.runner.events.subscribe(self.events.append)

    def starts(self):
        return [e for e in self.events if e.type == "delegate.started"]

    def terminals(self):
        return [e for e in self.events if e.type in {"delegate.completed", "delegate.failed", "delegate.cancelled"}]

    def assert_settled(self, result):
        self.assertCountEqual([e.data["delegation_id"] for e in self.starts()],
                              [e.data["delegation_id"] for e in self.terminals()])
        rows = self.runner.sessions.tool_calls(result.session_id)
        self.assertNotIn("running", {row["status"] for row in rows})
        self.assertTrue(all(row["metadata"].get("delegation_id") for row in rows
                            if row["run_id"] != result.run_id))
        validate_tool_call_order(self.runner.sessions.messages(result.session_id))
        self.assertEqual(result.usage.to_dict(), self.runner.sessions.usage(result.session_id).to_dict())
        if self.terminals():
            self.assertEqual(self.terminals()[-1].data["limits"]["active_requests"], 0)

    async def test_three_levels_compose_evidence_exact_once_and_single_slot(self):
        self.runner.config.agent.swarm_max_concurrent = 1
        requests = []

        def handler(request, call):
            requests.append(request)
            depth = request.metadata.get("delegation_depth", 0)
            if request.messages[-1].role == Role.TOOL:
                return response(text="Evidence: " + request.messages[-1].content)
            if depth < 2:
                return response(calls=[delegate("grandchild" if depth else "child")])
            return response(calls=[ToolCall(id="read", name="list_directory", arguments={
                "path": ".", "depth": 1, "include_hidden": False, "max_entries": 10,
            })])

        (self.root / "evidence.txt").write_text("source")
        self.provider.handler = handler
        result = await asyncio.wait_for(self.runner.run("Investigate"), 5)
        self.assertEqual(result.stop_reason.value, "end_turn")
        self.assertIn("evidence.txt", result.text)
        self.assertEqual(result.usage.requests, 6)
        self.assertAlmostEqual(result.usage.cost_usd, 0.06)
        self.assertEqual([e.data["depth"] for e in self.starts()], [1, 2])
        self.assertEqual(self.starts()[1].data["parent_delegation_id"], self.starts()[0].data["delegation_id"])
        self.assertEqual(self.starts()[1].data["parent_run_id"], self.starts()[0].data["delegated_run_id"])
        self.assertTrue(all(e.data["limits"]["active_requests"] <= 1 for e in self.events if e.type.startswith("delegate.")))
        self.assertEqual(requests[1].messages[0].content, "child")
        self.assertEqual(requests[2].messages[0].content, "grandchild")
        self.assertNotIn("Investigate", requests[1].system)
        self.assert_settled(result)

    async def test_depth_total_disabled_and_fresh_run(self):
        for depth, total, expected in ((0, 8, 0), (2, 0, 0), (1, 8, 1), (2, 1, 1), (2, 8, 2)):
            with self.subTest(depth=depth, total=total):
                self.events.clear()
                self.runner.config.agent.swarm_max_depth = depth
                self.runner.config.agent.swarm_max_descendants = total

                def handler(request, call):
                    if request.messages[-1].role == Role.TOOL:
                        return response(text="Concluded: " + request.messages[-1].content)
                    return response(calls=[delegate()])

                self.provider.handler = handler
                result = await self.runner.run("Nested bounds")
                self.assertEqual(len(self.starts()), expected)
                self.assertEqual(len([e for e in self.events if e.type == "delegate.refused"]), 1)
                self.assertEqual(result.usage.requests, 2 + 2 * expected)
                self.assert_settled(result)

    async def test_parallel_and_sequential_cumulative_admission(self):
        self.runner.config.agent.swarm_max_descendants = 2
        self.runner.config.agent.parallel_reads = 8

        def handler(request, call):
            if request.metadata.get("delegated"):
                return response(text="Child complete")
            batches = sum(bool(message.tool_calls) for message in request.messages)
            if batches < 2:
                return response(calls=[delegate(f"task {batches}-{i}", 1, f"batch{batches}-{i}") for i in range(5)])
            return response(text="Done")

        self.provider.handler = handler
        result = await self.runner.run("Bound racing siblings")
        self.assertEqual(len(self.starts()), 2)
        self.assertEqual(len([e for e in self.events if e.type == "delegate.refused"]), 8)
        self.assertEqual(result.usage.requests, 5)
        self.assert_settled(result)

    async def test_turn_ceiling_cannot_be_raised_by_model(self):
        self.runner.config.agent.swarm_max_turns = 1
        seen = []

        def handler(request, call):
            if call == 1:
                return response(calls=[delegate(turns=16)])
            if request.metadata.get("delegated"):
                seen.append(request)
                return response(calls=[write()])
            return response(text="Incomplete child")

        self.provider.handler = handler
        result = await self.runner.run("Turn ceiling")
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].tools, [])
        self.assertFalse((self.root / "child.txt").exists())
        self.assertEqual(self.terminals()[0].type, "delegate.failed")
        self.assert_settled(result)

    async def test_opt_in_child_and_grandchild_writes_are_durable_and_verified(self):
        self.runner.config.agent.swarm_allow_writes = True
        self.runner.config.agent.auto_verify = True
        verify = []

        async def verify_changes(context, steps):
            verify.append(set(context.changed_files))
            return VerificationReport(ok=True)

        def handler(request, call):
            depth = request.metadata.get("delegation_depth", 0)
            if request.messages[-1].role == Role.TOOL:
                return response(text="Changes complete; " + request.messages[-1].content)
            if depth == 0:
                return response(calls=[delegate()])
            if depth == 1:
                return response(calls=[write("child.txt"), delegate("grandchild")])
            return response(calls=[ToolCall(id="patch", name="apply_patch", arguments={
                "patch": "*** Begin Patch\n*** Add File: grandchild.txt\n+grandchild evidence\n*** End Patch",
            })])

        self.provider.handler = handler
        with patch("borealis_coder.agent.runner.VerificationPlanner.run", side_effect=verify_changes):
            result = await asyncio.wait_for(self.runner.run("Implement files"), 5)
        self.assertEqual((self.root / "child.txt").read_text(), "child evidence\n")
        self.assertEqual((self.root / "grandchild.txt").read_text(), "grandchild evidence\n")
        self.assertEqual(set(result.changed_files), {"child.txt", "grandchild.txt"})
        self.assertEqual(verify, [{"child.txt", "grandchild.txt"}])
        assert result.verification is not None
        self.assertTrue(result.verification["checks_ok"])
        self.assertTrue(result.verification["ok"])
        self.assertEqual(result.mutation_tracking, "complete")
        self.assertEqual(result.verification["mutation_tracking"], "complete")
        rows = self.runner.sessions.tool_calls(result.session_id)
        writes = [row for row in rows if row["tool_name"] in {"write_file", "apply_patch"}]
        self.assertEqual(len(writes), 2)
        self.assertTrue(all(row["metadata"]["delegation_id"] for row in writes))
        self.assertTrue(all(row["metadata"]["checkpoint_id"] for row in writes))
        root_messages = self.runner.sessions.messages(result.session_id)
        self.assertFalse(any(m.tool_name in {"write_file", "apply_patch"} for m in root_messages))
        self.assertTrue(all(e.data["capability"] == "write" for e in self.starts()))
        self.assert_settled(result)

    async def test_steered_nested_write_invalidates_verification_on_request_budget(self):
        await self._assert_steered_write_interrupted("requests")

    async def test_steered_nested_write_invalidates_verification_on_cost_budget(self):
        await self._assert_steered_write_interrupted("cost")

    async def test_steered_nested_write_invalidates_verification_on_cancel(self):
        await self._assert_steered_write_interrupted("cancel")

    async def _assert_steered_write_interrupted(self, interruption):
        self.runner.config.agent.swarm_allow_writes = True
        self.runner.config.agent.auto_verify = True
        if interruption == "requests":
            # Include ancestor closure reservations; still stop on the exact hard
            # request bound after a real nested mutation, not before admission.
            self.runner.config.agent.max_model_requests = 9
        if interruption == "cost":
            self.runner.config.agent.max_cost_usd = 0.1
        after_write = asyncio.Event()
        drained = asyncio.Event()
        original = self.provider.complete

        def handler(request, call):
            depth = request.metadata.get("delegation_depth", 0)
            if depth == 2:
                return response(calls=[write("latest.txt", "latest revision\n")])
            if depth == 1:
                if interruption == "requests" and request.messages[-1].role == Role.TOOL:
                    return response(text="Nested change observed; current verification is required.")
                return response(calls=[delegate("grandchild")])
            if call == 1:
                return response(calls=[write("first.txt", "verified revision\n")])
            if call == 2:
                return response(text="Initial candidate.")
            if call == 3:
                return response(text="Checks passed for the first revision.")
            if call == 4:
                return response(calls=[delegate()])
            if interruption == "requests" and call == 8:
                return response(text="<tool_call>not a conclusion")
            self.fail("Unexpected provider work after interrupted delegation")

        async def controlled(request):
            if request.metadata.get("delegation_depth") == 2 and request.messages[-1].role == Role.TOOL:
                # Gate only after the real nested writer and durable finalizer returned.
                rows = self.runner.sessions.tool_calls(request.metadata["parent_session_id"])
                self.assertTrue(any(r["tool_name"] == "write_file" and r["run_id"] != rows[0]["run_id"]
                                    and r["status"] == "completed" for r in rows))
                after_write.set()
                if interruption == "cancel":
                    try:
                        await asyncio.Event().wait()
                    finally:
                        drained.set()
                if interruption == "requests":
                    return response(text="Nested write observed; remaining checks incomplete.")
                self.assertEqual(interruption, "cost")
                return response(text="Billed over budget", cost=0.2)
            return await original(request)

        steered = []

        def steer(event):
            if event.type == "verification.completed" and not steered:
                steered.append(event.session_id)
                self.runner.steer(event.session_id, "Make a nested change after verification.")

        self.runner.events.subscribe(steer)
        self.provider.handler = handler
        with patch("borealis_coder.agent.runner.VerificationPlanner.run", new_callable=AsyncMock,
                   return_value=VerificationReport(ok=True)) as verify, patch.object(
                       self.provider, "complete", side_effect=controlled):
            task = asyncio.create_task(self.runner.run("Make and verify the first change"))
            if interruption == "cancel":
                await asyncio.wait_for(after_write.wait(), 3)
                task.cancel()
            result = await asyncio.wait_for(task, 5)
        self.assertEqual(len(steered), 1)
        self.assertEqual(verify.await_count, 1)
        self.assertEqual(result.stop_reason.value, "cancelled" if interruption == "cancel" else "budget", result.error)
        self.assertEqual((self.root / "latest.txt").read_text(), "latest revision\n")
        self.assertEqual(set(result.changed_files), {"first.txt", "latest.txt"})
        # A dispatched cancelled request is settled once as unknown usage, not omitted.
        self.assertEqual(result.usage.requests, 9 if interruption == "requests" else 7)
        self.assertAlmostEqual(result.usage.cost_usd, 0.26 if interruption == "cost" else 0.09 if interruption == "requests" else 0.06)
        self.assertEqual(drained.is_set(), interruption == "cancel")
        self.assert_settled(result)
        assert result.verification is not None
        self.assertFalse(result.verification["checks_ok"])
        self.assertFalse(result.verification["ok"])
        self.assertEqual(result.verification["steps"], [])
        self.assertIn("latest state was not verified", result.verification["error"])
        self.assertIn("latest state was not verified", result.text)
        self.assertNotIn("Checks passed for the first revision", result.text)

    async def test_resume_recovers_nested_delegated_write_without_replay(self):
        await self._assert_abandoned_delegation_recovered(write_effect=True)

    async def test_resume_recovers_nested_delegated_read_without_mutation_uncertainty(self):
        await self._assert_abandoned_delegation_recovered(write_effect=False)

    async def _assert_abandoned_delegation_recovered(self, *, write_effect):
        config = self.runner.config
        config.agent.auto_verify = True
        config.agent.swarm_allow_writes = True
        session = self.runner.sessions.create_session(workspace=self.root, provider="mock", model="fixture")
        root_call = delegate(identifier="crashed-delegate")
        original = Message(role=Role.ASSISTANT, tool_calls=[root_call])
        self.runner.sessions.append_message(session.id, original)
        self.runner.sessions.start_tool_call(session.id, "crashed-root", root_call.id,
                                             root_call.name, root_call.arguments)
        nested = [ToolCall(id="child.read", name="read_file", arguments={"path": "partial.txt"})]
        if write_effect:
            nested.append(write("partial.txt", "survived crash\n"))
            nested[-1].id = "grandchild.write"
        lineage = {}
        for depth, call in enumerate(nested, start=1):
            lineage[call.id] = {"delegation_id": f"delegation-{depth}", "delegation_depth": depth,
                                "fixture_evidence": "preserve attribution"}
            self.runner.sessions.start_tool_call(session.id, f"crashed-child-{depth}", call.id,
                                                 call.name, call.arguments, metadata=lineage[call.id])
        # True crash window: effects reached disk but no durable receipt/finalizer ran.
        if write_effect:
            (self.root / "partial.txt").write_text("survived crash\n")
        await self.runner.close()

        previous_rows = None
        previous_usage = Usage()
        for _attempt in range(2):
            runner = await build_runner(self.root, config=config, interactive=False)
            try:
                provider = runner.providers[0].provider
                assert isinstance(provider, MockProvider)
                requests = []

                def handler(request, call, requests=requests):
                    validate_tool_call_order(request.messages)
                    self.assertFalse(any(m.role == Role.TOOL and m.tool_call_id in lineage
                                         for m in request.messages))
                    requests.append(request)
                    return response(text="Resume candidate." if call == 1 else "Resume finalizer.")

                provider.handler = handler
                with patch("borealis_coder.agent.runner.VerificationPlanner.run", new_callable=AsyncMock,
                           return_value=VerificationReport(ok=True)) as verify:
                    result = await runner.run("continue", session_id=session.id)
                rows = runner.sessions.tool_calls(session.id)
                self.assertNotIn("running", {r["status"] for r in rows})
                for row in rows:
                    self.assertEqual(row["status"], "error")
                    self.assertTrue(row["metadata"]["unknown_outcome"])
                    if row["tool_call_id"] in lineage:
                        for key, value in lineage[row["tool_call_id"]].items():
                            self.assertEqual(row["metadata"][key], value)
                        self.assertEqual(row["metadata"].get("workspace_change_tracking") == "incomplete",
                                         row["tool_name"] == "write_file")
                        self.assertEqual(row["run_id"], f"crashed-child-{row['metadata']['delegation_depth']}")
                if previous_rows is not None:
                    self.assertEqual(rows, previous_rows, "Recovery must not rewrite terminal receipts")
                previous_rows = rows
                messages = runner.sessions.messages(session.id)
                self.assertEqual(messages[0], original)
                validate_tool_call_order(messages)
                self.assertEqual([m.tool_call_id for m in messages if m.role == Role.TOOL], [root_call.id])
                self.assertEqual(verify.await_count, int(write_effect))
                self.assertEqual(result.mutation_tracking, "incomplete" if write_effect else "complete")
                self.assertEqual(result.changed_files, [])  # Unknown effects are not invented diffs.
                if write_effect:
                    assert result.verification is not None
                    # Passing checks do not resolve unknown mutation coverage.
                    self.assertTrue(result.verification["checks_ok"])
                    self.assertEqual(result.verification["mutation_tracking"], "incomplete")
                    self.assertIn("Mutation tracking: incomplete", result.text)
                    self.assertEqual((self.root / "partial.txt").read_text(), "survived crash\n")
                else:
                    self.assertIsNone(result.verification)
                    self.assertFalse((self.root / "partial.txt").exists())
                self.assertEqual(len(requests), 2 if write_effect else 1)
                self.assertEqual(result.usage.requests, len(requests))
                self.assertAlmostEqual(result.usage.cost_usd, 0.01 * len(requests))
                previous_usage.add(result.usage)
                self.assertEqual(runner.sessions.usage(session.id).to_dict(), previous_usage.to_dict())
                self.assertEqual(len(rows), 3 if write_effect else 2)
                self.assertEqual(result.stop_reason.value, "end_turn")
            finally:
                await runner.close()

    async def test_read_only_and_plan_and_paths_and_approval_remain_enforced(self):
        for writes, mode, path, approval in (
            (False, "workspace-write", "blocked.txt", "never"),
            (True, "plan", "blocked.txt", "never"),
            (True, "workspace-write", ".git/config", "never"),
            (True, "workspace-write", "../outside.txt", "never"),
            (True, "workspace-write", "blocked.txt", "always"),
            (True, "workspace-write", "allowed.txt", "never"),
        ):
            with self.subTest(writes=writes, mode=mode, path=path, approval=approval):
                self.events.clear()
                self.runner.config.agent.swarm_allow_writes = writes
                self.runner.config.safety.mode = mode
                self.runner.config.safety.approval = approval
                child_results = []
                self.runner.tool_context.policy.interactive = True
                self.runner.tool_context.approvals.callback = lambda request: request.tool_name != "write_file"

                def handler(request, call, child_results=child_results, path=path):
                    if request.messages[-1].role == Role.TOOL:
                        if request.metadata.get("delegated"):
                            child_results.append(request.messages[-1])
                        return response(text="Reviewed")
                    return response(calls=[write(path)]) if request.metadata.get("delegated") else response(calls=[delegate()])

                self.provider.handler = handler
                result = await self.runner.run("Check authority")
                allowed = path == "allowed.txt"
                self.assertEqual(child_results[0].is_error, not allowed)
                self.assertEqual(set(result.changed_files), {"allowed.txt"} if allowed else set())
                self.assertFalse((self.root / "blocked.txt").exists())
                self.assert_settled(result)

    async def test_model_cannot_enable_descendant_writes_or_raise_bounds(self):
        def handler(request, call):
            if request.messages[-1].role == Role.TOOL:
                return response(text="Concluded")
            if request.metadata.get("delegated"):
                elevated = delegate("escalate")
                elevated.arguments["allow_writes"] = True
                elevated.arguments["max_depth"] = 16
                return response(calls=[elevated])
            return response(calls=[delegate()])

        self.provider.handler = handler
        result = await self.runner.run("Do not elevate")
        self.assertEqual(len(self.starts()), 1)
        self.assertEqual(self.starts()[0].data["capability"], "read-only")
        errors = [row for row in self.runner.sessions.tool_calls(result.session_id) if row["status"] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("unknown keys", errors[0]["output"])
        self.assertEqual(result.changed_files, [])
        self.assert_settled(result)

    async def test_conflicting_writes_keep_hash_guards_and_serialize(self):
        from borealis_coder.util import sha256_text

        self.runner.config.agent.swarm_allow_writes = True
        (self.root / "shared.txt").write_text("before")

        def handler(request, call):
            if request.messages[-1].role == Role.TOOL:
                return response(text="Concluded")
            if not request.metadata.get("delegated"):
                return response(calls=[delegate("one", identifier="one"), delegate("two", identifier="two")])
            return response(calls=[write("shared.txt", request.messages[0].content, sha256_text("before"))])

        self.provider.handler = handler
        result = await self.runner.run("Conflict")
        rows = [row for row in self.runner.sessions.tool_calls(result.session_id) if row["tool_name"] == "write_file"]
        self.assertCountEqual([row["status"] for row in rows], ["completed", "error"])
        self.assertTrue(any("Stale write rejected" in row["output"] for row in rows))
        self.assertIn((self.root / "shared.txt").read_text(), {"one", "two"})
        self.assertEqual(result.changed_files, ["shared.txt"])
        self.assert_settled(result)

    async def test_provider_errors_overflow_retries_and_single_slot_release(self):
        self.runner.config.agent.swarm_max_concurrent = 1
        self.provider.config.max_retries = 1
        self.provider.config.initial_backoff_seconds = 0
        self.provider.config.max_backoff_seconds = 0
        attempts = {}

        def handler(request, call):
            if not request.metadata.get("delegated"):
                return response(text="Done") if request.messages[-1].role == Role.TOOL else response(calls=[
                    delegate("retry", 1, "retry"), delegate("overflow", 1, "overflow"), delegate("failure", 1, "failure"),
                ])
            task = request.messages[0].content
            attempts[task] = attempts.get(task, 0) + 1
            if task == "failure":
                raise ProviderError("Billed failure", usage=Usage(cost_usd=0.03, requests=1))
            if attempts[task] == 1:
                cls = ProviderUnavailableError if task == "retry" else ProviderContextOverflowError
                raise cls("Billed recoverable error", retryable=task == "retry", usage=Usage(cost_usd=0.02, requests=1))
            return response(text="Recovered")

        self.provider.handler = handler
        result = await asyncio.wait_for(self.runner.run("Recover"), 5)
        self.assertEqual(attempts, {"retry": 2, "overflow": 2, "failure": 1})
        self.assertEqual(result.usage.requests, 7)
        self.assertAlmostEqual(result.usage.cost_usd, 0.11)
        self.assertCountEqual([e.type for e in self.terminals()], ["delegate.completed", "delegate.completed", "delegate.failed"])
        self.assert_settled(result)

    async def test_recursive_fatal_budget_cancels_and_drains_billed_sibling(self):
        blocked = asyncio.Event()
        drained = asyncio.Event()
        original = self.provider.complete

        async def controlled(request):
            task = request.messages[0].content
            depth = request.metadata.get("delegation_depth", 0)
            if depth == 2 and task == "blocked":
                blocked.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    drained.set()
            if depth == 2 and task == "fail":
                await blocked.wait()
                return response(text="Over budget", cost=0.2)
            return await original(request)

        def handler(request, call):
            if not request.metadata.get("delegated"):
                return response(calls=[delegate()])
            return response(calls=[delegate("blocked", identifier="blocked"), delegate("fail", identifier="fail")])

        self.provider.handler = handler
        self.runner.config.agent.max_cost_usd = 0.1
        with patch.object(self.provider, "complete", side_effect=controlled):
            result = await asyncio.wait_for(self.runner.run("Fatal nested budget"), 5)
        self.assertTrue(drained.is_set())
        self.assertEqual(result.stop_reason.value, "budget")
        self.assertEqual(result.usage.requests, 4)
        self.assertAlmostEqual(result.usage.cost_usd, 0.22)
        self.assert_settled(result)

    async def test_cancellation_during_nested_start_persistence_leaves_no_running_row(self):
        entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        original = self.runner.sessions.start_tool_call

        def blocked_start(session_id, run_id, call_id, name, arguments, **kwargs):
            if name == "list_directory":
                loop.call_soon_threadsafe(entered.set)
                if not release.wait(timeout=3):
                    raise RuntimeError("Test did not release nested persistence")
            return original(session_id, run_id, call_id, name, arguments, **kwargs)

        self.provider.enqueue(response(calls=[delegate()]), response(calls=[ToolCall(
            name="list_directory", arguments={"path": ".", "depth": 1, "include_hidden": False, "max_entries": 10},
        )]))
        with patch.object(self.runner.sessions, "start_tool_call", side_effect=blocked_start):
            task = asyncio.create_task(self.runner.run("Cancel nested persistence"))
            try:
                await asyncio.wait_for(entered.wait(), 3)
                task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(task.done())
            finally:
                release.set()
            result = await asyncio.wait_for(task, 3)
        self.assertEqual(result.stop_reason.value, "cancelled")
        self.assert_settled(result)

    async def test_cancel_recursive_subtree_drains_and_releases_all_slots(self):
        entered = asyncio.Event()
        drained = asyncio.Event()
        original = self.provider.complete

        async def controlled(request):
            if request.metadata.get("delegation_depth") == 2:
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    drained.set()
            return await original(request)

        self.provider.handler = lambda request, call: response(calls=[delegate()])
        self.runner.config.agent.swarm_max_concurrent = 1
        with patch.object(self.provider, "complete", side_effect=controlled):
            task = asyncio.create_task(self.runner.run("Cancel nested"))
            await asyncio.wait_for(entered.wait(), 3)
            task.cancel()
            result = await asyncio.wait_for(task, 3)
        self.assertTrue(drained.is_set())
        self.assertEqual(result.stop_reason.value, "cancelled")
        self.assertEqual(result.usage.requests, 3)
        self.assert_settled(result)
        self.events.clear()
        self.provider.handler = lambda request, call: response(text="Fresh") if request.metadata.get("delegated") or request.messages[-1].role == Role.TOOL else response(calls=[delegate()])
        fresh = await asyncio.wait_for(self.runner.run("Fresh run"), 3)
        self.assertEqual(self.starts()[0].data["limits"]["admitted"], 1)
        self.assert_settled(fresh)

    async def test_retries_share_root_request_budget(self):
        self.runner.config.agent.max_model_requests = 2
        self.provider.config.max_retries = 2
        self.provider.config.initial_backoff_seconds = 0
        self.provider.config.max_backoff_seconds = 0

        def handler(request, call):
            if call == 1:
                return response(calls=[delegate()])
            raise ProviderUnavailableError("Billed retry", retryable=True,
                                           usage=Usage(cost_usd=0.02, requests=1))

        self.provider.handler = handler
        result = await self.runner.run("Bound retries")
        self.assertEqual(self.provider.calls, 2)
        self.assertEqual(result.usage.requests, 2)
        self.assertEqual(result.stop_reason.value, "budget")
        self.assertAlmostEqual(result.usage.cost_usd, 0.03)
        self.assert_settled(result)

    async def test_provider_concurrency_caps_racing_siblings(self):
        self.runner.config.agent.swarm_max_concurrent = 2
        waiting = asyncio.Event()
        two_started = asyncio.Event()
        third_started = asyncio.Event()
        release = asyncio.Event()
        active = 0
        maximum = 0
        entered = 0
        original = self.provider.complete

        def observe(event):
            if (event.type == "delegate.state" and event.data.get("state") == "waiting"
                    and sum(e.type == "delegate.state" and e.data.get("state") == "waiting" for e in self.events) == 3):
                waiting.set()

        self.runner.events.subscribe(observe)

        async def controlled(request):
            nonlocal active, maximum, entered
            if request.metadata.get("delegated"):
                active += 1
                entered += 1
                maximum = max(maximum, active)
                if entered == 2:
                    two_started.set()
                if entered == 3:
                    third_started.set()
                try:
                    await release.wait()
                    return response(text="Complete")
                finally:
                    active -= 1
            return await original(request)

        self.provider.handler = lambda request, call: response(text="Done") if request.messages[-1].role == Role.TOOL else response(calls=[delegate(str(i), 1, str(i)) for i in range(3)])
        with patch.object(self.provider, "complete", side_effect=controlled):
            task = asyncio.create_task(self.runner.run("Concurrent work"))
            try:
                await asyncio.wait_for(asyncio.gather(waiting.wait(), two_started.wait()), 3)
                self.assertFalse(third_started.is_set())
                self.assertEqual(active, 2)
            finally:
                release.set()
            result = await asyncio.wait_for(task, 3)
        self.assertTrue(third_started.is_set())
        self.assertEqual(maximum, 2)
        self.assertEqual(active, 0)
        self.assert_settled(result)

    async def test_recursive_request_limit_starts_no_extra_provider_work(self):
        self.runner.config.agent.max_model_requests = 2
        self.provider.handler = lambda request, call: response(calls=[delegate()])
        result = await self.runner.run("Request bound")
        self.assertEqual(self.provider.calls, 2)
        self.assertEqual(result.stop_reason.value, "budget")
        self.assertEqual(result.usage.requests, 2)
        self.assert_settled(result)

    async def test_script_aliases_and_unapproved_tool_classes_are_denied(self):
        class AliasDelegate(DelegateTaskTool):
            name = "delegate_alias"
            script_accessible = True

        self.runner.tools.register(AliasDelegate())
        attempted = []
        self.runner.tools.register(FunctionTool(
            name="plugin_write", description="Unapproved writer", parameters=object_schema({}),
            function=lambda args, context: attempted.append("mutated") or ToolResult("bad"),
            effect=Effect.WRITE,
        ))
        self.runner.config.agent.swarm_allow_writes = True
        seen = []

        def handler(request, call):
            if request.messages[-1].role == Role.TOOL:
                if request.metadata.get("delegated"):
                    seen.extend(m for m in request.messages if m.role == Role.TOOL)
                return response(text="Concluded")
            if request.metadata.get("delegated"):
                names = {tool["name"] for tool in request.tools}
                self.assertNotIn("script", names)
                self.assertNotIn("shell", names)
                self.assertNotIn("plugin_write", names)
                return response(calls=[ToolCall(name="plugin_write", arguments={}), ToolCall(name="script", arguments={"source": "text('no')", "background": False})])
            return response(calls=[delegate()])

        self.provider.handler = handler
        result = await self.runner.run("Check tool boundary")
        self.assertEqual(attempted, [])
        self.assertTrue(all(m.is_error for m in seen))
        self.assert_settled(result)
        self.provider.handler = None
        self.provider.enqueue(response(calls=[ToolCall(name="script", arguments={
            "source": 'await tools.delegate_alias({task:"forbidden", max_turns:1});', "background": False,
        })]), response(text="Script denied"))
        self.events.clear()
        result = await self.runner.run("Script alias")
        self.assertEqual(self.starts(), [])
        self.assertIn("delegation are unavailable", next(row["output"] for row in self.runner.sessions.tool_calls(result.session_id) if row["tool_name"] == "script"))


class SwarmConfigurationTests(unittest.TestCase):
    def test_validated_trusted_configuration(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {}, clear=True):
            root = Path(td)
            for key, value in (("swarm_max_depth", -1), ("swarm_max_depth", 17),
                               ("swarm_max_descendants", 1.5), ("swarm_max_concurrent", 0),
                               ("swarm_max_turns", 17), ("swarm_max_depth", True),
                               ("swarm_allow_writes", "yes")):
                with self.subTest(key=key, value=value), self.assertRaises(ConfigurationError):
                    make_config(root, agent={key: value})
            config = make_config(root, agent={"swarm_allow_writes": True, "swarm_max_depth": 3})
            self.assertTrue(config.agent.swarm_allow_writes)
            (root / ".borealis").mkdir(exist_ok=True)
            (root / ".borealis/config.toml").write_text("[agent]\nswarm_allow_writes=true\nswarm_max_depth=3\n")
            with self.assertRaisesRegex(ConfigurationError, "Workspace authority-bearing"):
                load_config(root)
