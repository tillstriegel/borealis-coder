from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from borealis_coder.agent import build_runner, validate_tool_call_order
from borealis_coder.models import Role, ToolCall
from borealis_coder.providers.mock import MockProvider
from borealis_coder.tools.verification import VerificationReport
from tests.helpers import make_config
from tests.test_swarm import delegate, response, write


class TerminalCancellationOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def test_execution_cancellation_keeps_effects_when_terminal_delivery_is_cancelled_again(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            runner = await build_runner(root, config=make_config(
                root, agent={"swarm_allow_writes": True}), interactive=False)
            provider = runner.providers[0].provider
            assert isinstance(provider, MockProvider)
            executing, publishing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
            events = []
            runner.events.subscribe(events.append)
            original = provider.complete

            async def complete(request):
                if request.metadata.get("delegated") and request.messages[-1].role == Role.TOOL:
                    executing.set()
                    await asyncio.Event().wait()
                return await original(request)

            async def observe(event):
                if event.type == "model.completed" and event.data.get("stop_reason") == "cancelled":
                    publishing.set()
                    await release.wait()

            runner.events.subscribe(observe)
            provider.handler = lambda request, call: response(calls=[
                write("survived.txt", "retained evidence\n") if request.metadata.get("delegated") else delegate(capability="write")
            ])
            try:
                with patch.object(provider, "complete", side_effect=complete):
                    task = asyncio.create_task(runner.run("Write then interrupt"))
                    try:
                        await asyncio.wait_for(executing.wait(), 3)
                        task.cancel()
                        await asyncio.wait_for(publishing.wait(), 3)
                        task.cancel()
                        await asyncio.sleep(0)
                        release.set()
                        result = await asyncio.wait_for(task, 5)
                    finally:
                        release.set()
                        if not task.done():
                            task.cancel()
                            await task
                self.assertEqual(result.stop_reason.value, "cancelled")
                self.assertTrue(result.incomplete)
                self.assertEqual(result.changed_files, ["survived.txt"])
                # Two completed responses plus the dispatched/interrupted child request.
                self.assertEqual(result.usage.requests, 3)
                self.assertEqual(result.usage.to_dict(), runner.sessions.usage(result.session_id).to_dict())
                self.assertIn("survived.txt", result.text)
                self.assertIn("Write then interrupt", result.text)
                messages = runner.sessions.messages(result.session_id)
                self.assertEqual([m.content for m in messages if m.metadata.get("terminal_result")], [result.text])
                self.assertEqual(sum(e.type == "run.completed" for e in events), 1)
                self.assertEqual([e.data["text"] for e in events if e.type == "model.completed"
                                  and e.run_id == result.run_id and e.data.get("stop_reason") == "cancelled"], [result.text])
                self.assertEqual((root / "survived.txt").read_text(), "retained evidence\n")
            finally:
                release.set()
                await runner.close()

    async def test_late_cancellation_preserves_one_canonical_executed_result(self):
        for boundary in ("before_publication", "publication", "idle", "ack", "completed", "lease", "deadline"):
            with self.subTest(boundary=boundary):
                await self.assert_late_cancellation(boundary)

    async def assert_late_cancellation(self, boundary):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            config = make_config(root, agent={"swarm_allow_writes": True, "auto_verify": True,
                                               "max_time_seconds": 1 if boundary == "deadline" else 60})
            runner = await build_runner(root, config=config, interactive=False)
            entered, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
            events, acknowledgements = [], []
            provider = runner.providers[0].provider
            assert isinstance(provider, MockProvider)

            def handler(request, call):
                if request.metadata.get("delegated"):
                    return response(text="Child wrote evidence.") if request.messages[-1].role == Role.TOOL else response(calls=[write("survived.txt", "retained evidence\n")])
                return response(text="Final checked answer.") if any(m.role == Role.TOOL for m in request.messages) else response(calls=[delegate(capability="write")])

            provider.handler = handler

            async def observe(event):
                events.append(event)
                if ((boundary in {"publication", "deadline"} and event.type == "model.completed"
                     and event.data.get("authoritative_verification"))
                        or (boundary == "completed" and event.type == "run.completed")):
                    entered.set()
                    await release.wait()

            runner.events.subscribe(observe)
            original_thread, original_cancel = asyncio.to_thread, runner.cancel
            original_publish = runner._publish_authoritative_result

            async def publish(*args, **kwargs):
                if boundary == "before_publication":
                    entered.set()
                    await release.wait()
                return await original_publish(*args, **kwargs)

            async def thread(function, *args, **kwargs):
                if function == runner.sessions.update_session:
                    is_ack = "settled_delegation_runs" in kwargs.get("metadata", {})
                    if (boundary == "idle" and kwargs.get("status") == "idle") or (boundary == "ack" and is_ack):
                        entered.set()
                        await release.wait()
                    if is_ack:
                        acknowledgements.append(kwargs["metadata"]["settled_delegation_runs"])
                if boundary == "lease" and function == runner.sessions.release_run_lease:
                    entered.set()
                    await release.wait()
                return await original_thread(function, *args, **kwargs)

            def cancel(session_id):
                original_cancel(session_id)
                cancelled.set()

            async def verify(context, steps):
                self.assertEqual(context.workspace, root)
                self.assertEqual((root / "survived.txt").read_text(), "retained evidence\n")
                return VerificationReport(ok=True, steps=[{"name": "Reopened content", "exit_code": 0}])

            try:
                with patch("borealis_coder.agent.runner.asyncio.to_thread", side_effect=thread), patch.object(runner, "cancel", side_effect=cancel), patch.object(runner, "_publish_authoritative_result", side_effect=publish), patch("borealis_coder.agent.runner.VerificationPlanner.run", side_effect=verify):
                    task = asyncio.create_task(runner.run("Write then report"))
                    try:
                        await asyncio.wait_for(entered.wait(), 5)
                        if boundary == "before_publication":
                            session_id = next(iter(runner._locks))
                            self.assertNotIn("settled_delegation_runs", runner.sessions.get_session(session_id).metadata)
                            self.assertFalse(any(e.type == "run.completed" for _, e in runner.sessions.events(session_id)))
                            self.assertFalse(any(m.metadata.get("terminal_result") for m in runner.sessions.messages(session_id)))
                        if boundary != "deadline":
                            task.cancel()
                        if boundary != "lease":
                            await asyncio.wait_for(cancelled.wait(), 3)
                        task.cancel()
                        # A scheduling turn, not a timing-based race trigger.
                        await asyncio.sleep(0)
                        release.set()
                        result = await asyncio.wait_for(task, 5)
                    finally:
                        release.set()
                        if not task.done():
                            task.cancel()
                            await task
                self.assertEqual(result.changed_files, ["survived.txt"])
                self.assertEqual(result.usage.requests, 5)
                self.assertEqual(result.usage.to_dict(), runner.sessions.usage(result.session_id).to_dict())
                # Execution and verification committed before these gates;
                # late cancellation must not retrospectively cancel success.
                self.assertEqual(result.stop_reason.value, "end_turn")
                self.assertTrue(result.text.startswith("Final checked answer."), result.text)
                self.assertNotIn("Run cancelled", result.text)
                assert result.verification is not None
                self.assertTrue(result.verification["checks_ok"])
                messages = runner.sessions.messages(result.session_id)
                validate_tool_call_order(messages)
                terminals = [m for m in messages if m.metadata.get("terminal_result")]
                self.assertEqual([m.content for m in terminals], [result.text])
                public = [e for e in events if e.type == "model.completed" and e.run_id == result.run_id
                          and e.data.get("authoritative_verification")]
                self.assertEqual([e.data["text"] for e in public], [result.text])
                self.assertEqual(sum(e.type == "run.completed" for e in events), 1)
                self.assertEqual(acknowledgements, [[result.run_id]])
                self.assertEqual(runner.sessions.get_session(result.session_id).metadata["settled_delegation_runs"], [result.run_id])
                runner.sessions.prune_events(max_count=1)
                provider.handler = lambda request, call: response(text="Next ordinary answer.")
                fresh = await runner.run("continue", session_id=result.session_id)
                self.assertEqual(fresh.changed_files, [])
                self.assertIsNone(fresh.verification)
                self.assertEqual((root / "survived.txt").read_text(), "retained evidence\n")
            finally:
                release.set()
                await runner.close()


class RecoveredRootOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.primary = self.base / "primary"
        self.auxiliary = self.base / "original" / "auxiliary"
        self.replacement = self.base / "replacement" / "auxiliary"
        for root in (self.primary, self.auxiliary, self.replacement):
            root.mkdir(parents=True)
        self.config = make_config(self.primary, agent={"auto_verify": True, "swarm_allow_writes": True})

    async def seed(self, *, literal=False, patch_tool=False):
        runner = await build_runner(self.primary, config=self.config, interactive=False,
                                    additional_roots=[self.auxiliary])
        try:
            session = runner.sessions.create_session(workspace=self.primary, provider="mock", model="fixture")
            target = self.primary / "auxiliary:changed.txt" if literal else self.auxiliary / "changed.txt"
            context = replace(runner.tool_context, session_id=session.id, run_id="run_interrupted.child",
                              metadata={**runner.tool_context.metadata, "delegated": True,
                                        "delegation_id": "child", "delegation_depth": 1,
                                        "delegation_allow_writes": True})
            call = ToolCall(id="child.patch", name="apply_patch", arguments={
                "patch": f"*** Begin Patch\n*** Add File: {target}\n+retained evidence\n*** End Patch",
            }) if patch_tool else write(str(target), "retained evidence\n")
            call.id = "child.mutation"
            await runner._execute_one(call, context, asyncio.Event(), nested_results=[])
            rows = runner.sessions.tool_calls(session.id)
            self.assertEqual(rows[0]["status"], "completed", rows)
            self.assertEqual(target.read_text(), "retained evidence\n")
            return session.id, rows
        finally:
            await runner.close()

    async def resume(self, session_id, roots):
        runner = await build_runner(self.primary, config=self.config, interactive=False, additional_roots=roots)
        scopes = []
        provider = runner.providers[0].provider
        assert isinstance(provider, MockProvider)
        provider.handler = lambda request, call: response(text="Qualified recovered report.")

        async def verify(planner, context, steps):
            scopes.append(planner.workspace)
            self.assertIn(planner.workspace, [self.primary, *roots])
            return VerificationReport(ok=True, steps=[{"name": "Owned fixture check", "exit_code": 0}])

        try:
            with patch("borealis_coder.agent.runner.VerificationPlanner.run", autospec=True, side_effect=verify):
                result = await runner.run("continue", session_id=session_id)
            rows = runner.sessions.tool_calls(session_id)
            messages = runner.sessions.messages(session_id)
            validate_tool_call_order(messages)
            self.assertFalse(any(m.role == Role.TOOL and m.tool_call_id == "child.mutation" for m in messages))
            self.assertEqual(messages[-1].content, result.text)
            return result, scopes, runner.sessions.get_session(session_id).metadata, rows
        finally:
            await runner.close()

    async def test_omitted_and_replaced_roots_remain_unacknowledged_until_restored(self):
        for roots in ([], [self.replacement]):
            for patch_tool in (False, True):
                with self.subTest(roots=bool(roots), patch_tool=patch_tool):
                    # A fresh path/session per schedule, never a tool replay.
                    target = self.auxiliary / "changed.txt"
                    if target.exists():
                        target.unlink()
                    session_id, rows = await self.seed(patch_tool=patch_tool)
                    result, scopes, metadata, recovered_rows = await self.resume(session_id, roots)
                    self.assertEqual(result.changed_files, ["auxiliary:changed.txt"])
                    self.assertEqual(result.mutation_tracking, "incomplete")
                    assert result.verification is not None
                    self.assertEqual(result.verification["mutation_tracking"], "incomplete")
                    self.assertIn("Mutation tracking: incomplete", result.text)
                    self.assertNotIn("run_interrupted", metadata.get("settled_delegation_runs", []))
                    self.assertEqual(recovered_rows, rows)
                    self.assertNotIn(self.auxiliary, scopes)
                    self.assertEqual(target.read_text(), "retained evidence\n")
                    self.assertFalse((self.replacement / "changed.txt").exists())
                    restored, scopes, metadata, restored_rows = await self.resume(session_id, [self.auxiliary])
                    self.assertEqual(restored.changed_files, ["auxiliary:changed.txt"])
                    self.assertEqual(restored.mutation_tracking, "complete")
                    assert restored.verification is not None
                    self.assertTrue(restored.verification["ok"])
                    self.assertEqual(scopes, [self.auxiliary])
                    self.assertIn("run_interrupted", metadata["settled_delegation_runs"])
                    self.assertEqual(restored_rows, rows)

    @unittest.skipIf(os.name == "nt", "Windows filenames cannot contain literal colons")
    async def test_primary_literal_colon_patch_and_genuine_additional_patch_keep_distinct_owners(self):
        for literal in (True, False):
            with self.subTest(literal=literal):
                session_id, rows = await self.seed(literal=literal, patch_tool=True)
                result, scopes, metadata, restored_rows = await self.resume(session_id, [self.auxiliary])
                self.assertEqual(result.changed_files, ["auxiliary:changed.txt"])
                self.assertEqual(result.mutation_tracking, "complete")
                self.assertEqual(scopes, [self.primary if literal else self.auxiliary])
                assert result.verification is not None
                self.assertTrue(result.verification["ok"])
                self.assertIn("run_interrupted", metadata["settled_delegation_runs"])
                self.assertEqual(restored_rows, rows)

    async def test_cancelled_partial_startup_recovery_keeps_context_without_acknowledging_unread_tail(self):
        session_id, _ = await self.seed(patch_tool=True)
        runner = await build_runner(self.primary, config=self.config, interactive=False,
                                    additional_roots=[self.auxiliary])
        entered = asyncio.Event()
        original_thread = asyncio.to_thread
        try:
            runner.sessions.start_tool_call(session_id, "run_interrupted.child", "child.unknown", "write_file", {},
                                            metadata={"delegation_id": "child", "delegation_depth": 1})

            async def thread(function, *args, **kwargs):
                if function == runner.sessions.complete_tool_call and args[1] == "child.unknown":
                    entered.set()
                    await asyncio.Event().wait()
                return await original_thread(function, *args, **kwargs)

            with patch("borealis_coder.agent.runner.asyncio.to_thread", side_effect=thread):
                task = asyncio.create_task(runner.run("continue", session_id=session_id))
                try:
                    await asyncio.wait_for(entered.wait(), 3)
                    task.cancel()
                    result = await asyncio.wait_for(task, 5)
                finally:
                    if not task.done():
                        task.cancel()
                        await task
            self.assertEqual(result.changed_files, ["auxiliary:changed.txt"])
            self.assertEqual(result.usage.requests, 0)
            self.assertTrue(result.incomplete)
            self.assertEqual(result.mutation_tracking, "incomplete")
            self.assertNotIn("run_interrupted", runner.sessions.get_session(session_id).metadata.get("settled_delegation_runs", []))
            self.assertEqual(runner.sessions.messages(session_id)[-1].content, result.text)
        finally:
            await runner.close()
        fresh, _, _, _ = await self.resume(session_id, [self.auxiliary])
        self.assertEqual(fresh.changed_files, ["auxiliary:changed.txt"])
        self.assertEqual(fresh.mutation_tracking, "incomplete")
        self.assertEqual((self.auxiliary / "changed.txt").read_text(), "retained evidence\n")

    async def test_mixed_known_and_unavailable_paths_do_not_acknowledge_the_whole_run(self):
        session_id, _ = await self.seed(patch_tool=True)
        runner = await build_runner(self.primary, config=self.config, interactive=False,
                                    additional_roots=[self.auxiliary])
        try:
            context = replace(runner.tool_context, session_id=session_id, run_id="run_interrupted.child",
                              metadata={**runner.tool_context.metadata, "delegated": True,
                                        "delegation_id": "child", "delegation_depth": 1,
                                        "delegation_allow_writes": True})
            call = write("known.txt", "known primary effect\n")
            call.id = "child.known"
            await runner._execute_one(call, context, asyncio.Event(), nested_results=[])
            rows = runner.sessions.tool_calls(session_id)
        finally:
            await runner.close()
        result, scopes, metadata, recovered_rows = await self.resume(session_id, [])
        self.assertEqual(result.changed_files, ["auxiliary:changed.txt", "known.txt"])
        self.assertEqual(result.mutation_tracking, "incomplete")
        self.assertEqual(scopes, [self.primary])
        self.assertNotIn("run_interrupted", metadata.get("settled_delegation_runs", []))
        self.assertEqual(rows, recovered_rows)
        self.assertEqual((self.primary / "known.txt").read_text(), "known primary effect\n")
        restored, scopes, metadata, restored_rows = await self.resume(session_id, [self.auxiliary])
        self.assertEqual(restored.changed_files, ["auxiliary:changed.txt", "known.txt"])
        self.assertEqual(restored.mutation_tracking, "complete")
        self.assertCountEqual(scopes, [self.primary, self.auxiliary])
        self.assertIn("run_interrupted", metadata["settled_delegation_runs"])
        self.assertEqual(rows, restored_rows)

    @unittest.skipIf(os.name == "nt", "Windows filenames cannot contain literal colons")
    async def test_one_patch_can_preserve_both_owners_of_the_same_display_label(self):
        runner = await build_runner(self.primary, config=self.config, interactive=False,
                                    additional_roots=[self.auxiliary])
        try:
            session = runner.sessions.create_session(workspace=self.primary, provider="mock", model="fixture")
            context = replace(runner.tool_context, session_id=session.id, run_id="run_interrupted.child",
                              metadata={**runner.tool_context.metadata, "delegated": True,
                                        "delegation_id": "child", "delegation_depth": 1,
                                        "delegation_allow_writes": True})
            call = ToolCall(id="child.mutation", name="apply_patch", arguments={"patch":
                f"*** Begin Patch\n*** Add File: {self.primary / 'auxiliary:changed.txt'}\n+primary evidence\n"
                f"*** Add File: {self.auxiliary / 'changed.txt'}\n+additional evidence\n*** End Patch"})
            await runner._execute_one(call, context, asyncio.Event(), nested_results=[])
            rows = runner.sessions.tool_calls(session.id)
            self.assertEqual(rows[0]["status"], "completed")
        finally:
            await runner.close()
        result, scopes, metadata, recovered_rows = await self.resume(session.id, [self.auxiliary])
        self.assertEqual(result.changed_files, ["auxiliary:changed.txt"])
        self.assertEqual(result.mutation_tracking, "complete")
        self.assertCountEqual(scopes, [self.primary, self.auxiliary])
        self.assertEqual((self.primary / "auxiliary:changed.txt").read_text(), "primary evidence\n")
        self.assertEqual((self.auxiliary / "changed.txt").read_text(), "additional evidence\n")
        self.assertEqual(recovered_rows, rows)
        self.assertIn("run_interrupted", metadata["settled_delegation_runs"])

    async def test_legacy_qualified_patch_without_identity_is_not_guessed(self):
        session_id, _ = await self.seed(patch_tool=True)
        runner = await build_runner(self.primary, config=self.config, interactive=False,
                                    additional_roots=[self.auxiliary])
        try:
            rows = runner.sessions.tool_calls(session_id)
            # Model a legacy stored receipt at the reader boundary. The real
            # producer above proves both the canonical display and file effects.
            rows[0]["metadata"].pop("changed_path_roots", None)
            context = replace(runner.tool_context, session_id=session_id, run_id="run_legacy")
            await runner._recover_nested_calls(rows, context)
            self.assertEqual(context.changed_files, {"auxiliary:changed.txt"})
            self.assertEqual(context.mutation_tracking, "incomplete")
            self.assertNotIn("run_interrupted", context.metadata["recovered_nested_runs"])
            self.assertEqual((self.auxiliary / "changed.txt").read_text(), "retained evidence\n")
        finally:
            await runner.close()
