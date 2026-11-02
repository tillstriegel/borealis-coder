from __future__ import annotations

import asyncio
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from borealis_coder.agent import build_runner, validate_tool_call_order
from borealis_coder.models import Event, Message, Role, ToolCall
from borealis_coder.providers.mock import MockProvider
from borealis_coder.tools.verification import VerificationReport
from borealis_coder.util import sha256_text
from tests.helpers import make_config
from tests.test_swarm import delegate, response, write


class CompletedDescendantRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.config = make_config(self.root, agent={"auto_verify": True, "swarm_allow_writes": True})

    async def seed(self, *, root_completed=False, healthy=False, read_only=False, incomplete=False):
        runner = await build_runner(self.root, config=self.config, interactive=False)
        try:
            session = runner.sessions.create_session(workspace=self.root, provider="mock", model="fixture")
            root_run = "run_crashed"
            call = delegate(identifier="root-delegate")
            runner.sessions.append_message(session.id, Message(role=Role.ASSISTANT, tool_calls=[call]))
            runner.sessions.start_tool_call(session.id, root_run, call.id, call.name, call.arguments)
            await runner.events.emit("run.started", session_id=session.id, run_id=root_run)
            # Canonical production producer: real tool, checkpoint, and nested
            # receipt; only the process-death schedule/ancestor state is seeded.
            context = replace(runner.tool_context, session_id=session.id,
                              run_id=root_run + ".delegate_child.delegate_grandchild",
                              metadata={**runner.tool_context.metadata, "delegated": True,
                                        "delegation_id": "delegate_grandchild", "delegation_depth": 2,
                                        "delegation_allow_writes": True})
            if read_only:
                (self.root / "survived.txt").write_text("read-only evidence\n")
                nested = ToolCall(id="delegate_grandchild.read", name="read_file",
                                  arguments={"path": "survived.txt", "start_line": None,
                                             "end_line": None, "max_chars": None})
            else:
                nested = write("survived.txt", "completed before crash\n")
                nested.id = "delegate_grandchild.write"
            await runner._execute_one(nested, context, asyncio.Event(), nested_results=[])
            if incomplete:
                # A canonical terminal error can carry uncertain effects even
                # without unknown_outcome (result collection failed after edit).
                failed = ToolCall(id="delegate_grandchild.partial", name="write_file", arguments={})
                runner.sessions.start_tool_call(session.id, context.run_id, failed.id, failed.name, {},
                                               metadata={"delegation_id": "delegate_grandchild"})
                runner.sessions.complete_tool_call(session.id, failed.id, output="partial effects", is_error=True,
                                                  metadata={"delegation_id": "delegate_grandchild",
                                                            "workspace_change_tracking": "incomplete"})
            if root_completed:
                message = Message(role=Role.TOOL, tool_call_id=call.id, tool_name=call.name,
                                  content="Descendant finished.")
                runner.sessions.complete_tool_call(session.id, call.id, output=message.content,
                                                  is_error=False, message=message)
            if healthy:
                self.assertTrue(root_completed)
                await runner.events.emit("run.completed", session_id=session.id, run_id=root_run,
                                         result={"changed_files": ["survived.txt"] if not read_only else []})
            return session.id
        finally:
            await runner.close()

    async def resume(self, session_id):
        runner = await build_runner(self.root, config=self.config, interactive=False)
        provider = runner.providers[0].provider
        assert isinstance(provider, MockProvider)
        provider.handler = lambda request, call: response(text="Recovered candidate." if call == 1 else "Recovered final answer.")
        try:
            async def verify_content(context, steps):
                self.assertEqual(context.changed_files, {"survived.txt"})
                self.assertEqual(context.changed_roots, {self.root})
                self.assertEqual((self.root / "survived.txt").read_text(), "completed before crash\n")
                return VerificationReport(ok=True, steps=[{"name": "Fixture content assertion", "exit_code": 0}])

            # Use a real reopened-file assertion as the fixture check. Native
            # subprocesses cannot guarantee descendant lifecycle containment.
            with patch("borealis_coder.agent.runner.VerificationPlanner.run", side_effect=verify_content):
                result = await asyncio.wait_for(runner.run("continue", session_id=session_id), 10)
            rows = runner.sessions.tool_calls(session_id)
            messages = runner.sessions.messages(session_id)
            validate_tool_call_order(messages)
            self.assertFalse(any(m.role == Role.TOOL and m.tool_call_id == "delegate_grandchild.write"
                                 for m in messages))
            self.assertEqual(provider.calls, 2 if result.verification else 1)
            self.assertTrue(any(m.role == Role.ASSISTANT and m.content == result.text for m in messages))
            return result, rows
        finally:
            await runner.close()

    async def test_completed_write_recovered_with_running_root_delegate(self):
        await self.assert_recovered(root_completed=False)

    async def test_completed_write_recovered_with_completed_root_delegate(self):
        await self.assert_recovered(root_completed=True)

    async def assert_recovered(self, *, root_completed):
        session_id = await self.seed(root_completed=root_completed)
        # Recovery itself may be followed by another crash before the next
        # run starts. Reopen again: no receipt may be rewritten/lost/replayed.
        runner = await build_runner(self.root, config=self.config, interactive=False)
        try:
            context = replace(runner.tool_context, session_id=session_id, run_id="run_recovery_crashed")
            before = runner.sessions.tool_calls(session_id)
            await runner._recover_nested_calls(before, context)
            self.assertEqual(context.changed_files, {"survived.txt"})
            self.assertEqual(context.changed_roots, {self.root})
            self.assertEqual(context.mutation_tracking, "complete")
            self.assertEqual(runner.sessions.tool_calls(session_id), before)
        finally:
            await runner.close()
        result, rows = await self.resume(session_id)
        self.assertEqual(result.changed_files, ["survived.txt"])
        self.assertEqual(result.mutation_tracking, "complete")
        self.assertIsNotNone(result.verification)
        assert result.verification is not None
        self.assertTrue(result.verification["checks_ok"], result.verification)
        self.assertEqual(result.verification["steps"][0]["name"], "Fixture content assertion")
        writer = next(r for r in rows if r["tool_name"] == "write_file")
        self.assertEqual(writer["status"], "completed")
        self.assertFalse(writer["metadata"].get("unknown_outcome", False))
        self.assertEqual((self.root / "survived.txt").read_text(), "completed before crash\n")
        # Once the resumed root is terminal, these are historical known writes.
        runner = await build_runner(self.root, config=self.config, interactive=False)
        try:
            runner.sessions.prune_events(max_count=1)
        finally:
            await runner.close()
        fresh, fresh_rows = await self.resume(session_id)
        self.assertEqual(fresh.changed_files, [])
        self.assertIsNone(fresh.verification)
        self.assertEqual(rows, fresh_rows)

    async def test_healthy_terminal_root_is_not_recovered_even_after_event_pagination(self):
        session_id = await self.seed(root_completed=True)
        runner = await build_runner(self.root, config=self.config, interactive=False)
        try:
            runner.sessions.append_events(Event(type="fixture.padding", session_id=session_id)
                                          for _ in range(10_001))
            await runner.events.emit("run.completed", session_id=session_id, run_id="run_crashed",
                                     result={"changed_files": ["survived.txt"]})
        finally:
            await runner.close()
        result, _ = await self.resume(session_id)
        self.assertEqual(result.changed_files, [])
        self.assertIsNone(result.verification)

    async def test_completed_read_only_descendant_does_not_trigger_verification(self):
        result, _ = await self.resume(await self.seed(root_completed=True, read_only=True))
        self.assertEqual(result.changed_files, [])
        self.assertEqual(result.mutation_tracking, "complete")
        self.assertIsNone(result.verification)

    async def test_terminal_incomplete_receipt_remains_fail_closed(self):
        result, _ = await self.resume(await self.seed(root_completed=True, incomplete=True))
        self.assertEqual(result.changed_files, ["survived.txt"])
        self.assertEqual(result.mutation_tracking, "incomplete")
        self.assertIn("Mutation tracking: incomplete", result.text)

    async def test_patch_delete_and_directory_receipts_recover_additional_root(self):
        session_id = await self.seed(root_completed=True)
        additional = self.root / "auxiliary"
        additional.mkdir()
        (additional / "deleted.txt").write_text("preimage")
        runner = await build_runner(self.root, config=self.config, interactive=False,
                                    additional_roots=[additional])
        try:
            context = replace(runner.tool_context, session_id=session_id,
                              run_id="run_crashed.delegate_child.delegate_grandchild",
                              metadata={**runner.tool_context.metadata, "delegated": True,
                                        "delegation_id": "delegate_grandchild", "delegation_depth": 2,
                                        "delegation_allow_writes": True})
            calls = [ToolCall(id="grandchild.patch", name="apply_patch", arguments={
                "patch": f"*** Begin Patch\n*** Add File: {additional / 'added.txt'}\n+patch evidence\n*** End Patch",
            }), ToolCall(id="grandchild.delete", name="delete_file", arguments={
                "path": str(additional / "deleted.txt"), "expected_sha256": sha256_text("preimage"),
            }), ToolCall(id="grandchild.mkdir", name="make_directory", arguments={
                "path": str(additional / "deep" / "folder"),
            })]
            for call in calls:
                await runner._execute_one(call, context, asyncio.Event(), nested_results=[])
            rows = runner.sessions.tool_calls(session_id)
            self.assertTrue(all(row["status"] == "completed" for row in rows))
        finally:
            await runner.close()
        runner = await build_runner(self.root, config=self.config, interactive=False,
                                    additional_roots=[additional])
        try:
            fresh = replace(runner.tool_context, session_id=session_id, run_id="run_resumed")
            await runner._recover_nested_calls(runner.sessions.tool_calls(session_id), fresh)
            self.assertEqual(fresh.changed_files, {"survived.txt", "auxiliary:added.txt", "auxiliary:deleted.txt",
                                                   "auxiliary:deep", "auxiliary:deep/folder"})
            self.assertEqual(fresh.changed_roots, {self.root, additional})
            self.assertEqual(fresh.mutation_tracking, "complete")
            self.assertFalse((additional / "deleted.txt").exists())
            self.assertEqual((additional / "added.txt").read_text(), "patch evidence\n")
            self.assertEqual(runner.sessions.tool_calls(session_id), rows)
        finally:
            await runner.close()
