"""Crash/reopen regressions for historical coordination and transaction ownership."""
from __future__ import annotations

import asyncio
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from borealis_coder.agent import CompactionError, build_runner, validate_tool_call_order
from borealis_coder.models import Message, Role, ToolCall
from borealis_coder.sessions import SessionStore
from tests.helpers import make_config
from tests.test_swarm import response


def seed_crash(root: Path, kind: str, *, status: str = "running") -> SessionStore:
    """Persist a valid crash boundary while holding the real process lease."""
    store = SessionStore(make_config(root).database_path)
    store.create_session(workspace=root, provider="mock", model="fixture", session_id="root")
    store.acquire_run_lease("root")
    store.update_session("root", status=status)
    store.append_message("root", Message(id="original-user", role=Role.USER,
                                         content="Inspect the original source task."))
    start = ToolCall(id="start-child", name="delegate_start", arguments={"task": "inspect"})
    store.append_message("root", Message(id="original-start", role=Role.ASSISTANT, tool_calls=[start]))
    store.start_tool_call("root", "old", start.id, start.name, start.arguments)
    store.complete_tool_call("root", start.id, output="child", is_error=False,
                             message=Message(id="started-child", role=Role.TOOL,
                                             tool_call_id=start.id, tool_name=start.name, content="child"))
    calls = [ToolCall(id="read-done", name="read_file", arguments={}),
             ToolCall(id="root-wait", name="delegate_wait", arguments={}),
             ToolCall(id="unstarted-write", name="write_file", arguments={})]
    store.append_message("root", Message(id="original-bundle", role=Role.ASSISTANT, tool_calls=calls))
    store.start_tool_call("root", "old", calls[0].id, calls[0].name, {})
    store.complete_tool_call("root", calls[0].id, output="retained source", is_error=False,
                             message=Message(id="original-read", role=Role.TOOL,
                                             tool_call_id=calls[0].id, tool_name=calls[0].name,
                                             content="retained source"))
    store.start_tool_call("root", "old", calls[1].id, calls[1].name, {})
    store.ensure_delegated_conversation("root", "old.child", "child", 1,
                                        delegation_call_id=start.id)
    store.start_coordination_child("root", "old", "child", "root", "old.child", start.id, True)
    child_call = ToolCall(id="child-wait", name="delegate_wait", arguments={})
    store.append_message("old.child", Message(id="child-original", role=Role.ASSISTANT,
                                              tool_calls=[child_call]))
    store.start_tool_call("root", "old.child", "child.child-wait", "delegate_wait", {},
                          metadata={"delegation_id": "child", "delegation_depth": 1})
    # A separate canonical nested receipt may have changed the workspace. Recovery
    # must keep its identity/uncertainty even when the root wait itself is harmless.
    store.start_tool_call("root", "old.child", "child.write", "write_file", {},
                          metadata={"delegation_id": "child", "delegation_depth": 1})
    (root / "survived.txt").write_text("committed effect before crash\n")
    if kind == "question":
        store.queue_coordination_message("root", "old", "question-id", "child", "root",
                                          "UNDELIVERED_QUESTION", inbox_limit=8, run_limit=16)
    else:
        store.finish_coordination_child("old", "child", "completed", {"text": "retained terminal"})
        store.start_tool_call("root", "old.child", "child.result", "delegate_result", {},
                              metadata={"delegation_id": "child", "coordination_terminal": True})
        store.complete_tool_call("root", "child.result", output="retained terminal", is_error=False,
                                 metadata={"delegation_id": "child", "coordination_terminal": True})
    store.queue_coordination_message("root", "old", "steering-id", "root", "child",
                                      "HISTORICAL_STEERING", inbox_limit=8, run_limit=16)
    # No cancellation/finalizer/store.close: process exit releases the OS lease.
    return store


def crash_process(root: Path, kind: str, status: str = "running") -> None:
    store = seed_crash(root, kind, status=status)
    assert "root" in store._run_leases
    os._exit(23)


def database_snapshot(path: Path) -> dict:
    connection = sqlite3.connect(path)
    try:
        return {table: connection.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                for table in ("sessions", "messages", "tool_calls", "coordination_children",
                              "coordination_messages")}
    finally:
        connection.close()


class CoordinationCrashRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_root_mutation_uncertainty_survives_store_repair_and_second_reopen(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            config = make_config(root)
            store = SessionStore(config.database_path)
            session = store.create_session(workspace=root, provider="mock", model="fixture")
            store.acquire_run_lease(session.id)
            store.update_session(session.id, status="running")
            call = ToolCall(id="root-write", name="write_file", arguments={})
            store.append_message(session.id, Message(role=Role.ASSISTANT, tool_calls=[call]))
            store.start_tool_call(session.id, "old", call.id, call.name, {})
            store.queue_coordination_message(session.id, "old", "root-note", "child", "root",
                                              "Historical question", inbox_limit=8, run_limit=16)
            (root / "effect.txt").write_text("Retained effect\n")
            store.close()
            for _ in range(2):
                runner = await build_runner(root, config=config, interactive=False)
                try:
                    with patch.object(runner.providers[0].provider, "complete", return_value=response("Inspect prior effects.")) as provider:
                        result = await runner.run("continue", session_id=session.id)
                    self.assertEqual(result.stop_reason.value, "end_turn", result.error)
                    self.assertEqual(result.mutation_tracking, "incomplete")
                    provider.assert_called_once()
                    rows = runner.sessions.tool_calls(session.id)
                    self.assertEqual(rows[0]["status"], "error")
                    self.assertEqual(rows[0]["metadata"]["workspace_change_tracking"], "incomplete")
                    messages = runner.sessions.messages(session.id)
                    validate_tool_call_order(messages)
                    self.assertEqual(sum(m.tool_call_id == call.id for m in messages), 1)
                    self.assertEqual(sum(m.id == "root-note" for m in messages), 1)
                    self.assertEqual((root / "effect.txt").read_text(), "Retained effect\n")
                finally:
                    await runner.close()

    async def test_crash_reopen_resume_preserves_bundles_and_receipts(self):
        for kind, status, entrypoint in (("question", "running", "runner"),
                                         ("terminal", "running", "list"),
                                         ("question", "idle", "runner")):
            with self.subTest(kind=kind, status=status, entrypoint=entrypoint), tempfile.TemporaryDirectory() as td:
                root = Path(td).resolve()
                command = [sys.executable, "-c", "from pathlib import Path; "
                           "from tests.test_coordination_recovery import crash_process; "
                           "import sys; crash_process(Path(sys.argv[1]), sys.argv[2], sys.argv[3])",
                           str(root), kind, status]
                crashed = await asyncio.to_thread(subprocess.run, command, capture_output=True, timeout=15)
                self.assertEqual(crashed.returncode, 23, crashed.stderr.decode())
                path = make_config(root).database_path
                original = database_snapshot(path)
                runner = await build_runner(root, config=make_config(root), interactive=False)
                requests = []

                async def complete(request, requests=requests):
                    self.assertFalse(request.metadata.get("delegated"))
                    validate_tool_call_order(request.messages)
                    requests.append(request)
                    return response("Previous work was interrupted. Inspect retained evidence before retrying.")

                try:
                    if entrypoint == "list":
                        runner.sessions.list_sessions()
                    for turn in range(2):
                        if turn:
                            await runner.close()
                            runner = await build_runner(root, config=make_config(root), interactive=False)
                        with patch.object(runner.providers[0].provider, "complete", side_effect=complete):
                            result = await asyncio.wait_for(runner.run("continue", session_id="root"), 10)
                            self.assertEqual(result.stop_reason.value, "end_turn", result.error)
                            self.assertEqual(result.mutation_tracking, "incomplete")
                    self.assertEqual(len(requests), 2)  # Only the two requested new root turns.
                    messages = runner.sessions.messages("root")
                    validate_tool_call_order(messages)
                    child_messages = runner.sessions.messages("old.child")
                    validate_tool_call_order(child_messages)
                    repairs = [m for m in messages if m.metadata.get("abandoned")]
                    self.assertCountEqual([m.tool_call_id for m in repairs], ["root-wait", "unstarted-write"])
                    self.assertTrue(all(m.is_error for m in repairs))
                    self.assertTrue(next(m for m in repairs if m.tool_call_id == "root-wait").metadata["unknown_outcome"])
                    self.assertTrue(next(m for m in repairs if m.tool_call_id == "unstarted-write").metadata["not_started"])
                    notice_ids = ["terminal:child"] + (["question-id"] if kind == "question" else [])
                    for notice_id in notice_ids:
                        self.assertEqual(sum(m.id == notice_id for m in messages), 1)
                        self.assertGreater(next(i for i, m in enumerate(messages) if m.id == notice_id),
                                           max(i for i, m in enumerate(messages) if m in repairs))
                    self.assertEqual(sum(m.id == "steering-id" for m in child_messages), 1)
                    self.assertTrue(child_messages[-2].metadata["unknown_outcome"])
                    self.assertIn("completed" if kind == "terminal" else "interrupted",
                                  next(m.content for m in messages if m.id == "terminal:child"))
                    rows = {r["tool_call_id"]: r for r in runner.sessions.tool_calls("root")}
                    self.assertEqual(rows["child.write"]["status"], "error")
                    self.assertEqual(rows["child.write"]["metadata"]["delegation_id"], "child")
                    self.assertEqual(rows["child.write"]["metadata"]["workspace_change_tracking"], "incomplete")
                    self.assertEqual(rows["root-wait"]["status"], "error")
                    self.assertNotIn("workspace_change_tracking", rows["root-wait"]["metadata"])
                    self.assertEqual(sum(r["tool_name"] == "delegate_start" for r in rows.values()), 1)
                    if kind == "terminal":
                        self.assertEqual(rows["child.result"]["output"], "retained terminal")
                        self.assertEqual(rows["child.result"]["status"], "completed")
                    self.assertEqual((root / "survived.txt").read_text(), "committed effect before crash\n")
                    self.assertFalse(any("UNDELIVERED" in i["text"] or "HISTORICAL" in i["text"]
                                         for i in runner.sessions.task_state("root")["instructions"]))
                    after = database_snapshot(path)
                    self.assertEqual(after["messages"][:len(original["messages"])], original["messages"])
                    for request in requests:
                        self.assertTrue(any(m.metadata.get("abandoned") for m in request.messages))
                finally:
                    await runner.close()
                reopened = SessionStore(path)
                try:
                    stable = database_snapshot(path)
                    reopened.list_sessions()
                    reopened.messages("root")
                    self.assertEqual(database_snapshot(path), stable)
                finally:
                    reopened.close()


class CoordinationTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        seed_crash(self.root, "question").close()
        self.path = make_config(self.root).database_path
        self.owner = SessionStore(self.path)
        self.addCleanup(self.owner.close)
        for identifier in ("plain", "live"):
            self.owner.create_session(workspace=self.root, provider="mock", model="fixture",
                                      session_id=identifier)
            self.owner.acquire_run_lease(identifier)
            self.owner.update_session(identifier, status="running")
        self.owner.release_run_lease("plain")
        self.owner.start_coordination_child("live", "live-run", "live-child", "root",
                                            "live-scope", "live-call", True)
        self.owner.queue_coordination_message("live", "live-run", "live-question", "live-child",
                                              "root", "Still live", inbox_limit=8, run_limit=16)
        self.store = SessionStore(self.path)
        self.addCleanup(self.store.close)

    def assert_reconciled(self):
        rows = self.store.list_sessions()
        self.assertEqual({row.id: row.status for row in rows},
                         {"root": "interrupted", "plain": "interrupted", "live": "running"})
        for identifier in ("root", "plain", "live"):
            self.assertEqual(self.store.get_session(identifier).status,
                             "running" if identifier == "live" else "interrupted")
        self.assertEqual(len([m for m in self.store.messages("root") if m.id == "question-id"]), 1)
        validate_tool_call_order(self.store.messages("root"))
        validate_tool_call_order(self.store.messages("old.child"))

    def test_multiple_stale_roots_and_live_lease_reopen_idempotently(self):
        original = database_snapshot(self.path)
        self.assert_reconciled()
        reconciled = database_snapshot(self.path)
        for table in ("coordination_children", "coordination_messages"):
            self.assertEqual([r for r in original[table] if r[0] == "live"],
                             [r for r in reconciled[table] if r[0] == "live"])
        self.assertEqual(reconciled["messages"][:len(original["messages"])], original["messages"])
        self.assert_reconciled()
        other = SessionStore(self.path)
        try:
            self.assertEqual(len(other.list_sessions()), 3)
            self.assertEqual(other.get_session("live").status, "running")
            self.assertEqual(database_snapshot(self.path), reconciled)
        finally:
            other.close()

    def test_status_failure_rolls_back_recovery_and_all_rows_then_retry_succeeds(self):
        for entrypoint in ("list", "get"):
            with self.subTest(entrypoint=entrypoint):
                before = database_snapshot(self.path)
                self.store._connection.execute("""
                    CREATE TEMP TRIGGER fail_reconciliation BEFORE UPDATE OF status ON sessions
                    WHEN NEW.id='root' AND NEW.status='interrupted'
                    BEGIN SELECT RAISE(ABORT, 'controlled status failure'); END
                """)
                try:
                    with self.assertRaisesRegex(sqlite3.DatabaseError, "controlled status failure"):
                        if entrypoint == "list":
                            self.store.list_sessions()
                        else:
                            self.store.get_session("root")
                    # An independent connection observes no committed notices, ACKs,
                    # status changes or transcript repairs from the failed attempt.
                    self.assertEqual(database_snapshot(self.path), before)
                finally:
                    self.store._connection.execute("DROP TRIGGER fail_reconciliation")
        self.assert_reconciled()
        stable = database_snapshot(self.path)
        self.assert_reconciled()
        self.assertEqual(database_snapshot(self.path), stable)

    def test_live_lease_keeps_unfinished_history_and_queues_untouched(self):
        # The observer must not close a bundle or interrupt a worker owned by a
        # different SessionStore, even when its durable state looks abandoned.
        self.owner.acquire_run_lease("root")
        try:
            before = database_snapshot(self.path)
            self.assertEqual(self.store.get_session("root").status, "running")
            self.store.messages("root")
            self.assertEqual(database_snapshot(self.path), before)
        finally:
            self.owner.release_run_lease("root")

    def test_internal_reconciliation_does_not_commit_callers_pending_change(self):
        before = database_snapshot(self.path)
        with self.assertRaisesRegex(RuntimeError, "caller failure"), self.store._lock, self.store._connection:
            self.store._connection.execute("BEGIN IMMEDIATE")
            self.store._connection.execute("UPDATE sessions SET title='pending caller change' WHERE id='plain'")
            row = self.store._connection.execute("SELECT * FROM sessions WHERE id='root'").fetchone()
            self.store._reconcile_running_row(row)
            self.assertEqual(database_snapshot(self.path), before)
            raise RuntimeError("caller failure")
        self.assertEqual(database_snapshot(self.path), before)
        self.assert_reconciled()

    def test_direct_interruption_rolls_back_after_notice_append_failure(self):
        self.owner.acquire_run_lease("root")
        before = database_snapshot(self.path)
        append = self.owner._append_message_locked

        def fail_terminal(scope, message):
            append(scope, message)
            if message.id == "terminal:child":
                raise RuntimeError("controlled terminal failure")

        try:
            with patch.object(self.owner, "_append_message_locked", side_effect=fail_terminal), self.assertRaisesRegex(RuntimeError, "controlled terminal failure"):
                self.owner.interrupt_coordination("root")
            self.assertEqual(database_snapshot(self.path), before)
            self.owner.interrupt_coordination("root")
            stable = database_snapshot(self.path)
            self.owner.interrupt_coordination("root")
            self.assertEqual(database_snapshot(self.path), stable)
            validate_tool_call_order(self.owner.messages("root"))
        finally:
            self.owner.release_run_lease("root")

    def test_child_receipt_committed_before_snapshot_keeps_its_real_result(self):
        self.owner.complete_tool_call("root", "child.child-wait", output="Retained completed wait result",
                                      is_error=False, metadata={"delegation_id": "child", "observed": True})
        self.store.get_session("root")
        results = [m for m in self.store.messages("old.child") if m.tool_call_id == "child-wait"]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "Retained completed wait result")
        self.assertFalse(results[0].is_error)
        self.assertEqual(results[0].metadata, {"delegation_id": "child", "observed": True})
        validate_tool_call_order(self.store.messages("old.child"))


class MalformedRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_internal_notice_does_not_heal_an_already_broken_history(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            store = seed_crash(root, "question")
            store.append_message("root", Message(role=Role.ASSISTANT, content="preexisting corruption"))
            store.close()
            before = database_snapshot(make_config(root).database_path)
            runner = await build_runner(root, config=make_config(root), interactive=False)
            try:
                with patch.object(runner.providers[0].provider, "complete") as provider, self.assertRaises(CompactionError):
                    await runner.run("continue", session_id="root")
                provider.assert_not_called()
                self.assertEqual(database_snapshot(make_config(root).database_path), before)
            finally:
                await runner.close()
