from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from borealis_coder.agent import build_runner, validate_tool_call_order
from borealis_coder.errors import ProviderError
from borealis_coder.models import Role, ToolCall
from borealis_coder.providers.base import ProviderStreamEvent
from borealis_coder.providers.mock import MockProvider
from borealis_coder.sessions import SessionStore
from borealis_coder.swarm_view import SwarmView
from borealis_coder.tools.verification import VerificationReport
from tests.helpers import make_config
from tests.test_swarm import response


def launch(task, identifier):
    return ToolCall(id=identifier, name="delegate_start", arguments={
        "task": task, "max_turns": 12, "capability": "read-only",
    })


def send(recipient, kind, text, identifier):
    return ToolCall(id=identifier, name="delegate_message", arguments={
        "recipient": recipient, "kind": kind, "message": text,
    })


def wait(identifier):
    return ToolCall(id=identifier, name="delegate_wait", arguments={})


class CoordinationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.runner = await build_runner(self.root, config=make_config(self.root), interactive=False)
        self.addAsyncCleanup(self.runner.close)
        self.provider = self.runner.providers[0].provider
        self.events = []
        self.runner.events.subscribe(self.events.append)

    async def test_root_wait_allows_child_writes_without_releasing_capability_or_approval_gates(self):
        for mode in ("write", "read-only", "approval-denied"):
            with self.subTest(mode=mode):
                root = self.root / mode
                root.mkdir()
                config = make_config(root, safety={"approval": "always"})
                runner = await build_runner(root, config=config, interactive=False)
                wait_entered, write_requested = asyncio.Event(), asyncio.Event()
                lock_states, approvals, events = [], [], []
                turns = {False: 0, True: 0}

                def approve(request, mode=mode, approvals=approvals):
                    approvals.append(request.tool_name)
                    return mode != "approval-denied" or request.tool_name != "write_file"

                runner.tool_context.approvals.callback = approve

                def observe(event, events=events, lock_states=lock_states, runner=runner, wait_entered=wait_entered):
                    events.append(event)
                    if event.type == "swarm.coordination" and event.data.get("state") == "waiting":
                        # This comes from inside the actual root wait operation.
                        lock_states.append(runner._mutation_lock.locked())
                        wait_entered.set()
                    if event.type == "tool.started" and event.data.get("tool") == "write_file":
                        lock_states.append(runner._mutation_lock.locked())

                runner.events.subscribe(observe)

                async def complete(request, mode=mode, turns=turns, wait_entered=wait_entered, write_requested=write_requested):
                    child = bool(request.metadata.get("delegated"))
                    turn = turns[child]
                    turns[child] += 1
                    if child and turn == 0:
                        await wait_entered.wait()
                        write_requested.set()
                        return response(calls=[ToolCall(id="gated-write", name="write_file", arguments={
                            "path": "answer.txt", "content": "useful child output", "expected_sha256": None,
                        })])
                    if child:
                        return response("The child write attempt is settled.")
                    if turn == 0:
                        return response(calls=[replace(launch("writer", "start-gated-writer"), arguments={
                            "task": "writer", "max_turns": 6,
                            "capability": "read-only" if mode == "read-only" else "write",
                        }), wait("wait-gated-writer")])
                    return response("The child work is settled.")

                with patch.object(runner.providers[0].provider, "complete", side_effect=complete):
                    task = asyncio.create_task(runner.run("Write through a background child, then wait"))
                    try:
                        await asyncio.wait_for(write_requested.wait(), 3)
                        result = await asyncio.wait_for(task, 2)
                        self.assertEqual(result.stop_reason.value, "end_turn", result.error)
                        self.assertFalse(result.incomplete)
                        self.assertEqual(lock_states, [False, True])
                        rows = runner.sessions.tool_calls(result.session_id)
                        write = next(row for row in rows if row["tool_name"] == "write_file")
                        self.assertEqual(write["is_error"], mode != "write")
                        if mode == "write":
                            self.assertEqual((root / "answer.txt").read_text(), "useful child output")
                            self.assertTrue(write["metadata"]["checkpoint_id"])
                            self.assertIn("write_file", approvals)
                        else:
                            self.assertFalse((root / "answer.txt").exists())
                            self.assertIn("ApprovalDenied" if mode == "approval-denied" else "PolicyError",
                                          write["metadata"]["error_type"])
                        self.assertNotIn("running", {row["status"] for row in rows})
                        self.assertFalse(runner._mutation_lock.locked())
                        self.assertFalse(runner._run_tasks)
                        self.assertFalse(runner._cancel)
                    finally:
                        wait_entered.set()
                        if not task.done():
                            runner.cancel(next(iter(runner._cancel)))
                            await asyncio.wait_for(task, 4)
                        await runner.close()

    async def test_shared_child_fatal_cancels_inflight_root_provider_and_preserves_cause(self):
        await self._assert_root_provider_stop("cancelled")

    async def test_shared_child_fatal_retains_completed_root_response_at_cancellation(self):
        await self._assert_root_provider_stop("completed")

    async def test_recoverable_child_failure_does_not_cancel_inflight_root_provider(self):
        await self._assert_root_provider_stop("recoverable")

    async def _assert_root_provider_stop(self, ending):
        self.runner.config.agent.max_cost_usd = 0 if ending == "recoverable" else .05
        root_entered, child_failed, root_cancelled, root_drained, release_root = (asyncio.Event() for _ in range(5))
        root_turn = 0
        existing_tasks = asyncio.all_tasks()

        def observe(event):
            if event.type == "delegate.failed":
                child_failed.set()

        self.runner.events.subscribe(observe)

        async def complete_child(request):
            await root_entered.wait()
            if ending == "recoverable":
                raise ProviderError("Isolated child provider failure", usage=response(cost=.02).usage)
            return response("Child exceeded the shared dollar budget.", cost=.1)

        async def stream_root(request):
            nonlocal root_turn
            root_turn += 1
            if root_turn == 1:
                yield ProviderStreamEvent(type="completed", response=response(calls=[launch("costly child", "start-fatal-child")]))
                return
            if root_turn > 2:
                yield ProviderStreamEvent(type="completed", response=response("Root synthesized the received failure.", cost=.01))
                return
            root_entered.set()
            try:
                await release_root.wait()
            except asyncio.CancelledError:
                root_cancelled.set()
                if ending != "completed":
                    raise
                # A real completed frame competes with cancellation. Its known
                # charge must survive, with no synthetic extra charge or retry.
                yield ProviderStreamEvent(type="completed", response=response("Root completed at cancellation.", cost=.03))
                return
            finally:
                root_drained.set()
            yield ProviderStreamEvent(type="completed", response=response("Root recovered the isolated failure.", cost=.03))

        with patch.object(self.provider, "complete", side_effect=complete_child), patch.object(
            self.provider, "stream", side_effect=stream_root,
        ):
            task = asyncio.create_task(self.runner.run("Handle a child stop during a root request"))
            try:
                await asyncio.wait_for(root_entered.wait(), 3)
                await asyncio.wait_for(child_failed.wait(), 3)
                if ending == "recoverable":
                    self.assertFalse(root_cancelled.is_set())
                    self.assertFalse(task.done())
                    release_root.set()
                try:
                    result = await asyncio.wait_for(asyncio.shield(task), 2)
                except TimeoutError:
                    self.fail("Shared child fatal did not cancel and drain the gated root provider")
                self.assertTrue(root_drained.is_set())
                self.assertEqual(root_cancelled.is_set(), ending != "recoverable")
                self.assertEqual(result.stop_reason.value, "end_turn" if ending == "recoverable" else "budget", result.error)
                self.assertEqual(root_turn, 3 if ending == "recoverable" else 2)
                self.assertEqual(result.usage.requests, 4 if ending == "recoverable" else 3)
                expected_cost = .07 if ending == "recoverable" else .14 if ending == "completed" else .11
                self.assertAlmostEqual(result.usage.cost_usd, expected_cost)
                self.assertEqual(result.usage.cost_status, "incomplete" if ending == "cancelled" else "known")
                self.assertEqual(result.usage.to_dict(), self.runner.sessions.usage(result.session_id).to_dict())
                if ending != "recoverable":
                    self.assertIn("Model cost $0.1100 exceeded $0.05", result.error or "")
                    self.assertTrue(result.incomplete)
                self.assertNotIn("running", {row["status"] for row in self.runner.sessions.tool_calls(result.session_id)})
                self.assertFalse(self.runner._cancel)
                self.assertFalse(self.runner._run_tasks)
                self.assertFalse(self.runner._mutation_lock.locked())
            finally:
                if not task.done():
                    self.runner.cancel(next(iter(self.runner._cancel)))
                    await asyncio.wait_for(task, 4)
                await self.runner.close()
        self.assertFalse(asyncio.all_tasks() - existing_tasks)

    async def test_live_question_reply_and_selected_relay_before_sibling_finishes(self):
        b_entered, release_b, a_answered, b_relayed = (asyncio.Event() for _ in range(4))
        steps = {"root": 0, "A": 0, "B": 0}
        handles = {}
        requests = []
        self.request_snapshots = requests

        async def complete(request):
            agent = request.messages[0].content if request.metadata.get("delegated") else "root"
            step = steps[agent]
            steps[agent] += 1
            requests.append((agent, request))
            notes = [m for m in request.messages if m.metadata.get("coordination")]
            if agent == "B":
                if step == 0:
                    b_entered.set()
                    await release_b.wait()
                    return response(calls=[wait("B-wait")])
                self.assertTrue(any("selected finding" in m.content for m in notes))
                b_relayed.set()
                return response("B completed with the selected finding.")
            if agent == "A":
                if step == 0:
                    await b_entered.wait()
                    return response(calls=[send("parent", "question", "Which case?", "A-question"), wait("A-wait")])
                if step == 1:
                    self.assertTrue(any("Use case two" in m.content for m in notes))
                    self.assertFalse(release_b.is_set())
                    a_answered.set()
                    return response(calls=[send("parent", "finding", "selected finding", "A-finding")])
                return response("A completed the selected case.")
            for m in request.messages:
                if m.role == Role.TOOL and m.tool_name == "delegate_start" and not m.is_error:
                    handles[m.tool_call_id] = json.loads(m.content)["handle"]
            if step == 0:
                return response(calls=[launch("A", "launch-A"), launch("B", "launch-B")])
            if any("Which case?" in m.content for m in notes) and not a_answered.is_set():
                self.assertFalse(release_b.is_set())
                return response(calls=[send(handles["launch-A"], "reply", "Use case two", "root-reply"), wait("root-answer-wait")])
            if any("selected finding" in m.content for m in notes) and not release_b.is_set():
                self.assertTrue(a_answered.is_set())
                release_b.set()
                return response(calls=[send(handles["launch-B"], "finding", "selected finding", "root-relay"), wait("root-relay-wait")])
            if b_relayed.is_set():
                return response("Both investigations are settled.")
            return response(calls=[wait(f"root-wait-{step}")])

        with patch.object(self.provider, "complete", side_effect=complete):
            try:
                result = await asyncio.wait_for(self.runner.run("Coordinate the investigations"), 8)
            finally:
                release_b.set()
        self.assertEqual(result.stop_reason.value, "end_turn", result.error)
        self.assertTrue(a_answered.is_set())
        self.assertTrue(b_relayed.is_set())
        a_reply = [m for agent, req in requests if agent == "A" for m in req.messages
                   if m.metadata.get("coordination") and "Use case two" in m.content]
        self.assertTrue(a_reply)
        self.assertEqual(len({m.id for m in a_reply}), 1)
        self.assertTrue(all(m.role != Role.USER for m in a_reply))
        validate_tool_call_order(self.runner.sessions.messages(result.session_id))
        self.assertNotIn("running", {r["status"] for r in self.runner.sessions.tool_calls(result.session_id)})
        self.assertEqual(result.usage.to_dict(), self.runner.sessions.usage(result.session_id).to_dict())
        view = SwarmView()
        visible_states = set()
        for event in self.events:
            view.adopt(event)
            visible_states.update(node.state for node in view.nodes.values())
        self.assertIn("question", visible_states)
        self.assertIn("blocked", visible_states)
        self.assertIn("waiting", visible_states)
        self.assertTrue(all(node.state == "done" for node in view.nodes.values()))

    async def test_recoverable_failure_allows_replacement_while_sibling_is_in_flight(self):
        b_entered, release_b, c_entered = (asyncio.Event() for _ in range(3))
        root_turn = 0

        async def complete(request):
            nonlocal root_turn
            task = request.messages[0].content if request.metadata.get("delegated") else "root"
            if task == "B":
                b_entered.set()
                await release_b.wait()
                return response("B preserved its evidence.")
            if task == "A":
                await b_entered.wait()
                raise ProviderError("Recoverable child provider failure")
            if task == "C":
                self.assertFalse(release_b.is_set())
                c_entered.set()
                release_b.set()
                return response("Replacement C completed.")
            root_turn += 1
            if root_turn == 1:
                return response(calls=[launch("A", "fail-A"), launch("B", "slow-B")])
            if any(m.metadata.get("child_handle") and m.is_error for m in request.messages):
                if not c_entered.is_set():
                    return response(calls=[launch("C", "replacement-C"), wait("replacement-wait")])
                return response("A failed; B and replacement C completed.")
            return response(calls=[wait(f"recovery-wait-{root_turn}")])

        with patch.object(self.provider, "complete", side_effect=complete):
            try:
                result = await asyncio.wait_for(self.runner.run("Recover one failed child"), 8)
            finally:
                release_b.set()
        self.assertTrue(c_entered.is_set())
        self.assertEqual(result.stop_reason.value, "end_turn", result.error)
        self.assertEqual(len([e for e in self.events if e.type == "delegate.started"]), 3)
        self.assertEqual(len([e for e in self.events if e.type == "delegate.failed"]), 1)
        self.assertEqual(len([e for e in self.events if e.type == "delegate.completed"]), 2)

    async def test_recursive_questions_with_one_provider_slot_and_idle_wait(self):
        self.runner.config.agent.swarm_max_concurrent = 1
        parent_waiting, root_entered, release_answer, grandchild_replied = (asyncio.Event() for _ in range(4))
        turns = {}
        calls_before_release = []
        completed = []

        async def complete(request):
            depth = request.metadata.get("delegation_depth", 0)
            turn = turns.get(depth, 0)
            turns[depth] = turn + 1
            notes = "\n".join(m.content for m in request.messages if m.metadata.get("coordination"))
            if depth == 2:
                if turn == 0:
                    return response(calls=[send("parent", "question", "Grandchild case?", "grand-question"), wait("grand-wait")])
                self.assertIn("Nested answer", notes)
                grandchild_replied.set()
                return response("Grandchild received the answer.")
            if depth == 1:
                handles = [json.loads(m.content)["handle"] for m in request.messages
                           if m.tool_name == "delegate_start" and not m.is_error]
                if turn == 0:
                    return response(calls=[launch("grandchild", "launch-grand"), wait("child-event")])
                if "Grandchild case?" in notes and "Root answer" not in notes:
                    parent_waiting.set()
                    return response(calls=[send("parent", "question", "Need root answer", "child-question"), wait("child-answer")])
                if "Root answer" in notes and not grandchild_replied.is_set():
                    return response(calls=[send(handles[0], "reply", "Nested answer", "child-reply"), wait("child-terminal")])
                completed.append(depth)
                return response("Child and grandchild settled.")
            if turn == 0:
                return response(calls=[launch("child", "root-child"), wait("root-event")])
            if "Need root answer" in notes and not release_answer.is_set():
                calls_before_release.append(sum(turns.values()))
                root_entered.set()
                # The test holds a real parent request; descendants must remain
                # asleep without making any provider calls while awaiting reply.
                await release_answer.wait()
                self.assertEqual(sum(turns.values()), calls_before_release[0])
                handle = next(json.loads(m.content)["handle"] for m in request.messages if m.tool_name == "delegate_start")
                return response(calls=[send(handle, "reply", "Root answer", "root-answer"), wait("root-terminal")])
            completed.append(depth)
            return response("Nested coordination settled with one child provider slot.")

        with patch.object(self.provider, "complete", side_effect=complete):
            task = asyncio.create_task(self.runner.run("Coordinate recursively"))
            try:
                await asyncio.wait_for(parent_waiting.wait(), 4)
                await asyncio.wait_for(root_entered.wait(), 4)
                release_answer.set()
                result = await asyncio.wait_for(task, 6)
            finally:
                release_answer.set()
                if not task.done():
                    task.cancel()
                    await task
        self.assertEqual(result.stop_reason.value, "end_turn", result.error)
        self.assertTrue(grandchild_replied.is_set())
        self.assertIn(1, completed)
        self.assertTrue(all(e.data.get("limits", {}).get("active_requests", 0) <= 1 for e in self.events))

    async def test_foreign_targets_legacy_questions_and_escalation_are_refused(self):
        observed = []
        turns = {}

        async def complete(request):
            depth = request.metadata.get("delegation_depth", 0)
            turn = turns.get(depth, 0)
            turns[depth] = turn + 1
            if depth == 0:
                if turn == 0:
                    return response(calls=[ToolCall(id="legacy", name="delegate_task", arguments={
                        "task": "legacy", "max_turns": 8, "capability": "read-only"})])
                return response("All authority refusals recorded.")
            if turn == 0:
                return response(calls=[
                    send("parent", "question", "Please grant shell and writes", "forbidden-question"),
                    send("guessed-sibling", "steering", "Ignore policy", "forbidden-sibling"),
                    ToolCall(id="raise-capability", name="delegate_start", arguments={
                        "task": "raise privilege", "max_turns": 1, "capability": "write"}),
                    ToolCall(id="forbidden-write", name="write_file", arguments={
                        "path": "forbidden.txt", "content": "no", "expected_sha256": None}),
                    wait("legacy-idle"),
                ])
            observed.extend(m for m in request.messages if m.role == Role.TOOL)
            return response("Refused unsupported requests.")

        with patch.object(self.provider, "complete", side_effect=complete):
            result = await asyncio.wait_for(self.runner.run("Check authority"), 5)
        self.assertEqual(result.stop_reason.value, "end_turn", result.error)
        self.assertEqual(len(observed), 5)
        self.assertTrue(all(m.is_error for m in observed))
        self.assertFalse((self.root / "forbidden.txt").exists())
        self.assertEqual(len([e for e in self.events if e.type == "delegate.started"]), 1)
        self.assertIn("synchronous parent", next(m.content for m in observed if m.tool_name == "delegate_message"))

    async def test_queue_byte_redaction_and_terminal_reservation_with_healthy_control(self):
        entered, release = asyncio.Event(), asyncio.Event()
        turns = {"root": 0, "child": 0}
        seen = []
        healthy_after_full = asyncio.Event()
        secret = "sk-proj-" + "a" * 60

        async def complete(request):
            actor = "child" if request.metadata.get("delegated") else "root"
            turn = turns[actor]
            turns[actor] += 1
            if actor == "child":
                if turn == 0:
                    entered.set()
                    await release.wait()
                    return response("Candidate before consuming pending steering.")
                if turn == 1:
                    return response(calls=[send("parent", "question", "Inbox drained; ready for a new message", "drained-question"), wait("after-drain-wait")])
                if any("healthy after full" in m.content for m in request.messages if m.metadata.get("coordination")):
                    healthy_after_full.set()
                return response("Terminal survives a saturated notification inbox.")
            if turn == 0:
                return response(calls=[launch("child", "launch-limit")])
            handle = next(json.loads(m.content)["handle"] for m in request.messages if m.tool_name == "delegate_start")
            if turn == 1:
                await entered.wait()
                return response(calls=[send(handle, "steering", "界" * 700, "too-many-bytes"),
                                       send(handle, "steering", "healthy " + secret, "healthy-redacted"),
                                       *[send(handle, "finding", f"notice {i}", f"notice-{i}") for i in range(32)]])
            seen.extend(m for m in request.messages if m.role == Role.TOOL and m.tool_name == "delegate_message")
            if turn == 2:
                release.set()
                return response(calls=[wait("wait-for-inbox-drain")])
            if turn == 3:
                return response(calls=[send(handle, "reply", "healthy after full", "healthy-after-full"), wait("wait-for-terminal")])
            return response("Notification limits applied; terminal child evidence preserved.")

        with patch.object(self.provider, "complete", side_effect=complete):
            try:
                result = await asyncio.wait_for(self.runner.run("Bound sparse traffic"), 6)
            finally:
                release.set()
        self.assertEqual(result.stop_reason.value, "end_turn", result.error)
        self.assertTrue(next(m for m in seen if m.tool_call_id == "too-many-bytes").is_error)
        self.assertFalse(next(m for m in seen if m.tool_call_id == "healthy-redacted").is_error)
        self.assertTrue(next(m for m in seen if m.tool_call_id == "notice-31").is_error)
        content = json.dumps(self.runner.sessions.export(result.session_id))
        # Root model arguments can retain their original input; stored coordination
        # and observer payloads must use the canonical redactor.
        queued = self.runner.sessions._connection.execute("SELECT content FROM coordination_messages").fetchall()
        self.assertEqual(len(queued), 34)
        self.assertTrue(healthy_after_full.is_set())
        self.assertNotIn(secret, "".join(row[0] for row in queued))
        self.assertIn("[REDACTED", "".join(row[0] for row in queued))
        self.assertIn("Terminal survives", content)

    async def test_cancel_and_close_drain_waiting_provider_and_settle_completed_usage(self):
        for action in ("cancel", "close", "cost"):
            with self.subTest(action=action):
                if action != "cancel":
                    self.runner = await build_runner(self.root, config=make_config(self.root), interactive=False)
                    self.addAsyncCleanup(self.runner.close)
                    self.provider = self.runner.providers[0].provider
                entered, finish, drained = (asyncio.Event() for _ in range(3))
                root_turns = 0
                if action == "cost":
                    self.runner.config.agent.max_cost_usd = .05

                async def complete(request, entered=entered, finish=finish, drained=drained, action=action):
                    nonlocal root_turns
                    if request.metadata.get("delegated"):
                        entered.set()
                        try:
                            await finish.wait()
                        except asyncio.CancelledError:
                            # Completed transport wins cancellation. Its charge
                            # must survive the shared stop exactly once.
                            return response("Completed child response at cancellation", cost=.03)
                        finally:
                            drained.set()
                        return response("Child exhausted the shared dollar budget", cost=.1)
                    root_turns += 1
                    if root_turns == 1:
                        return response(calls=[launch("gated", f"launch-{action}"), wait(f"wait-{action}")])
                    return response("Final synthesis after settlement")

                with patch.object(self.provider, "complete", side_effect=complete):
                    task = asyncio.create_task(self.runner.run("Stop an owned tree"))
                    try:
                        await asyncio.wait_for(entered.wait(), 3)
                        if action == "cancel":
                            self.runner.cancel(next(iter(self.runner._cancel)))
                        elif action == "close":
                            await asyncio.wait_for(self.runner.close(), 5)
                        else:
                            finish.set()
                        result = await asyncio.wait_for(task, 5)
                    finally:
                        finish.set()
                        if not task.done():
                            task.cancel()
                            await task
                self.assertTrue(drained.is_set())
                self.assertEqual(result.stop_reason.value, "budget" if action == "cost" else "cancelled", result.error)
                self.assertEqual(result.usage.requests, 2)
                self.assertAlmostEqual(result.usage.cost_usd, .11 if action == "cost" else .04)
                if action != "close":
                    self.assertNotIn("running", {r["status"] for r in self.runner.sessions.tool_calls(result.session_id)})
                    self.assertEqual(result.usage.to_dict(), self.runner.sessions.usage(result.session_id).to_dict())

    async def test_last_turn_cancels_unanswered_question_without_publishing_success(self):
        self.runner.config.agent.max_turns = 2
        asked = asyncio.Event()
        root_turn = 0

        async def complete(request):
            nonlocal root_turn
            if request.metadata.get("delegated"):
                asked.set()
                return response(calls=[send("parent", "question", "Need an answer before writing", "last-question"), wait("last-child-wait")])
            root_turn += 1
            if root_turn == 1:
                return response(calls=[launch("questioner", "launch-last"), wait("last-root-wait")])
            return response("Premature success must remain private")

        with patch.object(self.provider, "complete", side_effect=complete):
            result = await asyncio.wait_for(self.runner.run("Use a hard turn bound"), 5)
        self.assertTrue(asked.is_set())
        self.assertTrue(result.incomplete)
        self.assertNotIn("Premature success", result.text)
        self.assertFalse(any("Premature success" in e.data.get("text", "") for e in self.events if e.type == "model.completed"))
        self.assertNotIn("running", {r["status"] for r in self.runner.sessions.tool_calls(result.session_id)})

    async def test_background_write_settles_before_verification_and_invalidates_old_revision(self):
        self.runner.config.agent.auto_verify = True
        write_done, release = asyncio.Event(), asyncio.Event()
        root_turn, child_turn = 0, 0
        verification_contents = []

        async def verify(context, steps):
            verification_contents.append((self.root / "owned.txt").read_text())
            return VerificationReport(ok=True)

        async def complete(request):
            nonlocal root_turn, child_turn
            if request.metadata.get("delegated"):
                child_turn += 1
                if child_turn == 1:
                    return response(calls=[ToolCall(id="owned-write", name="write_file", arguments={
                        "path": "owned.txt", "content": "child revision", "expected_sha256": None})])
                write_done.set()
                await release.wait()
                return response("Child completed its write.")
            root_turn += 1
            if root_turn == 1:
                return response(calls=[replace(launch("writer", "launch-writer"), arguments={
                    "task": "writer", "max_turns": 6, "capability": "write"})])
            if root_turn == 2:
                await write_done.wait()
                self.assertEqual(verification_contents, [])
                release.set()
                return response("Candidate before terminal child settlement")
            return response("Child revision is settled and verified.")

        with patch.object(self.provider, "complete", side_effect=complete), patch(
            "borealis_coder.agent.runner.VerificationPlanner.run", side_effect=verify,
        ):
            try:
                result = await asyncio.wait_for(self.runner.run("Implement and verify child revision"), 6)
            finally:
                release.set()
        self.assertEqual(result.stop_reason.value, "end_turn", result.error)
        self.assertEqual(result.changed_files, ["owned.txt"])
        self.assertEqual(verification_contents, ["child revision"])
        self.assertTrue(result.verification and result.verification["ok"])

    async def test_terminal_evidence_is_searchable_readable_and_quotable_after_preview(self):
        fact = "BEYOND_PREVIEW: the verified value is 731."
        (self.root / "source.txt").write_text("filler\n" * 4000 + fact + "\n")
        self.runner.config.context.tool_output_chars = 12000
        root_turn, child_turn = 0, 0
        references = {}
        results = []

        async def complete(request):
            nonlocal root_turn, child_turn
            if request.metadata.get("delegated"):
                child_turn += 1
                if child_turn == 1:
                    return response(calls=[ToolCall(id="evidence-source", name="read_file", arguments={
                        "path": "source.txt", "start_line": None, "end_line": None, "max_chars": 100000})])
                return response("Source read; preserve the complete evidence.")
            root_turn += 1
            notes = [m for m in request.messages if m.metadata.get("child_handle")]
            if root_turn == 1:
                return response(calls=[launch("source reader", "source-reader"), wait("source-ready")])
            if notes and root_turn == 2:
                handle = notes[0].metadata["child_handle"]
                return response(calls=[ToolCall(id="search-terminal", name="search_history", arguments={
                    "query": handle + ".result", "after": 0, "offset": 0, "artifact_after": None})])
            last = request.messages[-1]
            if last.tool_call_id == "search-terminal":
                payload = json.loads(last.content.split("\n", 1)[1])
                for row in payload["tools"]:
                    references.update({item["tool"]: item for item in row["metadata"].get("evidence", [])})
                source = references["read_file"]
                return response(calls=[ToolCall(id="read-returned", name="read_artifact", arguments={
                    "artifact_id": source["output_artifact"]["artifact_id"], "offset": 0, "limit": 500, "query": fact}),
                    ToolCall(id="quote-returned", name="record_finding", arguments={
                        "tool_call_id": source["tool_call_id"], "quote": fact}),
                    ToolCall(id="search-fact", name="search_history", arguments={
                        "query": fact, "after": 0, "offset": 0, "artifact_after": None})])
            results.extend(m for m in request.messages if m.tool_call_id in {"read-returned", "quote-returned", "search-fact"})
            return response("Returned evidence was read, searched and quoted.")

        with patch.object(self.provider, "complete", side_effect=complete):
            result = await asyncio.wait_for(self.runner.run("Inspect child evidence beyond preview"), 6)
        self.assertEqual(result.stop_reason.value, "end_turn", result.error)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(not m.is_error for m in results))
        self.assertTrue(all(fact in m.content for m in results))

    async def test_reopen_preserves_undelivered_history_without_replaying_workers(self):
        root_entered, question_sent = asyncio.Event(), asyncio.Event()
        root_turn = 0

        def observe(event):
            if event.type == "delegate.coordination" and event.data.get("state") == "question":
                question_sent.set()

        self.runner.events.subscribe(observe)

        async def complete(request):
            nonlocal root_turn
            if request.metadata.get("delegated"):
                await root_entered.wait()
                return response(calls=[send("parent", "question", "PENDING_RECOVERY_QUESTION", "persist-question"), wait("persist-wait")])
            root_turn += 1
            if root_turn == 1:
                return response(calls=[launch("persistent questioner", "persist-launch")])
            root_entered.set()
            await asyncio.Event().wait()
            self.fail("Unreachable")

        with patch.object(self.provider, "complete", side_effect=complete):
            task = asyncio.create_task(self.runner.run("Persist original user task"))
            try:
                await asyncio.wait_for(question_sent.wait(), 4)
                session_id = next(iter(self.runner._cancel))
                self.runner.cancel(session_id)
                result = await asyncio.wait_for(task, 5)
            finally:
                if not task.done():
                    task.cancel()
                    await task
        self.assertTrue(result.incomplete)
        path = self.runner.sessions.path
        await self.runner.close()
        # Reopen the actual SQLite file and inspect its committed pending message.
        store = SessionStore(path)
        self.assertEqual(store.get_session(session_id).status, "idle")
        pending = store._connection.execute("SELECT message_id FROM coordination_messages WHERE state='pending'").fetchall()
        self.assertEqual(len(pending), 1)
        message_id = pending[0][0]
        store.close()
        self.runner = await build_runner(self.root, config=make_config(self.root), interactive=False)
        self.addAsyncCleanup(self.runner.close)
        self.provider = self.runner.providers[0].provider
        resumed = []

        async def resume(request):
            self.assertFalse(request.metadata.get("delegated"))
            resumed.append(request)
            return response("Previous child work was interrupted; no automatic retry.")

        with patch.object(self.provider, "complete", side_effect=resume):
            for _ in range(2):
                continuation = await self.runner.run("continue", session_id=session_id)
                self.assertEqual(continuation.stop_reason.value, "end_turn", continuation.error)
                self.assertEqual(continuation.mutation_tracking, "complete")
        durable = self.runner.sessions.messages(session_id)
        recovered = [m for m in durable if m.id == message_id]
        self.assertEqual(len(recovered), 1)
        self.assertTrue(recovered[0].metadata["interrupted"])
        self.assertEqual(recovered[0].role, Role.ASSISTANT)
        self.assertIn("PENDING_RECOVERY_QUESTION", recovered[0].content)
        self.assertEqual(len(resumed), 2)
        self.assertEqual(len([r for r in self.runner.sessions.tool_calls(session_id) if r["tool_name"] == "delegate_start"]), 1)
        instructions = self.runner.sessions.task_state(session_id)["instructions"]
        self.assertFalse(any("PENDING_RECOVERY" in item["text"] for item in instructions))

    async def test_existing_v7_database_migrates_without_losing_source_history(self):
        result = await self.runner.run("Original persisted task")
        path = self.runner.sessions.path
        await self.runner.close()
        import sqlite3

        connection = sqlite3.connect(path)
        try:
            with connection:
                connection.execute("DROP TABLE coordination_messages")
                connection.execute("DROP TABLE coordination_children")
                connection.execute("UPDATE schema_meta SET value='7' WHERE key='version'")
        finally:
            connection.close()
        for _ in range(2):
            store = SessionStore(path)
            try:
                self.assertEqual(store._connection.execute("SELECT value FROM schema_meta WHERE key='version'").fetchone()[0], "8")
                self.assertEqual(store.messages(result.session_id)[0].content, "Original persisted task")
                self.assertEqual(store._connection.execute("SELECT count(*) FROM coordination_messages").fetchone()[0], 0)
            finally:
                store.close()

    async def test_fresh_steering_survives_full_compaction_and_incremental_reuse(self):
        self.runner.config.agent.deterministic_compaction = True
        self.runner.config.context.compact_tool_output_tokens = 1000
        (self.root / "large.txt").write_text("large historical evidence " * 2500)
        question_ready = asyncio.Event()
        turns = {"root": 0, "child": 0}
        seen_notes = []

        async def complete(request):
            actor = "child" if request.metadata.get("delegated") else "root"
            turn = turns[actor]
            turns[actor] += 1
            notes = [m for m in request.messages if m.metadata.get("coordination")]
            if actor == "child":
                if turn < 2:
                    return response(calls=[ToolCall(id=f"large-read-{turn}", name="read_file", arguments={
                        "path": "large.txt", "start_line": None, "end_line": None, "max_chars": 100000})])
                if turn == 2:
                    question_ready.set()
                    return response(calls=[send("parent", "question", "Need compacted steering", "compact-question"), wait("compact-answer")])
                seen_notes.extend(m for m in notes if "FRESH_UNTRUSTED_STEERING" in m.content)
                if turn < 23:
                    return response(calls=[ToolCall(id=f"after-answer-read-{turn}", name="list_directory", arguments={
                        "path": ".", "depth": 1, "include_hidden": False, "max_entries": 10})])
                return response("Child completed using fresh steering across compaction.")
            if turn == 0:
                return response(calls=[replace(launch("child", "compact-launch"), arguments={
                    "task": "child", "max_turns": 30, "capability": "read-only"}), wait("compact-event")])
            if question_ready.is_set() and not seen_notes:
                handle = next(json.loads(m.content)["handle"] for m in request.messages if m.tool_name == "delegate_start")
                return response(calls=[send(handle, "reply", "FRESH_UNTRUSTED_STEERING", "compact-reply"), wait("compact-terminal")])
            return response("Coordination and compacted history are preserved.")

        with patch.object(self.provider, "complete", side_effect=complete):
            result = await asyncio.wait_for(self.runner.run("Compact coordinated child history"), 8)
        self.assertEqual(result.stop_reason.value, "end_turn", result.error)
        self.assertTrue(seen_notes)
        self.assertEqual(len({m.id for m in seen_notes}), 1)
        self.assertTrue(any(e.type == "context.compacted" and e.run_id != result.run_id for e in self.events))
        self.assertTrue(any(e.type == "context.reused" and e.run_id != result.run_id for e in self.events))
        scope = next(e.data["delegated_run_id"] for e in self.events if e.type == "delegate.started")
        self.assertTrue(self.runner.sessions.search_history(scope, self.root, "FRESH_UNTRUSTED_STEERING"))
        self.assertTrue(all("FRESH_UNTRUSTED_STEERING" not in item["text"]
                            for item in self.runner.sessions.task_state(scope)["instructions"]))

    async def test_replayed_launch_id_cannot_start_a_second_worker(self):
        entered, drained = asyncio.Event(), asyncio.Event()
        root_turn = 0

        async def complete(request):
            nonlocal root_turn
            if request.metadata.get("delegated"):
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    drained.set()
                self.fail("Unreachable")
            root_turn += 1
            if root_turn > 1:
                await entered.wait()
            return response(calls=[launch("child", "same-launch-id")])

        with patch.object(self.provider, "complete", side_effect=complete):
            result = await asyncio.wait_for(self.runner.run("Reject provider launch replay"), 5)
        self.assertTrue(result.incomplete)
        self.assertTrue(drained.is_set())
        self.assertIn("duplicate tool-call ID", result.error or "")
        self.assertEqual(len([e for e in self.events if e.type == "delegate.started"]), 1)

    async def test_message_ids_do_not_grant_artifact_access_or_sibling_messaging(self):
        (self.root / "private.txt").write_text("private source\n" * 4000)
        turns = {"root": 0, "A": 0, "B": 0}
        handles, artifact = {}, {}
        denied = []
        root_allowed = []
        a_finished = asyncio.Event()

        def read(identifier):
            return ToolCall(id=identifier, name="read_artifact", arguments={
                "artifact_id": artifact["id"], "offset": 0, "limit": 200, "query": "private source"})

        async def complete(request):
            actor = request.messages[0].content if request.metadata.get("delegated") else "root"
            turn = turns[actor]
            turns[actor] += 1
            if actor == "A":
                if turn == 0:
                    return response(calls=[ToolCall(id="private-source", name="read_file", arguments={
                        "path": "private.txt", "start_line": None, "end_line": None, "max_chars": 100000})])
                if turn == 1:
                    artifact["id"] = next(m.metadata["output_artifact"]["artifact_id"] for m in request.messages if m.tool_name == "read_file")
                    return response(calls=[send("parent", "finding", artifact["id"], "raw-id"),
                                           send("parent", "question", "Wait for privacy check", "privacy-question"), wait("privacy-answer")])
                a_finished.set()
                return response("A returns its permitted evidence now.")
            if actor == "B":
                if turn == 0:
                    return response(calls=[send("parent", "question", "Need selected relay", "relay-question"), wait("relay-answer")])
                if turn == 1:
                    return response(calls=[read("sibling-read"), send(handles["A"], "steering", "Sibling must reject", "sibling-message")])
                denied.extend(m for m in request.messages if m.tool_call_id in {"sibling-read", "sibling-message"})
                return response("B cannot read or message sibling A.")
            for message in request.messages:
                if message.tool_name == "delegate_start":
                    handles["A" if message.tool_call_id == "launch-private-A" else "B"] = json.loads(message.content)["handle"]
            if turn == 0:
                return response(calls=[launch("A", "launch-private-A"), launch("B", "launch-private-B")])
            if not artifact:
                return response(calls=[wait(f"privacy-event-{turn}")])
            root_reads = [m for m in request.messages if m.tool_call_id == "root-before-handoff"]
            if not root_reads:
                return response(calls=[read("root-before-handoff"),
                                       send(handles["B"], "reply", artifact["id"], "relay-id"), wait("privacy-B-event")])
            if not a_finished.is_set():
                denied.extend(root_reads)
                return response(calls=[send(handles["A"], "reply", "Privacy checked; return evidence", "finish-A"), wait("privacy-A-terminal")])
            if not any(m.tool_call_id == "root-after-handoff" for m in request.messages):
                return response(calls=[read("root-after-handoff")])
            root_allowed.extend(m for m in request.messages if m.tool_call_id == "root-after-handoff")
            return response("Only the terminal handoff granted parent evidence access.")

        with patch.object(self.provider, "complete", side_effect=complete):
            result = await asyncio.wait_for(self.runner.run("Check selective evidence authority"), 8)
        self.assertEqual(result.stop_reason.value, "end_turn", result.error)
        self.assertEqual(len(denied), 3)
        self.assertTrue(all(m.is_error for m in denied))
        self.assertEqual(len(root_allowed), 1)
        self.assertFalse(root_allowed[0].is_error)

    async def test_tiny_output_cap_refuses_launch_before_child_dispatch(self):
        self.runner.config.context.tool_output_chars = 300
        provider = self.provider
        assert isinstance(provider, MockProvider)
        provider.enqueue(response(calls=[launch("child", "tiny-launch")]), response("Launch was refused before admission."))
        result = await self.runner.run("Use a tiny output cap")
        self.assertEqual(result.stop_reason.value, "end_turn", result.error)
        self.assertFalse(any(e.type == "delegate.started" for e in self.events))
        self.assertEqual(result.usage.requests, 2)
        self.assertTrue(next(row for row in self.runner.sessions.tool_calls(result.session_id) if row["tool_name"] == "delegate_start")["is_error"])

    async def test_cancellation_during_terminal_storage_commits_once_and_preserves_handoff(self):
        entered, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()
        original = self.runner.sessions.handoff_output_artifacts

        def gated_handoff(*args, **kwargs):
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(4):
                raise AssertionError("Terminal handoff release was not reached")
            return original(*args, **kwargs)

        async def complete(request):
            if request.metadata.get("delegated"):
                return response("Completed evidence before cancellation.")
            return response(calls=[launch("terminal child", "terminal-launch"), wait("terminal-wait")])

        with patch.object(self.provider, "complete", side_effect=complete), patch.object(
            self.runner.sessions, "handoff_output_artifacts", side_effect=gated_handoff,
        ):
            task = asyncio.create_task(self.runner.run("Cancel at terminal commit"))
            try:
                await asyncio.wait_for(entered.wait(), 3)
                session_id = next(iter(self.runner._cancel))
                self.runner.cancel(session_id)
                self.assertFalse(task.done())
                release.set()
                result = await asyncio.wait_for(task, 5)
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                    await task
        self.assertEqual(result.stop_reason.value, "cancelled", result.error)
        receipts = [row for row in self.runner.sessions.tool_calls(session_id) if row["tool_name"] == "delegate_result"]
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0]["status"], "completed")
        artifact = receipts[0]["metadata"]["output_artifact"]["artifact_id"]
        retained = self.runner.sessions.read_output_artifact(session_id, self.root, artifact,
                                                           offset=0, limit=1000, query="")
        self.assertIn("Completed evidence", retained["content"])
        self.assertEqual(result.usage.requests, 2)

    async def test_child_services_a_queued_grandchild_terminal_before_its_own_conclusion(self):
        child_candidate, grandchild_settled = asyncio.Event(), asyncio.Event()
        turns = {0: 0, 1: 0, 2: 0}
        child_observed = []

        def observe(event):
            if event.type == "delegate.completed" and event.data["depth"] == 2:
                grandchild_settled.set()

        self.runner.events.subscribe(observe)

        async def complete(request):
            depth = request.metadata.get("delegation_depth", 0)
            turn = turns[depth]
            turns[depth] += 1
            if depth == 2:
                await child_candidate.wait()
                return response("GRANDCHILD_TERMINAL_VALUE is ready.")
            if depth == 1:
                if turn == 0:
                    return response(calls=[launch("grandchild", "queued-grandchild")])
                if turn == 1:
                    child_candidate.set()
                    await grandchild_settled.wait()
                    return response("Premature child conclusion before observing queued result")
                notes = [m for m in request.messages if m.metadata.get("coordination")]
                child_observed.extend(m for m in notes if "GRANDCHILD_TERMINAL_VALUE" in m.content)
                return response("Child observed the queued grandchild terminal before concluding.")
            if turn == 0:
                return response(calls=[launch("child", "queued-child"), wait("queued-root-wait")])
            return response("The nested tree is settled.")

        with patch.object(self.provider, "complete", side_effect=complete):
            result = await asyncio.wait_for(self.runner.run("Observe queued child terminals"), 6)
        self.assertEqual(result.stop_reason.value, "end_turn", result.error)
        self.assertTrue(child_observed)
        self.assertEqual(turns[1], 3)
