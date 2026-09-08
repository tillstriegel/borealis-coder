from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from borealis_coder.agent import build_runner
from borealis_coder.errors import ProtocolError, SessionError
from borealis_coder.models import Event
from borealis_coder.protocol import ACPServer
from borealis_coder.sessions import SessionStore


class FakeConnection:
    def __init__(self):
        self.notifications = []
        self.requests = []

    async def notify(self, method, params):
        self.notifications.append((method, params))

    async def request(self, method, params, timeout=None):
        self.requests.append((method, params))
        return {"optionId": "allow_once"}


class ACPTests(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_session_resume_closes_the_unowned_runner(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with patch.dict(
                os.environ,
                {"BOREALIS_DATA_DIR": str(root / "data"), "BOREALIS_PROVIDER": "mock"},
            ):
                runner = await build_runner(root)
                server = ACPServer()
                await server.handle("initialize", {"protocolVersion": 2})
                try:
                    with (
                        patch("borealis_coder.protocol.acp.build_runner", return_value=runner),
                        self.assertRaises(SessionError),
                    ):
                        await server.handle(
                            "session/resume", {"cwd": str(root), "sessionId": "missing"}
                        )
                    self.assertEqual(server.runners, {})
                    with self.assertRaises(sqlite3.ProgrammingError):
                        runner.sessions.list_sessions()
                finally:
                    await runner.close()

    async def test_failed_session_setup_closes_the_unowned_runner(self):
        for method, store_method in (
            ("session/new", "create_session"),
            ("session/resume", "get_session"),
        ):
            for error_type in (SessionError, asyncio.CancelledError):
                with (
                    self.subTest(method=method, error=error_type.__name__),
                    tempfile.TemporaryDirectory() as td,
                ):
                    root = Path(td)
                    with patch.dict(
                        os.environ,
                        {
                            "BOREALIS_DATA_DIR": str(root / "data"),
                            "BOREALIS_PROVIDER": "mock",
                        },
                    ):
                        runner = await build_runner(root)
                        server = ACPServer()
                        await server.handle("initialize", {"protocolVersion": 2})
                        try:
                            with (
                                patch(
                                    "borealis_coder.protocol.acp.build_runner",
                                    return_value=runner,
                                ),
                                patch.object(runner.sessions, store_method, side_effect=error_type),
                                self.assertRaises(error_type),
                            ):
                                await server.handle(
                                    method, {"cwd": str(root), "sessionId": "missing"}
                                )
                            self.assertEqual(server.runners, {})
                            with self.assertRaises(sqlite3.ProgrammingError):
                                runner.sessions.list_sessions()
                        finally:
                            await runner.close()

    async def test_prompt_waits_for_a_finishing_run(self):
        class FinishingRunner:
            def __init__(self):
                self.run_arguments = None

            def accepts_steering(self, session_id):
                return False

            async def run(self, prompt, **arguments):
                self.run_arguments = (prompt, arguments)

        server = ACPServer()
        fake = FakeConnection()
        server.connection = cast(Any, fake)
        runner = FinishingRunner()
        server.runners["session_1"] = cast(Any, runner)

        accepted = await server._session_prompt(
            {"sessionId": "session_1", "prompt": [{"type": "text", "text": "continue"}]}
        )
        self.assertEqual(accepted, {})
        await server.tasks["session_1"]

        assert runner.run_arguments is not None
        prompt, arguments = runner.run_arguments
        self.assertEqual(prompt, "continue")
        self.assertTrue(arguments["wait_for_active_run"])
        updates = [params["update"] for _, params in fake.notifications]
        self.assertEqual([item["sessionUpdate"] for item in updates], ["user_message"])

    async def test_resume_stops_active_prompts_before_closing_the_old_runner(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with patch.dict(
                os.environ,
                {"BOREALIS_DATA_DIR": str(root / "data"), "BOREALIS_PROVIDER": "mock"},
            ):
                server = ACPServer()
                server.connection = cast(Any, FakeConnection())
                await server.handle("initialize", {"protocolVersion": 2})
                created = await server.handle("session/new", {"cwd": str(root)})
                session_id = created["sessionId"]
                old = server.runners[session_id]
                started = asyncio.Event()
                stopped = asyncio.Event()

                async def busy(*args, **kwargs):
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        # Prompt cleanup still has access to its open session store.
                        old.sessions.get_session(session_id)
                        stopped.set()

                try:
                    with patch.object(old, "run", side_effect=busy):
                        await server.handle(
                            "session/prompt",
                            {
                                "sessionId": session_id,
                                "prompt": [{"type": "text", "text": "work"}],
                            },
                        )
                        await asyncio.wait_for(started.wait(), timeout=2)
                        task = server.tasks[session_id]
                        await server.handle(
                            "session/resume", {"cwd": str(root), "sessionId": session_id}
                        )
                    self.assertTrue(task.done())
                    self.assertTrue(stopped.is_set())
                    self.assertNotIn(session_id, server.tasks)
                    self.assertIsNot(server.runners[session_id], old)
                    with self.assertRaises(sqlite3.ProgrammingError):
                        old.sessions.list_sessions()
                    self.assertEqual(
                        server.runners[session_id].sessions.get_session(session_id).id, session_id
                    )
                finally:
                    await server.close()

    async def test_cancelled_session_close_still_closes_the_runner(self):
        exception_handler = patch.object(asyncio.get_running_loop(), "call_exception_handler")
        errors = exception_handler.start()
        self.addCleanup(exception_handler.stop)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with patch.dict(
                os.environ,
                {"BOREALIS_DATA_DIR": str(root / "data"), "BOREALIS_PROVIDER": "mock"},
            ):
                runner = await build_runner(root)
                server = ACPServer()
                server.runners["session_1"] = runner
                started = asyncio.Event()
                stopping = asyncio.Event()

                async def work():
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        stopping.set()
                        await asyncio.Event().wait()

                task = asyncio.create_task(work())
                server.tasks["session_1"] = task
                await started.wait()
                closing = asyncio.create_task(server._session_close({"sessionId": "session_1"}))
                try:
                    await asyncio.wait_for(stopping.wait(), timeout=2)
                    closing.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(closing, timeout=2)
                    self.assertTrue(task.done())
                    with self.assertRaises(sqlite3.ProgrammingError):
                        runner.sessions.list_sessions()
                    errors.assert_not_called()
                finally:
                    task.cancel()
                    closing.cancel()
                    await asyncio.gather(task, closing, return_exceptions=True)
                    await runner.close()

    async def test_cancelled_server_close_still_closes_all_runners(self):
        exception_handler = patch.object(asyncio.get_running_loop(), "call_exception_handler")
        errors = exception_handler.start()
        self.addCleanup(exception_handler.stop)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with patch.dict(
                os.environ,
                {"BOREALIS_DATA_DIR": str(root / "data"), "BOREALIS_PROVIDER": "mock"},
            ):
                runners = [await build_runner(root), await build_runner(root)]
                server = ACPServer()
                server.runners = dict(zip(("first", "second"), runners, strict=True))
                started = asyncio.Event()
                stopping = asyncio.Event()

                async def work():
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        stopping.set()
                        await asyncio.Event().wait()

                task = asyncio.create_task(work())
                server.tasks["first"] = task
                server._prompt_tasks["first"] = [task]
                await started.wait()
                closing = asyncio.create_task(server.close())
                try:
                    await asyncio.wait_for(stopping.wait(), timeout=2)
                    closing.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(closing, timeout=2)
                    self.assertTrue(task.done())
                    for runner in runners:
                        with self.assertRaises(sqlite3.ProgrammingError):
                            runner.sessions.list_sessions()
                    self.assertEqual(server.runners, {})
                    self.assertEqual(server.tasks, {})
                    self.assertEqual(server._prompt_tasks, {})
                    errors.assert_not_called()
                finally:
                    task.cancel()
                    closing.cancel()
                    await asyncio.gather(task, closing, return_exceptions=True)
                    await asyncio.gather(*(runner.close() for runner in runners))

    async def test_concurrent_resumes_serialize_only_the_same_session(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with patch.dict(
                os.environ,
                {"BOREALIS_DATA_DIR": str(root / "data"), "BOREALIS_PROVIDER": "mock"},
            ):
                server = ACPServer()
                await server.handle("initialize", {"protocolVersion": 2})
                created = await server.handle("session/new", {"cwd": str(root)})
                unrelated = await server.handle("session/new", {"cwd": str(root)})
                session_id = created["sessionId"]
                old = server.runners[session_id]
                original_close = old.close
                entered = asyncio.Event()
                release = asyncio.Event()
                replacements = []
                requests = []

                async def capture_runner(*args, **kwargs):
                    runner = await build_runner(*args, **kwargs)
                    replacements.append(runner)
                    return runner

                async def slow_close():
                    entered.set()
                    await release.wait()
                    await original_close()

                params = {"cwd": str(root), "sessionId": session_id}
                second_started = asyncio.Event()

                async def second_resume():
                    second_started.set()
                    return await server.handle("session/resume", params)

                try:
                    with (
                        patch("borealis_coder.protocol.acp.build_runner", capture_runner),
                        patch.object(old, "close", side_effect=slow_close),
                    ):
                        requests.append(asyncio.create_task(server.handle("session/resume", params)))
                        await asyncio.wait_for(entered.wait(), timeout=2)
                        await asyncio.wait_for(
                            server.handle("session/close", unrelated), timeout=2
                        )
                        requests.append(asyncio.create_task(second_resume()))
                        await asyncio.wait_for(second_started.wait(), timeout=2)
                        self.assertEqual(len(replacements), 1)
                        release.set()
                        self.assertEqual(await asyncio.gather(*requests), [{}, {}])
                    self.assertEqual(len(replacements), 2)
                    self.assertIs(server.runners[session_id], replacements[1])
                    with self.assertRaises(sqlite3.ProgrammingError):
                        replacements[0].sessions.list_sessions()
                    self.assertEqual(len(server._session_locks), 0)
                finally:
                    release.set()
                    for request in requests:
                        request.cancel()
                    await asyncio.gather(*requests, return_exceptions=True)
                    await server.close()
                    await old.close()
                    for runner in replacements:
                        await runner.close()

    async def test_session_listing_owns_its_connection_during_concurrent_close(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with patch.dict(
                os.environ,
                {"BOREALIS_DATA_DIR": str(root / "data"), "BOREALIS_PROVIDER": "mock"},
            ):
                server = ACPServer()
                await server.handle("initialize", {"protocolVersion": 2})
                created = await server.handle("session/new", {"cwd": str(root)})
                original_list = SessionStore.list_sessions
                entered = threading.Event()
                release = threading.Event()
                queried = []

                def delayed_list(store, **kwargs):
                    queried.append(store)
                    entered.set()
                    if not release.wait(2):
                        raise TimeoutError("test synchronization timeout")
                    return original_list(store, **kwargs)

                with patch.object(SessionStore, "list_sessions", delayed_list):
                    listing = asyncio.create_task(server.handle("session/list", {"cwd": str(root)}))
                    try:
                        self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                        await server.handle("session/close", created)
                        release.set()
                        result = await asyncio.wait_for(listing, timeout=2)
                        self.assertEqual(
                            [item["sessionId"] for item in result["sessions"]],
                            [created["sessionId"]],
                        )
                    finally:
                        release.set()
                        await asyncio.gather(listing, return_exceptions=True)
                        await server.close()
                with self.assertRaises(sqlite3.ProgrammingError):
                    queried[0].list_sessions()

    async def test_cancel_stops_all_queued_prompt_tasks(self):
        class QueuedRunner:
            def __init__(self):
                self.started = 0
                self.all_started = asyncio.Event()
                self.release = asyncio.Event()
                self.executed = 0
                self.cancelled_sessions = []

            def accepts_steering(self, session_id):
                return False

            def cancel(self, session_id):
                self.cancelled_sessions.append(session_id)
                return True

            async def run(self, prompt, **arguments):
                del prompt, arguments
                self.started += 1
                if self.started == 2:
                    self.all_started.set()
                await self.release.wait()
                self.executed += 1

        server = ACPServer()
        server.connection = cast(Any, FakeConnection())
        runner = QueuedRunner()
        server.runners["session_1"] = cast(Any, runner)

        await server._session_prompt(
            {"sessionId": "session_1", "prompt": [{"type": "text", "text": "first"}]}
        )
        first = server.tasks["session_1"]
        await server._session_prompt(
            {"sessionId": "session_1", "prompt": [{"type": "text", "text": "second"}]}
        )
        second = server.tasks["session_1"]
        await asyncio.wait_for(runner.all_started.wait(), timeout=1)

        cancelled = await server._session_cancel({"sessionId": "session_1"})
        self.assertEqual(cancelled, {})
        results = await asyncio.gather(first, second, return_exceptions=True)

        self.assertTrue(all(isinstance(result, asyncio.CancelledError) for result in results))
        self.assertEqual(runner.cancelled_sessions, ["session_1"])
        self.assertEqual(runner.executed, 0)
        self.assertNotIn("session_1", server.tasks)
        self.assertNotIn("session_1", server._prompt_tasks)

    async def test_run_started_sends_running_state(self):
        server = ACPServer()
        fake = FakeConnection()
        server.connection = cast(Any, fake)

        await server._event_update(
            "session_1",
            cast(Any, None),
            Event(type="run.started", session_id="session_1"),
        )

        update = fake.notifications[0][1]["update"]
        self.assertEqual(update, {"sessionUpdate": "state_update", "state": "running"})

    async def test_reasoning_summary_is_forwarded_as_agent_thought(self):
        server = ACPServer()
        fake = FakeConnection()
        server.connection = cast(Any, fake)

        await server._event_update(
            "session_1",
            cast(Any, None),
            Event(
                type="model.reasoning_delta",
                session_id="session_1",
                data={"message_id": "message_1", "text": "Checked the plan."},
            ),
        )

        update = fake.notifications[0][1]["update"]
        self.assertEqual(update["sessionUpdate"], "agent_thought_chunk")
        self.assertEqual(update["messageId"], "message_1")
        self.assertEqual(update["content"], {"type": "text", "text": "Checked the plan."})

    async def test_max_turns_sends_recovery_message_before_idle_state(self):
        server = ACPServer()
        fake = FakeConnection()
        server.connection = cast(Any, fake)

        recovery = (
            "Run incomplete: maximum 60 model turns reached. "
            "Session preserved; send 'continue' to resume."
        )
        await server._event_update(
            "session_1",
            cast(Any, None),
            Event(
                type="run.completed",
                session_id="session_1",
                data={
                    "result": {
                        "stop_reason": "max_turns",
                        "incomplete": True,
                        "error": recovery,
                    }
                },
            ),
        )

        updates = [params["update"] for _, params in fake.notifications]
        self.assertEqual([item["sessionUpdate"] for item in updates], ["agent_message", "state_update"])
        self.assertEqual(updates[0]["content"], [{"type": "text", "text": recovery}])
        self.assertEqual(updates[1]["state"], "idle")
        self.assertEqual(updates[1]["stopReason"], "max_turns")

    async def test_incomplete_mutation_tracking_sends_warning_before_idle_state(self):
        server = ACPServer()
        fake = FakeConnection()
        server.connection = cast(Any, fake)

        await server._event_update(
            "session_1",
            cast(Any, None),
            Event(
                type="run.completed",
                session_id="session_1",
                data={
                    "result": {
                        "stop_reason": "cancelled",
                        "incomplete": False,
                        "mutation_tracking": "incomplete",
                    }
                },
            ),
        )

        updates = [params["update"] for _, params in fake.notifications]
        self.assertEqual(
            [item["sessionUpdate"] for item in updates],
            ["agent_message", "state_update"],
        )
        self.assertEqual(
            updates[0]["content"],
            [
                {
                    "type": "text",
                    "text": (
                        "Workspace mutation tracking is incomplete; "
                        "verification is required."
                    ),
                }
            ],
        )
        self.assertEqual(updates[1]["state"], "idle")
        self.assertEqual(updates[1]["stopReason"], "cancelled")

    async def test_mutation_warning_is_not_duplicated_in_max_turn_recovery(self):
        server = ACPServer()
        fake = FakeConnection()
        server.connection = cast(Any, fake)
        recovery = (
            "Run incomplete: maximum turns reached. Workspace mutation tracking "
            "is incomplete; verification is required."
        )

        await server._event_update(
            "session_1",
            cast(Any, None),
            Event(
                type="run.completed",
                session_id="session_1",
                data={
                    "result": {
                        "stop_reason": "max_turns",
                        "incomplete": True,
                        "error": recovery,
                        "mutation_tracking": "incomplete",
                    }
                },
            ),
        )

        updates = [params["update"] for _, params in fake.notifications]
        self.assertEqual(
            [item["sessionUpdate"] for item in updates],
            ["agent_message", "state_update"],
        )
        self.assertEqual(updates[0]["content"], [{"type": "text", "text": recovery}])

    async def test_inactive_session_list_does_not_build_a_provider_runtime(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "data"
            store = SessionStore(data / "sessions.sqlite3")
            try:
                session = store.create_session(
                    workspace=root,
                    provider="offline",
                    model="stored-model",
                    title="Stored session",
                )
            finally:
                store.close()

            server = ACPServer()
            await server.handle("initialize", {"protocolVersion": 2, "capabilities": {}})
            with (
                patch.dict(
                    os.environ,
                    {"BOREALIS_DATA_DIR": str(data)},
                    clear=False,
                ),
                patch(
                    "borealis_coder.protocol.acp.build_runner",
                    side_effect=AssertionError("runtime must not start"),
                ),
            ):
                listed = await server.handle("session/list", {"cwd": str(root)})
            self.assertEqual([item["sessionId"] for item in listed["sessions"]], [session.id])

    async def test_session_lifecycle_and_updates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "data"
            server = ACPServer()
            fake = FakeConnection()
            server.connection = cast(Any, fake)
            with patch.dict(
                os.environ,
                {"BOREALIS_DATA_DIR": str(data), "BOREALIS_PROVIDER": "mock"},
                clear=False,
            ):
                init = await server.handle(
                    "initialize",
                    {
                        "protocolVersion": 2,
                        "capabilities": {},
                        "info": {"name": "test", "version": "1"},
                    },
                )
                self.assertEqual(init["protocolVersion"], 2)
                created = await server.handle("session/new", {"cwd": str(root), "mcpServers": []})
                session_id = created["sessionId"]
                accepted = await server.handle(
                    "session/prompt",
                    {
                        "sessionId": session_id,
                        "prompt": [{"type": "text", "text": "OFFLINE_WRITE_DEMO"}],
                    },
                )
                self.assertEqual(accepted, {})
                task = server.tasks[session_id]
                await asyncio.wait_for(task, timeout=5)
                self.assertTrue((root / "borealis-demo.txt").exists())
                update_types = [
                    params["update"]["sessionUpdate"]
                    for method, params in fake.notifications
                    if method == "session/update"
                ]
                self.assertIn("tool_call_update", update_types)
                self.assertIn("agent_message", update_types)
                self.assertIn("agent_message_chunk", update_types)
                self.assertIn("state_update", update_types)
                user_update = next(
                    params["update"]
                    for method, params in fake.notifications
                    if method == "session/update"
                    and params["update"]["sessionUpdate"] == "user_message"
                )
                persisted = server.runners[session_id].sessions.messages(session_id)
                self.assertEqual(persisted[0].id, user_update["messageId"])
                assistant_ids = {item.id for item in persisted if item.role.value == "assistant"}
                full_agent_updates = [
                    params["update"]
                    for method, params in fake.notifications
                    if method == "session/update"
                    and params["update"]["sessionUpdate"] == "agent_message"
                ]
                self.assertTrue(
                    any(item["messageId"] in assistant_ids for item in full_agent_updates)
                )
                internal_ids = {
                    item.id for item in persisted if item.metadata.get("internal")
                }
                authoritative_ids = {
                    item.id
                    for item in persisted
                    if item.metadata.get("authoritative_verification")
                }
                self.assertTrue(internal_ids)
                self.assertTrue(authoritative_ids)
                listed = await server.handle("session/list", {"cwd": str(root)})
                self.assertTrue(any(item["sessionId"] == session_id for item in listed["sessions"]))
                await server.handle("session/close", {"sessionId": session_id})
                fake.notifications.clear()
                await server.handle(
                    "session/resume",
                    {
                        "sessionId": session_id,
                        "cwd": str(root),
                        "mcpServers": [],
                        "replayFrom": {"type": "start"},
                    },
                )
                replayed_ids = {
                    params["update"]["messageId"]
                    for method, params in fake.notifications
                    if method == "session/update"
                    and params["update"]["sessionUpdate"]
                    in {"user_message", "agent_message"}
                }
                self.assertTrue(authoritative_ids.issubset(replayed_ids))
                self.assertTrue(internal_ids.isdisjoint(replayed_ids))
                await server.handle("session/close", {"sessionId": session_id})
                await server.handle("session/delete", {"sessionId": session_id})
                listed = await server.handle("session/list", {"cwd": str(root)})
                self.assertFalse(
                    any(item["sessionId"] == session_id for item in listed["sessions"])
                )
            await server.close()

    async def test_rejects_relative_roots_and_invalid_mcp(self):
        server = ACPServer()
        server.connection = cast(Any, FakeConnection())
        await server.handle("initialize", {"protocolVersion": 2, "capabilities": {}})
        with self.assertRaises(ProtocolError):
            await server.handle("session/new", {"cwd": ".", "mcpServers": []})
        with (
            tempfile.TemporaryDirectory() as td,
            patch.dict(
                os.environ,
                {"BOREALIS_DATA_DIR": str(Path(td) / "data")},
                clear=False,
            ),
            self.assertRaises(ProtocolError),
        ):
            await server.handle(
                "session/new",
                {
                    "cwd": td,
                    "mcpServers": [{"type": "stdio", "name": "bad", "command": "python"}],
                },
            )


if __name__ == "__main__":
    unittest.main()
