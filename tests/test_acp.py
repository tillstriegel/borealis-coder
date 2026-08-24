from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from borealis_coder.errors import ProtocolError
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
                listed = await server.handle("session/list", {"cwd": str(root)})
                self.assertTrue(any(item["sessionId"] == session_id for item in listed["sessions"]))
                await server.handle("session/close", {"sessionId": session_id})
                await server.handle(
                    "session/resume",
                    {
                        "sessionId": session_id,
                        "cwd": str(root),
                        "mcpServers": [],
                        "replayFrom": {"type": "start"},
                    },
                )
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
