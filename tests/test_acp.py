from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from borealis_coder.errors import ProtocolError
from borealis_coder.protocol import ACPServer


class FakeConnection:
    def __init__(self):
        self.notifications=[]
        self.requests=[]
    async def notify(self, method, params):
        self.notifications.append((method, params))
    async def request(self, method, params, timeout=None):
        self.requests.append((method, params))
        return {"optionId":"allow_once"}


class ACPTests(unittest.IsolatedAsyncioTestCase):
    async def test_session_lifecycle_and_updates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "data"
            server = ACPServer()
            fake = FakeConnection()
            server.connection = cast(Any, fake)
            with patch.dict(os.environ, {"BOREALIS_DATA_DIR": str(data), "BOREALIS_PROVIDER":"mock"}, clear=False):
                init = await server.handle("initialize", {"protocolVersion":2,"capabilities":{},"info":{"name":"test","version":"1"}})
                self.assertEqual(init["protocolVersion"], 2)
                created = await server.handle("session/new", {"cwd":str(root),"mcpServers":[]})
                session_id = created["sessionId"]
                accepted = await server.handle("session/prompt", {"sessionId":session_id,"prompt":[{"type":"text","text":"OFFLINE_WRITE_DEMO"}]})
                self.assertEqual(accepted, {})
                task = server.tasks[session_id]
                await asyncio.wait_for(task, timeout=5)
                self.assertTrue((root/"borealis-demo.txt").exists())
                update_types = [params["update"]["sessionUpdate"] for method, params in fake.notifications if method=="session/update"]
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
                self.assertTrue(any(item["messageId"] in assistant_ids for item in full_agent_updates))
                listed = await server.handle("session/list", {"cwd":str(root)})
                self.assertTrue(any(item["sessionId"]==session_id for item in listed["sessions"]))
                await server.handle("session/close", {"sessionId":session_id})
                await server.handle("session/resume", {
                    "sessionId": session_id,
                    "cwd": str(root),
                    "mcpServers": [],
                    "replayFrom": {"type": "start"},
                })
                await server.handle("session/close", {"sessionId": session_id})
                await server.handle("session/delete", {"sessionId": session_id})
                listed = await server.handle("session/list", {"cwd": str(root)})
                self.assertFalse(any(item["sessionId"] == session_id for item in listed["sessions"]))
            await server.close()

    async def test_rejects_relative_roots_and_invalid_mcp(self):
        server = ACPServer()
        server.connection = cast(Any, FakeConnection())
        await server.handle("initialize", {"protocolVersion": 2, "capabilities": {}})
        with self.assertRaises(ProtocolError):
            await server.handle("session/new", {"cwd": ".", "mcpServers": []})
        with (
            tempfile.TemporaryDirectory() as td, patch.dict(
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
                    "mcpServers": [
                        {"type": "stdio", "name": "bad", "command": "python"}
                    ],
                },
            )


if __name__ == "__main__":
    unittest.main()
