from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from borealis_coder.events import JsonlTrace
from borealis_coder.models import Event, Message, Role, Usage
from borealis_coder.sessions import SessionStore


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = SessionStore(self.root / "sessions.sqlite3")
        self.session = self.store.create_session(workspace=self.root, provider="mock", model="deterministic", title="Test")

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_message_event_usage_and_export(self):
        message = Message(role=Role.USER, content="hello")
        self.store.append_message(self.session.id, message)
        event = Event(type="test", session_id=self.session.id, data={"x": 1})
        self.store.append_event(event)
        self.store.add_usage(self.session.id, Usage(input_tokens=5, output_tokens=2, requests=1, cost_usd=.1))
        self.store.start_tool_call(self.session.id, "run_1", "call_1", "read_file", {"path": "x"})
        self.store.complete_tool_call(
            self.session.id,
            "call_1",
            output="ok",
            is_error=False,
            metadata={"cached": False},
        )
        exported = self.store.export(self.session.id)
        self.assertEqual(exported["messages"][0]["content"], "hello")
        self.assertEqual(exported["events"][0]["type"], "test")
        self.assertEqual(exported["usage"]["total_tokens"], 7)
        self.assertEqual(exported["tool_calls"][0]["status"], "completed")

    def test_message_upsert_preserves_order_and_tool_ids_are_session_scoped(self):
        first = Message(role=Role.USER, content="first")
        second = Message(role=Role.ASSISTANT, content="second")
        self.store.append_message(self.session.id, first)
        self.store.append_message(self.session.id, second)
        first.content = "updated"
        self.store.append_message(self.session.id, first)
        self.assertEqual([item.content for item in self.store.messages(self.session.id)], ["updated", "second"])

        other = self.store.create_session(
            workspace=self.root,
            provider="mock",
            model="deterministic",
            title="Other",
        )
        self.store.start_tool_call(self.session.id, "run_a", "same", "read_file", {"path": "a"})
        self.store.start_tool_call(other.id, "run_b", "same", "read_file", {"path": "b"})
        self.store.complete_tool_call(self.session.id, "same", output="a", is_error=False)
        self.store.complete_tool_call(other.id, "same", output="b", is_error=False)
        self.assertEqual(self.store.tool_calls(self.session.id)[0]["output"], "a")
        self.assertEqual(self.store.tool_calls(other.id)[0]["output"], "b")

    def test_v1_tool_call_schema_migrates(self):
        self.store.close()
        path = self.root / "legacy.sqlite3"
        connection = sqlite3.connect(path)
        connection.executescript(
            """
            CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO schema_meta VALUES('version', '1');
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, workspace TEXT NOT NULL, title TEXT NOT NULL,
                provider TEXT NOT NULL, model TEXT NOT NULL, status TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            INSERT INTO sessions VALUES(
                'sess_legacy', '/tmp', 'Legacy', 'mock', 'deterministic', 'idle',
                '2020-01-01T00:00:00Z', '2020-01-01T00:00:00Z', '{}'
            );
            CREATE TABLE tool_calls (
                tool_call_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                run_id TEXT, tool_name TEXT NOT NULL, arguments_json TEXT NOT NULL,
                output TEXT, is_error INTEGER, status TEXT NOT NULL,
                started_at TEXT NOT NULL, completed_at TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            INSERT INTO tool_calls VALUES(
                'call_legacy', 'sess_legacy', 'run_legacy', 'read_file', '{}',
                'ok', 0, 'completed', '2020-01-01T00:00:00Z',
                '2020-01-01T00:00:01Z', '{}'
            );
            """
        )
        connection.close()
        legacy = SessionStore(path)
        try:
            self.assertEqual(legacy.tool_calls("sess_legacy")[0]["output"], "ok")
            version = legacy._connection.execute(
                "SELECT value FROM schema_meta WHERE key='version'"
            ).fetchone()[0]
            self.assertEqual(version, "2")
        finally:
            legacy.close()
        self.store = SessionStore(self.root / "sessions.sqlite3")

    def test_update_list_delete(self):
        updated = self.store.update_session(self.session.id, status="running", title="Updated")
        self.assertEqual(updated.title, "Updated")
        self.assertEqual(self.store.list_sessions(workspace=self.root)[0].id, self.session.id)
        self.store.delete_session(self.session.id)
        self.assertEqual(self.store.list_sessions(), [])

    def test_key_values(self):
        self.store.set_value(self.session.id, "plan", {"x": 1})
        self.assertEqual(self.store.get_value(self.session.id, "plan"), {"x": 1})
        self.assertEqual(self.store.get_value(self.session.id, "missing", 9), 9)

    @unittest.skipUnless(os.name == "posix", "POSIX permissions only")
    def test_state_files_are_private(self):
        database_mode = self.store.path.stat().st_mode & 0o777
        directory_mode = self.store.path.parent.stat().st_mode & 0o777
        self.assertEqual(database_mode, 0o600)
        self.assertEqual(directory_mode, 0o700)

        trace_path = self.root / "traces" / "events.jsonl"
        trace = JsonlTrace(trace_path)
        trace.append(Event(type="private"))
        self.assertEqual(trace_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(trace_path.parent.stat().st_mode & 0o777, 0o700)


if __name__ == "__main__":
    unittest.main()
