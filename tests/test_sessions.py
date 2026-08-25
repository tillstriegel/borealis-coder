from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from borealis_coder.errors import SessionError
from borealis_coder.events import EventBus, JsonlTrace
from borealis_coder.models import Event, Message, Role, Usage
from borealis_coder.sessions import SessionStore


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = SessionStore(self.root / "sessions.sqlite3")
        self.session = self.store.create_session(
            workspace=self.root, provider="mock", model="deterministic", title="Test"
        )

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_message_event_usage_and_export(self):
        message = Message(role=Role.USER, content="hello")
        self.store.append_message(self.session.id, message)
        event = Event(type="test", session_id=self.session.id, data={"x": 1})
        self.store.append_event(event)
        self.store.add_usage(
            self.session.id, Usage(input_tokens=5, output_tokens=2, requests=1, cost_usd=0.1)
        )
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

    def test_export_contains_more_than_one_event_query_page(self):
        self.store.append_events(
            Event(type="model.text_delta", session_id=self.session.id, data={"text": "x"})
            for _ in range(10_005)
        )

        self.assertEqual(len(self.store.events(self.session.id)), 10_000)
        exported = self.store.export(self.session.id)
        self.assertEqual(len(exported["events"]), 10_005)
        self.assertEqual(exported["events"][-1]["sequence"], 10_005)

    def test_event_maintenance_dry_run_changes_nothing(self):
        self.store.append_events(
            Event(type="run.started", session_id=self.session.id) for _ in range(5)
        )

        report = self.store.prune_events(max_count=2, dry_run=True)

        self.assertEqual(report["pruned_events"], 3)
        self.assertEqual(len(self.store.events(self.session.id)), 5)
        applied = self.store.prune_events(max_count=2)
        self.assertEqual(applied["events_after"], 2)
        self.assertEqual(len(self.store.events(self.session.id)), 2)

    def test_event_maintenance_respects_age_without_pruning_recent_rows(self):
        self.store.append_events(
            [
                Event(
                    type="run.started",
                    session_id=self.session.id,
                    created_at="2000-01-01T00:00:00+00:00",
                ),
                Event(type="run.completed", session_id=self.session.id),
            ]
        )

        report = self.store.prune_events(max_count=100, max_age_seconds=60)

        self.assertEqual(report["pruned_events"], 1)
        self.assertEqual(
            [event.type for _, event in self.store.events(self.session.id)],
            ["run.completed"],
        )

    def test_message_upsert_preserves_order_and_tool_ids_are_session_scoped(self):
        first = Message(role=Role.USER, content="first")
        second = Message(role=Role.ASSISTANT, content="second")
        self.store.append_message(self.session.id, first)
        self.store.append_message(self.session.id, second)
        first.content = "updated"
        self.store.append_message(self.session.id, first)
        self.assertEqual(
            [item.content for item in self.store.messages(self.session.id)], ["updated", "second"]
        )

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
            self.assertEqual(version, "3")
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

    def test_response_cache_is_bounded_and_tracks_usage(self):
        original = Usage(input_tokens=10, output_tokens=2, requests=1, cost_usd=0.25)
        self.store.put_cached_response(
            "first",
            provider="mock",
            model="deterministic",
            response={"text": "one"},
            usage=original,
            ttl_seconds=60,
            max_entries=1,
        )
        self.store.put_cached_response(
            "second",
            provider="mock",
            model="deterministic",
            response={"text": "two"},
            usage=original,
            ttl_seconds=60,
            max_entries=1,
        )
        self.assertIsNone(self.store.get_cached_response("first"))
        cached = self.store.get_cached_response("second")
        assert cached is not None
        self.assertEqual(cached["response"]["text"], "two")
        self.assertEqual(cached["usage"]["total_tokens"], 12)

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


class EventBusTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_deltas_are_live_but_only_assembled_event_is_durable_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = SessionStore(root / "sessions.sqlite3")
            session = store.create_session(
                workspace=root,
                provider="mock",
                model="deterministic",
            )
            batches: list[int] = []

            def persist(events: list[Event]) -> None:
                batches.append(len(events))
                store.append_events(events)

            trace_path = root / "events.jsonl"
            bus = EventBus(trace=JsonlTrace(trace_path), persist=persist)
            live: list[str] = []
            bus.subscribe(lambda event: live.append(event.type))
            for index in range(10):
                await bus.emit(
                    "model.text_delta",
                    session_id=session.id,
                    run_id="run_1",
                    text=str(index),
                )
            self.assertEqual(len(live), 10)
            self.assertEqual(batches, [])

            await bus.emit(
                "model.completed",
                session_id=session.id,
                run_id="run_1",
                text="0123456789",
            )
            self.assertEqual(batches, [1])
            persisted = [event for _, event in store.events(session.id)]
            self.assertEqual([event.type for event in persisted], ["model.completed"])
            self.assertEqual(persisted[0].data["text"], "0123456789")
            self.assertEqual(len(trace_path.read_text().splitlines()), 11)
            await bus.flush()
            store.close()

    async def test_verbose_delta_persistence_can_be_enabled(self):
        persisted: list[str] = []
        bus = EventBus(
            persist=lambda events: persisted.extend(event.type for event in events),
            persist_deltas=True,
        )

        await bus.emit("model.text_delta", text="part")
        await bus.emit("model.completed", text="part")

        self.assertEqual(persisted, ["model.text_delta", "model.completed"])

    async def test_trace_rotation_preserves_valid_jsonl_records(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "events.jsonl"
            trace = JsonlTrace(path, max_bytes=300, backup_count=2)
            for index in range(8):
                trace.append(Event(type="model.completed", data={"index": index}))

            files = [Path(f"{path}.2"), Path(f"{path}.1"), path]
            records = []
            for item in files:
                if item.is_file():
                    for line in item.read_text(encoding="utf-8").splitlines():
                        records.append(json.loads(line))
            self.assertTrue(records)
            self.assertTrue(all(record["type"] == "model.completed" for record in records))
            self.assertEqual(records[-1]["data"]["index"], 7)

    async def test_trace_maintenance_prunes_backups_above_current_limit(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "events.jsonl"
            trace = JsonlTrace(path, max_bytes=240, backup_count=3)
            for index in range(12):
                trace.append(Event(type="model.completed", data={"index": index}))
            self.assertTrue(Path(f"{path}.2").is_file())

            reduced = JsonlTrace(
                path, max_bytes=240, backup_count=1, create=False
            )
            preview = reduced.maintenance(dry_run=True)
            self.assertGreater(preview["records_before"], preview["records_after"])
            self.assertTrue(Path(f"{path}.2").is_file())

            reduced.maintenance()

            self.assertFalse(Path(f"{path}.2").exists())
            records = []
            for item in (Path(f"{path}.1"), path):
                if item.is_file():
                    records.extend(
                        json.loads(line)
                        for line in item.read_text(encoding="utf-8").splitlines()
                    )
            self.assertTrue(records)
            self.assertEqual(records[-1]["data"]["index"], 11)

    async def test_trace_maintenance_streams_existing_files(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "events.jsonl"
            trace = JsonlTrace(path, max_bytes=300, backup_count=2)
            for index in range(8):
                trace.append(Event(type="model.completed", data={"index": index}))

            with patch.object(
                Path,
                "read_text",
                side_effect=AssertionError("maintenance must stream trace files"),
            ):
                report = trace.maintenance(dry_run=True)

            self.assertGreater(report["records_before"], 0)
            self.assertGreater(report["records_after"], 0)

    async def test_persistence_failure_keeps_order_for_lossless_retry(self):
        durable: list[str] = []
        attempted: list[list[str]] = []
        attempts = 0

        def persist(events: list[Event]) -> None:
            nonlocal attempts
            attempts += 1
            attempted.append([event.id for event in events])
            if attempts == 1:
                raise OSError("disk full")
            durable.extend(event.id for event in events)

        bus = EventBus(persist=persist)
        await bus.emit("model.text_delta")
        await bus.emit("model.text_delta")
        with self.assertRaisesRegex(SessionError, "disk full"):
            await bus.emit("model.completed")
        self.assertEqual(durable, [])

        await bus.flush()
        fourth = await bus.emit("run.completed")
        self.assertEqual(durable, [*attempted[0], fourth.id])
        self.assertEqual(len(durable), len(set(durable)))

    async def test_trace_retry_does_not_repeat_authoritative_persistence(self):
        persisted: list[str] = []

        class FlakyTrace:
            def __init__(self) -> None:
                self.calls = 0
                self.traced: list[str] = []

            def append_many(self, events: list[Event]) -> None:
                self.calls += 1
                if self.calls == 1:
                    raise OSError("trace unavailable")
                self.traced.extend(event.id for event in events)

        trace = FlakyTrace()
        bus = EventBus(
            trace=trace,  # type: ignore[arg-type]
            persist=lambda events: persisted.extend(event.id for event in events),
        )
        with self.assertRaisesRegex(OSError, "trace unavailable"):
            await bus.emit("run.started")
        self.assertEqual(len(persisted), 1)

        await bus.flush()
        self.assertEqual(len(persisted), 1)
        self.assertEqual(trace.traced, persisted)


if __name__ == "__main__":
    unittest.main()
