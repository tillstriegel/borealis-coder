from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from borealis_coder.errors import SessionError
from borealis_coder.events import EventBus, JsonlTrace
from borealis_coder.models import CompactionArtifact, Event, Message, Role, ToolCall, Usage
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

    def test_run_lease_distinguishes_live_and_interrupted_sessions(self):
        other = SessionStore(self.root / "sessions.sqlite3")
        try:
            self.store.acquire_run_lease(self.session.id)
            self.store.update_session(self.session.id, status="running")

            self.assertEqual(other.get_session(self.session.id).status, "running")

            self.store.release_run_lease(self.session.id)
            self.assertEqual(other.get_session(self.session.id).status, "interrupted")
        finally:
            self.store.release_run_lease(self.session.id)
            other.close()

    def test_pending_compaction_summary_reports_single_owner(self):
        value = {
            "version": 2,
            "text": "winner",
            "usage": Usage(input_tokens=3, requests=1).to_dict(),
        }

        self.assertTrue(
            self.store.put_pending_compaction_summary(
                self.session.id, "compaction_summary:key", value
            )
        )
        self.assertFalse(
            self.store.put_pending_compaction_summary(
                self.session.id, "compaction_summary:key", value
            )
        )
        self.assertFalse(
            self.store.put_pending_compaction_summary(
                self.session.id,
                "compaction_summary:key",
                {**value, "text": "loser"},
            )
        )
        cached = self.store.get_compaction_summary(
            self.session.id, "compaction_summary:key"
        )
        assert cached is not None
        self.assertEqual(cached["text"], "winner")
        self.assertFalse(cached["usage_settled"])

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

    def test_messages_are_append_only_and_tool_ids_are_session_scoped(self):
        first = Message(role=Role.USER, content="first")
        second = Message(role=Role.ASSISTANT, content="second")
        self.store.append_message(self.session.id, first)
        self.store.append_message(self.session.id, second)
        first.content = "updated"
        with self.assertRaisesRegex(SessionError, "cannot be overwritten"):
            self.store.append_message(self.session.id, first)
        self.assertEqual(
            [item.content for item in self.store.messages(self.session.id)], ["first", "second"]
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

    def test_tool_completion_cannot_overwrite_a_durable_result_message(self):
        self.store.start_tool_call(
            self.session.id, "run", "call", "read_file", {"path": "a"}
        )
        result = Message(
            id="result",
            role=Role.TOOL,
            content="original",
            tool_call_id="call",
            tool_name="read_file",
        )
        self.store.complete_tool_call(
            self.session.id,
            "call",
            output="original",
            is_error=False,
            message=result,
        )
        result.content = "overwritten"

        with self.assertRaisesRegex(SessionError, "cannot be overwritten"):
            self.store.complete_tool_call(
                self.session.id,
                "call",
                output="overwritten",
                is_error=False,
                message=result,
            )

        self.assertEqual(self.store.messages(self.session.id)[0].content, "original")

    def test_running_tool_call_start_is_exactly_idempotent(self):
        self.store.start_tool_call(
            self.session.id,
            "run",
            "call",
            "read_file",
            {"path": "a", "start": 1},
        )
        original = self.store.tool_calls(self.session.id)[0]

        self.store.start_tool_call(
            self.session.id,
            "run",
            "call",
            "read_file",
            {"start": 1, "path": "a"},
        )

        self.assertEqual(self.store.tool_calls(self.session.id)[0], original)
        for run_id, name, arguments in (
            ("other-run", "read_file", {"path": "a", "start": 1}),
            ("run", "write_file", {"path": "a", "start": 1}),
            ("run", "read_file", {"path": "b", "start": 1}),
        ):
            with self.subTest(run_id=run_id, name=name, arguments=arguments):
                with self.assertRaisesRegex(SessionError, "cannot be overwritten"):
                    self.store.start_tool_call(
                        self.session.id,
                        run_id,
                        "call",
                        name,
                        arguments,
                    )
                self.assertEqual(self.store.tool_calls(self.session.id)[0], original)

    def test_empty_tool_call_ids_are_rejected_without_ledger_mutation(self):
        for call_id in ("", "   ", "\t\n"):
            with self.subTest(call_id=repr(call_id)), self.assertRaisesRegex(
                SessionError,
                "must be non-empty",
            ):
                self.store.start_tool_call(
                    self.session.id,
                    "run",
                    call_id,
                    "read_file",
                    {"path": "a"},
                )

        self.assertEqual(self.store.tool_calls(self.session.id), [])

    def test_terminal_tool_completion_accepts_only_exact_replay(self):
        self.store.start_tool_call(
            self.session.id,
            "run",
            "call",
            "read_file",
            {"path": "a"},
        )
        message = Message(
            id="result",
            role=Role.TOOL,
            content="original",
            tool_call_id="call",
            tool_name="read_file",
            metadata={"stable": True},
        )
        self.store.complete_tool_call(
            self.session.id,
            "call",
            output="original",
            is_error=False,
            metadata={"stable": True},
            message=message,
        )
        original = self.store.tool_calls(self.session.id)[0]

        self.store.complete_tool_call(
            self.session.id,
            "call",
            output="original",
            is_error=False,
            metadata={"stable": True},
            message=message,
        )

        self.assertEqual(self.store.tool_calls(self.session.id)[0], original)
        self.assertEqual(len(self.store.messages(self.session.id)), 1)
        for output, is_error, metadata in (
            ("replacement", False, {"stable": True}),
            ("original", True, {"stable": True}),
            ("original", False, {"stable": False}),
        ):
            with self.subTest(output=output, is_error=is_error, metadata=metadata):
                with self.assertRaisesRegex(SessionError, "terminal"):
                    self.store.complete_tool_call(
                        self.session.id,
                        "call",
                        output=output,
                        is_error=is_error,
                        metadata=metadata,
                        message=message,
                    )
                self.assertEqual(self.store.tool_calls(self.session.id)[0], original)

    def test_concurrent_cancellation_wins_over_stale_completion(self):
        call = ToolCall(id="raced-call", name="read_file", arguments={"path": "a"})
        assistant = Message(role=Role.ASSISTANT, tool_calls=[call])
        self.store.append_message(self.session.id, assistant)
        self.store.start_tool_call(
            self.session.id, "run", call.id, call.name, call.arguments
        )
        completion_store = SessionStore(self.store.path)
        update_started = threading.Event()
        release_update = threading.Event()
        completion_errors: list[BaseException] = []

        class PausingConnection:
            def __init__(self, connection):
                self.connection = connection

            def __enter__(self):
                self.connection.__enter__()
                return self

            def __exit__(self, *args):
                return self.connection.__exit__(*args)

            def __getattr__(self, name):
                return getattr(self.connection, name)

            def execute(self, statement, parameters=()):
                if (
                    "UPDATE tool_calls SET" in statement
                    and "status=?" in statement
                ):
                    update_started.set()
                    release_update.wait(timeout=3)
                return self.connection.execute(statement, parameters)

        completion_message = Message(
            role=Role.TOOL,
            content="stale success",
            tool_call_id=call.id,
            tool_name=call.name,
        )

        def complete() -> None:
            try:
                completion_store.complete_tool_call(
                    self.session.id,
                    call.id,
                    output=completion_message.content,
                    is_error=False,
                    message=completion_message,
                )
            except BaseException as error:
                completion_errors.append(error)

        original_connection = completion_store._connection
        try:
            with patch.object(
                completion_store,
                "_connection",
                PausingConnection(original_connection),
            ):
                thread = threading.Thread(target=complete)
                thread.start()
                self.assertTrue(update_started.wait(timeout=1))
                cancellation_message = Message(
                    role=Role.TOOL,
                    content="cancel won",
                    tool_call_id=call.id,
                    tool_name=call.name,
                    is_error=True,
                    metadata={"cancelled": True},
                )
                self.store.cancel_tool_call(
                    self.session.id,
                    call.id,
                    reason=cancellation_message.content,
                    message=cancellation_message,
                )
                release_update.set()
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive())
        finally:
            release_update.set()
            completion_store.close()

        self.assertEqual(len(completion_errors), 1)
        self.assertIsInstance(completion_errors[0], SessionError)
        ledger = self.store.tool_calls(self.session.id)[0]
        self.assertEqual(ledger["status"], "cancelled")
        self.assertEqual(ledger["output"], "cancel won")
        messages = self.store.messages(self.session.id)
        self.assertEqual(messages, [assistant, cancellation_message])

    def test_tool_cancellation_accepts_only_exact_cancelled_replay(self):
        with self.assertRaisesRegex(SessionError, "does not exist"):
            self.store.cancel_tool_call(self.session.id, "missing")

        self.store.start_tool_call(
            self.session.id, "run", "cancelled-call", "read_file", {"path": "a"}
        )
        message = Message(
            id="cancelled-result",
            role=Role.TOOL,
            content="stopped",
            tool_call_id="cancelled-call",
            tool_name="read_file",
            is_error=True,
            metadata={"cancelled": True},
        )
        self.store.cancel_tool_call(
            self.session.id, "cancelled-call", reason="stopped", message=message
        )
        cancelled = self.store.tool_calls(self.session.id)[0]
        self.store.cancel_tool_call(
            self.session.id, "cancelled-call", reason="stopped", message=message
        )
        self.assertEqual(self.store.tool_calls(self.session.id)[0], cancelled)
        self.assertEqual(self.store.messages(self.session.id), [message])

        for status in ("completed", "error"):
            call_id = f"terminal-{status}"
            self.store.start_tool_call(
                self.session.id, "run", call_id, "read_file", {"path": status}
            )
            self.store.complete_tool_call(
                self.session.id,
                call_id,
                output=f"{status} result",
                is_error=status == "error",
                metadata={"stable": True},
            )
            original = next(
                row
                for row in self.store.tool_calls(self.session.id)
                if row["tool_call_id"] == call_id
            )
            with self.assertRaisesRegex(SessionError, "terminal"):
                self.store.cancel_tool_call(
                    self.session.id, call_id, reason="stale cancellation"
                )
            current = next(
                row
                for row in self.store.tool_calls(self.session.id)
                if row["tool_call_id"] == call_id
            )
            self.assertEqual(current, original)

    def test_terminal_tool_calls_cannot_be_restarted_or_overwritten(self):
        for status in ("completed", "error", "cancelled"):
            with self.subTest(status=status):
                call_id = f"call-{status}"
                self.store.start_tool_call(
                    self.session.id,
                    "original-run",
                    call_id,
                    "read_file",
                    {"path": "original"},
                )
                if status == "cancelled":
                    self.store.cancel_tool_call(self.session.id, call_id)
                else:
                    self.store.complete_tool_call(
                        self.session.id,
                        call_id,
                        output="original output",
                        is_error=status == "error",
                        metadata={"original": True},
                    )
                original = next(
                    row
                    for row in self.store.tool_calls(self.session.id)
                    if row["tool_call_id"] == call_id
                )

                with self.assertRaisesRegex(SessionError, "cannot be overwritten"):
                    self.store.start_tool_call(
                        self.session.id,
                        "new-run",
                        call_id,
                        "write_file",
                        {"path": "replacement"},
                    )

                current = next(
                    row
                    for row in self.store.tool_calls(self.session.id)
                    if row["tool_call_id"] == call_id
                )
                self.assertEqual(current, original)

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
            self.assertEqual(version, "4")
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

    def test_compaction_artifacts_are_immutable_reusable_and_exported(self):
        artifact = CompactionArtifact(
            id="cmp-test",
            session_id=self.session.id,
            version=2,
            strategy="deterministic",
            source_message_ids=["m1", "m2"],
            source_hash="source-hash",
            summary_text="exact compacted context",
            config_fingerprint="config-hash",
            estimated_tokens_before=100,
            estimated_tokens_after=25,
            usage=Usage(input_tokens=5, output_tokens=2, requests=1, cost_usd=0.01),
            parent_artifact_id="cmp-parent",
            metadata={"retained_message_ids": ["m3"]},
        )

        self.store.append_compaction_artifact(artifact)

        latest = self.store.latest_compaction_artifact(self.session.id)
        reusable = self.store.reusable_compaction_artifact(
            self.session.id,
            source_hash="source-hash",
            config_fingerprint="config-hash",
            strategy="deterministic",
        )
        assert latest is not None and reusable is not None
        self.assertEqual(latest.summary_text, "exact compacted context")
        self.assertEqual(reusable.id, artifact.id)
        self.assertEqual(reusable.usage.cost_usd, 0.01)
        self.assertEqual(
            self.store.export(self.session.id)["compaction_artifacts"][0]["id"],
            artifact.id,
        )
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.append_compaction_artifact(artifact)

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

    async def test_independent_trace_writers_rotate_without_racing(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "events.jsonl"
            traces = [
                JsonlTrace(path, max_bytes=300, backup_count=3)
                for _ in range(4)
            ]
            barrier = threading.Barrier(len(traces))
            errors: list[Exception] = []

            def write_events(writer: int, trace: JsonlTrace) -> None:
                try:
                    barrier.wait()
                    for index in range(20):
                        trace.append(
                            Event(
                                type="model.completed",
                                data={
                                    "writer": writer,
                                    "index": index,
                                    "payload": "x" * 80,
                                },
                            )
                        )
                except Exception as error:
                    errors.append(error)

            threads = [
                threading.Thread(target=write_events, args=(index, trace))
                for index, trace in enumerate(traces)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [])
            records = []
            for item in [Path(f"{path}.{index}") for index in range(3, 0, -1)] + [path]:
                if item.is_file():
                    records.extend(
                        json.loads(line)
                        for line in item.read_text(encoding="utf-8").splitlines()
                    )
            self.assertTrue(records)
            self.assertTrue(all(record["type"] == "model.completed" for record in records))

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
