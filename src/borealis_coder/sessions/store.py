"""Durable SQLite session/event/message store with WAL and explicit migrations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..errors import SessionError
from ..models import CompactionArtifact, Event, Message, SessionInfo, Usage
from ..util import ensure_private_directory, ensure_private_file, json_dumps, new_id, utc_now

_SCHEMA_VERSION = 8
_DEDUPLICATED_PROVIDER_CONTEXT_STORAGE = "deduplicated-v1"
_EVENT_EXPORT_PAGE_SIZE = 1_000


def _truncated_envelope_has_quote(content: str, quote: str) -> bool:
    """Decode source value strings in a quota-retained JSON prefix.

    Follow JSON structure so keys and metadata cannot become source evidence.
    An unfinished string contributes only its fully retained escape sequences.
    """
    decoder = json.JSONDecoder()
    position = 0
    matched = False

    def whitespace() -> None:
        nonlocal position
        while position < len(content) and content[position] in " \r\n\t":
            position += 1

    def string() -> tuple[str, bool]:
        nonlocal position
        try:
            text, position = decoder.raw_decode(content, position)
            if not isinstance(text, str):
                raise ValueError("Expected string")
            return text, True
        except json.JSONDecodeError:
            prefix = re.match(
                r'(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*',
                content[position + 1 :],
            )
            assert prefix is not None
            return json.loads('"' + prefix[0] + '"'), False

    def value(source: bool = False, *, root: bool = False) -> None:
        nonlocal position, matched
        whitespace()
        token = content[position]
        if token == '"':
            text, complete = string()
            matched = matched or (source and quote in text)
            if not complete:
                raise ValueError("End of retained string")
        elif token in "{[":
            closing = "}" if token == "{" else "]"
            position += 1
            whitespace()
            if content[position] == closing:
                position += 1
                return
            while True:
                selected = source
                if token == "{":
                    if content[position] != '"':
                        raise ValueError("Expected key")
                    key, complete = string()
                    if not complete:
                        raise ValueError("End of retained key")
                    whitespace()
                    if content[position] != ":":
                        raise ValueError("Expected colon")
                    position += 1
                    selected = source or (root and key in {"output", "data"})
                value(selected)
                whitespace()
                delimiter = content[position]
                position += 1
                if delimiter == closing:
                    return
                if delimiter != ",":
                    raise ValueError("Expected delimiter")
                whitespace()
        else:
            _, position = decoder.raw_decode(content, position)

    try:
        if content.lstrip().startswith("{"):
            value(root=True)
    except (ValueError, IndexError, RecursionError):
        pass  # A quota prefix normally ends inside a value or container.
    return matched


class _SessionRunLease:
    """A process-bound file lease used to distinguish live and stale runs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file: Any | None = None

    def acquire(self) -> bool:
        ensure_private_directory(self.path.parent)
        handle = self.path.open("a+b")
        ensure_private_file(self.path)
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self._file = handle
        return True

    def release(self) -> None:
        handle = self._file
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._file = None


class SessionStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        ensure_private_directory(self.path.parent)
        self._lock = threading.RLock()
        self._run_leases: dict[str, _SessionRunLease] = {}
        self._connection = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._migrate()
            self._secure_files()

    def close(self) -> None:
        with self._lock:
            for lease in self._run_leases.values():
                lease.release()
            self._run_leases.clear()
            self._connection.close()
            self._secure_files()

    def _secure_files(self) -> None:
        ensure_private_file(self.path)
        ensure_private_file(Path(f"{self.path}-wal"))
        ensure_private_file(Path(f"{self.path}-shm"))

    def _migrate(self) -> None:
        conn = self._connection
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                workspace TEXT NOT NULL,
                title TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);
            CREATE TABLE IF NOT EXISTS messages (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                message_id TEXT NOT NULL,
                role TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(session_id, message_id)
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, sequence);
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
                run_id TEXT,
                type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, sequence);
            CREATE TABLE IF NOT EXISTS tool_calls (
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                tool_call_id TEXT NOT NULL,
                run_id TEXT,
                tool_name TEXT NOT NULL,
                arguments_json TEXT NOT NULL,
                output TEXT,
                is_error INTEGER,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY(session_id, tool_call_id)
            );
            CREATE INDEX IF NOT EXISTS idx_tool_calls_session ON tool_calls(session_id, started_at);
            CREATE TABLE IF NOT EXISTS usage (
                session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                cached_input_tokens INTEGER NOT NULL DEFAULT 0,
                cache_write_tokens INTEGER NOT NULL DEFAULT 0,
                reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                requests INTEGER NOT NULL DEFAULT 0,
                cost_usd REAL NOT NULL DEFAULT 0,
                cost_status TEXT NOT NULL DEFAULT 'unknown',
                cache_savings_usd REAL NOT NULL DEFAULT 0,
                application_cache_hits INTEGER NOT NULL DEFAULT 0,
                application_cache_misses INTEGER NOT NULL DEFAULT 0,
                application_cache_saved_tokens INTEGER NOT NULL DEFAULT 0,
                application_cache_saved_cost_usd REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS response_cache (
                cache_key TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                response_json TEXT NOT NULL,
                usage_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at REAL NOT NULL,
                last_accessed_at REAL NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_response_cache_expiry
                ON response_cache(expires_at);
            CREATE TABLE IF NOT EXISTS output_artifacts (
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                artifact_id TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY(session_id, artifact_id)
            );
            CREATE TABLE IF NOT EXISTS delegated_artifact_handoffs (
                recipient_scope_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                owner_scope_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                PRIMARY KEY(recipient_scope_id, artifact_id),
                FOREIGN KEY(owner_scope_id, artifact_id)
                    REFERENCES output_artifacts(session_id, artifact_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS key_values (
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                PRIMARY KEY(session_id, key)
            );
            CREATE TABLE IF NOT EXISTS delegated_conversations (
                scope_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
                root_session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                delegation_id TEXT NOT NULL,
                depth INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS coordination_children (
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                root_run_id TEXT NOT NULL,
                handle TEXT NOT NULL,
                parent TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                call_id TEXT NOT NULL,
                coordinated INTEGER NOT NULL,
                state TEXT NOT NULL DEFAULT 'running',
                result_json TEXT,
                delivered INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(root_run_id, handle)
            );
            CREATE TABLE IF NOT EXISTS coordination_messages (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                root_run_id TEXT NOT NULL,
                message_id TEXT NOT NULL UNIQUE,
                sender TEXT NOT NULL,
                recipient TEXT NOT NULL,
                content TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending'
            );
            CREATE INDEX IF NOT EXISTS idx_coordination_inbox
                ON coordination_messages(root_run_id, recipient, state);
            CREATE TABLE IF NOT EXISTS compaction_artifacts (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                artifact_id TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                version INTEGER NOT NULL,
                strategy TEXT NOT NULL,
                source_start_sequence INTEGER,
                source_end_sequence INTEGER,
                source_message_ids_json TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                provider TEXT,
                model TEXT,
                config_fingerprint TEXT NOT NULL,
                estimated_tokens_before INTEGER NOT NULL,
                estimated_tokens_after INTEGER NOT NULL,
                usage_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                parent_artifact_id TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_compaction_artifacts_session
                ON compaction_artifacts(session_id, sequence DESC);
            CREATE INDEX IF NOT EXISTS idx_compaction_artifacts_reuse
                ON compaction_artifacts(session_id, source_hash, config_fingerprint, strategy);
            CREATE TABLE IF NOT EXISTS compaction_provider_messages (
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                message_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY(session_id, message_hash)
            );
            CREATE TABLE IF NOT EXISTS compaction_artifact_message_refs (
                artifact_id TEXT NOT NULL REFERENCES compaction_artifacts(artifact_id)
                    ON DELETE CASCADE,
                context_index INTEGER NOT NULL,
                position INTEGER NOT NULL,
                message_hash TEXT NOT NULL,
                PRIMARY KEY(artifact_id, context_index, position)
            );
            """
        )
        current = conn.execute("SELECT value FROM schema_meta WHERE key='version'").fetchone()
        if current is None:
            conn.execute(
                "INSERT INTO schema_meta(key, value) VALUES('version', ?)", (str(_SCHEMA_VERSION),)
            )
        else:
            version = int(current[0])
            if version > _SCHEMA_VERSION:
                raise SessionError(
                    f"Database schema {current[0]} is newer than supported {_SCHEMA_VERSION}"
                )
            if version < 2:
                self._migrate_tool_calls_v2()
                version = 2
            if version < 3:
                self._migrate_cache_v3()
                version = 3
            if version < 4:
                self._migrate_compaction_v4()
                version = 4
            if version < 5:
                self._migrate_compaction_v5()
            if version < 6:
                self._migrate_cache_v3()
            conn.execute(
                "UPDATE schema_meta SET value=? WHERE key='version'", (str(_SCHEMA_VERSION),)
            )
        conn.commit()

    def _migrate_tool_calls_v2(self) -> None:
        self._connection.executescript(
            """
            ALTER TABLE tool_calls RENAME TO tool_calls_v1;
            CREATE TABLE tool_calls (
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                tool_call_id TEXT NOT NULL,
                run_id TEXT,
                tool_name TEXT NOT NULL,
                arguments_json TEXT NOT NULL,
                output TEXT,
                is_error INTEGER,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY(session_id, tool_call_id)
            );
            INSERT INTO tool_calls(
                session_id,tool_call_id,run_id,tool_name,arguments_json,output,
                is_error,status,started_at,completed_at,metadata_json
            )
            SELECT
                session_id,tool_call_id,run_id,tool_name,arguments_json,output,
                is_error,status,started_at,completed_at,metadata_json
            FROM tool_calls_v1;
            DROP TABLE tool_calls_v1;
            CREATE INDEX idx_tool_calls_session ON tool_calls(session_id, started_at);
            """
        )

    def _migrate_cache_v3(self) -> None:
        columns = {
            row["name"] for row in self._connection.execute("PRAGMA table_info(usage)").fetchall()
        }
        additions = {
            "cost_status": "TEXT NOT NULL DEFAULT 'unknown'",
            "cache_savings_usd": "REAL NOT NULL DEFAULT 0",
            "application_cache_hits": "INTEGER NOT NULL DEFAULT 0",
            "application_cache_misses": "INTEGER NOT NULL DEFAULT 0",
            "application_cache_saved_tokens": "INTEGER NOT NULL DEFAULT 0",
            "application_cache_saved_cost_usd": "REAL NOT NULL DEFAULT 0",
        }
        for name, declaration in additions.items():
            if name not in columns:
                self._connection.execute(f"ALTER TABLE usage ADD COLUMN {name} {declaration}")

    def _migrate_compaction_v4(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS compaction_artifacts (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                artifact_id TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                version INTEGER NOT NULL,
                strategy TEXT NOT NULL,
                source_start_sequence INTEGER,
                source_end_sequence INTEGER,
                source_message_ids_json TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                provider TEXT,
                model TEXT,
                config_fingerprint TEXT NOT NULL,
                estimated_tokens_before INTEGER NOT NULL,
                estimated_tokens_after INTEGER NOT NULL,
                usage_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                parent_artifact_id TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_compaction_artifacts_session
                ON compaction_artifacts(session_id, sequence DESC);
            CREATE INDEX IF NOT EXISTS idx_compaction_artifacts_reuse
                ON compaction_artifacts(session_id, source_hash, config_fingerprint, strategy);
            """
        )

    def _migrate_compaction_v5(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS compaction_provider_messages (
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                message_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY(session_id, message_hash)
            );
            CREATE TABLE IF NOT EXISTS compaction_artifact_message_refs (
                artifact_id TEXT NOT NULL REFERENCES compaction_artifacts(artifact_id)
                    ON DELETE CASCADE,
                context_index INTEGER NOT NULL,
                position INTEGER NOT NULL,
                message_hash TEXT NOT NULL,
                PRIMARY KEY(artifact_id, context_index, position)
            );
            """
        )
        rows = self._connection.execute(
            "SELECT artifact_id,session_id,metadata_json FROM compaction_artifacts"
        ).fetchall()
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            stored_metadata, messages, references = self._deduplicate_provider_contexts(
                metadata
            )
            if not references:
                continue
            self._connection.execute(
                "UPDATE compaction_artifacts SET metadata_json=? WHERE artifact_id=?",
                (json_dumps(stored_metadata), row["artifact_id"]),
            )
            self._store_compaction_message_references(
                artifact_id=str(row["artifact_id"]),
                session_id=str(row["session_id"]),
                messages=messages,
                references=references,
            )

    def create_session(
        self,
        *,
        workspace: Path,
        provider: str,
        model: str,
        title: str = "New coding session",
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> SessionInfo:
        session_id = session_id or new_id("sess")
        now = utc_now()
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO sessions(id,workspace,title,provider,model,status,created_at,updated_at,metadata_json) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    session_id,
                    str(workspace.resolve()),
                    title,
                    provider,
                    model,
                    "idle",
                    now,
                    now,
                    json_dumps(metadata or {}),
                ),
            )
            self._connection.execute("INSERT INTO usage(session_id) VALUES(?)", (session_id,))
        return self.get_session(session_id)

    def ensure_delegated_conversation(
        self, root_session_id: str, scope_id: str, delegation_id: str, depth: int,
        *, delegation_call_id: str | None = None,
    ) -> None:
        """Create a hidden durable conversation. This never resumes child execution."""
        root = self.get_session(root_session_id)
        if root.metadata.get("delegated_conversation"):
            raise SessionError("Delegated conversations require a root session owner")
        with self._lock, self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            row = self._connection.execute(
                "SELECT root_session_id,delegation_id,depth FROM delegated_conversations WHERE scope_id=?",
                (scope_id,),
            ).fetchone()
            if row is not None:
                if tuple(row) != (root_session_id, delegation_id, depth):
                    raise SessionError("Delegated conversation identity mismatch")
                if delegation_call_id is not None:
                    saved = self._connection.execute(
                        "SELECT metadata_json FROM sessions WHERE id=?", (scope_id,),
                    ).fetchone()
                    metadata = json.loads(saved[0])
                    if metadata.get("delegation_call_id") not in (None, delegation_call_id):
                        raise SessionError("Delegated conversation call identity mismatch")
                    metadata["delegation_call_id"] = delegation_call_id
                    self._connection.execute("UPDATE sessions SET metadata_json=? WHERE id=?",
                                             (json_dumps(metadata), scope_id))
                return
            now = utc_now()
            self._connection.execute(
                "INSERT INTO sessions(id,workspace,title,provider,model,status,created_at,updated_at,metadata_json) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (scope_id, root.workspace, "Internal delegated conversation", root.provider,
                 root.model, "idle", now, now, json_dumps({
                     "delegated_conversation": True, "root_session_id": root_session_id,
                     "delegation_id": delegation_id, "depth": depth,
                     "delegation_call_id": delegation_call_id,
                 })),
            )
            self._connection.execute("INSERT INTO usage(session_id) VALUES(?)", (scope_id,))
            self._connection.execute(
                "INSERT INTO delegated_conversations VALUES(?,?,?,?)",
                (scope_id, root_session_id, delegation_id, depth),
            )

    def sync_conversation_messages(self, scope_id: str, messages: list[Message]) -> None:
        """Append a validated snapshot suffix; never replace an earlier child message."""
        with self._lock, self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            if self._connection.execute(
                "SELECT 1 FROM delegated_conversations WHERE scope_id=?", (scope_id,),
            ).fetchone() is None:
                raise SessionError("Unknown delegated conversation")
            saved = [Message.from_dict(json.loads(row[0])) for row in self._connection.execute(
                "SELECT payload_json FROM messages WHERE session_id=? ORDER BY sequence", (scope_id,),
            )]
            if any(
                left.to_dict() != right.to_dict()
                for left, right in zip(saved, messages, strict=False)
            ):
                raise SessionError("Delegated conversation snapshot is not append-only")
            for message in messages[len(saved):]:
                self._append_message_locked(scope_id, message)

    def get_session(self, session_id: str) -> SessionInfo:
        with self._lock, self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            row = self._connection.execute(
                "SELECT * FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
            row = self._reconcile_running_row(row)
        if row is None:
            raise SessionError(f"Unknown session: {session_id}")
        return _session_info(row)

    def list_sessions(
        self, *, workspace: Path | None = None, limit: int = 100
    ) -> list[SessionInfo]:
        query = ("SELECT * FROM sessions WHERE NOT EXISTS "
                 "(SELECT 1 FROM delegated_conversations WHERE scope_id=sessions.id)")
        params: list[Any] = []
        if workspace is not None:
            query += " AND workspace=?"
            params.append(str(workspace.resolve()))
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, limit))
        with self._lock, self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            rows = self._connection.execute(query, params).fetchall()
            rows = [self._reconcile_running_row(row) for row in rows]
        return [_session_info(row) for row in rows]

    def acquire_run_lease(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._run_leases:
                raise SessionError(f"Session {session_id} already has a local run lease")
            exists = self._connection.execute(
                "SELECT 1 FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
            if exists is None:
                raise SessionError(f"Unknown session: {session_id}")
            lease = _SessionRunLease(self._run_lease_path(session_id))
            if not lease.acquire():
                raise SessionError(f"Session {session_id} is already running")
            self._run_leases[session_id] = lease

    def release_run_lease(self, session_id: str) -> None:
        with self._lock:
            lease = self._run_leases.pop(session_id, None)
            if lease is not None:
                lease.release()

    def _reconcile_running_row(self, row: sqlite3.Row | None) -> sqlite3.Row | None:
        if (
            row is None
            or row["status"] != "running"
            or str(row["id"]) in self._run_leases
        ):
            return row
        session_id = str(row["id"])
        lease = _SessionRunLease(self._run_lease_path(session_id))
        if not lease.acquire():
            return row
        try:
            self._interrupt_coordination_locked(session_id)
            self._connection.execute(
                "UPDATE sessions SET status='interrupted',updated_at=? "
                "WHERE id=? AND status='running'",
                (utc_now(), session_id),
            )
            return self._connection.execute(
                "SELECT * FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
        finally:
            lease.release()

    def _run_lease_path(self, session_id: str) -> Path:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return self.path.parent / "run-leases" / f"{digest}.lock"

    def update_session(
        self,
        session_id: str,
        *,
        status: str | None = None,
        title: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionInfo:
        updates = ["updated_at=?"]
        values: list[Any] = [utc_now()]
        for column, value in (
            ("status", status),
            ("title", title),
            ("provider", provider),
            ("model", model),
        ):
            if value is not None:
                updates.append(f"{column}=?")
                values.append(value)
        if metadata is not None:
            updates.append("metadata_json=?")
            values.append(json_dumps(metadata))
        values.append(session_id)
        with self._lock, self._connection:
            cursor = self._connection.execute(
                f"UPDATE sessions SET {', '.join(updates)} WHERE id=?", values
            )
            if cursor.rowcount == 0:
                raise SessionError(f"Unknown session: {session_id}")
        return self.get_session(session_id)

    def delete_session(self, session_id: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM sessions WHERE id IN "
                "(SELECT scope_id FROM delegated_conversations WHERE root_session_id=?)", (session_id,),
            )
            cursor = self._connection.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            if cursor.rowcount == 0:
                raise SessionError(f"Unknown session: {session_id}")

    def append_message(self, session_id: str, message: Message) -> None:
        with self._lock, self._connection:
            self._append_message_locked(session_id, message)

    def start_coordination_child(
        self, session_id: str, root: str, handle: str, parent: str,
        scope: str, call_id: str, coordinated: bool,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO coordination_children(session_id,root_run_id,handle,parent,scope_id,call_id,coordinated) VALUES(?,?,?,?,?,?,?)",
                (session_id, root, handle, parent, scope, call_id, int(coordinated)),
            )

    def finish_coordination_child(self, root: str, handle: str, state: str, result: dict) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE coordination_children SET state=?,result_json=? WHERE root_run_id=? AND handle=?",
                (state, json_dumps(result), root, handle),
            )

    def ack_coordination_child(self, root: str, handle: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE coordination_children SET delivered=1 WHERE root_run_id=? AND handle=?", (root, handle),
            )

    def queue_coordination_message(
        self, session_id: str, root: str, identifier: str, sender: str, recipient: str,
        content: str, *, inbox_limit: int, run_limit: int,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            retained = self._connection.execute(
                "SELECT count(*) FROM coordination_messages WHERE root_run_id=?", (root,),
            ).fetchone()[0]
            pending = self._connection.execute(
                "SELECT count(*) FROM coordination_messages WHERE root_run_id=? AND recipient=? AND state='pending'",
                (root, recipient),
            ).fetchone()[0]
            if retained >= run_limit or pending >= inbox_limit:
                raise SessionError("Coordination queue/retention limit reached; message was not queued. Shorten the exchange or return a terminal handoff")
            self._connection.execute(
                "INSERT INTO coordination_messages(session_id,root_run_id,message_id,sender,recipient,content) VALUES(?,?,?,?,?,?)",
                (session_id, root, identifier, sender, recipient, content),
            )

    def has_coordination_delivery(self, root: str, recipient: str) -> bool:
        with self._lock:
            return bool(self._connection.execute(
                "SELECT 1 FROM coordination_messages WHERE root_run_id=? AND recipient=? AND state='pending' "
                "UNION ALL SELECT 1 FROM coordination_children WHERE root_run_id=? AND parent=? AND state!='running' AND delivered=0 LIMIT 1",
                (root, recipient, root, recipient),
            ).fetchone())

    def deliver_coordination(self, root: str, recipient: str, scope: str, limit: int,
                             frame: str, redactor: Any) -> list[Message]:
        from ..agent.coordination import byte_prefix
        from ..models import Role

        delivered: list[Message] = []
        with self._lock, self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            for row in self._connection.execute(
                "SELECT * FROM coordination_messages WHERE root_run_id=? AND recipient=? AND state='pending' ORDER BY sequence",
                (root, recipient),
            ).fetchall():
                size = len(row["content"].encode("utf-8"))
                if size > limit:
                    break
                message = Message(role=Role.ASSISTANT, id=row["message_id"], content=row["content"],
                                  metadata={"internal": "coordination", "coordination": True})
                self._append_message_locked(scope, message)
                self._connection.execute("UPDATE coordination_messages SET state='delivered' WHERE message_id=?", (message.id,))
                delivered.append(message)
                limit -= size
            for row in self._connection.execute(
                "SELECT * FROM coordination_children WHERE root_run_id=? AND parent=? AND state!='running' AND delivered=0 ORDER BY handle",
                (root, recipient),
            ).fetchall():
                if limit < 512:
                    break
                result = json.loads(row["result_json"] or '{}')
                header = (frame + json_dumps({"handle": row["handle"], "state": row["state"],
                          "source": row["handle"] + ".result"})
                          + "\nUse search_history with the source ID for the retained terminal receipt and evidence.\n")
                content = header + byte_prefix(redactor.text(result.get("output", "Interrupted previous work; no replay.")),
                                                min(4096, limit) - len(header.encode("utf-8")))
                message = Message(role=Role.ASSISTANT, id="terminal:" + row["handle"], content=content,
                                  is_error=row["state"] != "completed",
                                  metadata={"internal": "coordination", "coordination": True, "child_handle": row["handle"]})
                self._append_message_locked(scope, message)
                self._connection.execute("UPDATE coordination_children SET delivered=1 WHERE root_run_id=? AND handle=?", (root, row["handle"]))
                delivered.append(message)
                limit -= len(content.encode("utf-8"))
        return delivered

    def interrupt_coordination(self, session_id: str) -> None:
        """Called only under the root run lease; never recreate a live worker."""
        with self._lock, self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            self._interrupt_coordination_locked(session_id)

    def _interrupt_coordination_locked(self, session_id: str) -> None:
        """Use the caller's lock and transaction for recovery and all deliveries."""
        from ..models import Role

        rows = self._connection.execute(
            "SELECT * FROM coordination_children WHERE session_id=? AND state='running'", (session_id,),
        ).fetchall()
        for row in rows:
            self._connection.execute(
                "UPDATE coordination_children SET state='interrupted' WHERE root_run_id=? AND handle=?",
                (row["root_run_id"], row["handle"]),
            )
        # Preserve pending work in its actual recipient's durable history. Only
        # the root can resume; stopped children are historical conversations.
        pending = self._connection.execute(
            "SELECT * FROM coordination_messages WHERE session_id=? AND state='pending' ORDER BY sequence", (session_id,),
        ).fetchall()
        for row in pending:
            child = self._connection.execute(
                "SELECT scope_id FROM coordination_children WHERE root_run_id=? AND handle=?",
                (row["root_run_id"], row["recipient"]),
            ).fetchone()
            scope = child[0] if child is not None else session_id
            if self._connection.execute("SELECT 1 FROM sessions WHERE id=?", (scope,)).fetchone():
                self._close_interrupted_bundle_locked(session_id, scope)
                self._append_message_locked(scope, Message(
                    role=Role.ASSISTANT, id=row["message_id"],
                    content="Interrupted previous run; historical coordination only.\n" + row["content"],
                    metadata={"internal": "coordination", "coordination": True, "interrupted": True},
                ))
            self._connection.execute("UPDATE coordination_messages SET state='interrupted' WHERE message_id=?", (row["message_id"],))
        for row in self._connection.execute(
            "SELECT * FROM coordination_children WHERE session_id=? AND delivered=0 AND state!='running'", (session_id,),
        ).fetchall():
            parent = self._connection.execute(
                "SELECT scope_id FROM coordination_children WHERE root_run_id=? AND handle=?", (row["root_run_id"], row["parent"]),
            ).fetchone()
            scope = parent[0] if parent else session_id
            if self._connection.execute("SELECT 1 FROM sessions WHERE id=?", (scope,)).fetchone():
                self._close_interrupted_bundle_locked(session_id, scope)
                self._append_message_locked(scope, Message(
                    role=Role.ASSISTANT, id="terminal:" + row["handle"],
                    content=f"Historical child {row['handle']}: {row['state']}. No tools or requests were replayed. Search retained receipt {row['handle']}.result before retrying.",
                    metadata={"internal": "coordination", "coordination": True, "interrupted": True},
                ))
            self._connection.execute("UPDATE coordination_children SET delivered=1 WHERE root_run_id=? AND handle=?", (row["root_run_id"], row["handle"]))

    def _close_interrupted_bundle_locked(self, session_id: str, scope: str) -> None:
        """Close only a trailing bundle before a historical notice is appended.

        Leave canonical receipts to the runner, which knows tool mutation scopes
        and recovers nested effects. A child transcript uses local call IDs while
        its receipts belong to the root and use namespaced IDs.
        """
        from ..agent.compaction import abandoned_tool_call_results

        messages = [Message.from_dict(json.loads(row[0])) for row in self._connection.execute(
            "SELECT payload_json FROM messages WHERE session_id=? ORDER BY sequence", (scope,),
        )]
        child = self._connection.execute(
            "SELECT delegation_id FROM delegated_conversations WHERE scope_id=? AND root_session_id=?",
            (scope, session_id),
        ).fetchone()
        prefix = child[0] + "." if child else ""
        receipts = {
            str(row["tool_call_id"])[len(prefix):]: row
            for row in self._connection.execute(
                "SELECT * FROM tool_calls WHERE session_id=?", (session_id,),
            )
            if not child or (row["run_id"] == scope and str(row["tool_call_id"]).startswith(prefix))
        }
        repairs = abandoned_tool_call_results(messages, running_call_ids=set(receipts))
        for repair in repairs:
            receipt = receipts.get(repair.tool_call_id or "")
            if receipt is not None:
                metadata = json.loads(receipt["metadata_json"] or "{}")
                if receipt["status"] != "running":
                    # A child can crash after its canonical result commits but
                    # before its conversation snapshot is appended.
                    repair.content = receipt["output"] or ""
                    repair.is_error = bool(receipt["is_error"])
                    repair.metadata = metadata
                else:
                    repair.metadata = {**metadata, **repair.metadata}
            self._append_message_locked(scope, repair)

    def replace_messages(self, session_id: str, messages: Iterable[Message]) -> None:
        del session_id, messages
        raise SessionError("Durable messages are append-only and cannot be replaced")

    def messages(self, session_id: str) -> list[Message]:
        self.get_session(session_id)
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload_json FROM messages WHERE session_id=? ORDER BY sequence",
                (session_id,),
            ).fetchall()
        return [Message.from_dict(json.loads(row[0])) for row in rows]

    def sequenced_messages(self, session_id: str) -> list[tuple[int, Message]]:
        self.get_session(session_id)
        with self._lock:
            rows = self._connection.execute(
                "SELECT sequence,payload_json FROM messages WHERE session_id=? ORDER BY sequence",
                (session_id,),
            ).fetchall()
        return [(int(row[0]), Message.from_dict(json.loads(row[1]))) for row in rows]

    def append_compaction_artifact(self, artifact: CompactionArtifact) -> None:
        """Persist an immutable artifact separately from durable messages."""

        stored_metadata, provider_messages, references = (
            self._deduplicate_provider_contexts(artifact.metadata)
        )
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO compaction_artifacts(
                    artifact_id,session_id,version,strategy,source_start_sequence,
                    source_end_sequence,source_message_ids_json,source_hash,summary_text,
                    provider,model,config_fingerprint,estimated_tokens_before,
                    estimated_tokens_after,usage_json,created_at,parent_artifact_id,metadata_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    artifact.id,
                    artifact.session_id,
                    artifact.version,
                    artifact.strategy,
                    artifact.source_start_sequence,
                    artifact.source_end_sequence,
                    json_dumps(artifact.source_message_ids),
                    artifact.source_hash,
                    artifact.summary_text,
                    artifact.provider,
                    artifact.model,
                    artifact.config_fingerprint,
                    artifact.estimated_tokens_before,
                    artifact.estimated_tokens_after,
                    json_dumps(artifact.usage.to_dict()),
                    artifact.created_at,
                    artifact.parent_artifact_id,
                    json_dumps(stored_metadata),
                ),
            )
            self._store_compaction_message_references(
                artifact_id=artifact.id,
                session_id=artifact.session_id,
                messages=provider_messages,
                references=references,
            )
            self._connection.execute(
                "UPDATE sessions SET updated_at=? WHERE id=?",
                (utc_now(), artifact.session_id),
            )

    def latest_compaction_artifact(
        self,
        session_id: str,
        *,
        config_fingerprint: str | None = None,
        strategy: str | None = None,
    ) -> CompactionArtifact | None:
        query = "SELECT * FROM compaction_artifacts WHERE session_id=?"
        parameters: list[Any] = [session_id]
        if config_fingerprint is not None:
            query += " AND config_fingerprint=?"
            parameters.append(config_fingerprint)
        if strategy is not None:
            query += " AND strategy=?"
            parameters.append(strategy)
        query += " ORDER BY sequence DESC LIMIT 1"
        with self._lock:
            row = self._connection.execute(query, parameters).fetchone()
            return self._load_compaction_artifact(row) if row is not None else None

    def reusable_compaction_artifact(
        self,
        session_id: str,
        *,
        source_hash: str,
        config_fingerprint: str,
        strategy: str,
    ) -> CompactionArtifact | None:
        with self._lock:
            row = self._connection.execute(
                """SELECT * FROM compaction_artifacts
                WHERE session_id=? AND source_hash=? AND config_fingerprint=? AND strategy=?
                ORDER BY sequence DESC LIMIT 1""",
                (session_id, source_hash, config_fingerprint, strategy),
            ).fetchone()
            return self._load_compaction_artifact(row) if row is not None else None

    @staticmethod
    def _deduplicate_provider_contexts(
        metadata: dict[str, Any],
    ) -> tuple[
        dict[str, Any],
        dict[str, str],
        list[tuple[int, int, str]],
    ]:
        stored = json.loads(json_dumps(metadata))
        contexts = stored.get("provider_contexts")
        if not isinstance(contexts, list):
            return stored, {}, []
        provider_messages: dict[str, str] = {}
        references: list[tuple[int, int, str]] = []
        for context_index, provider_context in enumerate(contexts):
            if not isinstance(provider_context, dict):
                continue
            messages = provider_context.get("messages")
            if not isinstance(messages, list):
                continue
            for position, message in enumerate(messages):
                payload = json_dumps(message)
                message_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
                provider_messages[message_hash] = payload
                references.append((context_index, position, message_hash))
            provider_context.pop("messages", None)
            provider_context["message_count"] = len(messages)
        if references:
            stored["provider_context_storage"] = (
                _DEDUPLICATED_PROVIDER_CONTEXT_STORAGE
            )
        return stored, provider_messages, references

    def _store_compaction_message_references(
        self,
        *,
        artifact_id: str,
        session_id: str,
        messages: dict[str, str],
        references: list[tuple[int, int, str]],
    ) -> None:
        for message_hash, payload in messages.items():
            self._connection.execute(
                """INSERT INTO compaction_provider_messages(
                    session_id,message_hash,payload_json
                ) VALUES(?,?,?) ON CONFLICT(session_id,message_hash) DO NOTHING""",
                (session_id, message_hash, payload),
            )
        self._connection.executemany(
            """INSERT INTO compaction_artifact_message_refs(
                artifact_id,context_index,position,message_hash
            ) VALUES(?,?,?,?)""",
            [
                (artifact_id, context_index, position, message_hash)
                for context_index, position, message_hash in references
            ],
        )

    def _load_compaction_artifact(self, row: sqlite3.Row) -> CompactionArtifact:
        metadata = json.loads(row["metadata_json"] or "{}")
        if metadata.get("provider_context_storage") != (
            _DEDUPLICATED_PROVIDER_CONTEXT_STORAGE
        ):
            return _compaction_artifact(row, metadata=metadata)
        contexts = metadata.get("provider_contexts")
        if not isinstance(contexts, list):
            raise SessionError("Deduplicated compaction artifact has no provider contexts")
        references = self._connection.execute(
            """SELECT refs.context_index,refs.position,messages.payload_json
            FROM compaction_artifact_message_refs AS refs
            JOIN compaction_provider_messages AS messages
              ON messages.session_id=? AND messages.message_hash=refs.message_hash
            WHERE refs.artifact_id=?
            ORDER BY refs.context_index,refs.position""",
            (row["session_id"], row["artifact_id"]),
        ).fetchall()
        messages_by_context: dict[int, list[dict[str, Any]]] = {}
        for reference in references:
            messages_by_context.setdefault(int(reference["context_index"]), []).append(
                json.loads(reference["payload_json"])
            )
        for context_index, provider_context in enumerate(contexts):
            if not isinstance(provider_context, dict):
                continue
            expected = int(provider_context.pop("message_count", 0))
            restored = messages_by_context.get(context_index, [])
            if len(restored) != expected:
                raise SessionError(
                    f"Compaction artifact {row['artifact_id']} has incomplete provider messages"
                )
            provider_context["messages"] = restored
        metadata.pop("provider_context_storage", None)
        return _compaction_artifact(row, metadata=metadata)

    def append_event(self, event: Event) -> None:
        self.append_events([event])

    def append_events(self, events: Iterable[Event]) -> None:
        rows = [
            (
                event.id,
                event.session_id,
                event.run_id,
                event.type,
                json_dumps(event.to_dict()),
                event.created_at,
            )
            for event in events
        ]
        if not rows:
            return
        with self._lock, self._connection:
            self._connection.executemany(
                "INSERT OR IGNORE INTO events(event_id,session_id,run_id,type,payload_json,created_at) VALUES(?,?,?,?,?,?)",
                rows,
            )

    def events(
        self, session_id: str, *, after_sequence: int = 0, limit: int = 10_000
    ) -> list[tuple[int, Event]]:
        return self._event_page(
            session_id,
            after_sequence=after_sequence,
            limit=limit,
        )

    def prune_events(
        self,
        *,
        max_count: int,
        max_age_seconds: int = 0,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Prune old event rows without changing durable messages or tool records."""

        with self._lock:
            before, pruned, where, parameters = self._event_prune_plan(
                self._connection,
                max_count=max_count,
                max_age_seconds=max_age_seconds,
            )
            if not dry_run and pruned:
                with self._connection:
                    self._connection.execute(
                        f"DELETE FROM events WHERE {where}", parameters
                    )
        return {
            "events_before": before,
            "events_after": before - pruned,
            "pruned_events": pruned,
            "dry_run": dry_run,
        }

    @classmethod
    def preview_event_prune(
        cls,
        path: Path,
        *,
        max_count: int,
        max_age_seconds: int = 0,
    ) -> dict[str, Any]:
        """Report event retention through a read-only database connection."""

        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'"
            ).fetchone()
            columns = (
                {
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(events)").fetchall()
                }
                if table is not None
                else set()
            )
            if not {"sequence", "created_at"}.issubset(columns):
                before = 0
                pruned = 0
            else:
                before, pruned, _, _ = cls._event_prune_plan(
                    connection,
                    max_count=max_count,
                    max_age_seconds=max_age_seconds,
                )
        finally:
            connection.close()
        return {
            "events_before": before,
            "events_after": before - pruned,
            "pruned_events": pruned,
            "dry_run": True,
        }

    @staticmethod
    def _event_prune_plan(
        connection: sqlite3.Connection,
        *,
        max_count: int,
        max_age_seconds: int,
    ) -> tuple[int, int, str, list[Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        before = int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])
        if before > max_count:
            boundary = connection.execute(
                "SELECT sequence FROM events ORDER BY sequence DESC LIMIT 1 OFFSET ?",
                (max_count - 1,),
            ).fetchone()
            if boundary is not None:
                clauses.append("sequence < ?")
                parameters.append(int(boundary[0]))
        if max_age_seconds:
            cutoff = (datetime.now(UTC) - timedelta(seconds=max_age_seconds)).isoformat()
            clauses.append("created_at < ?")
            parameters.append(cutoff)
        where = " OR ".join(f"({clause})" for clause in clauses) or "0"
        pruned = int(
            connection.execute(
                f"SELECT COUNT(*) FROM events WHERE {where}", parameters
            ).fetchone()[0]
        )
        return before, pruned, where, parameters

    def _event_page(
        self,
        session_id: str,
        *,
        after_sequence: int,
        limit: int,
        through_sequence: int | None = None,
    ) -> list[tuple[int, Event]]:
        upper_bound = "" if through_sequence is None else " AND sequence<=?"
        parameters: tuple[Any, ...]
        if through_sequence is None:
            parameters = (session_id, after_sequence, limit)
        else:
            parameters = (session_id, after_sequence, through_sequence, limit)
        with self._lock:
            rows = self._connection.execute(
                "SELECT sequence,payload_json FROM events "
                f"WHERE session_id=? AND sequence>?{upper_bound} "
                "ORDER BY sequence LIMIT ?",
                parameters,
            ).fetchall()
        result: list[tuple[int, Event]] = []
        for row in rows:
            data = json.loads(row["payload_json"])
            result.append(
                (
                    int(row["sequence"]),
                    Event(
                        id=data["id"],
                        type=data["type"],
                        session_id=data.get("session_id"),
                        run_id=data.get("run_id"),
                        data=data.get("data") or {},
                        created_at=data["created_at"],
                    ),
                )
            )
        return result

    def _export_events(self, session_id: str) -> list[tuple[int, Event]]:
        with self._lock:
            row = self._connection.execute(
                "SELECT MAX(sequence) FROM events WHERE session_id=?",
                (session_id,),
            ).fetchone()
        through_sequence = int(row[0] or 0)
        after_sequence = 0
        result: list[tuple[int, Event]] = []
        while after_sequence < through_sequence:
            page = self._event_page(
                session_id,
                after_sequence=after_sequence,
                limit=_EVENT_EXPORT_PAGE_SIZE,
                through_sequence=through_sequence,
            )
            if not page:
                break
            result.extend(page)
            after_sequence = page[-1][0]
        return result

    def start_tool_call(
        self, session_id: str, run_id: str, call_id: str, name: str, arguments: dict[str, Any],
        *, metadata: dict[str, Any] | None = None,
    ) -> None:
        if not call_id.strip():
            raise SessionError("Tool call ID must be non-empty")
        arguments_json = json_dumps(arguments)
        normalized_arguments = json.loads(arguments_json)
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """INSERT INTO tool_calls(
                    session_id,tool_call_id,run_id,tool_name,arguments_json,status,started_at,metadata_json
                ) VALUES(?,?,?,?,?,'running',?,?)
                ON CONFLICT(session_id,tool_call_id) DO NOTHING""",
                (session_id, call_id, run_id, name, arguments_json, utc_now(), json_dumps(metadata or {})),
            )
            if cursor.rowcount:
                return
            existing = self._connection.execute(
                """SELECT run_id,tool_name,arguments_json,status
                FROM tool_calls WHERE session_id=? AND tool_call_id=?""",
                (session_id, call_id),
            ).fetchone()
            if (
                existing is not None
                and existing["status"] == "running"
                and existing["run_id"] == run_id
                and existing["tool_name"] == name
                and json.loads(existing["arguments_json"]) == normalized_arguments
            ):
                return
            raise SessionError(
                f"Tool call {call_id!r} already exists and cannot be overwritten"
            )

    def complete_tool_call(
        self,
        session_id: str,
        call_id: str,
        *,
        output: str,
        is_error: bool,
        metadata: dict[str, Any] | None = None,
        message: Message | None = None,
    ) -> None:
        status = "error" if is_error else "completed"
        metadata_json = json_dumps(metadata or {})
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """UPDATE tool_calls SET
                    output=?,is_error=?,status=?,completed_at=?,metadata_json=?
                WHERE session_id=? AND tool_call_id=? AND status='running'""",
                (
                    output,
                    int(is_error),
                    status,
                    utc_now(),
                    metadata_json,
                    session_id,
                    call_id,
                ),
            )
            if cursor.rowcount == 0:
                existing = self._connection.execute(
                    """SELECT output,is_error,status,metadata_json
                    FROM tool_calls WHERE session_id=? AND tool_call_id=?""",
                    (session_id, call_id),
                ).fetchone()
                if existing is None:
                    raise SessionError(f"Tool call {call_id!r} does not exist")
                if not (
                    existing["output"] == output
                    and existing["is_error"] == int(is_error)
                    and existing["status"] == status
                    and existing["metadata_json"] == metadata_json
                ):
                    raise SessionError(
                        f"Tool call {call_id!r} is terminal and cannot be overwritten"
                    )
                if message is not None:
                    self._append_message_locked(session_id, message)
                return
            if message is not None:
                self._append_message_locked(session_id, message)

    def cancel_tool_call(
        self,
        session_id: str,
        call_id: str,
        *,
        reason: str = "cancelled",
        message: Message | None = None,
    ) -> None:
        with self._lock, self._connection:
            previous = self._connection.execute(
                "SELECT metadata_json FROM tool_calls WHERE session_id=? AND tool_call_id=?",
                (session_id, call_id),
            ).fetchone()
            metadata = json.loads(previous["metadata_json"] or "{}") if previous else {}
            metadata_json = json_dumps({**metadata, "cancelled": True})
            cursor = self._connection.execute(
                """UPDATE tool_calls SET
                    output=?,is_error=1,status='cancelled',completed_at=?,metadata_json=?
                WHERE session_id=? AND tool_call_id=? AND status='running'""",
                (reason, utc_now(), metadata_json, session_id, call_id),
            )
            if cursor.rowcount == 0:
                existing = self._connection.execute(
                    """SELECT output,is_error,status,metadata_json
                    FROM tool_calls WHERE session_id=? AND tool_call_id=?""",
                    (session_id, call_id),
                ).fetchone()
                if existing is None:
                    raise SessionError(f"Tool call {call_id!r} does not exist")
                if not (
                    existing["output"] == reason
                    and existing["is_error"] == 1
                    and existing["status"] == "cancelled"
                    and existing["metadata_json"] == metadata_json
                ):
                    raise SessionError(
                        f"Tool call {call_id!r} is terminal and cannot be overwritten"
                    )
            if message is not None:
                self._append_message_locked(session_id, message)

    def _append_message_locked(self, session_id: str, message: Message) -> None:
        payload = json_dumps(message.to_dict())
        existing = self._connection.execute(
            "SELECT role,payload_json,created_at FROM messages "
            "WHERE session_id=? AND message_id=?",
            (session_id, message.id),
        ).fetchone()
        if existing is not None:
            if (
                existing["role"] != message.role.value
                or existing["payload_json"] != payload
                or existing["created_at"] != message.created_at
            ):
                raise SessionError(
                    f"Durable message {message.id} already exists and cannot be overwritten"
                )
            return
        self._connection.execute(
            "INSERT INTO messages(session_id,message_id,role,payload_json,created_at) "
            "VALUES(?,?,?,?,?)",
            (
                session_id,
                message.id,
                message.role.value,
                payload,
                message.created_at,
            ),
        )
        self._connection.execute(
            "UPDATE sessions SET updated_at=? WHERE id=?", (utc_now(), session_id)
        )

    def tool_calls(self, session_id: str) -> list[dict[str, Any]]:
        self.get_session(session_id)
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM tool_calls WHERE session_id=? ORDER BY started_at, tool_call_id",
                (session_id,),
            ).fetchall()
        return [
            {
                "session_id": row["session_id"],
                "tool_call_id": row["tool_call_id"],
                "run_id": row["run_id"],
                "tool_name": row["tool_name"],
                "arguments": json.loads(row["arguments_json"]),
                "output": row["output"],
                "is_error": None if row["is_error"] is None else bool(row["is_error"]),
                "status": row["status"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "metadata": json.loads(row["metadata_json"] or "{}"),
            }
            for row in rows
        ]

    def finding_source(self, session_id: str, workspace: Path, call_id: str) -> dict[str, Any]:
        """Read one completed source from the owning session, not a global history scan."""
        self.authorize_workspace(session_id, workspace)
        with self._lock:
            scope = self._connection.execute(
                "SELECT root_session_id,delegation_id FROM delegated_conversations WHERE scope_id=?", (session_id,),
            ).fetchone()
            owner = str(scope[0]) if scope is not None else session_id
            scoped_call_id = str(scope[1]) + "." + call_id if scope is not None else call_id
            run_scope = session_id if scope is not None else None
            row = self._connection.execute(
                "SELECT tool_name,output,is_error,status,metadata_json FROM tool_calls "
                "WHERE session_id=? AND tool_call_id IN (?,?) AND (? IS NULL OR run_id=? "
                "OR EXISTS (SELECT 1 FROM delegated_artifact_handoffs AS handoff "
                "WHERE handoff.recipient_scope_id=? AND handoff.owner_scope_id=tool_calls.run_id "
                "AND handoff.artifact_id=json_extract(tool_calls.metadata_json,'$.output_artifact.artifact_id')))",
                (owner, call_id, scoped_call_id, run_scope, run_scope, session_id),
            ).fetchone()
        if row is None or row["status"] != "completed" or row["is_error"]:
            raise SessionError("Finding source must be a successful completed call in this session")
        return {"tool": row["tool_name"], "output": row["output"],
                "metadata": json.loads(row["metadata_json"] or "{}")}

    def add_usage(self, session_id: str, usage: Usage) -> Usage:
        with self._lock, self._connection:
            self._add_usage_locked(session_id, usage)
        return self.usage(session_id)

    def _add_usage_locked(self, session_id: str, usage: Usage) -> None:
        aggregate = self.usage(session_id)
        aggregate.add(usage)
        self._connection.execute("UPDATE usage SET cost_status=? WHERE session_id=?", (aggregate.cost_status, session_id))
        cursor = self._connection.execute(
            """UPDATE usage SET
                input_tokens=input_tokens+?, output_tokens=output_tokens+?,
                cached_input_tokens=cached_input_tokens+?, cache_write_tokens=cache_write_tokens+?,
                reasoning_tokens=reasoning_tokens+?, requests=requests+?, cost_usd=cost_usd+?,
                cache_savings_usd=cache_savings_usd+?,
                application_cache_hits=application_cache_hits+?,
                application_cache_misses=application_cache_misses+?,
                application_cache_saved_tokens=application_cache_saved_tokens+?,
                application_cache_saved_cost_usd=application_cache_saved_cost_usd+?
                WHERE session_id=?""",
            (
                usage.input_tokens,
                usage.output_tokens,
                usage.cached_input_tokens,
                usage.cache_write_tokens,
                usage.reasoning_tokens,
                usage.requests,
                usage.cost_usd,
                usage.cache_savings_usd,
                usage.application_cache_hits,
                usage.application_cache_misses,
                usage.application_cache_saved_tokens,
                usage.application_cache_saved_cost_usd,
                session_id,
            ),
        )
        if cursor.rowcount == 0:
            raise SessionError(f"Unknown session: {session_id}")
        for attempt in usage.attempt_usage:
            event = Event(type="model.attempt_usage", session_id=session_id, data=attempt)
            self._connection.execute(
                "INSERT INTO events(event_id,session_id,run_id,type,payload_json,created_at) VALUES(?,?,?,?,?,?)",
                (event.id, session_id, None, event.type, json_dumps(event.to_dict()), event.created_at),
            )

    def put_pending_compaction_summary(
        self,
        session_id: str,
        key: str,
        value: dict[str, Any],
    ) -> bool:
        """Persist a provider summary and return whether this caller owns it."""

        pending = {**value, "usage_settled": False}
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """INSERT INTO key_values(session_id,key,value_json) VALUES(?,?,?)
                ON CONFLICT(session_id,key) DO NOTHING""",
                (session_id, key, json_dumps(pending)),
            )
            if cursor.rowcount:
                return True
            row = self._connection.execute(
                "SELECT value_json FROM key_values WHERE session_id=? AND key=?",
                (session_id, key),
            ).fetchone()
            existing = json.loads(row["value_json"]) if row is not None else None
            if not isinstance(existing, dict):
                raise SessionError(f"Compaction summary {key!r} has invalid durable state")
            # The cache key identifies the request, not the nondeterministic response.
            # An identical or different provider response is still a losing write.
            return False

    def get_compaction_summary(
        self,
        session_id: str,
        key: str,
    ) -> dict[str, Any] | None:
        value = self.get_value(session_id, key)
        return value if isinstance(value, dict) else None

    def settle_compaction_summary_usage(
        self,
        session_id: str,
        key: str,
    ) -> tuple[dict[str, Any], Usage, bool]:
        """Atomically charge pending summary usage exactly once."""

        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT value_json FROM key_values WHERE session_id=? AND key=?",
                (session_id, key),
            ).fetchone()
            if row is None:
                raise SessionError(f"Compaction summary {key!r} does not exist")
            value = json.loads(row["value_json"])
            if not isinstance(value, dict):
                raise SessionError(f"Compaction summary {key!r} has invalid durable state")
            usage = Usage.from_dict(value.get("usage"))
            if value.get("usage_settled") is True:
                return value, usage, False
            original_json = row["value_json"]
            value["usage_settled"] = True
            cursor = self._connection.execute(
                """UPDATE key_values SET value_json=?
                WHERE session_id=? AND key=? AND value_json=?""",
                (json_dumps(value), session_id, key, original_json),
            )
            if cursor.rowcount == 0:
                current = self._connection.execute(
                    "SELECT value_json FROM key_values WHERE session_id=? AND key=?",
                    (session_id, key),
                ).fetchone()
                current_value = (
                    json.loads(current["value_json"]) if current is not None else None
                )
                if isinstance(current_value, dict) and current_value.get(
                    "usage_settled"
                ) is True:
                    return current_value, Usage.from_dict(current_value.get("usage")), False
                raise SessionError(
                    f"Compaction summary {key!r} changed during usage settlement"
                )
            self._add_usage_locked(session_id, usage)
            scope = self._connection.execute(
                "SELECT root_session_id FROM delegated_conversations WHERE scope_id=?", (session_id,),
            ).fetchone()
            if scope is not None:
                self._add_usage_locked(str(scope[0]), usage)
            return value, usage, True

    def usage(self, session_id: str) -> Usage:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM usage WHERE session_id=?", (session_id,)
            ).fetchone()
        if row is None:
            raise SessionError(f"Unknown session: {session_id}")
        return Usage(
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            cached_input_tokens=row["cached_input_tokens"],
            cache_write_tokens=row["cache_write_tokens"],
            reasoning_tokens=row["reasoning_tokens"],
            requests=row["requests"],
            cost_usd=row["cost_usd"],
            cost_status=row["cost_status"],
            cache_savings_usd=row["cache_savings_usd"],
            application_cache_hits=row["application_cache_hits"],
            application_cache_misses=row["application_cache_misses"],
            application_cache_saved_tokens=row["application_cache_saved_tokens"],
            application_cache_saved_cost_usd=row["application_cache_saved_cost_usd"],
        )

    def get_cached_response(self, cache_key: str) -> dict[str, Any] | None:
        now = time.time()
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM response_cache WHERE expires_at<=?", (now,))
            row = self._connection.execute(
                "SELECT response_json,usage_json FROM response_cache WHERE cache_key=?",
                (cache_key,),
            ).fetchone()
            if row is None:
                return None
            self._connection.execute(
                "UPDATE response_cache SET last_accessed_at=?,hit_count=hit_count+1 WHERE cache_key=?",
                (now, cache_key),
            )
        return {
            "response": json.loads(row["response_json"]),
            "usage": json.loads(row["usage_json"]),
        }

    def put_cached_response(
        self,
        cache_key: str,
        *,
        provider: str,
        model: str,
        response: dict[str, Any],
        usage: Usage,
        ttl_seconds: int,
        max_entries: int,
    ) -> None:
        if ttl_seconds <= 0:
            return
        now = time.time()
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM response_cache WHERE expires_at<=?", (now,))
            self._connection.execute(
                """INSERT INTO response_cache(
                    cache_key,provider,model,response_json,usage_json,created_at,
                    expires_at,last_accessed_at,hit_count
                ) VALUES(?,?,?,?,?,?,?,?,0)
                ON CONFLICT(cache_key) DO UPDATE SET
                    provider=excluded.provider,model=excluded.model,
                    response_json=excluded.response_json,usage_json=excluded.usage_json,
                    created_at=excluded.created_at,expires_at=excluded.expires_at,
                    last_accessed_at=excluded.last_accessed_at,hit_count=0""",
                (
                    cache_key,
                    provider,
                    model,
                    json_dumps(response),
                    json_dumps(usage.to_dict()),
                    utc_now(),
                    now + ttl_seconds,
                    now,
                ),
            )
            excess = self._connection.execute(
                "SELECT MAX(0, COUNT(*) - ?) FROM response_cache", (max_entries,)
            ).fetchone()[0]
            if excess:
                self._connection.execute(
                    """DELETE FROM response_cache WHERE cache_key IN (
                        SELECT cache_key FROM response_cache
                        ORDER BY last_accessed_at ASC LIMIT ?
                    )""",
                    (excess,),
                )

    def task_state(self, session_id: str, messages: list[Message] | None = None) -> dict[str, Any]:
        """References are durable; original instruction text remains in messages."""
        messages = self.messages(session_id) if messages is None else messages
        instructions = [m for m in messages if m.role.value == "user"
                        and not m.metadata.get("recovery_continuation")
                        and not m.metadata.get("internal")
                        and not m.metadata.get("authoritative_verification")]
        superseded: dict[str, str] = {}
        for index, message in enumerate(instructions):
            # Only an explicit user source reference changes durable status.
            # Free-form revocations remain ordered user instructions, never guesses.
            # Only a leading, unquoted command has destructive meaning. Examples
            # elsewhere in a message remain ordinary chronological instructions.
            for match in re.finditer(r"\A(?:revoke|supersede) user:(\d+)(?=[.:](?:\s|$)|\r?\n|[ \t]*$)", message.content.strip("\r\n"), re.IGNORECASE):
                target = int(match.group(1)) - 1
                if 0 <= target < index:
                    superseded[instructions[target].id] = message.id
        objective = instructions[0].id if instructions else None
        while objective in superseded:
            objective = superseded[objective]
        state = {"version": 1, "objective_source": objective,
                 "instruction_sources": [m.id for m in instructions], "superseded": superseded}
        if self.get_value(session_id, "task_state") != state:
            self.set_value(session_id, "task_state", state)
        return {**state, "instructions": [{"source": m.id, "text": m.content,
                                           "status": "superseded" if m.id in superseded else "active",
                                           "superseded_by": superseded.get(m.id)} for m in instructions],
                "plan": self.get_value(session_id, "plan", []),
                "acceptance_criteria": self.get_value(session_id, "acceptance_criteria", [])}

    def save_output_artifact(
        self, session_id: str, workspace: Path, content: str, *,
        source: str, redactor: Any, status: str = "complete",
        max_bytes: int = 2_000_000, session_max_bytes: int = 20_000_000,
        observed_bytes: int | None = None, collection_status: str | None = None,
    ) -> dict[str, Any]:
        """Store permitted evidence once. Hash and size describe retained UTF-8 data."""
        self.authorize_workspace(session_id, workspace)
        data = redactor.text(content).encode("utf-8")
        collection_status = collection_status or status
        with self._lock, self._connection:
            count = self._connection.execute("SELECT COUNT(*) FROM output_artifacts WHERE session_id=?", (session_id,)).fetchone()[0]
            if count >= 1000:
                return {"artifact_id": None, "status": "quota_exhausted", "stored_bytes": 0,
                        "observed_bytes": observed_bytes if observed_bytes is not None else len(data),
                        "collection_status": collection_status}
            used = self._connection.execute(
                "SELECT COALESCE(SUM(length(CAST(content AS BLOB))),0) FROM output_artifacts WHERE session_id=?",
                (session_id,),
            ).fetchone()[0]
            limit = max(0, min(max_bytes, session_max_bytes - used))
            retained = data[:limit].decode("utf-8", errors="ignore")
            if len(data) > limit:
                status = "quota_limited"
            metadata = {
                "artifact_id": new_id("out"), "source": source,
                "sha256": hashlib.sha256(retained.encode()).hexdigest(),
                "stored_bytes": len(retained.encode()), "observed_bytes": observed_bytes if observed_bytes is not None else len(data),
                "status": status,
                "collection_status": collection_status,
            }
            self._connection.execute(
                "INSERT INTO output_artifacts VALUES(?,?,?,?)",
                (session_id, metadata["artifact_id"], retained, json_dumps(metadata)),
            )
        return metadata

    def authorize_workspace(self, session_id: str, workspace: Path) -> None:
        if Path(self.get_session(session_id).workspace).resolve() != workspace.resolve():
            raise SessionError("Session does not belong to this workspace")

    def _output_artifact_owner_locked(self, scope_id: str, artifact_id: str) -> str:
        rows = self._connection.execute(
            "SELECT session_id FROM output_artifacts AS artifact WHERE artifact_id=? "
            "AND (session_id=? OR EXISTS (SELECT 1 FROM delegated_artifact_handoffs AS handoff "
            "WHERE handoff.recipient_scope_id=? AND handoff.owner_scope_id=artifact.session_id "
            "AND handoff.artifact_id=artifact.artifact_id))",
            (artifact_id, scope_id, scope_id),
        ).fetchall()
        if len(rows) != 1:
            raise SessionError("Artifact missing or unavailable in this session")
        return str(rows[0][0])

    def handoff_output_artifacts(
        self, sender_scope_id: str, recipient_scope_id: str, workspace: Path,
        delegation_call_id: str, evidence: list[dict[str, Any]],
    ) -> None:
        """Expose only receipt-backed artifacts returned to this child's immediate parent.

        Links retain original ownership, shape and storage quotas; they copy no output.
        Model text and arbitrary artifact IDs cannot create handoff authority.
        """
        self.authorize_workspace(sender_scope_id, workspace)
        self.authorize_workspace(recipient_scope_id, workspace)
        with self._lock, self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            child = self._connection.execute(
                "SELECT root_session_id,delegation_id FROM delegated_conversations WHERE scope_id=?",
                (sender_scope_id,),
            ).fetchone()
            if child is None:
                raise SessionError("Artifact handoff requires a delegated conversation")
            sender = self._connection.execute(
                "SELECT metadata_json FROM sessions WHERE id=?", (sender_scope_id,),
            ).fetchone()
            if json.loads(sender[0]).get("delegation_call_id") != delegation_call_id:
                raise SessionError("Artifact handoff does not match the delegation call identity")
            parent_call = self._connection.execute(
                "SELECT run_id FROM tool_calls WHERE session_id=? AND tool_call_id=?",
                (child[0], delegation_call_id),
            ).fetchone()
            if parent_call is None or sender_scope_id != str(parent_call[0]) + "." + str(child[1]):
                raise SessionError("Artifact handoff does not match the delegation receipt")
            parent = self._connection.execute(
                "SELECT root_session_id FROM delegated_conversations WHERE scope_id=?", (parent_call[0],),
            ).fetchone()
            expected_recipient = str(parent_call[0]) if parent is not None else str(child[0])
            if recipient_scope_id != expected_recipient or (parent is not None and parent[0] != child[0]):
                raise SessionError("Artifact handoff must target the immediate parent")
            for reference in evidence:
                artifact_id = (reference.get("output_artifact") or {}).get("artifact_id")
                if not artifact_id:
                    continue
                owner = self._output_artifact_owner_locked(sender_scope_id, artifact_id)
                source = self._connection.execute(
                    "SELECT metadata_json FROM tool_calls WHERE session_id=? AND run_id=? "
                    "AND tool_call_id=? AND status IN ('completed','error')",
                    (child[0], owner, reference.get("tool_call_id")),
                ).fetchone()
                artifact = self._connection.execute(
                    "SELECT metadata_json FROM output_artifacts WHERE session_id=? AND artifact_id=?",
                    (owner, artifact_id),
                ).fetchone()
                if (source is None or artifact is None
                        or json.loads(artifact[0]).get("source") != reference.get("tool_call_id")
                        or (json.loads(source[0]).get("output_artifact") or {}).get("artifact_id") != artifact_id):
                    raise SessionError("Artifact handoff source is not a completed evidence receipt")
                existing = self._connection.execute(
                    "SELECT owner_scope_id FROM delegated_artifact_handoffs "
                    "WHERE recipient_scope_id=? AND artifact_id=?", (recipient_scope_id, artifact_id),
                ).fetchone()
                if existing is not None and existing[0] != owner:
                    raise SessionError("Artifact handoff ownership cannot change")
                self._connection.execute(
                    "INSERT OR IGNORE INTO delegated_artifact_handoffs VALUES(?,?,?)",
                    (recipient_scope_id, owner, artifact_id),
                )

    def output_artifact_has_quote(
        self, session_id: str, workspace: Path, artifact_id: str, quote: str,
        *, json_envelope: bool = False,
    ) -> bool:
        """Match retained text without loading full evidence into model history."""
        self.authorize_workspace(session_id, workspace)
        if not quote or len(quote) > 600:
            raise SessionError("Invalid finding quote length")
        expression = "instr(content, ?) > 0"
        partial_content = "NULL"
        parameters: list[Any] = [quote]
        if json_envelope:
            # SQLite decodes JSON string values before matching. Searching encoded
            # fragments would confuse literal backslashes with JSON escapes.
            expression = (
                "CASE WHEN json_valid(content) THEN "
                "instr(json_extract(content, '$.output'), ?) > 0 OR EXISTS ("
                "SELECT 1 FROM json_tree(content, '$.data') "
                "WHERE type='text' AND instr(value, ?) > 0) ELSE 0 END"
            )
            parameters.append(quote)
            partial_content = "CASE WHEN NOT json_valid(content) THEN content END"
        with self._lock:
            owner = self._output_artifact_owner_locked(session_id, artifact_id)
            row = self._connection.execute(
                f"SELECT {expression} AS matched, {partial_content} AS partial_content, "
                "metadata_json FROM output_artifacts "
                "WHERE session_id=? AND artifact_id=?",
                [*parameters, owner, artifact_id],
            ).fetchone()
        if row is None:
            raise SessionError("Artifact missing or unavailable in this session")
        if row["partial_content"] is not None and json.loads(row["metadata_json"]).get("status") == "quota_limited":
            return _truncated_envelope_has_quote(row["partial_content"], quote)
        return bool(row["matched"])

    def read_output_artifact(
        self, session_id: str, workspace: Path, artifact_id: str, *,
        offset: int = 0, limit: int = 8000, query: str = "",
    ) -> dict[str, Any]:
        self.authorize_workspace(session_id, workspace)
        if offset < 0 or not 1 <= limit <= 16_000:
            raise SessionError("Invalid artifact range")
        with self._lock:
            owner = self._output_artifact_owner_locked(session_id, artifact_id)
            row = self._connection.execute(
                "SELECT content, metadata_json FROM output_artifacts WHERE session_id=? AND artifact_id=?",
                (owner, artifact_id),
            ).fetchone()
        if row is None:
            raise SessionError("Artifact missing or unavailable in this session")
        content = row["content"]
        if query:
            found = content.find(query, offset)
            if found < 0:
                return {**json.loads(row["metadata_json"]), "match": False, "next_offset": None}
            offset = found
        end = min(len(content), offset + limit)
        return {**json.loads(row["metadata_json"]), "offset": offset,
                "next_offset": end if end < len(content) else None,
                "content": content[offset:end]}

    def search_tool_history(
        self, session_id: str, workspace: Path, query: str, *, after: int = 0, offset: int = 0,
    ) -> list[dict[str, Any]]:
        self.authorize_workspace(session_id, workspace)
        if offset < 0:
            raise SessionError("History offset must be nonnegative")
        with self._lock:
            scope = self._connection.execute(
                "SELECT root_session_id FROM delegated_conversations WHERE scope_id=?", (session_id,),
            ).fetchone()
            owner = str(scope[0]) if scope is not None else session_id
            run_scope = session_id if scope is not None else None
            rows = self._connection.execute(
                "SELECT sequence,payload_json FROM events WHERE session_id=? AND sequence>? "
                "AND (? IS NULL OR run_id=?) "
                "AND type='tool.completed' AND (instr(json_extract(payload_json,'$.data.output'),?)>0 "
                "OR json_extract(payload_json,'$.data.tool_call_id')=?) ORDER BY sequence LIMIT 20",
                (owner, after, run_scope, run_scope, query, query),
            ).fetchall()
        results = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            data = payload.get("data", {})
            output = str(data.get("output", ""))
            start = min(offset, len(output)) if data.get("tool_call_id") == query else max(0, output.find(query))
            end = min(len(output), start + 1000)
            results.append({"sequence": row["sequence"], "source": data.get("tool_call_id"),
                            "preview": output[start:end], "offset": start,
                            "next_offset": end if end < len(output) else None,
                            "total_chars": len(output), "metadata": data.get("metadata", {})})
        return results

    def search_output_artifacts(
        self, session_id: str, workspace: Path, query: str, *, after: str = "",
    ) -> list[dict[str, Any]]:
        self.authorize_workspace(session_id, workspace)
        with self._lock:
            rows = self._connection.execute(
                "SELECT metadata_json, instr(content,?) - 1 AS offset, "
                "substr(content,max(1,instr(content,?)),500) AS preview FROM output_artifacts AS artifact "
                "WHERE (session_id=? OR EXISTS (SELECT 1 FROM delegated_artifact_handoffs AS handoff "
                "WHERE handoff.recipient_scope_id=? AND handoff.owner_scope_id=artifact.session_id "
                "AND handoff.artifact_id=artifact.artifact_id)) "
                "AND artifact_id>? AND (instr(content,?)>0 OR instr(metadata_json,?)>0) "
                "ORDER BY artifact_id LIMIT 20", (query, query, session_id, session_id, after, query, query),
            ).fetchall()
        return [{**json.loads(row["metadata_json"]), "offset": max(0, row["offset"]), "preview": row["preview"]} for row in rows]

    def search_history(
        self, session_id: str, workspace: Path, query: str, *, after: int = 0, limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        self.authorize_workspace(session_id, workspace)
        if not query or len(query) > 500 or offset < 0:
            raise SessionError("History query must contain 1-500 characters and offset must be nonnegative")

        def excerpt(content: str, *, exact: bool) -> dict[str, Any]:
            start = min(offset, len(content))
            if not exact:
                match = content.find(query, start)
                start = max(start, match - max(0, (500 - len(query)) // 2))
            end = min(len(content), start + (8000 if exact else 500))
            return {"preview": content[start:end], "offset": start,
                    "next_offset": end if end < len(content) else None,
                    "total_chars": len(content)}

        if query.startswith("user:") and query[5:].isdigit():
            instructions = self.task_state(session_id)["instructions"]
            index = int(query[5:]) - 1
            if 0 <= index < len(instructions):
                item = instructions[index]
                return [{"source": query, "message_id": item["source"],
                         **excerpt(item["text"], exact=True)}]
            return []
        with self._lock:
            rows = self._connection.execute(
                "SELECT sequence,message_id,payload_json FROM messages "
                "WHERE session_id=? AND sequence>? AND (instr(substr(json_extract(payload_json,'$.content'),?),?)>0 "
                "OR message_id=?) ORDER BY sequence LIMIT ?",
                (session_id, max(0, after), offset + 1, query, query, min(20, max(1, limit))),
            ).fetchall()
        return [{"sequence": row["sequence"], "message_id": row["message_id"],
                 **excerpt(json.loads(row["payload_json"]).get("content", ""), exact=row["message_id"] == query)}
                for row in rows]

    def set_value(self, session_id: str, key: str, value: Any) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO key_values(session_id,key,value_json) VALUES(?,?,?) ON CONFLICT(session_id,key) DO UPDATE SET value_json=excluded.value_json",
                (session_id, key, json_dumps(value)),
            )

    def get_value(self, session_id: str, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._connection.execute(
                "SELECT value_json FROM key_values WHERE session_id=? AND key=?", (session_id, key)
            ).fetchone()
        return default if row is None else json.loads(row[0])

    def export(self, session_id: str) -> dict[str, Any]:
        session = self.get_session(session_id)
        with self._lock:
            artifacts = self._connection.execute("SELECT content,metadata_json FROM output_artifacts WHERE session_id=? ORDER BY artifact_id", (session_id,)).fetchall()
        return {
            "task_state": self.get_value(session_id, "task_state"),
            "plan": self.get_value(session_id, "plan", []),
            "acceptance_criteria": self.get_value(session_id, "acceptance_criteria", []),
            "output_artifacts": [{**json.loads(row["metadata_json"]), "content": row["content"]} for row in artifacts],
            "session": {
                "id": session.id,
                "workspace": session.workspace,
                "title": session.title,
                "provider": session.provider,
                "model": session.model,
                "status": session.status,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "metadata": session.metadata,
            },
            "messages": [item.to_dict() for item in self.messages(session_id)],
            "compaction_artifacts": [
                artifact.to_dict() for artifact in self.compaction_artifacts(session_id)
            ],
            "tool_calls": self.tool_calls(session_id),
            "usage": self.usage(session_id).to_dict(),
            "events": [
                {"sequence": sequence, **event.to_dict()}
                for sequence, event in self._export_events(session_id)
            ],
        }

    def compaction_artifacts(self, session_id: str) -> list[CompactionArtifact]:
        self.get_session(session_id)
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM compaction_artifacts WHERE session_id=? ORDER BY sequence",
                (session_id,),
            ).fetchall()
            return [self._load_compaction_artifact(row) for row in rows]


def _session_info(row: sqlite3.Row) -> SessionInfo:
    return SessionInfo(
        id=row["id"],
        workspace=row["workspace"],
        title=row["title"],
        provider=row["provider"],
        model=row["model"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


def _compaction_artifact(
    row: sqlite3.Row,
    *,
    metadata: dict[str, Any] | None = None,
) -> CompactionArtifact:
    return CompactionArtifact(
        id=row["artifact_id"],
        session_id=row["session_id"],
        version=int(row["version"]),
        strategy=row["strategy"],
        source_start_sequence=row["source_start_sequence"],
        source_end_sequence=row["source_end_sequence"],
        source_message_ids=list(json.loads(row["source_message_ids_json"] or "[]")),
        source_hash=row["source_hash"],
        summary_text=row["summary_text"],
        provider=row["provider"],
        model=row["model"],
        config_fingerprint=row["config_fingerprint"],
        estimated_tokens_before=int(row["estimated_tokens_before"]),
        estimated_tokens_after=int(row["estimated_tokens_after"]),
        usage=Usage.from_dict(json.loads(row["usage_json"] or "{}")),
        created_at=row["created_at"],
        parent_artifact_id=row["parent_artifact_id"],
        metadata=(
            metadata
            if metadata is not None
            else json.loads(row["metadata_json"] or "{}")
        ),
    )
