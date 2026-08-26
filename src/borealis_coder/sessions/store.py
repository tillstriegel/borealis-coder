"""Durable SQLite session/event/message store with WAL and explicit migrations."""

from __future__ import annotations

import json
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

_SCHEMA_VERSION = 4
_EVENT_EXPORT_PAGE_SIZE = 1_000


class SessionStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        ensure_private_directory(self.path.parent)
        self._lock = threading.RLock()
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
            CREATE TABLE IF NOT EXISTS key_values (
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                PRIMARY KEY(session_id, key)
            );
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

    def get_session(self, session_id: str) -> SessionInfo:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
        if row is None:
            raise SessionError(f"Unknown session: {session_id}")
        return _session_info(row)

    def list_sessions(
        self, *, workspace: Path | None = None, limit: int = 100
    ) -> list[SessionInfo]:
        query = "SELECT * FROM sessions"
        params: list[Any] = []
        if workspace is not None:
            query += " WHERE workspace=?"
            params.append(str(workspace.resolve()))
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, limit))
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return [_session_info(row) for row in rows]

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
            cursor = self._connection.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            if cursor.rowcount == 0:
                raise SessionError(f"Unknown session: {session_id}")

    def append_message(self, session_id: str, message: Message) -> None:
        with self._lock, self._connection:
            self._append_message_locked(session_id, message)

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
                    json_dumps(artifact.metadata),
                ),
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
        return _compaction_artifact(row) if row is not None else None

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
        return _compaction_artifact(row) if row is not None else None

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
        self, session_id: str, run_id: str, call_id: str, name: str, arguments: dict[str, Any]
    ) -> None:
        if not call_id.strip():
            raise SessionError("Tool call ID must be non-empty")
        arguments_json = json_dumps(arguments)
        normalized_arguments = json.loads(arguments_json)
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """INSERT INTO tool_calls(
                    session_id,tool_call_id,run_id,tool_name,arguments_json,status,started_at
                ) VALUES(?,?,?,?,?,'running',?)
                ON CONFLICT(session_id,tool_call_id) DO NOTHING""",
                (session_id, call_id, run_id, name, arguments_json, utc_now()),
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
        metadata_json = json_dumps({"cancelled": True})
        with self._lock, self._connection:
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

    def add_usage(self, session_id: str, usage: Usage) -> Usage:
        with self._lock, self._connection:
            self._add_usage_locked(session_id, usage)
        return self.usage(session_id)

    def _add_usage_locked(self, session_id: str, usage: Usage) -> None:
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
        return {
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
        return [_compaction_artifact(row) for row in rows]


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


def _compaction_artifact(row: sqlite3.Row) -> CompactionArtifact:
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
        metadata=json.loads(row["metadata_json"] or "{}"),
    )
