"""Event bus and append-only JSONL tracing."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import threading
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

from .errors import SessionError
from .models import Event
from .safety.redaction import Redactor
from .util import atomic_write_text, ensure_private_directory, ensure_private_file, json_dumps

EventHandler = Callable[[Event], Awaitable[None] | None]
EventBatchHandler = Callable[[list[Event]], None]
_BUFFERED_EVENT_TYPES = frozenset(
    {"model.reasoning_delta", "model.text_delta", "model.tool_call_delta"}
)
_EVENT_BATCH_SIZE = 64


class JsonlTrace:
    """Append-only local trace with process-local serialization."""

    def __init__(
        self,
        path: Path,
        redactor: Redactor | None = None,
        *,
        max_bytes: int = 10_000_000,
        backup_count: int = 3,
        create: bool = True,
    ) -> None:
        self.path = path
        self.redactor = redactor or Redactor()
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._lock = threading.Lock()
        if create:
            ensure_private_directory(path.parent)
            ensure_private_file(path)

    def append(self, event: Event) -> None:
        self.append_many([event])

    def append_many(self, events: Iterable[Event]) -> None:
        records = [
            json_dumps(self.redactor.value(event.to_dict())) + "\n" for event in events
        ]
        if not records:
            return
        with self._lock:
            for record in records:
                current_size = self.path.stat().st_size if self.path.exists() else 0
                if current_size and current_size + len(record.encode("utf-8")) > self.max_bytes:
                    self._rotate()
                fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
                with os.fdopen(fd, "a", encoding="utf-8", newline="\n") as handle:
                    handle.write(record)
                    handle.flush()

    def reset(self) -> None:
        atomic_write_text(self.path, "", mode=0o600)

    def maintenance(self, *, dry_run: bool = False) -> dict[str, int]:
        """Repack existing trace files into bounded, valid JSONL segments."""

        with self._lock:
            existing = self._existing_trace_paths()
            before_bytes = sum(path.stat().st_size for path in existing)
            capacity = self.max_bytes * (self.backup_count + 1)
            kept: deque[tuple[str, int]] = deque()
            valid_records = 0
            invalid_records = 0
            kept_bytes = 0
            for path in existing:
                with path.open(encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        line = line.rstrip("\r\n")
                        try:
                            json.loads(line)
                        except json.JSONDecodeError:
                            invalid_records += 1
                            continue
                        record = line + "\n"
                        size = len(record.encode("utf-8"))
                        valid_records += 1
                        kept.append((record, size))
                        kept_bytes += size
                        while len(kept) > 1 and kept_bytes > capacity:
                            _, removed_size = kept.popleft()
                            kept_bytes -= removed_size
            report = {
                "records_before": valid_records + invalid_records,
                "records_after": len(kept),
                "invalid_records_removed": invalid_records,
                "bytes_before": before_bytes,
                "bytes_after": kept_bytes,
            }
            if dry_run:
                return report
            chunks: list[list[str]] = []
            current: list[str] = []
            current_bytes = 0
            for record, size in reversed(kept):
                if current and current_bytes + size > self.max_bytes:
                    chunks.append(list(reversed(current)))
                    current = []
                    current_bytes = 0
                current.append(record)
                current_bytes += size
            if current:
                chunks.append(list(reversed(current)))
            output_paths = [self.path] + [
                self._backup_path(index) for index in range(1, self.backup_count + 1)
            ]
            for index, output in enumerate(output_paths):
                if index < len(chunks):
                    atomic_write_text(output, "".join(chunks[index]), mode=0o600)
                elif output.exists():
                    output.unlink()
            for obsolete in set(existing) - set(output_paths):
                if obsolete.is_file() and not obsolete.is_symlink():
                    obsolete.unlink()
            return report

    def _rotate(self) -> None:
        if self.backup_count == 0:
            self.reset()
            return
        oldest = self._backup_path(self.backup_count)
        if oldest.exists():
            oldest.unlink()
        for index in range(self.backup_count - 1, 0, -1):
            source = self._backup_path(index)
            if source.exists():
                os.replace(source, self._backup_path(index + 1))
        if self.path.exists():
            os.replace(self.path, self._backup_path(1))
        ensure_private_file(self.path)

    def _backup_path(self, index: int) -> Path:
        return Path(f"{self.path}.{index}")

    def _existing_trace_paths(self) -> list[Path]:
        numbered: list[tuple[int, Path]] = []
        prefix = f"{self.path.name}."
        if self.path.parent.is_dir():
            for candidate in self.path.parent.glob(f"{self.path.name}.*"):
                suffix = candidate.name.removeprefix(prefix)
                if suffix.isdigit() and int(suffix) > 0 and not candidate.is_symlink():
                    numbered.append((int(suffix), candidate))
        paths = [path for _, path in sorted(numbered, reverse=True)]
        if self.path.is_file() and not self.path.is_symlink():
            paths.append(self.path)
        return paths


class EventBus:
    def __init__(
        self,
        trace: JsonlTrace | None = None,
        redactor: Redactor | None = None,
        persist: EventBatchHandler | None = None,
        persist_deltas: bool = False,
    ) -> None:
        self._handlers: list[EventHandler] = []
        self.trace = trace
        self.redactor = redactor or Redactor()
        self._persist = persist
        self._persist_deltas = persist_deltas
        self._persist_pending: list[Event] = []
        self._trace_pending: list[Event] = []
        self._flush_lock = asyncio.Lock()

    def subscribe(self, handler: EventHandler) -> Callable[[], None]:
        self._handlers.append(handler)

        def unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._handlers.remove(handler)

        return unsubscribe

    async def emit(
        self,
        event_type: str,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        **data: Any,
    ) -> Event:
        event = Event(
            type=event_type,
            session_id=session_id,
            run_id=run_id,
            data=self.redactor.value(data),
        )
        if self._persist is not None and (
            self._persist_deltas or event_type not in _BUFFERED_EVENT_TYPES
        ):
            self._persist_pending.append(event)
        if self.trace is not None:
            self._trace_pending.append(event)
        if self.trace is not None or self._persist is not None:
            pending = max(len(self._persist_pending), len(self._trace_pending))
            if event_type not in _BUFFERED_EVENT_TYPES or pending >= _EVENT_BATCH_SIZE:
                await self.flush()
        for handler in list(self._handlers):
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                # Observers must not be able to crash the agent runtime.
                continue
        return event

    async def flush(self) -> None:
        async with self._flush_lock:
            if self._persist is not None and self._persist_pending:
                events = list(self._persist_pending)
                try:
                    await asyncio.to_thread(self._persist, events)
                except SessionError:
                    raise
                except Exception as error:
                    raise SessionError(f"Failed to persist runtime events: {error}") from error
                del self._persist_pending[: len(events)]
            if self.trace is not None and self._trace_pending:
                events = list(self._trace_pending)
                await asyncio.to_thread(self.trace.append_many, events)
                del self._trace_pending[: len(events)]
