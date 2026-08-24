"""Event bus and append-only JSONL tracing."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import threading
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

    def __init__(self, path: Path, redactor: Redactor | None = None) -> None:
        self.path = path
        self.redactor = redactor or Redactor()
        self._lock = threading.Lock()
        ensure_private_directory(path.parent)
        ensure_private_file(path)

    def append(self, event: Event) -> None:
        self.append_many([event])

    def append_many(self, events: Iterable[Event]) -> None:
        text = "".join(json_dumps(self.redactor.value(event.to_dict())) + "\n" for event in events)
        if not text:
            return
        with self._lock:
            fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()

    def reset(self) -> None:
        atomic_write_text(self.path, "", mode=0o600)


class EventBus:
    def __init__(
        self,
        trace: JsonlTrace | None = None,
        redactor: Redactor | None = None,
        persist: EventBatchHandler | None = None,
    ) -> None:
        self._handlers: list[EventHandler] = []
        self.trace = trace
        self.redactor = redactor or Redactor()
        self._persist = persist
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
        if self._persist is not None:
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
