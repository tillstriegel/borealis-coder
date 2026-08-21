"""Event bus and append-only JSONL tracing."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .models import Event
from .safety.redaction import Redactor
from .util import atomic_write_text, ensure_private_directory, ensure_private_file, json_dumps

EventHandler = Callable[[Event], Awaitable[None] | None]


class JsonlTrace:
    """Append-only local trace with process-local serialization."""

    def __init__(self, path: Path, redactor: Redactor | None = None) -> None:
        self.path = path
        self.redactor = redactor or Redactor()
        self._lock = threading.Lock()
        ensure_private_directory(path.parent)
        ensure_private_file(path)

    def append(self, event: Event) -> None:
        line = json_dumps(self.redactor.value(event.to_dict())) + "\n"
        with self._lock:
            fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.flush()

    def reset(self) -> None:
        atomic_write_text(self.path, "", mode=0o600)


class EventBus:
    def __init__(self, trace: JsonlTrace | None = None, redactor: Redactor | None = None) -> None:
        self._handlers: list[EventHandler] = []
        self.trace = trace
        self.redactor = redactor or Redactor()

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
        if self.trace:
            await asyncio.to_thread(self.trace.append, event)
        for handler in list(self._handlers):
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                # Observers must not be able to crash the agent runtime.
                continue
        return event
