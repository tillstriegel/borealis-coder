"""Stable asynchronous input for the interactive terminal."""

from __future__ import annotations

import contextlib
import datetime
import os
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.input import Input
from prompt_toolkit.output import Output
from prompt_toolkit.patch_stdout import StdoutProxy
from prompt_toolkit.shortcuts import CompleteStyle

_HISTORY_TIMESTAMP = re.compile(r"^# \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
_LIBEDIT_HEADER = "_HiStOrY_V2_"
_LIBEDIT_ESCAPE = re.compile(r"\\([0-7]{3})")
_MAX_HISTORY_ENTRIES = 1_000


class LineBufferedStdout(StdoutProxy):
    """Keep unfinished lines out of prompt-toolkit redraws."""

    def __init__(self) -> None:
        super().__init__(raw=True)
        self._line_open = False

    def write(self, data: str) -> int:
        if data:
            self._line_open = not data.endswith("\n")
        return super().write(data)

    def flush(self) -> None:
        # StdoutProxy already queues complete lines in write(). Flushing a
        # partial line lets the next prompt redraw overwrite response chunks.
        pass

    def close(self) -> None:
        if not self.closed and self._line_open:
            self.write("\n")
        super().close()


class TerminalInputInterrupted(Exception):
    """Ordinary exception used to report Ctrl+C to the shell coordinator."""


class CompatibleFileHistory(FileHistory):
    """Read legacy readline lines together with prompt-toolkit records."""

    def load_history_strings(self) -> Iterable[str]:
        path = Path(os.fsdecode(self.filename))
        return reversed(read_history_entries(path))

    def compact(self, max_entries: int = _MAX_HISTORY_ENTRIES) -> None:
        """Keep the most recent entries in the shared history file."""

        path = Path(os.fsdecode(self.filename))
        entries = read_history_entries(path)
        if len(entries) > max_entries:
            write_history_entries(path, entries[-max_entries:])


def read_history_entries(path: Path) -> list[str]:
    """Read prompt-toolkit, GNU readline, and macOS libedit history."""

    if not path.exists():
        return []
    raw_lines = path.read_bytes().decode("utf-8", errors="replace").splitlines()
    libedit = bool(raw_lines and raw_lines[0] == _LIBEDIT_HEADER)
    entries: list[str] = []
    index = 1 if libedit else 0
    while index < len(raw_lines):
        line = raw_lines[index]
        if (
            _HISTORY_TIMESTAMP.match(line)
            and index + 1 < len(raw_lines)
            and raw_lines[index + 1].startswith("+")
        ):
            index += 1
            record: list[str] = []
            while index < len(raw_lines) and raw_lines[index].startswith("+"):
                record.append(raw_lines[index][1:])
                index += 1
            entries.append("\n".join(record))
            continue
        if line:
            entries.append(_decode_libedit(line) if libedit else line)
        index += 1
    return entries


def write_history_entries(path: Path, entries: Iterable[str]) -> None:
    """Replace a history file atomically with prompt-toolkit records."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            for entry in entries:
                timestamp = datetime.datetime.now().isoformat(sep=" ")
                temporary.write(f"\n# {timestamp}\n".encode())
                for line in entry.split("\n"):
                    temporary.write(f"+{line}\n".encode("utf-8", errors="replace"))
        with contextlib.suppress(OSError):
            os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            with contextlib.suppress(OSError):
                temporary_path.unlink()


def _decode_libedit(value: str) -> str:
    return _LIBEDIT_ESCAPE.sub(lambda match: chr(int(match.group(1), 8)), value)


class TerminalInput:
    """Read editable terminal lines without blocking asynchronous output."""

    def __init__(
        self,
        history_file: Path,
        *,
        history_enabled: bool,
        completions: tuple[str, ...],
        completion_descriptions: dict[str, str],
        input: Input | None = None,
        output: Output | None = None,
    ) -> None:
        self._history_file = history_file
        if history_enabled:
            history_file.parent.mkdir(parents=True, exist_ok=True)
            history_file.touch(mode=0o600, exist_ok=True)
            with contextlib.suppress(OSError):
                os.chmod(history_file, 0o600)
        history = CompatibleFileHistory(history_file) if history_enabled else InMemoryHistory()
        completer = WordCompleter(
            completions,
            meta_dict=completion_descriptions,
            sentence=True,
        )
        self._session: PromptSession[str] = PromptSession(
            history=history,
            completer=completer,
            complete_while_typing=True,
            complete_style=CompleteStyle.MULTI_COLUMN,
            reserve_space_for_menu=8,
            input=input,
            output=output,
        )
        self._history_enabled = history_enabled
        self.reading = False

    @property
    def current_text(self) -> str:
        """Return the draft currently visible in the prompt buffer."""

        if not self.reading:
            return ""
        return self._session.default_buffer.text

    @property
    def current_document(self) -> Document:
        """Return the draft and cursor position visible in the prompt buffer."""

        if not self.reading:
            return Document()
        return self._session.default_buffer.document

    async def read(
        self,
        prompt: str,
        *,
        default: str | Document = "",
    ) -> str:
        """Read one line while preserving cursor and wrapped-line state."""

        self.reading = True
        try:
            try:
                return await self._session.prompt_async(
                    ANSI(prompt.replace("\001", "").replace("\002", "")),
                    default=default,
                )
            except KeyboardInterrupt:
                raise TerminalInputInterrupted from None
        finally:
            self.reading = False

    def close(self) -> None:
        """Restrict the persisted prompt history to the current user."""

        if self._history_enabled and self._history_file.exists():
            history = self._session.history
            if isinstance(history, CompatibleFileHistory):
                history.compact()
            with contextlib.suppress(OSError):
                os.chmod(self._history_file, 0o600)
