"""Stable asynchronous input for the interactive terminal."""

from __future__ import annotations

import contextlib
import os
import re
from collections.abc import Iterable
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.input import Input
from prompt_toolkit.output import Output

_HISTORY_TIMESTAMP = re.compile(r"^# \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")


class TerminalInputInterrupted(Exception):
    """Ordinary exception used to report Ctrl+C to the shell coordinator."""


class CompatibleFileHistory(FileHistory):
    """Read legacy readline lines together with prompt-toolkit records."""

    def load_history_strings(self) -> Iterable[str]:
        path = Path(os.fsdecode(self.filename))
        if not path.exists():
            return []
        raw_lines = path.read_bytes().decode("utf-8", errors="replace").splitlines()
        strings: list[str] = []
        index = 0
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
                strings.append("\n".join(record))
                continue
            if line:
                strings.append(line)
            index += 1
        return reversed(strings)


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
            complete_while_typing=False,
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
            with contextlib.suppress(OSError):
                os.chmod(self._history_file, 0o600)
