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
from prompt_toolkit.filters import Condition, is_done
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    FloatContainer,
    HSplit,
    VerticalAlign,
    Window,
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.output import ColorDepth, Output
from prompt_toolkit.patch_stdout import StdoutProxy
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.styles import Style

from .swarm_view import SwarmView
from .usage_view import UsageView

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
        swarm: SwarmView | None = None,
        usage: UsageView | None = None,
    ) -> None:
        self.swarm = swarm or SwarmView()
        self.usage = usage or UsageView()
        bindings = KeyBindings()

        @bindings.add("f6")
        def focus_swarm(event):
            self.swarm.visible = True
            self.swarm.focused = not self.swarm.focused if self._inspector_visible() else False

        @bindings.add("f7")
        def toggle_swarm(event):
            self.swarm.visible = not self.swarm.visible
            self.swarm.focused = False

        focused = Condition(lambda: self._inspector_visible() and self.swarm.focused)

        @bindings.add("up", filter=focused)
        @bindings.add("down", filter=focused)
        def select_agent(event):
            self.swarm.move(-1 if event.key_sequence[0].key == "up" else 1)

        @bindings.add("left", filter=focused)
        @bindings.add("right", filter=focused)
        def fold_agent(event):
            self.swarm.collapse(event.key_sequence[0].key == "left")

        @bindings.add("pageup", filter=focused)
        @bindings.add("pagedown", filter=focused)
        def scroll_detail(event):
            self.swarm.detail_offset = max(0, self.swarm.detail_offset +
                                          (-3 if event.key_sequence[0].key == "pageup" else 3))

        @bindings.add("enter", filter=focused)
        @bindings.add("escape", filter=focused)
        def return_to_input(event):
            self.swarm.focused = False

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
            key_bindings=bindings,
            style=Style.from_dict({
                "swarm.title": "ansicyan bold", "swarm.muted": "ansibrightblack",
                "swarm.text": "", "swarm.selected": "ansigreen bold",
            }) if "NO_COLOR" not in os.environ and os.getenv("TERM") != "dumb" else Style([]),
            color_depth=ColorDepth.DEPTH_1_BIT if "NO_COLOR" in os.environ else None,
            history=history,
            completer=completer,
            complete_while_typing=True,
            complete_style=CompleteStyle.MULTI_COLUMN,
            reserve_space_for_menu=0,
            input=input,
            output=output,
        )
        layout = self._session.app.layout
        # Render only the form, leaving recent stdout untouched above it.
        # Pin on entry/resize; normal-screen scrolling then keeps both stdout
        # and the compact form visible when StdoutProxy prints above the app.
        layout.current_window.dont_extend_height = Condition(lambda: True)
        layout.current_window.height = lambda: Dimension(
            min=1, max=max(1, self._session.app.output.get_size().rows - 3)
        )
        self._form = layout.container
        # PromptSession normally bounds completion floats to its input region.
        # Give completion floats temporary room above the compact form. Leave
        # the right prompt in the form and release menu space after stdout.
        input_container = next(container for container in layout.walk()
                               if isinstance(container, FloatContainer))
        completion_floats = [floating for floating in input_container.floats if floating.ycursor]
        input_container.floats = [floating for floating in input_container.floats if not floating.ycursor]
        layout.container = FloatContainer(HSplit([
            ConditionalContainer(
                Window(height=self._completion_height),
                filter=Condition(lambda: self._completion_height() > 0) & ~is_done,
            ),
            ConditionalContainer(
                Window(FormattedTextControl(self._swarm_toolbar), dont_extend_height=True,
                       wrap_lines=False),
                filter=Condition(self._inspector_visible) & ~is_done,
            ),
            ConditionalContainer(
                Window(FormattedTextControl(self._usage_toolbar), dont_extend_height=True,
                       wrap_lines=True), filter=~is_done,
            ),
            self._form,
        ], align=VerticalAlign.BOTTOM), completion_floats)
        self._last_size = None
        self._session.app.before_render += self._pin_on_resize
        self._history_enabled = history_enabled
        self.reading = False

    def _inspector_visible(self) -> bool:
        visible = self.swarm.visible and self._swarm_height() >= 3
        if not visible:
            self.swarm.focused = False
        return visible

    def _pin_on_resize(self, app) -> None:
        size = app.output.get_size()
        if size != self._last_size:
            app.renderer.erase(leave_alternate_screen=False)
            try:
                # Native consoles move within a scrollback buffer, not just the
                # viewport. Use their actual remaining visible rows.
                down = max(0, app.output.get_rows_below_cursor_position() - 1)
            except NotImplementedError:
                # VT cursor-down clamps at the last physical row.
                down = size.rows
            app.output.cursor_down(down)
            self._last_size = size
        elif not app.is_done:
            screen = app.renderer._last_screen
            if screen is None:
                return
            height = app.layout.container.preferred_height(size.columns, size.rows).preferred
            if screen.height > height:
                # Release the old form's padding before stdout can erase/reset
                # the renderer. Keep the new compact form at the same bottom.
                down = screen.height - height
                app.renderer.erase(leave_alternate_screen=False)
                app.output.cursor_down(down)

    def _completion_height(self) -> int:
        state = self._session.default_buffer.complete_state
        if state is None:
            return 0
        size = self._session.app.output.get_size()
        return max(0, min(8, len(state.completions) + 1, size.rows - 4))

    def _usage_toolbar(self) -> str:
        return self.usage.render(self._session.app.output.get_size().columns)

    def _swarm_height(self) -> int:
        size = self._session.app.output.get_size()
        form_height = self._form.preferred_height(
            size.columns, size.rows
        ).preferred
        hud_height = len(self.usage.render(size.columns).splitlines())
        return max(0, min(16, size.rows - form_height - hud_height - self._completion_height() - 1))

    def _swarm_toolbar(self):
        size = self._session.app.output.get_size()
        return self.swarm.render(size.columns, self._swarm_height())

    def invalidate(self) -> None:
        self._session.app.invalidate()

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
        self._last_size = None
        try:
            try:
                return await self._session.prompt_async(
                    ANSI(prompt.replace("\001", "").replace("\002", "")),
                    default=default,
                    pre_run=lambda: self._pin_on_resize(self._session.app),
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
