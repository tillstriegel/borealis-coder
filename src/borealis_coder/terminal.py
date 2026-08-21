"""Human-oriented terminal rendering and optional readline integration."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TextIO

from .models import Event
from .util import json_dumps, truncate_text


class ConsoleRenderer:
    """Render the authoritative EventBus stream without losing final model text.

    A single user request can contain several model exchanges separated by tool
    calls. Rendering state is therefore reset on every ``model.started`` event,
    not only once per user turn. This avoids suppressing a final answer merely
    because an earlier model exchange already streamed some text.
    """

    def __init__(
        self,
        *,
        json_events: bool = False,
        quiet: bool = False,
        stream_text: bool = True,
        interactive: bool = False,
        show_tool_output: bool = False,
        tool_output_chars: int = 2_000,
        text_stream: TextIO | None = None,
        status_stream: TextIO | None = None,
    ) -> None:
        self.json_events = json_events
        self.quiet = quiet
        self.stream_text = stream_text
        self.interactive = interactive
        self.show_tool_output = show_tool_output
        self.tool_output_chars = max(200, tool_output_chars)
        self._text_stream = text_stream
        self._status_stream = status_stream
        self.printed_text = False
        self._rendered_texts: list[str] = []
        self._streamed_text = ""
        self._line_open = False
        self._assistant_started = False

    @property
    def text_stream(self) -> TextIO:
        return self._text_stream or sys.stdout

    @property
    def status_stream(self) -> TextIO:
        return self._status_stream or sys.stderr

    def reset_turn(self) -> None:
        self.printed_text = False
        self._rendered_texts = []
        self._streamed_text = ""
        self._line_open = False
        self._assistant_started = False

    def finish_turn(self) -> None:
        self._ensure_line_break()

    def has_rendered(self, text: str) -> bool:
        return bool(text) and text in self._rendered_texts

    async def handle(self, event: Event) -> None:
        if self.json_events:
            print(json_dumps({"event": event.to_dict()}), file=self.status_stream)
            return
        if self.quiet:
            return

        if event.type == "model.started":
            self._begin_model_exchange()
            return
        if event.type == "model.text_delta":
            self._render_text_delta(str(event.data.get("text") or ""))
            return
        if event.type == "model.completed":
            self._render_completed_text(str(event.data.get("text") or ""))
            return
        if event.type == "tool.started":
            self._ensure_line_break()
            print(
                f"→ {_tool_label(event.data.get('tool'), event.data.get('arguments'))}",
                file=self.status_stream,
                flush=True,
            )
            return
        if event.type == "tool.completed":
            self._ensure_line_break()
            marker = "✗" if event.data.get("is_error") else "✓"
            duration = event.data.get("metadata", {}).get("duration_ms", 0)
            print(
                f"{marker} {event.data.get('tool')} ({duration} ms)",
                file=self.status_stream,
                flush=True,
            )
            output = str(event.data.get("output") or "")
            if output and (self.show_tool_output or bool(event.data.get("is_error"))):
                _print_indented(
                    truncate_text(output, self.tool_output_chars),
                    stream=self.status_stream,
                )
            return
        if event.type == "tool.cancelled":
            self._ensure_line_break()
            print(f"■ {event.data.get('tool')} cancelled", file=self.status_stream)
            return
        if event.type == "verification.completed":
            self._ensure_line_break()
            marker = "✓" if event.data.get("ok") else "✗"
            count = len(event.data.get("commands") or [])
            print(
                f"{marker} verification ({count} command{'s' if count != 1 else ''})",
                file=self.status_stream,
            )
            return
        if event.type == "context.compacted":
            self._ensure_line_break()
            print("↻ compacted conversation context", file=self.status_stream)
            return
        if event.type == "model.route_failed":
            self._ensure_line_break()
            print(
                f"↻ provider route failed: {event.data.get('provider')}; trying fallback",
                file=self.status_stream,
            )
            return
        if event.type == "run.error":
            self._ensure_line_break()
            print(f"✗ {event.data.get('error')}", file=self.status_stream)

    def _begin_model_exchange(self) -> None:
        self._ensure_line_break()
        self.printed_text = False
        self._streamed_text = ""
        self._assistant_started = False

    def _start_assistant_block(self) -> None:
        if self._assistant_started:
            return
        self._assistant_started = True
        if self.interactive:
            self._ensure_line_break()
            print("\nBorealis", file=self.text_stream, flush=True)

    def _render_text_delta(self, text: str) -> None:
        if not text:
            return
        self._streamed_text += text
        if not self.stream_text:
            return
        self._start_assistant_block()
        print(text, end="", file=self.text_stream, flush=True)
        self._line_open = not text.endswith(("\n", "\r"))
        self.printed_text = True

    def _render_completed_text(self, text: str) -> None:
        if text:
            self._start_assistant_block()
            if not self.stream_text or not self._streamed_text:
                print(text, end="", file=self.text_stream, flush=True)
                self._line_open = not text.endswith(("\n", "\r"))
            elif text.startswith(self._streamed_text):
                suffix = text[len(self._streamed_text) :]
                if suffix:
                    print(suffix, end="", file=self.text_stream, flush=True)
                    self._line_open = not suffix.endswith(("\n", "\r"))
            elif text != self._streamed_text:
                # The completed response is the source of truth. A provider can
                # omit/rewrite stream deltas, so show the complete result rather
                # than silently presenting an incomplete answer.
                self._ensure_line_break()
                print(text, end="", file=self.text_stream, flush=True)
                self._line_open = not text.endswith(("\n", "\r"))
            self.printed_text = True
            self._rendered_texts.append(text)
        self._ensure_line_break()

    def _ensure_line_break(self) -> None:
        if self._line_open:
            print(file=self.text_stream, flush=True)
            self._line_open = False


class ReadlineHistory:
    """Best-effort command history and slash-command completion.

    The standard-library ``readline`` module is optional on some platforms. The
    CLI remains fully functional when it is unavailable.
    """

    def __init__(
        self,
        path: Path,
        *,
        enabled: bool = True,
        completions: tuple[str, ...] = (),
        max_entries: int = 1_000,
    ) -> None:
        self.path = path
        self.enabled = enabled
        self.completions = completions
        self.max_entries = max(1, max_entries)
        self._readline = None
        self._previous_completer = None

    def __enter__(self) -> ReadlineHistory:
        if not self.enabled:
            return self
        try:
            import readline  # type: ignore[import-not-found]
        except ImportError:
            return self
        self._readline = readline
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            try:
                readline.read_history_file(str(self.path))
            except OSError:
                pass
        readline.set_history_length(self.max_entries)
        self._previous_completer = readline.get_completer()
        readline.set_completer(self._complete)
        try:
            readline.parse_and_bind("tab: complete")
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        readline = self._readline
        if readline is None:
            return
        try:
            readline.write_history_file(str(self.path))
            if os.name != "nt":
                os.chmod(self.path, 0o600)
        except OSError:
            pass
        finally:
            readline.set_completer(self._previous_completer)

    def _complete(self, text: str, state: int) -> str | None:
        candidates = [item for item in self.completions if item.startswith(text)]
        return candidates[state] if state < len(candidates) else None


def _tool_label(tool: object, arguments: object) -> str:
    name = str(tool or "tool")
    if not isinstance(arguments, dict):
        return name
    for key in ("path", "query", "pattern", "command", "url", "task"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            summary = value.strip().replace("\n", " ")
            return f"{name}  {truncate_text(summary, 100)}"
    return name


def _print_indented(value: str, *, stream: TextIO) -> None:
    for line in value.splitlines() or [value]:
        print(f"    {line}", file=stream)
