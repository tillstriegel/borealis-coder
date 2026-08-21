"""Human-oriented terminal rendering and optional readline integration."""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from .models import Event
from .util import json_dumps, truncate_text

_PROMPT_GLYPH = "❯"  # noqa: RUF001 - intentional terminal prompt glyph
_ERROR_GLYPH = "×"  # noqa: RUF001 - intentional terminal status glyph


class AuroraUI:
    """Small, dependency-free visual system for the interactive terminal.

    The interface keeps scrollback intact instead of entering an alternate
    screen. ANSI color is used only for capable terminals; structure and labels
    carry the same meaning when output is redirected or ``NO_COLOR`` is set.
    """

    _RESET = "0"
    _BOLD = "1"
    _MINT = "38;2;112;255;185"
    _CYAN = "38;2;79;214;255"
    _VIOLET = "38;2;183;139;255"
    _AMBER = "38;2;255;198;92"
    _RED = "38;2;255;112;128"
    # Inherit the terminal theme for readable text on both light and dark
    # backgrounds. Standard bright black provides a theme-aware secondary tone.
    _FOREGROUND = "39"
    _MUTED = "90"

    def __init__(self, stream: TextIO, *, color: bool | None = None) -> None:
        self.stream = stream
        self.is_tty = _stream_is_tty(stream)
        if color is None:
            forced = os.environ.get("CLICOLOR_FORCE", "") not in {"", "0"}
            disabled = "NO_COLOR" in os.environ or os.environ.get("TERM") == "dumb"
            color = (self.is_tty or forced) and not disabled
        self.color = color

    @property
    def live(self) -> bool:
        """Whether a carriage-return activity pulse is safe to render."""

        return self.is_tty and self.color

    @property
    def width(self) -> int:
        columns = shutil.get_terminal_size((88, 24)).columns if self.is_tty else 88
        return max(40, min(columns, 100))

    def paint(self, value: object, *codes: str) -> str:
        text = str(value)
        if not self.color or not codes:
            return text
        return f"\033[{';'.join(codes)}m{text}\033[{self._RESET}m"

    def accent(self, value: object) -> str:
        return self.paint(value, self._MINT, self._BOLD)

    def subdued(self, value: object) -> str:
        return self.paint(value, self._MUTED)

    def prompt(self, session: str) -> str:
        """Return a two-line readline-safe prompt."""

        rail = self._readline_paint("╭─", self._MUTED)
        you = self._readline_paint("YOU", self._CYAN, self._BOLD)
        session_label = self._readline_paint(f"session {session}", self._MUTED)
        corner = self._readline_paint("╰─", self._MUTED)
        arrow = self._readline_paint(_PROMPT_GLYPH, self._MINT, self._BOLD)
        return f"\n{rail} {you}  {session_label}\n{corner}{arrow} "

    def inline_prompt(self, label: str) -> str:
        rail = self._readline_paint("│", self._MUTED)
        arrow = self._readline_paint(_PROMPT_GLYPH, self._MINT, self._BOLD)
        hint = self._readline_paint(label, self._MUTED)
        return f"{rail} {hint} {arrow} "

    def banner(
        self,
        *,
        version: str,
        workspace: Path,
        route: str,
        safety: str,
        session: str,
    ) -> None:
        width = self.width
        top_label = f" AURORA SHELL  {version} "
        top = "╭" + top_label + "─" * max(0, width - len(top_label) - 2) + "╮"
        bottom = "╰" + "─" * (width - 2) + "╯"
        title = self._gradient("BOREALIS") + "  " + self.paint(
            "CODER", self._FOREGROUND, self._BOLD
        )
        print(self.paint(top, self._MUTED), file=self.stream)
        print(self._box_line(f"  ◢◤  {title}", width), file=self.stream)
        subtitle = _middle_truncate(
            "      policy-first autonomous coding · interactive mode",
            width - 2,
        )
        print(self._box_line(self.subdued(subtitle), width), file=self.stream)
        print(self._box_divider(width), file=self.stream)
        self._box_field("WORKSPACE", str(workspace), width)
        self._box_field("ROUTE", route, width)
        self._box_field("GUARDRAIL", safety, width)
        self._box_field("SESSION", session, width)
        print(self.paint(bottom, self._MUTED), file=self.stream)
        hint = " /help commands   // literal slash   Ctrl+C cancel turn"
        print(self.subdued(_middle_truncate(hint, width)), file=self.stream)

    def panel(
        self,
        title: str,
        rows: Sequence[tuple[str, object]],
        *,
        tone: str = "mint",
    ) -> None:
        width = self.width
        color = self._tone(tone)
        heading = f" {title.upper()} "
        print(file=self.stream)
        print(
            self.paint(
                "╭─" + heading + "─" * max(0, width - len(heading) - 3) + "╮",
                color,
            ),
            file=self.stream,
        )
        for label, value in rows:
            label_text = self.paint(f"{label.upper():<11}", self._MUTED, self._BOLD)
            available = max(10, width - 18)
            rendered = _middle_truncate(str(value), available)
            print(self._framed_line(f"  {label_text} {rendered}", width, color), file=self.stream)
        print(self.paint("╰" + "─" * (width - 2) + "╯", color), file=self.stream)

    def notice(self, kind: str, message: str, detail: str = "") -> None:
        markers = {
            "success": ("◆", self._MINT),
            "error": (_ERROR_GLYPH, self._RED),
            "warning": ("!", self._AMBER),
            "info": ("◇", self._CYAN),
        }
        marker, color = markers.get(kind, markers["info"])
        suffix = f"  {self.subdued(detail)}" if detail else ""
        print(f"{self.paint(marker, color, self._BOLD)}  {message}{suffix}", file=self.stream, flush=True)

    def activity(self, marker: str, label: str, detail: str = "", *, tone: str = "muted") -> None:
        color = self._tone(tone)
        suffix = f"  {self.subdued(detail)}" if detail else ""
        print(
            f"{self.paint('│', self._MUTED)} {self.paint(marker, color, self._BOLD)} "
            f"{self.paint(label, self._FOREGROUND)}{suffix}",
            file=self.stream,
            flush=True,
        )

    def live_pulse(self, phase: str, elapsed_seconds: float, frame: int) -> str:
        glyph = ("◐", "◓", "◑", "◒")[frame % 4]
        return (
            f"{self.paint('│', self._MUTED)} "
            f"{self.paint(glyph, self._CYAN, self._BOLD)} "
            f"{self.paint(phase, self._FOREGROUND)}  "
            f"{self.subdued(f'{elapsed_seconds:.1f}s')}"
        )

    def assistant_header(self) -> None:
        label = self.paint("✦ BOREALIS", self._VIOLET, self._BOLD)
        print(f"\n{self.paint('╭─', self._VIOLET)} {label}", file=self.stream, flush=True)

    def assistant_footer(self) -> None:
        print(self.paint("╰─", self._VIOLET), file=self.stream, flush=True)

    def turn_footer(self, summary: str) -> None:
        print(
            f"{self.paint('◆', self._MINT, self._BOLD)} "
            f"{self.paint('TURN COMPLETE', self._MINT, self._BOLD)}  {self.subdued(summary)}",
            file=self.stream,
        )

    def role_header(self, label: str) -> None:
        color = self._CYAN if label == "user" else self._VIOLET
        print(f"\n{self.paint('◆', color)} {self.paint(label.upper(), color, self._BOLD)}", file=self.stream)

    def help(self, groups: Sequence[tuple[str, Sequence[tuple[str, str]]]]) -> None:
        width = self.width
        print(file=self.stream)
        print(
            self.paint(
                "╭─ COMMAND CONSTELLATION " + "─" * max(0, width - 25) + "╮",
                self._CYAN,
            ),
            file=self.stream,
        )
        for group, commands in groups:
            group_label = self.paint(group.upper(), self._MINT, self._BOLD)
            print(self._framed_line(f"  {group_label}", width, self._CYAN), file=self.stream)
            for command, description in commands:
                description = _middle_truncate(description, max(8, width - 32))
                content = (
                    f"    {self.paint(f'{command:<24}', self._FOREGROUND)}"
                    f"{self.subdued(description)}"
                )
                print(
                    self._framed_line(content, width, self._CYAN),
                    file=self.stream,
                )
        print(self.paint("╰" + "─" * (width - 2) + "╯", self._CYAN), file=self.stream)

    def command_selector(
        self,
        commands: Sequence[tuple[str, str]],
        *,
        query: str,
        hidden: int = 0,
    ) -> None:
        """Render a compact readline completion menu without taking over the screen."""

        width = self.width
        heading = " COMMAND DECK "
        print(file=self.stream)
        print(
            self.paint(
                "╭─" + heading + "─" * max(0, width - len(heading) - 3) + "╮",
                self._FOREGROUND,
                self._BOLD,
            ),
            file=self.stream,
        )
        for command, description in commands:
            available = max(8, width - 27)
            content = (
                f"  {self.paint(f'{command:<16}', self._FOREGROUND, self._BOLD)}"
                f"{self.subdued(_middle_truncate(description, available))}"
            )
            print(
                self._framed_line(content, width, self._FOREGROUND),
                file=self.stream,
            )
        if hidden:
            detail = f"{hidden} more · keep typing to narrow"
            print(
                self._framed_line(
                    f"  {self.subdued(detail)}", width, self._FOREGROUND
                ),
                file=self.stream,
            )
        hint = f"  {query or '/'}  · type to narrow · Tab completes · Enter runs"
        print(
            self._framed_line(
                self.paint(hint, self._FOREGROUND),
                width,
                self._FOREGROUND,
            ),
            file=self.stream,
        )
        print(
            self.paint("╰" + "─" * (width - 2) + "╯", self._FOREGROUND),
            file=self.stream,
        )

    def _gradient(self, value: str) -> str:
        colors = (self._MINT, self._CYAN, self._VIOLET)
        return "".join(
            self.paint(
                char,
                colors[index * len(colors) // max(1, len(value))],
                self._BOLD,
            )
            for index, char in enumerate(value)
        )

    def _tone(self, tone: str) -> str:
        return {
            "mint": self._MINT,
            "cyan": self._CYAN,
            "violet": self._VIOLET,
            "warning": self._AMBER,
            "error": self._RED,
            "muted": self._MUTED,
        }.get(tone, self._MUTED)

    def _box_line(self, content: str, width: int) -> str:
        return self._framed_line(content, width, self._MUTED)

    def _framed_line(self, content: str, width: int, color: str) -> str:
        visible = _strip_ansi(content)
        padding = max(0, width - len(visible) - 2)
        return (
            f"{self.paint('│', color)}{content}{' ' * padding}"
            f"{self.paint('│', color)}"
        )

    def _box_divider(self, width: int) -> str:
        return self.paint("├" + "─" * (width - 2) + "┤", self._MUTED)

    def _box_field(self, label: str, value: str, width: int) -> None:
        available = max(10, width - 19)
        content = (
            f"  {self.paint(f'{label:<11}', self._MUTED, self._BOLD)} "
            f"{self.paint(_middle_truncate(value, available), self._FOREGROUND)}"
        )
        print(self._box_line(content, width), file=self.stream)

    def _readline_paint(self, value: str, *codes: str) -> str:
        if not self.color or not codes:
            return value
        # Readline ignores bytes between \001 and \002 when calculating cursor
        # position. Without these guards, long prompts wrap incorrectly.
        return f"\001\033[{';'.join(codes)}m\002{value}\001\033[{self._RESET}m\002"


def _stream_is_tty(stream: object) -> bool:
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except OSError:
        return False


def _strip_ansi(value: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", value)


def _middle_truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars < 8:
        return value[:max_chars]
    side = (max_chars - 1) // 2
    return f"{value[:side]}…{value[-(max_chars - side - 1):]}"


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
        self._ui = AuroraUI(self.status_stream)
        self.printed_text = False
        self._rendered_texts: list[str] = []
        self._streamed_text = ""
        self._line_open = False
        self._assistant_started = False
        self._tool_call_announced = False
        self._activity_generation = 0
        self._activity_phase = ""
        self._assistant_block_open = False
        self._live_line_open = False
        self._streamed_tool_output_chars: dict[str, int] = {}
        self._truncated_tool_outputs: set[str] = set()
        self._tool_output_line_open = False

    @property
    def text_stream(self) -> TextIO:
        return self._text_stream or sys.stdout

    @property
    def status_stream(self) -> TextIO:
        return self._status_stream or sys.stderr

    @property
    def ui(self) -> AuroraUI:
        if self._ui.stream is not self.status_stream:
            self._ui = AuroraUI(self.status_stream)
        return self._ui

    @property
    def activity_generation(self) -> int:
        return self._activity_generation

    def reset_turn(self) -> None:
        self.printed_text = False
        self._rendered_texts = []
        self._streamed_text = ""
        self._line_open = False
        self._assistant_started = False
        self._tool_call_announced = False
        self._activity_phase = ""
        self._assistant_block_open = False
        self._streamed_tool_output_chars = {}
        self._truncated_tool_outputs = set()
        self._tool_output_line_open = False
        self._clear_live_line()

    @property
    def live_activity(self) -> bool:
        return self.interactive and self.ui.live

    def pulse(self, elapsed_seconds: float, frame: int) -> None:
        """Animate the current phase on capable interactive terminals."""

        if not self.live_activity or not self._activity_phase:
            return
        self._clear_live_line()
        value = self.ui.live_pulse(self._activity_phase, elapsed_seconds, frame)
        print(value, end="", file=self.status_stream, flush=True)
        self._live_line_open = True

    def heartbeat(self, elapsed_seconds: int) -> None:
        if not self._activity_phase:
            return
        self._ensure_line_break()
        if self.interactive:
            self.ui.activity(
                "◌",
                "Still working",
                f"{self._activity_phase} · {elapsed_seconds}s elapsed",
                tone="cyan",
            )
        else:
            print(
                f"… still working · {self._activity_phase} · {elapsed_seconds}s elapsed",
                file=self.status_stream,
                flush=True,
            )

    def finish_turn(self) -> None:
        self._clear_live_line()
        self._ensure_tool_output_line_break()
        self._ensure_line_break()
        self._close_assistant_block()

    def has_rendered(self, text: str) -> bool:
        return bool(text) and text in self._rendered_texts

    async def handle(self, event: Event) -> None:
        if self.json_events:
            print(json_dumps({"event": event.to_dict()}), file=self.status_stream)
            return
        if self.quiet:
            return

        self._activity_generation += 1
        self._clear_live_line()

        if event.type == "run.started":
            self._activity_phase = "preparing workspace context"
            self._ensure_line_break()
            if self.interactive:
                self.ui.activity("◌", "Context", "preparing workspace context", tone="cyan")
            else:
                print("… preparing workspace context", file=self.status_stream, flush=True)
            return
        if event.type == "model.started":
            self._begin_model_exchange()
            provider = str(event.data.get("provider") or "provider")
            model = str(event.data.get("model") or "model")
            turn = event.data.get("turn")
            turn_label = f" · turn {turn}" if turn is not None else ""
            self._activity_phase = f"waiting for {provider}/{model}"
            if self.interactive:
                self.ui.activity(
                    "◌",
                    "Model",
                    f"model working · {provider}/{model}{turn_label}",
                    tone="violet",
                )
            else:
                print(
                    f"… model working · {provider}/{model}{turn_label}",
                    file=self.status_stream,
                    flush=True,
                )
            return
        if event.type == "model.text_delta":
            self._activity_phase = ""
            self._render_text_delta(str(event.data.get("text") or ""))
            return
        if event.type == "model.tool_call_delta":
            self._activity_phase = "preparing tool call"
            if not self._tool_call_announced:
                self._ensure_line_break()
                name = str(event.data.get("name") or "").strip()
                suffix = f" · {name}" if name else ""
                if self.interactive:
                    self.ui.activity(
                        "◇",
                        "Tool request",
                        f"preparing tool call{suffix}",
                        tone="cyan",
                    )
                else:
                    print(
                        f"… preparing tool call{suffix}",
                        file=self.status_stream,
                        flush=True,
                    )
                self._tool_call_announced = True
            return
        if event.type == "model.retrying":
            self._ensure_line_break()
            attempt = event.data.get("attempt")
            max_attempts = event.data.get("max_attempts")
            delay = float(event.data.get("delay_seconds") or 0)
            self._activity_phase = f"waiting to retry {event.data.get('provider') or 'provider'}"
            detail = f"provider retry {attempt}/{max_attempts} in {delay:g}s"
            if self.interactive:
                self.ui.activity("↻", "Retry", detail, tone="warning")
            else:
                print(f"↻ {detail}", file=self.status_stream, flush=True)
            return
        if event.type == "model.cache_hit":
            self._ensure_line_break()
            saved_tokens = int(event.data.get("saved_tokens") or 0)
            detail = f"exact response reused · {saved_tokens} tokens avoided"
            if self.interactive:
                self.ui.activity("◆", "Response cache", detail, tone="mint")
            else:
                print(f"◆ response cache hit · {saved_tokens} tokens avoided", file=self.status_stream, flush=True)
            return
        if event.type == "model.cache_miss":
            self._activity_phase = "response cache miss · contacting model"
            return
        if event.type == "cache.adaptive":
            self._ensure_line_break()
            detail = (
                f"low hit rate ({float(event.data.get('hit_rate') or 0):.0%}) · "
                f"{event.data.get('action') or 'stable-prefix mode'}"
            )
            if self.interactive:
                self.ui.activity("◇", "Cache tuning", detail, tone="warning")
            else:
                print(f"◇ cache tuning · {detail}", file=self.status_stream, flush=True)
            return
        if event.type == "model.completed":
            self._activity_phase = "processing model response"
            self._render_completed_text(str(event.data.get("text") or ""))
            usage = event.data.get("usage") or {}
            cached = int(usage.get("cached_input_tokens") or 0)
            written = int(usage.get("cache_write_tokens") or 0)
            total_input = int(usage.get("input_tokens") or 0)
            if cached or written:
                self._ensure_line_break()
                rate = cached / total_input if total_input else 0.0
                detail = f"{rate:.0%} hit · {cached} read · {written} written"
                if self.interactive:
                    self.ui.activity("◆", "Prompt cache", detail, tone="mint")
                else:
                    print(f"◆ prompt cache · {detail}", file=self.status_stream, flush=True)
            return
        if event.type == "tool.started":
            self._activity_phase = f"running {event.data.get('tool') or 'tool'}"
            self._ensure_line_break()
            self._close_assistant_block()
            label = _tool_label(event.data.get("tool"), event.data.get("arguments"))
            if self.interactive:
                self.ui.activity("◇", "Tool", label, tone="cyan")
            else:
                print(f"→ {label}", file=self.status_stream, flush=True)
            return
        if event.type == "tool.policy":
            decision = str(event.data.get("decision") or "checking")
            risk = str(event.data.get("risk") or "unknown")
            self._activity_phase = f"policy {decision} · {risk} risk"
            return
        if event.type == "tool.output":
            if not self.show_tool_output:
                return
            self._activity_phase = ""
            text = str(event.data.get("text") or "")
            if not text:
                return
            call_id = str(event.data.get("tool_call_id") or event.data.get("tool") or "tool")
            used = self._streamed_tool_output_chars.get(call_id, 0)
            remaining = max(0, self.tool_output_chars - used)
            chunk = text[:remaining]
            if chunk:
                print(chunk, end="", file=self.status_stream, flush=True)
                self._streamed_tool_output_chars[call_id] = used + len(chunk)
                self._tool_output_line_open = not chunk.endswith(("\n", "\r"))
            if len(text) > remaining and call_id not in self._truncated_tool_outputs:
                self._ensure_tool_output_line_break()
                print("… output truncated …", file=self.status_stream, flush=True)
                self._truncated_tool_outputs.add(call_id)
            return
        if event.type == "tool.completed":
            self._activity_phase = "processing tool result"
            self._ensure_tool_output_line_break()
            self._ensure_line_break()
            marker = "✗" if event.data.get("is_error") else "✓"
            duration = event.data.get("metadata", {}).get("duration_ms", 0)
            if self.interactive:
                tone = "error" if event.data.get("is_error") else "mint"
                self.ui.activity(
                    _ERROR_GLYPH if event.data.get("is_error") else "◆",
                    "Tool failed" if event.data.get("is_error") else "Tool complete",
                    f"{event.data.get('tool')} · {duration} ms",
                    tone=tone,
                )
            else:
                print(
                    f"{marker} {event.data.get('tool')} ({duration} ms)",
                    file=self.status_stream,
                    flush=True,
                )
            output = str(event.data.get("output") or "")
            call_id = str(event.data.get("tool_call_id") or event.data.get("tool") or "tool")
            already_streamed = call_id in self._streamed_tool_output_chars
            if output and (
                (self.show_tool_output and not already_streamed)
                or (not self.show_tool_output and bool(event.data.get("is_error")))
            ):
                _print_indented(
                    truncate_text(output, self.tool_output_chars),
                    stream=self.status_stream,
                )
            return
        if event.type == "tool.cancelled":
            self._ensure_line_break()
            if self.interactive:
                self.ui.activity(
                    "■", "Tool cancelled", str(event.data.get("tool") or "tool"), tone="warning"
                )
            else:
                print(f"■ {event.data.get('tool')} cancelled", file=self.status_stream)
            return
        if event.type == "verification.started":
            self._activity_phase = "running verification"
            self._ensure_line_break()
            self._close_assistant_block()
            if self.interactive:
                self.ui.activity("◌", "Verification", "running verification", tone="cyan")
            else:
                print("… running verification", file=self.status_stream, flush=True)
            return
        if event.type == "verification.policy":
            command = str(event.data.get("command") or "verification command")
            self._activity_phase = f"verifying · {truncate_text(command, 72)}"
            return
        if event.type == "verification.completed":
            self._ensure_line_break()
            marker = "✓" if event.data.get("ok") else "✗"
            count = len(event.data.get("commands") or [])
            detail = f"{count} command{'s' if count != 1 else ''}"
            if self.interactive:
                tone = "mint" if event.data.get("ok") else "error"
                self.ui.activity(
                    "◆" if event.data.get("ok") else _ERROR_GLYPH,
                    "Verification passed" if event.data.get("ok") else "Verification failed",
                    detail,
                    tone=tone,
                )
            else:
                print(f"{marker} verification ({detail})", file=self.status_stream)
            return
        if event.type == "plan.updated":
            items = event.data.get("items") or []
            if self.interactive and isinstance(items, list):
                rows = []
                markers = {
                    "completed": "◆",
                    "in_progress": "◌",
                    "blocked": _ERROR_GLYPH,
                    "pending": "◇",
                }
                for index, item in enumerate(items, 1):
                    if not isinstance(item, dict):
                        continue
                    status = str(item.get("status") or "pending")
                    rows.append(
                        (f"{markers.get(status, '◇')} {index}", str(item.get("content") or ""))
                    )
                self.ui.panel("Plan constellation", rows, tone="violet")
            return
        if event.type == "context.compacted":
            self._ensure_line_break()
            if self.interactive:
                self.ui.activity("↻", "Context", "compacted conversation context", tone="warning")
            else:
                print("↻ compacted conversation context", file=self.status_stream)
            return
        if event.type == "model.route_failed":
            self._ensure_line_break()
            detail = f"provider route failed: {event.data.get('provider')}; trying fallback"
            if self.interactive:
                self.ui.activity("↻", "Fallback", detail, tone="warning")
            else:
                print(f"↻ {detail}", file=self.status_stream)
            return
        if event.type == "model.fallback_succeeded":
            if self.interactive:
                self.ui.activity(
                    "◆",
                    "Fallback connected",
                    f"{event.data.get('provider')}/{event.data.get('model')}",
                    tone="mint",
                )
            return
        if event.type == "run.error":
            self._activity_phase = ""
            self._ensure_line_break()
            self._close_assistant_block()
            if self.interactive:
                self.ui.notice("error", "Run failed", str(event.data.get("error") or ""))
            else:
                print(f"✗ {event.data.get('error')}", file=self.status_stream)
            return
        if event.type == "run.completed":
            self._activity_phase = ""
            self._close_assistant_block()

    def _begin_model_exchange(self) -> None:
        self._close_assistant_block()
        self._ensure_line_break()
        self.printed_text = False
        self._streamed_text = ""
        self._assistant_started = False
        self._tool_call_announced = False

    def _start_assistant_block(self) -> None:
        if self._assistant_started:
            return
        self._assistant_started = True
        if self.interactive:
            self._ensure_line_break()
            if self.text_stream is self.status_stream:
                self.ui.assistant_header()
            else:
                AuroraUI(self.text_stream).assistant_header()
            self._assistant_block_open = True

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
        self._close_assistant_block()

    def _ensure_line_break(self) -> None:
        if self._line_open:
            print(file=self.text_stream, flush=True)
            self._line_open = False

    def _ensure_tool_output_line_break(self) -> None:
        if self._tool_output_line_open:
            print(file=self.status_stream, flush=True)
            self._tool_output_line_open = False

    def _close_assistant_block(self) -> None:
        if not self._assistant_block_open:
            return
        self._ensure_line_break()
        if self.text_stream is self.status_stream:
            self.ui.assistant_footer()
        else:
            AuroraUI(self.text_stream).assistant_footer()
        self._assistant_block_open = False

    def _clear_live_line(self) -> None:
        if not self._live_line_open:
            return
        print("\r\033[2K", end="", file=self.status_stream, flush=True)
        self._live_line_open = False


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
        completion_descriptions: dict[str, str] | None = None,
        max_entries: int = 1_000,
    ) -> None:
        self.path = path
        self.enabled = enabled
        self.completions = completions
        self.completion_descriptions = dict(completion_descriptions or {})
        self.max_entries = max(1, max_entries)
        self._readline = None
        self._previous_completer = None
        self._previous_delimiters: str | None = None
        self._selector_bound = False

    def __enter__(self) -> ReadlineHistory:
        try:
            import readline  # type: ignore[import-not-found]
        except ImportError:
            return self
        self._readline = readline
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.is_file():
                with contextlib.suppress(OSError):
                    readline.read_history_file(str(self.path))
            readline.set_history_length(self.max_entries)
        self._previous_completer = readline.get_completer()
        self._previous_delimiters = readline.get_completer_delims()
        readline.set_completer(self._complete)
        readline.set_completer_delims(" \t\n")
        with contextlib.suppress(Exception):
            if "libedit" in str(readline.__doc__ or "").lower():
                readline.parse_and_bind("bind ^I rl_complete")
            else:
                readline.parse_and_bind("tab: complete")
        if self.completion_descriptions and _stream_is_tty(sys.stdin):
            with contextlib.suppress(Exception):
                _bind_slash_selector(readline)
                self._selector_bound = True
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        readline = self._readline
        if readline is None:
            return
        try:
            if self.enabled:
                readline.write_history_file(str(self.path))
                if os.name != "nt":
                    os.chmod(self.path, 0o600)
        except OSError:
            pass
        finally:
            readline.set_completer(self._previous_completer)
            if self._previous_delimiters is not None:
                readline.set_completer_delims(self._previous_delimiters)
            if self._selector_bound:
                with contextlib.suppress(Exception):
                    _restore_slash_binding(readline)

    def _complete(self, text: str, state: int) -> str | None:
        candidates = [item for item in self.completions if item.startswith(text)]
        if state == 0 and candidates and self._selector_bound:
            line = str(self._readline.get_line_buffer() if self._readline else text)
            if line.startswith("/") and " " not in line:
                self._show_selector(candidates, query=line)
        return candidates[state] if state < len(candidates) else None

    def _show_selector(self, candidates: list[str], *, query: str) -> None:
        visible = candidates[:8]
        rows = [
            (command, self.completion_descriptions.get(command, "Run command"))
            for command in visible
        ]
        AuroraUI(sys.stdout).command_selector(
            rows,
            query=query,
            hidden=max(0, len(candidates) - len(visible)),
        )
        if self._readline is not None:
            self._readline.redisplay()


def _bind_slash_selector(readline: Any) -> None:
    parse_and_bind = readline.parse_and_bind
    documentation = str(getattr(readline, "__doc__", "") or "").lower()
    if "libedit" in documentation:
        parse_and_bind('bind -s "/" "^V/\\t^E"')
    else:
        parse_and_bind('"/": "\\C-v/\\C-i\\C-e"')


def _restore_slash_binding(readline: Any) -> None:
    parse_and_bind = readline.parse_and_bind
    documentation = str(getattr(readline, "__doc__", "") or "").lower()
    if "libedit" in documentation:
        parse_and_bind('bind "/" ed-insert')
    else:
        parse_and_bind('"/": self-insert')


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
