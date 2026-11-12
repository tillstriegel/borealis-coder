"""Backend cursor contract for native consoles with a scrollback buffer."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from prompt_toolkit.input.defaults import create_pipe_input
from scripts.swarm_demo import CaptureOutput

from borealis_coder.terminal_input import TerminalInput


class NativeOutput(CaptureOutput):
    """Model Win32's buffer-relative, non-clamping cursor movement."""

    def __init__(self, row: int) -> None:
        super().__init__(80, 24)
        self.row = row
        self.bottom = 103  # Visible viewport is buffer rows 80 through 103.
        self.scrolled = 0
        self.moves: list[int] = []

    def get_rows_below_cursor_position(self) -> int:
        return self.bottom - self.row + 1

    def cursor_down(self, amount: int) -> None:
        self.moves.append(amount)
        self.row += amount
        if self.row > self.bottom:
            self.scrolled += self.row - self.bottom
            self.bottom = self.row


class TerminalAnchorTests(unittest.TestCase):
    def test_native_pin_uses_visible_rows_without_scrolling_buffer(self):
        for row in (80, 91, 103):
            with self.subTest(row=row), tempfile.TemporaryDirectory() as td, create_pipe_input() as pipe:
                output = NativeOutput(row)
                terminal = TerminalInput(Path(td) / "history", history_enabled=False,
                                         completions=(), completion_descriptions={}, input=pipe, output=output)
                try:
                    terminal._pin_on_resize(terminal._session.app)
                    self.assertEqual(output.row, 103)
                    self.assertEqual(output.scrolled, 0)
                    self.assertEqual(output.moves, [103 - row])
                    terminal._pin_on_resize(terminal._session.app)
                    self.assertEqual(output.moves, [103 - row])
                finally:
                    terminal.close()
