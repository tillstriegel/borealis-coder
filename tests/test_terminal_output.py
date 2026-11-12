"""Physical terminal acceptance using real PTY bytes and a VT100 consumer."""
from __future__ import annotations

import asyncio
import codecs
import errno
import json
import os
import select
import signal
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pyte
from prompt_toolkit.document import Document
from pyte.modes import DECAWM

from borealis_coder.models import Usage
from borealis_coder.terminal import AuroraUI
from borealis_coder.terminal_input import LineBufferedStdout, TerminalInput
from borealis_coder.usage_view import UsageView


async def child(root: Path) -> None:
    terminal = TerminalInput(root / "history", history_enabled=False,
                             completions=tuple(f"draft 界X{name}" for name in ("one", "two", "three", "four", "five")),
                             completion_descriptions={},
                             usage=UsageView(Usage(input_tokens=100, cached_input_tokens=50, output_tokens=10), 6000, 12000))
    terminal._session.complete_while_typing = False
    app = terminal._session.app
    app.ttimeoutlen = 0.05
    frame = asyncio.Event()
    app.after_render += lambda app: frame.set()

    async def stage(name: str) -> None:
        await asyncio.sleep(0.3)  # StdoutProxy coalesces complete lines in its worker.
        frame.clear()
        terminal.invalidate()
        await asyncio.wait_for(frame.wait(), 3)
        (root / f"{name}.json").write_text(json.dumps({
            "draft": terminal.current_text, "cursor": terminal.current_document.cursor_position,
            "focused": app.layout.current_buffer is terminal._session.default_buffer,
            "completing": terminal._session.default_buffer.complete_state is not None,
        }))

    async def command(value: str) -> None:
        while not (root / value).exists():
            await asyncio.sleep(0.01)

    with LineBufferedStdout() as output, redirect_stdout(output):
        read = asyncio.create_task(terminal.read(AuroraUI(sys.stdout).prompt("new"), default=Document("draft 界 text", 7)))
        try:
            await stage("idle")
            await command("partial")
            print("STREAM-0", end="", flush=True)
            await stage("partial")
            await command("stream")
            print("-joined")
            for index in range(1, 4):
                print(f"STREAM-{index}")
            ui = AuroraUI(sys.stdout)
            ui.panel("Approval required", [("tool", "synthetic"), ("risk", "low"),
                                           ("reason", "Offline test"), ("request", "synthetic.txt")])
            ui.notice("info", "Reply y once · a for session · n to reject")
            await stage("stream")
            await command("flood")
            for index in range(40):
                print(f"OLDER-{index:02d}")
            await stage("flood")
            for transition in ("swarm", "completion"):
                await command(transition + "-open")
                await stage(transition + "-open")
                await command(transition + "-close")
                await stage(transition + "-close")
                print("AFTER-" + transition.upper())
                await stage(transition + "-output")
            await command("resize")
            await stage("resize")
            result = await read
            (root / "result.json").write_text(json.dumps({"result": result}, ensure_ascii=False))
        finally:
            if not read.done():
                read.cancel()
            await asyncio.gather(read, return_exceptions=True)
    terminal.close()


class ReplyScreen(pyte.HistoryScreen):
    input_fd: int

    def write_process_input(self, data: str) -> None:
        os.write(self.input_fd, data.encode())


@unittest.skipUnless(os.name == "posix", "PTY acceptance requires POSIX")
class TerminalOutputTests(unittest.TestCase):
    def test_stdout_stays_visible_and_enters_history_once(self):
        for width, cpr in ((80, False), (48, False), (80, True), (48, True)):
            with self.subTest(width=width, cpr=cpr):
                self.check_terminal(width, cpr)

    def check_terminal(self, width: int, cpr: bool) -> None:
        if sys.platform == "win32":
            self.skipTest("PTY acceptance requires POSIX")
        import fcntl
        import pty
        import termios
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            master, slave = pty.openpty()
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, width, 0, 0))
            env = dict(os.environ, TERM="xterm-256color", PROMPT_TOOLKIT_NO_CPR="1",
                       PYTHONPATH="src", TMPDIR=td)
            if cpr:
                env.pop("PROMPT_TOOLKIT_NO_CPR", None)
            process = subprocess.Popen([sys.executable, "-m", "tests.test_terminal_output", "--child", td],
                                       stdin=slave, stdout=slave, stderr=slave, env=env,
                                       start_new_session=True)
            os.close(slave)
            screen = ReplyScreen(width, 24, history=1000)
            screen.input_fd = master
            stream = pyte.Stream(screen)
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            raw = bytearray()
            captures = {}

            def drain():
                while select.select([master], [], [], 0.05)[0]:
                    try:
                        data = os.read(master, 65536)
                    except OSError as error:
                        if error.errno != errno.EIO:
                            raise
                        return
                    if not data:
                        return
                    raw.extend(data)
                    stream.feed(decoder.decode(data))

            def wait(name):
                deadline = time.monotonic() + 8
                path = root / f"{name}.json"
                while not path.exists():
                    drain()
                    if path.exists():
                        break
                    if process.poll() is not None:
                        self.fail(f"Terminal child exited {process.returncode}: {raw!r}")
                    if time.monotonic() > deadline:
                        self.fail(f"Terminal child timed out at {name}: {raw!r}")
                drain()
                history = ["".join(line[x].data for x in range(screen.columns))
                           for line in screen.history.top]
                captures[name] = {"raw_length": len(raw), "emulator_cursor": [screen.cursor.x, screen.cursor.y], "visible": list(screen.display), "history": history,
                                  "document": json.loads(path.read_text())}
                return captures[name]

            def send(name):
                (root / name).touch()
                return wait(name)

            def bottom(capture):
                self.assertTrue(capture["visible"][-1].startswith("╰─❯ draft 界"), capture)  # noqa: RUF001 - production prompt glyph.
                self.assertIn("Context est.", "\n".join(capture["visible"][-6:-1]))
                self.assertIn("Cache read 50%", "\n".join(capture["visible"][-6:-1]))
                self.assertIn("Tokens 110", "\n".join(capture["visible"][-6:-1]))
                self.assertTrue(capture["document"]["focused"])

            try:
                bottom(wait("idle"))
                os.write(master, b"X")
                partial = send("partial")
                self.assertNotIn("STREAM-0", "\n".join(partial["visible"]))
                live = send("stream")
                bottom(live)
                visible = "\n".join(live["visible"][:-2])
                self.assertIn("STREAM-0-joined", visible)
                self.assertIn("STREAM-3", visible)
                self.assertIn("APPROVAL REQUIRED", visible)
                self.assertIn("synthetic.txt", visible)
                self.assertIn("Reply y once", visible)
                flooded = send("flood")
                bottom(flooded)
                self.assertIn("OLDER-39", "\n".join(flooded["visible"][:-2]))
                all_lines = flooded["history"] + flooded["visible"]
                for marker in ["STREAM-0-joined", "STREAM-1", "STREAM-2", "STREAM-3",
                               *[f"OLDER-{i:02d}" for i in range(40)]]:
                    self.assertEqual(sum(line.strip() == marker for line in all_lines), 1, marker)
                transitions = {}
                for transition, opening, closing in (("swarm", b"\x1b[18~", b"\x1b[18~"),
                                                       ("completion", b"\t", b"\x07")):
                    os.write(master, opening)
                    expanded = send(transition + "-open")
                    if transition == "swarm":
                        self.assertIn("SWARM", "\n".join(expanded["visible"]))
                    else:
                        self.assertTrue(expanded["document"]["completing"])
                        self.assertIn("draft 界Xone", "\n".join(expanded["visible"]))
                    os.write(master, closing)
                    closed = send(transition + "-close")
                    if transition == "completion":
                        self.assertFalse(closed["document"]["completing"])
                    else:
                        self.assertNotIn("╭─ SWARM", "\n".join(closed["visible"]))
                    transitions[transition] = wait(transition + "-output")
                for transition, capture in transitions.items():
                    with self.subTest(transition=transition):
                        bottom(capture)
                        marker = "AFTER-" + transition.upper()
                        self.assertIn(marker, "\n".join(capture["visible"]))
                        hud = next(i for i, line in enumerate(capture["visible"]) if "Context est." in line)
                        self.assertEqual(capture["visible"][hud - 1].strip(), marker)
                        all_lines = capture["history"] + capture["visible"]
                        markers = ["STREAM-0-joined", "STREAM-1", "STREAM-2", "STREAM-3",
                                   *[f"OLDER-{i:02d}" for i in range(40)], "AFTER-SWARM"]
                        if transition == "completion":
                            markers.append("AFTER-COMPLETION")
                        for output_marker in markers:
                            self.assertEqual(sum(line.strip() == output_marker for line in all_lines), 1, output_marker)
                        self.assertEqual(capture["document"]["draft"], "draft 界X text")
                        self.assertEqual(capture["document"]["cursor"], 8)
                fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 12, 48, 0, 0))
                screen.resize(lines=12, columns=48)
                screen.ensure_vbounds()
                screen.ensure_hbounds()
                os.kill(process.pid, signal.SIGWINCH)
                resized = send("resize")
                bottom(resized)
                self.assertIn("AFTER-COMPLETION", "\n".join(resized["visible"][:-3]))
                self.assertEqual(sum("Context est." in line for line in resized["visible"]), 1)
                self.assertEqual(resized["document"]["draft"], "draft 界X text")
                self.assertEqual(resized["document"]["cursor"], 8)
                os.write(master, b"\x1b[DZ\r")
                self.assertEqual(wait("result")["document"]["result"], "draft 界ZX text")
                deadline = time.monotonic() + 5
                while process.poll() is None and time.monotonic() < deadline:
                    drain()
                self.assertEqual(process.poll(), 0)
                self.assertNotIn(b"\x1b[?1049h", raw)
                self.assertIn(b"\x1b[?2004l", raw)
                self.assertFalse(screen.cursor.hidden)
                self.assertIn(DECAWM, screen.mode)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                os.close(master)
                destination = os.getenv("BOREALIS_TERMINAL_CAPTURE_DIR")
                if destination:
                    target = Path(destination)
                    target.mkdir(parents=True, exist_ok=True)
                    (target / f"terminal-{width}-cpr-{cpr}.ansi").write_bytes(raw)
                    (target / f"terminal-{width}-cpr-{cpr}.json").write_text(json.dumps(captures, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        asyncio.run(child(Path(sys.argv[2])))
    else:
        unittest.main()
