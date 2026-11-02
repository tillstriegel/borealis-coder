from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prompt_toolkit.document import Document
from prompt_toolkit.input.defaults import create_pipe_input
from scripts.swarm_demo import CaptureOutput, demo

from borealis_coder.models import Event
from borealis_coder.swarm_view import SwarmView, get_cwidth, literal
from borealis_coder.terminal_input import TerminalInput, TerminalInputInterrupted


class SwarmViewTests(unittest.IsolatedAsyncioTestCase):
    def view(self):
        view = SwarmView()
        view.adopt(Event(type="swarm.started", run_id="run_root", data={
            "task": "Root task", "limits": {"max_descendants": 8, "max_concurrent": 2},
            "max_model_requests": 40,
        }))
        for identifier, parent, depth in (("child", None, 1), ("grandchild", "child", 2), ("sibling", None, 1)):
            view.adopt(Event(type="delegate.started", run_id="run_root", data={
                "delegation_id": identifier, "parent_delegation_id": parent,
                "delegated_run_id": "run_root." + identifier, "depth": depth,
                "task": identifier + "\x1b[2J\u202e\r\nspoof", "capability": "read-only",
            }))
        return view

    async def test_footer_counts_tree_entries_not_admitted_agents(self):
        view = self.view()
        view.adopt(Event(type="delegate.refused", data={
            "delegated_run_id": "run_root.refused", "delegation_id": "refused",
            "depth": 1, "task": "At capacity", "capability": "read-only",
            "limits": {"admitted": 3, "max_descendants": 3, "max_concurrent": 2},
        }))
        text = "".join(fragment[1] for fragment in view.render(120, 20))
        self.assertIn("Agents 3/3", text)
        self.assertIn("F6 inspect 5 entries", text)  # Root + three admitted + refused.
        view.selected = "child"
        view.collapse(True)
        text = "".join(fragment[1] for fragment in view.render(120, 20))
        self.assertIn("Agents 3/3", text)
        self.assertIn("F6 inspect 4 entries", text)  # Folded grandchild is not counted.

    async def test_live_key_bindings_preserve_draft_cursor_and_resize(self):
        for width, height in ((120, 40), (80, 24), (60, 24)):
            with self.subTest(width=width), tempfile.TemporaryDirectory() as td, create_pipe_input() as pipe:
                view = self.view()
                output = CaptureOutput(width, height)
                terminal = TerminalInput(Path(td) / "history", history_enabled=False, completions=(),
                                         completion_descriptions={}, input=pipe, output=output, swarm=view)
                rendered = asyncio.Event()
                app = terminal._session.app
                app.after_render += lambda app, rendered=rendered: rendered.set()
                task = asyncio.create_task(terminal.read("Follow-up> ", default=Document("draft text", 5)))

                async def until(predicate, rendered=rendered, app=app):
                    while not predicate():
                        rendered.clear()
                        app.invalidate()
                        await asyncio.wait_for(rendered.wait(), 3)

                try:
                    await until(lambda app=app: app.renderer._last_screen is not None)
                    pipe.send_text("\x1b[17~\x1b[B")  # F6 focuses and Down selects child.
                    await until(lambda view=view: view.focused and view.selected == "child")
                    self.assertEqual(terminal.current_document, Document("draft text", 5))
                    pipe.send_text("\x1b[D")
                    await until(lambda view=view: view.nodes["child"].collapsed)
                    self.assertEqual([row.id for row in view.rows()], ["root", "child", "sibling"])
                    pipe.send_text("\x1b[C\x1b[B")
                    await until(lambda view=view: view.selected == "grandchild")
                    view.nodes["grandchild"].detail = "Untrusted evidence\x1b[2J\n" + "long result " * 200
                    view.nodes["grandchild"].detail = literal(view.nodes["grandchild"].detail)
                    output.size = type(output.size)(rows=24, columns=60)
                    app._on_resize()
                    rendered.clear()
                    app.invalidate()
                    await asyncio.wait_for(rendered.wait(), 3)
                    self.assertEqual(view.selected, "grandchild")
                    self.assertEqual(terminal.current_document, Document("draft text", 5))
                    screen = app.renderer._last_screen
                    self.assertIsNotNone(screen)
                    assert screen is not None
                    text = "\n".join("".join(screen.data_buffer[y][x].char for x in range(60)) for y in range(24))
                    self.assertIn("Follow-up> draft text", text)
                    self.assertIn("SWARM", text)
                    self.assertIn("grandchild", text)
                    self.assertNotIn("\x1b", text)
                    self.assertNotIn("\u202e", text)
                    self.assertTrue(all(get_cwidth(line) <= 60 for line in text.splitlines()))
                    pipe.send_text("\x1b[6~")  # Page down inspects bounded detail.
                    await until(lambda view=view: view.detail_offset > 0)
                    pipe.send_text("\rX\n")  # Enter returns focus; edit at the original cursor.
                    self.assertEqual(await asyncio.wait_for(task, 3), "draftX text")
                finally:
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    terminal.close()

    async def test_live_tree_does_not_capture_ctrl_c_or_eof(self):
        for key, exception in (("\x03", TerminalInputInterrupted), ("\x04", EOFError)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as td, create_pipe_input() as pipe:
                terminal = TerminalInput(Path(td) / "history", history_enabled=False, completions=(),
                                         completion_descriptions={}, input=pipe, output=CaptureOutput(60, 24), swarm=self.view())
                terminal.swarm.focused = True
                task = asyncio.create_task(terminal.read("Follow-up> "))
                await asyncio.sleep(0)
                pipe.send_text(key)
                with self.assertRaises(exception):
                    await asyncio.wait_for(task, 3)
                terminal.close()

    async def test_bounded_literal_state_and_monochrome(self):
        view = self.view()
        view.visible = False
        view.adopt(Event(type="delegate.started", data={
            "delegated_run_id": "run_root.late", "delegation_id": "late",
            "depth": 1, "task": "Late child", "capability": "read-only",
        }))
        self.assertFalse(view.visible)
        view.visible = True
        view.adopt(Event(type="delegate.failed", data={
            "delegation_id": "child", "delegated_run_id": "run_root.child", "error": "bad\x1b[2J\u202e\x07",
        }))
        self.assertEqual(view.nodes["child"].state, "error")
        self.assertNotIn("\x1b", view.nodes["child"].detail)
        self.assertNotIn("\u202e", view.nodes["child"].detail)
        self.assertFalse(view.adopt(Event(type="model.reasoning_delta", data={"text": "private"})))
        self.assertFalse(view.adopt(Event(type="delegate.started", data={"delegated_run_id": "other.child"})))
        for i in range(300):
            view.adopt(Event(type="delegate.refused", data={
                "delegated_run_id": f"run_root.{i}", "delegation_id": str(i),
                "parent_delegation_id": None, "depth": 1, "task": "refused", "capability": "read-only",
            }))
        self.assertEqual(len(view.nodes), 256)
        self.assertGreater(view.omitted, 0)
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"NO_COLOR": "1"}), create_pipe_input() as pipe:
            terminal = TerminalInput(Path(td) / "history", history_enabled=False, completions=(),
                                     completion_descriptions={}, input=pipe, output=CaptureOutput(60, 24), swarm=view)
            style = terminal._session.style
            assert style is not None
            self.assertFalse(style.get_attrs_for_style_str("class:swarm.title").color)
            terminal.close()

    async def test_synthetic_production_runner_composes_live_and_terminal_screens(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            await asyncio.wait_for(demo(root, 80, 24), 10)
            live = (root / "swarm-80x24-live.txt").read_text()
            done = (root / "swarm-80x24-cancelled.txt").read_text()
            events = json.loads((root / "swarm-80x24-events.json").read_text())
            self.assertIn("SWARM", live)
            self.assertIn("keep this draft", live)
            self.assertIn("cancelled", done)
            self.assertTrue(any(event["type"] == "delegate.refused" for event in events))
            self.assertTrue(any(event["type"] == "delegate.failed" for event in events))
            self.assertTrue(any(event["type"] == "tool.completed" and event["data"].get("tool") == "write_file" for event in events))
            self.assertTrue(any(event["type"] == "delegate.started" and event["data"]["depth"] == 2 for event in events))
            self.assertTrue((root / "swarm-80x24-live.svg").is_file())
