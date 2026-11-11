"""Offline synthetic swarm demo and real prompt-toolkit compositor captures.

Run with --capture-dir PATH for deterministic cell/text/SVG output, or without it
for editable input. No providers, credentials, or network connections are used.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from prompt_toolkit.data_structures import Size
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from borealis_coder.agent import build_runner
from borealis_coder.config import load_config
from borealis_coder.errors import ProviderError
from borealis_coder.models import ModelResponse, Role, ToolCall, Usage
from borealis_coder.providers.mock import MockProvider
from borealis_coder.swarm_view import SwarmView
from borealis_coder.terminal_input import TerminalInput, TerminalInputInterrupted


class CaptureOutput(DummyOutput):
    def __init__(self, columns: int, rows: int) -> None:
        self.size = Size(rows=rows, columns=columns)

    def get_size(self) -> Size:
        return self.size


def capture_screen(terminal: TerminalInput, destination: Path) -> None:
    """Export the renderer's actual composed cells, including resolved styles."""
    app = terminal._session.app
    screen = app.renderer._last_screen
    if screen is None:
        raise RuntimeError("Prompt-toolkit has not composed a screen")
    size = app.output.get_size()
    cells = []
    lines = []
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size.columns * 9}" height="{size.rows * 19}" viewBox="0 0 {size.columns * 9} {size.rows * 19}">',
           '<rect width="100%" height="100%" fill="#181b21"/>',
           '<g font-family="Menlo,DejaVu Sans Mono,monospace" font-size="14">']
    for y in range(size.rows):
        line = ""
        for x in range(size.columns):
            char = screen.data_buffer[y][x]
            attrs = app._merged_style.get_attrs_for_style_str(char.style)
            foreground = attrs.color or "d8dee9"
            background = attrs.bgcolor or "181b21"
            ansi = {"ansicyan": "83dbe5", "ansigreen": "9ce5c0", "ansibrightblack": "9299a5",
                    "ansiblack": "181b21", "ansiwhite": "d8dee9", "ansired": "e89292"}
            foreground = ansi.get(foreground, foreground)
            background = ansi.get(background, background)
            if attrs.reverse:
                foreground, background = background, foreground
            cells.append({"x": x, "y": y, "char": char.char, "style": char.style,
                          "foreground": foreground, "background": background})
            line += char.char
            if background != "181b21":
                svg.append(f'<rect x="{x * 9}" y="{y * 19}" width="9" height="19" fill="#{background}"/>')
            if char.char.strip():
                weight = ' font-weight="bold"' if attrs.bold else ""
                svg.append(f'<text x="{x * 9}" y="{y * 19 + 15}" fill="#{foreground}"{weight}>{html.escape(char.char)}</text>')
        lines.append(line.rstrip())
    svg.append("</g></svg>")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.with_suffix(".cells.json").write_text(json.dumps({"columns": size.columns, "rows": size.rows, "cells": cells}, ensure_ascii=False, indent=2) + "\n")
    destination.with_suffix(".txt").write_text("\n".join(lines) + "\n")
    destination.with_suffix(".svg").write_text("\n".join(svg) + "\n")


async def demo(capture_dir: Path | None, columns: int, rows: int) -> None:
    with tempfile.TemporaryDirectory(prefix="borealis-synthetic-swarm-") as temp:
        root = Path(temp)
        config = load_config(root, overrides={
            "agent": {"provider": "mock", "auto_verify": False, "swarm_allow_writes": True,
                      "swarm_max_descendants": 5, "swarm_max_concurrent": 2, "swarm_max_depth": 3},
            "safety": {"approval": "never"}, "storage": {"directory": str(root / "state")},
        })
        runner = await build_runner(root, config=config, interactive=False)
        provider = runner.providers[0].provider
        assert isinstance(provider, MockProvider)
        view = SwarmView()
        events = []
        release = asyncio.Event()
        held = asyncio.Event()

        def call(task: str, identifier: str, *, capability: str = "read-only") -> ToolCall:
            return ToolCall(id=identifier, name="delegate_task", arguments={
                "task": task, "max_turns": 4, "capability": capability,
            })

        def handler(request, count):
            task = request.messages[0].content
            if request.messages[-1].role == Role.TOOL:
                return ModelResponse(text="Complete. " + request.messages[-1].content)
            if not request.metadata.get("delegated"):
                return ModelResponse(tool_calls=[call("Implement example file", "build", capability="write"),
                                                call("Inspect evidence", "inspect"), call("Report failure", "failure")])
            if task == "Implement example file":
                return ModelResponse(tool_calls=[ToolCall(id="write", name="write_file", arguments={
                    "path": "swarm-example.txt", "content": "Written by the synthetic swarm demo.\n", "expected_sha256": None,
                }), call("Check nested result", "nested")])
            if task == "Inspect evidence":
                return ModelResponse(tool_calls=[call("Waiting investigation", "waiting")])
            if task == "Check nested result":
                return ModelResponse(tool_calls=[call("Capacity refusal", "refused")])
            if task == "Report failure":
                raise ProviderError("Synthetic provider failure (no spend)", usage=Usage(requests=1, cost_status="known"))
            return ModelResponse(text="Synthetic nested evidence complete. swarm-example.txt")

        original = provider.complete

        async def controlled(request):
            if request.messages[0].content == "Check nested result":
                await held.wait()
            if request.messages[0].content == "Waiting investigation":
                held.set()
                await release.wait()
            return await original(request)

        provider.handler = handler
        try:
            if capture_dir is None:
                print("Synthetic offline swarm demo · no spend · Enter cancels/exits · F6 inspect · F7 hide")
                terminal = TerminalInput(root / "history", history_enabled=False, completions=(), completion_descriptions={}, swarm=view)
                def observe_live(event):
                    if view.adopt(event):
                        terminal.invalidate()

                runner.events.subscribe(observe_live)
                with patch.object(provider, "complete", side_effect=controlled):
                    run = asyncio.create_task(runner.run("Synthetic demo: bounded file-editing swarm"))
                    read = asyncio.create_task(terminal.read("Demo follow-up> "))
                    await held.wait()
                    try:
                        await read
                    except (EOFError, TerminalInputInterrupted):
                        pass
                    finally:
                        run.cancel()
                        release.set()
                        await asyncio.gather(run, return_exceptions=True)
                        terminal.close()
                return
            output = CaptureOutput(columns, rows)
            with create_pipe_input() as pipe:
                terminal = TerminalInput(root / "history", history_enabled=False, completions=(),
                                         completion_descriptions={}, swarm=view, input=pipe, output=output)
                frame_ready = asyncio.Event()
                terminal._session.app.after_render += lambda app: frame_ready.set()

                failed = asyncio.Event()
                refused = asyncio.Event()

                def observe(event):
                    if event.type == "delegate.failed":
                        failed.set()
                    if event.type == "delegate.refused":
                        refused.set()
                    events.append(event.to_dict())
                    if view.adopt(event):
                        terminal.invalidate()

                runner.events.subscribe(observe)
                read = asyncio.create_task(terminal.read("Follow-up> ", default="keep this draft"))

                async def frame(name, ready=lambda: True):
                    while True:
                        frame_ready.clear()
                        terminal.invalidate()
                        await asyncio.wait_for(frame_ready.wait(), 3)
                        if ready():
                            break
                    capture_screen(terminal, capture_dir / name)

                with patch.object(provider, "complete", side_effect=controlled):
                    run = asyncio.create_task(runner.run("Synthetic demo: bounded file-editing swarm"))
                    try:
                        await asyncio.wait_for(held.wait(), 5)
                        await asyncio.wait_for(asyncio.gather(failed.wait(), refused.wait()), 5)
                        await frame(f"swarm-{columns}x{rows}-live")
                        failure = next(node for node in view.rows() if node.label == "Report failure")
                        index = view.rows().index(failure)
                        pipe.send_text("\x1b[17~" + "\x1b[B" * index)  # Real F6 and arrow bindings.
                        await frame(f"swarm-{columns}x{rows}-selected",
                                    lambda: view.focused and view.selected == failure.id)
                        run.cancel()
                        await asyncio.wait_for(run, 5)
                        await frame(f"swarm-{columns}x{rows}-cancelled")
                        (capture_dir / f"swarm-{columns}x{rows}-events.json").write_text(json.dumps(events, indent=2) + "\n")
                        assert (root / "swarm-example.txt").is_file()
                    finally:
                        release.set()
                        if not run.done():
                            run.cancel()
                        await asyncio.gather(run, return_exceptions=True)
                        pipe.send_text("\x1b[18~\n")  # F7 hides without changing the draft.
                        await asyncio.wait_for(read, 3)
                        terminal.close()
        finally:
            await runner.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--width", type=int, default=120)
    parser.add_argument("--height", type=int, default=40)
    args = parser.parse_args()
    asyncio.run(demo(args.capture_dir, args.width, args.height))


if __name__ == "__main__":
    main()
