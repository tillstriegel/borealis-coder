"""Offline input/HUD acceptance launcher with real runner and compositor captures.

Without --capture-dir, type while the synthetic response streams; Enter exits.
No provider credentials or external requests are used.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import tempfile
from contextlib import redirect_stdout, suppress
from pathlib import Path
from unittest.mock import patch

from prompt_toolkit.document import Document
from prompt_toolkit.input.defaults import create_pipe_input
from scripts.swarm_demo import CaptureOutput, capture_screen

from borealis_coder.config import load_config
from borealis_coder.interactive import InteractiveCLI
from borealis_coder.models import ModelResponse, Usage
from borealis_coder.providers.base import ProviderStreamEvent
from borealis_coder.providers.mock import MockProvider
from borealis_coder.terminal_input import (
    LineBufferedStdout,
    TerminalInput,
    TerminalInputInterrupted,
)


async def demo(destination: Path | None, width: int, height: int) -> None:
    with tempfile.TemporaryDirectory(prefix="borealis-input-") as directory:
        root = Path(directory)
        config = load_config(root, overrides={
            "agent": {"provider": "mock", "auto_verify": False},
            "storage": {"directory": str(root / "state")},
            "safety": {"approval": "never"},
        })
        shell = InteractiveCLI(workspace=root, config=config, approval_callback=None, history_enabled=False)
        shell.runner = await shell._new_runner(config)
        provider = shell.runner.providers[0].provider
        assert isinstance(provider, MockProvider)
        terminal = None
        run = None
        read = None
        streaming = asyncio.Event()
        release = asyncio.Event()

        async def stream(request):
            yield ProviderStreamEvent(type="text_delta", text="Synthetic streaming response.\n")
            streaming.set()
            if destination is not None:
                await release.wait()
            else:
                for _ in range(100):
                    await asyncio.sleep(0.4)
                    yield ProviderStreamEvent(type="text_delta", text="Offline progress.\n")
            yield ProviderStreamEvent(type="completed", response=ModelResponse(
                text="Synthetic streaming response.\nComplete.",
                usage=Usage(input_tokens=900, cached_input_tokens=0, output_tokens=100, requests=1),
            ))

        try:
            # Seed actual persisted usage in an earlier completed turn.
            provider.enqueue(ModelResponse(text="Synthetic first turn.", usage=Usage(
                input_tokens=100, cached_input_tokens=100, output_tokens=10, requests=1)))
            with redirect_stdout(io.StringIO()):
                shell.renderer._status_stream = io.StringIO()
                shell.renderer._text_stream = io.StringIO()
                await shell._submit("Synthetic first task")
            with create_pipe_input() if destination is not None else io.StringIO() as pipe:
                terminal = TerminalInput(root / "history", history_enabled=False,
                                         completions=("/help",), completion_descriptions={},
                                         swarm=shell.swarm, usage=shell.usage_view,
                                         input=pipe if destination is not None else None,
                                         output=CaptureOutput(width, height) if destination is not None else None)
                shell._input = terminal
                frame = asyncio.Event()
                terminal._session.app.after_render += lambda app: frame.set()

                async def snapshot(name):
                    frame.clear()
                    terminal.invalidate()
                    await asyncio.wait_for(frame.wait(), 3)
                    if destination is not None:
                        capture_screen(terminal, destination / name)

                with LineBufferedStdout() as output, redirect_stdout(output):
                    shell.renderer._text_stream = output
                    shell.renderer._status_stream = output
                    read = asyncio.create_task(terminal.read("Demo> ", default=Document("draft 界 text", 7)))
                    await snapshot(f"input-{width}x{height}-idle")
                    with patch.object(provider, "stream", side_effect=stream):
                        run = asyncio.create_task(shell._submit("Synthetic streaming task"))
                        await asyncio.wait_for(streaming.wait(), 5)
                        await snapshot(f"input-{width}x{height}-streaming")
                        if destination is not None:
                            document = terminal.current_document
                            terminal._session.app.output.size = type(terminal._session.app.output.get_size())(rows=12, columns=48)
                            terminal._session.app._on_resize()
                            await snapshot(f"input-{width}x{height}-resized")
                            assert terminal.current_document == document
                            assert terminal._session.app.layout.current_buffer is terminal._session.default_buffer
                            release.set()
                            await asyncio.wait_for(run, 5)
                            await snapshot(f"input-{width}x{height}-complete")
                            assert shell.usage_view.usage.total_tokens == 1110
                            assert shell.usage_view.usage.provider_cache_hit_rate == 0.1
                            pipe.send_text("X\n")
                            assert await asyncio.wait_for(read, 3) == "draft 界X text"
                        else:
                            with suppress(EOFError, TerminalInputInterrupted):
                                await read
        finally:
            release.set()
            for task in (run, read):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(*(task for task in (run, read) if task is not None), return_exceptions=True)
            if terminal is not None:
                terminal.close()
            await shell.runner.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--height", type=int, default=24)
    args = parser.parse_args()
    asyncio.run(demo(args.capture_dir, args.width, args.height))


if __name__ == "__main__":
    main()
