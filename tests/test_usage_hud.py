from __future__ import annotations

import asyncio
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from prompt_toolkit.document import Document
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import ColorDepth
from scripts.swarm_demo import CaptureOutput

from borealis_coder.agent.budget import ContextBudget
from borealis_coder.config import load_config
from borealis_coder.errors import ProviderError, ProviderUnavailableError
from borealis_coder.interactive import InteractiveCLI
from borealis_coder.models import Event, Message, ModelResponse, Role, ToolCall, Usage
from borealis_coder.providers.mock import MockProvider
from borealis_coder.terminal_input import TerminalInput
from borealis_coder.usage_view import UsageView


class UsageHUDTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_compositor_bottom_geometry_resize_and_editing(self):
        for width, height in ((120, 40), (80, 24), (48, 24), (48, 6)):
            with self.subTest(width=width, height=height), tempfile.TemporaryDirectory() as td, create_pipe_input() as pipe:
                output = CaptureOutput(width, height)
                hud = UsageView(Usage(input_tokens=1000, cached_input_tokens=750, output_tokens=100), 6000, 12000)
                terminal = TerminalInput(Path(td) / "history", history_enabled=False,
                                         completions=("/help",), completion_descriptions={},
                                         input=pipe, output=output, usage=hud)
                app = terminal._session.app
                frame = asyncio.Event()
                app.after_render += lambda app, frame=frame: frame.set()
                document = Document("draft 界 e\u0301 " * 12, 8)
                read = asyncio.create_task(terminal.read("> ", default=document))

                async def redraw(frame=frame, app=app):
                    frame.clear()
                    app.invalidate()
                    await asyncio.wait_for(frame.wait(), 3)

                def lines(app=app, output=output):
                    screen = app.renderer._last_screen
                    assert screen is not None
                    size = output.get_size()
                    return ["".join(screen.data_buffer[y][x].char for x in range(size.columns))
                            for y in range(screen.height)]

                try:
                    await redraw()
                    rendered = lines()
                    self.assertLessEqual(len(rendered), height)
                    self.assertTrue(rendered[-1].strip(), rendered)
                    self.assertIn("Context est.", "\n".join(rendered))
                    self.assertIn("Cache read 75%", "\n".join(rendered))
                    self.assertEqual(terminal.current_document, document)
                    self.assertFalse(app.full_screen)
                    # HUD refresh and swarm expansion cannot replace the buffer.
                    terminal.swarm.adopt(Event(type="swarm.started", run_id="root", data={"task": "Synthetic", "limits": {}, "max_model_requests": 10}))
                    hud.usage.add(Usage(input_tokens=1000, output_tokens=50))
                    await redraw()
                    self.assertIn("Cache read 38%", "\n".join(lines()))
                    self.assertEqual(terminal.current_document, document)
                    output.size = type(output.size)(rows=height, columns=48)
                    app._on_resize()
                    await redraw()
                    self.assertEqual(terminal.current_document, document)
                    self.assertIs(app.layout.current_buffer, terminal._session.default_buffer)
                    pipe.send_text("X\n")
                    expected = document.text[:8] + "X" + document.text[8:]
                    self.assertEqual(await asyncio.wait_for(read, 3), expected)
                finally:
                    if not read.done():
                        read.cancel()
                    await asyncio.gather(read, return_exceptions=True)
                    terminal.close()

    async def test_idle_bottom_hud_adjacency_and_completion_history(self):
        with tempfile.TemporaryDirectory() as td, create_pipe_input() as pipe:
            terminal = TerminalInput(Path(td) / "history", history_enabled=False,
                                     completions=("/help",), completion_descriptions={},
                                     input=pipe, output=CaptureOutput(48, 24))
            app = terminal._session.app
            frame = asyncio.Event()
            app.after_render += lambda app: frame.set()
            read = asyncio.create_task(terminal.read("> "))
            await asyncio.wait_for(frame.wait(), 3)
            screen = app.renderer._last_screen
            assert screen is not None
            def line(y):
                return "".join(screen.data_buffer[y][x].char for x in range(48))
            self.assertTrue(line(screen.height - 1).startswith("> "))
            self.assertIn("Context unavailable", line(screen.height - 2))
            pipe.send_text("/he")
            for _ in range(50):
                await asyncio.sleep(0.01)
                if terminal._session.default_buffer.complete_state is not None:
                    break
            self.assertIsNotNone(terminal._session.default_buffer.complete_state)
            frame.clear()
            terminal.invalidate()
            await asyncio.wait_for(frame.wait(), 3)
            menu_screen = app.renderer._last_screen
            assert menu_screen is not None
            menu_text = "\n".join("".join(menu_screen.data_buffer[y][x].char for x in range(48))
                                  for y in range(menu_screen.height - 1))
            self.assertIn("/help", menu_text)
            self.assertTrue("".join(menu_screen.data_buffer[menu_screen.height - 1][x].char for x in range(48)).startswith("> /he"))
            pipe.send_text("\t\n")
            self.assertEqual(await asyncio.wait_for(read, 3), "/help")
            read = asyncio.create_task(terminal.read("> "))
            await asyncio.sleep(0)
            pipe.send_text("\x1b[A\n")
            self.assertEqual(await asyncio.wait_for(read, 3), "/help")
            terminal.close()

    async def test_production_runner_session_lifecycle_and_single_ledger(self):
        with tempfile.TemporaryDirectory() as td, redirect_stdout(io.StringIO()):
            root = Path(td)
            config = load_config(root, overrides={
                "agent": {"provider": "mock", "auto_verify": False},
                "safety": {"approval": "never"},
                "storage": {"directory": str(root / "state")},
            })
            shell = InteractiveCLI(workspace=root, config=config, approval_callback=None, history_enabled=False)
            shell.runner = await shell._new_runner(config)
            runner = shell.runner
            provider = runner.providers[0].provider
            assert isinstance(provider, MockProvider)
            shell.renderer._status_stream = io.StringIO()
            shell.renderer._text_stream = io.StringIO()
            seen = []
            ledger_snapshots = []
            requests = []

            def handler(request, count):
                requests.append(request)
                if request.metadata.get("delegated"):
                    return ModelResponse(text="Child done", usage=Usage(input_tokens=300, cached_input_tokens=0, output_tokens=30))
                if request.messages[-1].role != Role.TOOL:
                    return ModelResponse(tool_calls=[ToolCall(id=f"child{count}", name="delegate_task", arguments={
                        "task": "Offline child", "capability": "read-only", "max_turns": 1,
                    })], usage=Usage(input_tokens=100, cached_input_tokens=100, output_tokens=10))
                return ModelResponse(text="Root done", usage=Usage(input_tokens=600, cached_input_tokens=100, output_tokens=60))

            async def observe(event):
                if event.type == "context.usage":
                    seen.append(event)
                if event.type == "model.completed" and event.run_id == shell._root_run_id:
                    ledger_snapshots.append((shell.usage_view.usage, runner.sessions.usage(event.session_id)))

            provider.handler = handler
            runner.events.subscribe(observe)
            try:
                await shell._submit("Synthetic task")
                first_id = shell.session_id
                assert first_id is not None
                self.assertEqual(shell.usage_view.usage.input_tokens, 1000)
                self.assertEqual(shell.usage_view.usage.provider_cache_hit_rate, 0.2)
                self.assertEqual(shell.usage_view.usage.total_tokens, 1100)
                self.assertTrue(seen)
                self.assertTrue(ledger_snapshots)
                self.assertTrue(all(left == right for left, right in ledger_snapshots))
                # Exact final prepared request, not the earlier pre-route estimate.
                request = requests[-1]
                expected = ContextBudget.calculate(config.agent, system=request.system, tools=request.tools,
                                                   messages=request.messages, provider="mock").estimated_total(request.messages)
                self.assertEqual(shell.usage_view.context_tokens, expected)
                self.assertEqual(shell.usage_view.context_limit, request.metadata["resolved_limits"]["effective_context_tokens"])
                root_context = shell.usage_view.context_tokens
                await shell._observe(Event(type="context.usage", session_id=first_id, run_id="child", data={
                    "estimated_input_tokens": 1, "effective_context_tokens": 1,
                }))
                self.assertEqual(shell.usage_view.context_tokens, root_context)
                await shell._submit("Synthetic second turn")
                self.assertEqual(shell.usage_view.usage.input_tokens, 2000)
                self.assertEqual(shell.usage_view.usage.total_tokens, 2200)
                await shell._command("/new")
                self.assertEqual(shell.usage_view.usage.total_tokens, 0)
                self.assertIsNone(shell.usage_view.context_tokens)
                await shell._resume(first_id, announce=False)
                self.assertEqual(shell.usage_view.usage.total_tokens, 2200)
                self.assertIsNone(shell.usage_view.context_tokens)
                await shell._command("/model offline-model")
                self.assertEqual(shell.usage_view.usage.total_tokens, 2200)
                self.assertIsNone(shell.usage_view.context_tokens)
                provider = shell.runner.providers[0].provider
                assert isinstance(provider, MockProvider)
                # Settled provider failure is still part of the persisted session.
                def fail(request, count):
                    raise ProviderError("Synthetic failure", usage=Usage(input_tokens=50, output_tokens=5))
                provider.handler = fail
                await shell._submit("Synthetic failed turn")
                self.assertEqual(shell.usage_view.usage, shell.runner.sessions.usage(first_id))
                self.assertGreater(shell.usage_view.usage.total_tokens, 2200)
                settled_before = shell.usage_view.usage.total_tokens
                retrying = asyncio.Event()
                def retry_observer(event):
                    if event.type == "model.retrying":
                        retrying.set()
                shell.runner.events.subscribe(retry_observer)
                provider.config.max_retries = 1
                provider.config.initial_backoff_seconds = 30
                def billed_failure(request, count):
                    raise ProviderUnavailableError("Synthetic retry", retryable=True,
                                                   usage=Usage(input_tokens=40, output_tokens=4))
                provider.handler = billed_failure
                turn = asyncio.create_task(shell._submit("Synthetic cancellation"))
                await asyncio.wait_for(retrying.wait(), 3)
                shell.runner.cancel(first_id)
                await asyncio.wait_for(turn, 3)
                self.assertEqual(shell.usage_view.usage.total_tokens, settled_before + 44)
                self.assertEqual(shell.usage_view.usage, shell.runner.sessions.usage(first_id))
                await shell._command("/provider mock")
                self.assertIsNone(shell.usage_view.context_tokens)
                self.assertEqual(shell.usage_view.usage.total_tokens, settled_before + 44)
            finally:
                await shell.runner.close()

    async def test_context_measures_compacted_restored_request_and_effective_limit(self):
        with tempfile.TemporaryDirectory() as td, redirect_stdout(io.StringIO()):
            root = Path(td)
            config = load_config(root, overrides={
                "agent": {"provider": "mock", "auto_verify": False},
                "storage": {"directory": str(root / "state")},
            })
            shell = InteractiveCLI(workspace=root, config=config, approval_callback=None)
            shell.runner = await shell._new_runner(config)
            runner = shell.runner
            shell.renderer._status_stream = io.StringIO()
            shell.renderer._text_stream = io.StringIO()
            provider = runner.providers[0].provider
            assert isinstance(provider, MockProvider)
            provider.config.context_tokens = 90000
            session = runner.sessions.create_session(workspace=root, provider="mock", model="deterministic")
            runner.sessions.append_message(session.id, Message(role=Role.USER, content="Preserve this synthetic requirement."))
            for i in range(16):
                call = ToolCall(id=f"old{i}", name="shell", arguments={"command": f"check-{i}"})
                runner.sessions.append_message(session.id, Message(role=Role.ASSISTANT, tool_calls=[call]))
                runner.sessions.append_message(session.id, Message(role=Role.TOOL, tool_call_id=call.id,
                                                                  tool_name="shell", content=f"Evidence {i} " + "x" * 24000))
            await shell._resume(session.id, announce=False)
            events = []
            requests = []
            runner.events.subscribe(events.append)
            def handler(request, count):
                requests.append(request)
                return ModelResponse(text="Synthetic complete")
            provider.handler = handler
            try:
                await shell._submit("Continue and retain the requirement.")
                self.assertTrue(any(e.type == "context.compacted" for e in events))
                request = requests[-1]
                self.assertIn("Preserve this synthetic requirement.", request.system)
                estimate = ContextBudget.calculate(config.agent, system=request.system, tools=request.tools,
                                                    messages=request.messages, provider="mock").estimated_total(request.messages)
                self.assertEqual(shell.usage_view.context_tokens, estimate)
                self.assertEqual(shell.usage_view.context_limit, 90000)
                self.assertLess(estimate, 90000)
            finally:
                await runner.close()

    async def test_offline_streaming_capture_launcher(self):
        from scripts.input_demo import demo
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for width in (80, 48):
                await demo(root, width, 24)
                idle = (root / f"input-{width}x24-idle.txt").read_text().splitlines()
                active = (root / f"input-{width}x24-streaming.txt").read_text().splitlines()
                self.assertTrue(idle[-1].startswith("Demo> draft"))
                self.assertTrue(active[-1].startswith("Demo> draft"))
                self.assertIn("Context est.", "\n".join(active[-3:]))
                complete = (root / f"input-{width}x24-complete.txt").read_text()
                self.assertIn("Cache read 10%", complete)
                self.assertIn("Tokens 1,110", complete)

    def test_no_color_disables_completion_colors_too(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"NO_COLOR": "1"}), create_pipe_input() as pipe:
            terminal = TerminalInput(Path(td) / "history", history_enabled=False,
                                     completions=("/help",), completion_descriptions={},
                                     input=pipe, output=CaptureOutput(48, 24))
            self.assertEqual(terminal._session.app.color_depth, ColorDepth.DEPTH_1_BIT)
            terminal.close()

    def test_weighted_ratio_and_canonical_billed_categories(self):
        usage = Usage(input_tokens=100, cached_input_tokens=100, output_tokens=10,
                      cache_write_tokens=20, reasoning_tokens=5)
        usage.add(Usage(input_tokens=900, output_tokens=30))
        view = UsageView(usage)
        self.assertIn("Cache read 10%", view.render(80))
        self.assertIn(f"Tokens {usage.total_tokens:,}", view.render(80))
        self.assertIn("Context unavailable", view.render(80))
        self.assertIn("Cache read n/a", UsageView().render(48))
