from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from prompt_toolkit.document import Document
from prompt_toolkit.input.defaults import create_pipe_input
from scripts.swarm_demo import CaptureOutput

from borealis_coder.agent import ProviderRoute
from borealis_coder.agent.budget import ContextBudget
from borealis_coder.config import ProviderConfig
from borealis_coder.interactive import InteractiveCLI
from borealis_coder.models import Event, Message, ModelResponse, Role, ToolCall, Usage
from borealis_coder.providers.base import Provider
from borealis_coder.providers.http import SSEEvent
from borealis_coder.providers.mock import MockProvider
from borealis_coder.providers.openai import OpenAICompatibleProvider
from borealis_coder.terminal_input import TerminalInput
from tests.helpers import make_config
from tests.test_child_compaction import echo_evidence


class HUDRepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_hidden_inspector_returns_arrow_and_first_enter_to_input(self):
        for scenario in ("tiny", "shrink", "grow"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as td, create_pipe_input() as pipe:
                output = CaptureOutput(48, 4 if scenario == "tiny" else 24)
                terminal = TerminalInput(Path(td) / "history", history_enabled=False,
                                         completions=(), completion_descriptions={}, input=pipe, output=output)
                terminal.swarm.adopt(Event(type="swarm.started", run_id="root", data={
                    "task": "Synthetic", "limits": {}, "max_model_requests": 10,
                }))
                app = terminal._session.app
                frame = asyncio.Event()
                app.after_render += lambda app, frame=frame: frame.set()
                read = asyncio.create_task(terminal.read("> ", default=Document("draft", 5)))
                async def redraw(frame=frame, app=app):
                    frame.clear()
                    app.invalidate()
                    await asyncio.wait_for(frame.wait(), 3)
                try:
                    await redraw()
                    pipe.send_text("\x1b[17~")
                    await asyncio.sleep(0.05)
                    if scenario != "tiny":
                        self.assertTrue(terminal.swarm.focused)
                    if scenario == "shrink":
                        output.size = type(output.size)(rows=4, columns=48)
                        app._on_resize()
                    elif scenario == "grow":
                        pipe.send_text("long " * 220)
                        for _ in range(50):
                            await asyncio.sleep(0.01)
                            if len(terminal.current_text) > 1000:
                                break
                        self.assertGreater(len(terminal.current_text), 1000)
                    await redraw()
                    self.assertFalse(terminal.swarm.focused)
                    doc = terminal.current_document
                    pipe.send_text("\x1b[DX\n")
                    expected = doc.text[:doc.cursor_position - 1] + "X" + doc.text[doc.cursor_position - 1:]
                    self.assertEqual(await asyncio.wait_for(read, 2), expected)
                    read = asyncio.create_task(terminal.read("> "))
                    await asyncio.sleep(0)
                    pipe.send_text("\x1b[A\n")
                    self.assertEqual(await asyncio.wait_for(read, 2), expected)
                finally:
                    if not read.done():
                        read.cancel()
                    await asyncio.gather(read, return_exceptions=True)
                    terminal.close()

    async def test_root_adapter_recovery_updates_context_after_schema_removal(self):
        for mode in ("stream", "complete"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                config = make_config(root, agent={"max_model_requests": 4, "reasoning_effort": "high"})
                shell = InteractiveCLI(workspace=root, config=config, approval_callback=None)
                shell.runner = await shell._new_runner(config)
                runner = shell.runner
                adapter = OpenAICompatibleProvider(ProviderConfig(type="openai-compatible", api_style="chat", max_retries=0))
                runner.providers[:] = [ProviderRoute(name="mock", model="offline", provider=adapter)]
                shell.renderer._status_stream = io.StringIO()
                shell.renderer._text_stream = io.StringIO()
                wire, estimates, prepared = [], [], []
                def dispatch(event, estimates=estimates):
                    if event.type == "context.usage":
                        estimates.append(event.data["estimated_input_tokens"])
                runner.events.subscribe(dispatch)
                original = runner._emit_context_usage
                async def context(route, request, session_id, run_id, prepared=prepared, original=original):
                    prepared.append((request.system, list(request.messages), list(request.tools)))
                    await original(route, request, session_id, run_id)
                def answer(payload, wire=wire):
                    wire.append(json.loads(json.dumps(payload)))
                    index = len(wire)
                    if index == 1:
                        message = {"tool_calls": [{"id": "child", "type": "function", "function": {
                            "name": "delegate_task", "arguments": json.dumps({"task": "Synthetic child", "max_turns": 1, "capability": "read-only"})}}]}
                    else:
                        message = {"content": "" if index == 3 else "Synthetic complete"}
                    return {"choices": [{"message": message, "finish_reason": "tool_calls" if index == 1 else "stop"}],
                            "usage": {"prompt_tokens": 30, "completion_tokens": 10}}
                async def post_json(url, *, headers, payload, answer=answer):
                    return SimpleNamespace(data=answer(payload))
                async def stream_sse(url, *, headers, payload, answer=answer):
                    data = answer(payload)
                    for choice in data["choices"]:
                        choice["delta"] = choice.pop("message")
                        for index, call in enumerate(choice["delta"].get("tool_calls", [])):
                            call["index"] = index
                    yield SSEEvent(event="message", data=json.dumps(data))
                async def complete_stream(request, adapter=adapter):
                    async for event in Provider.stream(adapter, request):
                        yield event
                try:
                    with patch.object(adapter.http, "post_json", side_effect=post_json), patch.object(adapter.http, "stream_sse", side_effect=stream_sse), patch.object(runner, "_emit_context_usage", side_effect=context):
                        if mode == "complete":
                            with patch.object(adapter, "stream", side_effect=complete_stream):
                                await shell._submit("Synthetic delegation and recovery")
                        else:
                            await shell._submit("Synthetic delegation and recovery")
                    self.assertEqual(len(wire), 4)
                    self.assertTrue(wire[2]["tools"])
                    self.assertFalse(wire[3].get("tools"))
                    self.assertEqual(len(estimates), 3)
                    self.assertLess(estimates[-1], estimates[-2])
                    system, messages, tools = prepared[-1]
                    expected = ContextBudget.calculate(config.agent, system=system, messages=messages,
                                                       tools=tools, provider=adapter.name).estimated_total(messages)
                    self.assertEqual(shell.usage_view.context_tokens, expected)
                    self.assertEqual(estimates[-1], expected)
                finally:
                    await runner.close()

    async def test_settled_root_and_delegated_helpers_reach_hud_before_next_response(self):
        for delegated in (False, True):
            with self.subTest(delegated=delegated), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                (root / "source.txt").write_text("historical source " * 3500)
                (root / "other.txt").write_text("different source " * 3500)
                config = make_config(root, agent={"deterministic_compaction": False, "small_model": "offline-summary", "swarm_max_turns": 6},
                                     context={"compact_tool_output_tokens": 20000, "tool_output_chars": 100000})
                shell = InteractiveCLI(workspace=root, config=config, approval_callback=None)
                shell.runner = await shell._new_runner(config)
                runner = shell.runner
                shell.renderer._status_stream = io.StringIO()
                shell.renderer._text_stream = io.StringIO()
                provider = runner.providers[0].provider
                assert isinstance(provider, MockProvider)
                if not delegated:
                    session = runner.sessions.create_session(workspace=root, provider="mock", model="deterministic")
                    runner.sessions.append_message(session.id, Message(role=Role.USER, content="Synthetic requirements"))
                    for index in range(16):
                        call = ToolCall(id=f"old{index}", name="shell", arguments={"command": f"check-{index}"})
                        runner.sessions.append_message(session.id, Message(role=Role.ASSISTANT, tool_calls=[call]))
                        runner.sessions.append_message(session.id, Message(role=Role.TOOL, tool_name="shell", tool_call_id=call.id, content="x" * 24000))
                    await shell._resume(session.id, announce=False)
                paused, release = asyncio.Event(), asyncio.Event()
                summaries = []
                child_calls = []
                events = []
                runner.events.subscribe(events.append)
                async def complete(request, summaries=summaries, child_calls=child_calls, delegated=delegated, paused=paused, release=release):
                    if request.metadata.get("purpose") == "compaction_summary":
                        summaries.append(request)
                        return ModelResponse(text=echo_evidence(request.messages[0].content), usage=Usage(input_tokens=700, cached_input_tokens=350, output_tokens=70, requests=1, cost_status="known"))
                    if request.metadata.get("delegated"):
                        child_calls.append(request)
                        if len(child_calls) <= 2:
                            return ModelResponse(tool_calls=[ToolCall(id=f"read{len(child_calls)}", name="read_file", arguments={
                                "path": "source.txt" if len(child_calls) == 1 else "other.txt", "start_line": None, "end_line": None, "max_chars": 100000})],
                                                 usage=Usage(input_tokens=30, output_tokens=10, requests=1))
                    elif delegated and request.messages[-1].role != Role.TOOL:
                        return ModelResponse(tool_calls=[ToolCall(id="child", name="delegate_task", arguments={
                            "task": "Inspect both files", "max_turns": 6, "capability": "read-only"})], usage=Usage(input_tokens=30, output_tokens=10, requests=1))
                    if summaries and not paused.is_set():
                        paused.set()
                        await release.wait()
                    return ModelResponse(text="Synthetic complete", usage=Usage(input_tokens=30, output_tokens=10, requests=1))
                task = None
                try:
                    with patch.object(provider, "complete", side_effect=complete):
                        task = asyncio.create_task(shell._submit("Synthetic compaction"))
                        await asyncio.wait_for(paused.wait(), 10)
                        assert shell.session_id is not None
                        ledger = runner.sessions.usage(shell.session_id)
                        self.assertTrue(summaries)
                        self.assertGreaterEqual(ledger.total_tokens, 770)
                        self.assertEqual(shell.usage_view.usage.to_dict(), ledger.to_dict())
                        self.assertTrue(any(e.type == "usage.settled" and e.run_id == shell._root_run_id for e in events))
                        release.set()
                        await asyncio.wait_for(task, 5)
                        self.assertEqual(shell.usage_view.usage.to_dict(), runner.sessions.usage(shell.session_id).to_dict())
                finally:
                    release.set()
                    if task is not None and not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                    await runner.close()
