from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from borealis_coder.agent import build_runner, compact_messages, compact_messages_with_summary
from borealis_coder.config import ProviderConfig, SafetyConfig, SandboxConfig
from borealis_coder.errors import BudgetExceeded
from borealis_coder.models import Message, ModelResponse, Role, ToolCall, Usage
from borealis_coder.providers.base import Provider
from borealis_coder.providers.mock import MockProvider
from borealis_coder.providers.registry import ProviderRegistry
from borealis_coder.safety import (
    DockerProcessDriver,
    NativeProcessDriver,
    ProcessResult,
    WorkspaceRoots,
)
from borealis_coder.tools import build_builtin_registry
from borealis_coder.tools.verification import VerificationPlanner, VerificationStep
from tests.helpers import make_config, make_context


class SteeringProvider(Provider):
    name="steering"
    def __init__(self, config, api_key=""):
        super().__init__(config, api_key)
        self.calls=0
    async def complete(self, request):
        self.calls += 1
        if self.calls == 1:
            await asyncio.sleep(.15)
            return ModelResponse(tool_calls=[ToolCall(name="read_file", arguments={"path":"a.txt","start_line":None,"end_line":None,"max_chars":None})], usage=Usage(requests=1))
        steering = any(message.content == "new direction" for message in request.messages)
        return ModelResponse(text="steering seen" if steering else "missing", usage=Usage(requests=1))


class RuntimeFeatureTests(unittest.IsolatedAsyncioTestCase):
    async def test_steering_is_injected_before_next_turn(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/"a.txt").write_text("a")
            config=make_config(root, agent={"provider":"steering"})
            config.providers["steering"]=ProviderConfig(type="steering", model="s", max_retries=0)
            registry=ProviderRegistry()
            registry.register("steering", lambda cfg,key: SteeringProvider(cfg,key))
            runner=await build_runner(root, config=config, interactive=False, provider_registry=registry)
            try:
                task=asyncio.create_task(runner.run("start"))
                while not runner._cancel:
                    await asyncio.sleep(.01)
                session_id=next(iter(runner._cancel))
                runner.steer(session_id, "new direction")
                result=await task
                self.assertEqual(result.text, "steering seen")
            finally:
                await runner.close()

    async def test_shell_bounds_and_policy(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            context=make_context(root)
            registry=build_builtin_registry()
            ok=await registry.execute(ToolCall(name="shell", arguments={"command":"printf 123","cwd":".","timeout_seconds":10,"description":"test"}), context)
            self.assertFalse(ok.is_error, ok.output)
            self.assertIn("123", ok.output)
            denied=await registry.execute(ToolCall(name="shell", arguments={"command":"rm -rf x","cwd":".","timeout_seconds":10,"description":"bad"}), context)
            self.assertTrue(denied.is_error)

    def test_compaction_keeps_recent_context(self):
        messages=[Message(role=Role.USER if i%2==0 else Role.ASSISTANT, content=f"m{i}") for i in range(30)]
        compacted=compact_messages(messages, keep_recent=6)
        self.assertTrue(compacted[0].metadata["compacted"])
        self.assertEqual(compacted[0].metadata["strategy"], "deterministic")
        self.assertEqual([item.content for item in compacted[-6:]], [f"m{i}" for i in range(24,30)])

    async def test_compaction_llm_summary_preserves_tool_output(self):
        tool_output = "FAILED tests/test_x.py::test_y - AssertionError: expected 4 got 5"
        messages = [
            Message(role=Role.USER, content="run the tests"),
            Message(role=Role.ASSISTANT, content="", tool_calls=[ToolCall(name="shell", arguments={"command":"pytest"})]),
            Message(role=Role.TOOL, content=tool_output, tool_name="shell"),
            *[Message(role=Role.USER if i%2==0 else Role.ASSISTANT, content=f"m{i}") for i in range(20)],
        ]
        seen = {}

        async def summarizer(transcript):
            seen["transcript"] = transcript
            return "LLM summary: tests failed with AssertionError."

        compacted = await compact_messages_with_summary(messages, summarizer, keep_recent=6)
        self.assertEqual(compacted[0].metadata["strategy"], "llm")
        self.assertIn("AssertionError", seen["transcript"])
        self.assertIn("LLM summary", compacted[0].content)
        self.assertEqual([item.content for item in compacted[-6:]], [f"m{i}" for i in range(14,20)])

    async def test_compaction_llm_failure_falls_back_to_deterministic(self):
        messages=[Message(role=Role.USER if i%2==0 else Role.ASSISTANT, content=f"m{i}") for i in range(30)]

        async def boom(transcript):
            raise RuntimeError("provider offline")

        compacted = await compact_messages_with_summary(messages, boom, keep_recent=6)
        self.assertEqual(compacted[0].metadata["strategy"], "deterministic")
        self.assertTrue(compacted[0].metadata["compacted"])

    async def test_compaction_without_summarizer_is_deterministic(self):
        messages=[Message(role=Role.USER if i%2==0 else Role.ASSISTANT, content=f"m{i}") for i in range(30)]
        compacted = await compact_messages_with_summary(messages, None, keep_recent=6)
        self.assertEqual(compacted[0].metadata["strategy"], "deterministic")

    async def test_runner_honors_deterministic_compaction_setting(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runner = await build_runner(root, config=make_config(root), interactive=False)
            try:
                self.assertIsNone(runner._summarizer(AsyncMock()))
            finally:
                await runner.close()

    async def test_runner_accounts_for_llm_compaction_usage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = make_config(root, agent={"deterministic_compaction": False})
            runner = await build_runner(root, config=config, interactive=False)
            usage = Usage(input_tokens=100, output_tokens=20, requests=1, cost_usd=0.25)
            response = ModelResponse(text="accounted summary", usage=usage)
            provider = runner.providers[0].provider
            self.assertIsInstance(provider, MockProvider)
            assert isinstance(provider, MockProvider)
            provider.enqueue(response)
            usage_sink = AsyncMock()
            messages = [
                Message(role=Role.USER if i % 2 == 0 else Role.ASSISTANT, content=f"m{i}")
                for i in range(30)
            ]
            try:
                compacted = await compact_messages_with_summary(
                    messages,
                    runner._summarizer(usage_sink),
                    keep_recent=6,
                )
                self.assertEqual(compacted[0].content, "accounted summary")
                usage_sink.assert_awaited_once_with(usage)
            finally:
                await runner.close()

    async def test_compaction_does_not_swallow_budget_errors(self):
        messages = [
            Message(role=Role.USER if i % 2 == 0 else Role.ASSISTANT, content=f"m{i}")
            for i in range(30)
        ]

        async def over_budget(_transcript):
            raise BudgetExceeded("cost", "summary exceeded budget")

        with self.assertRaises(BudgetExceeded):
            await compact_messages_with_summary(messages, over_budget, keep_recent=6)

    async def test_shell_streams_output_events(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            context=make_context(root)
            registry=build_builtin_registry()
            events=[]
            original_emit = context.events.emit

            async def capture(event_type, **kwargs):
                events.append((event_type, kwargs))
                return await original_emit(event_type, **kwargs)

            with patch.object(context.events, "emit", side_effect=capture):
                result = await registry.execute(
                    ToolCall(name="shell", arguments={"command":"printf line1\\nline2\\n","cwd":".","timeout_seconds":10,"description":"stream"}),
                    context,
                )
            self.assertFalse(result.is_error, result.output)
            output_events = [item for item in events if item[0] == "tool.output"]
            self.assertTrue(output_events, "expected incremental tool.output events")
            combined = "".join(item[1]["text"] for item in output_events)
            self.assertIn("line1", combined)
            self.assertIn("line2", combined)
            for _, kwargs in output_events:
                self.assertIn(kwargs["stream"], {"stdout", "stderr"})

    async def test_process_stream_decodes_split_utf8(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            driver = NativeProcessDriver(
                WorkspaceRoots(root), SafetyConfig(), SandboxConfig()
            )
            code = (
                "import sys,time;"
                "sys.stdout.buffer.write(b'\\xe2');sys.stdout.buffer.flush();"
                "time.sleep(0.05);"
                "sys.stdout.buffer.write(b'\\x82\\xac');sys.stdout.buffer.flush()"
            )
            chunks = []
            result = await driver.run(
                [sys.executable, "-c", code],
                cwd=root,
                timeout=5,
                on_output=lambda _stream, text: chunks.append(text),
            )
            self.assertEqual(result.stdout, "€")
            self.assertEqual("".join(chunks), "€")

    async def test_process_bounds_streamed_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            driver = NativeProcessDriver(
                WorkspaceRoots(root),
                SafetyConfig(max_process_output_chars=10),
                SandboxConfig(),
            )
            chunks = []
            result = await driver.run(
                [sys.executable, "-c", "print('x' * 100, end='')"],
                cwd=root,
                timeout=5,
                on_output=lambda _stream, text: chunks.append(text),
            )
            streamed = "".join(chunks)
            self.assertEqual(streamed.count("x"), 10)
            self.assertEqual(streamed.count("output truncated"), 1)
            self.assertEqual(result.stdout, "x" * 10)

    @unittest.skipUnless(os.name == "posix", "SIGTERM output is POSIX-specific")
    async def test_process_drains_output_after_timeout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            driver = NativeProcessDriver(
                WorkspaceRoots(root), SafetyConfig(), SandboxConfig()
            )
            code = (
                "import signal,sys,time;"
                "signal.signal(signal.SIGTERM, lambda *_: "
                "(sys.stdout.write('final\\n'),sys.stdout.flush(),sys.exit(0)));"
                "sys.stdout.write('start\\n');sys.stdout.flush();time.sleep(10)"
            )
            chunks = []
            result = await driver.run(
                [sys.executable, "-c", code],
                cwd=root,
                timeout=1,
                on_output=lambda _stream, text: chunks.append(text),
            )
            self.assertTrue(result.timed_out)
            self.assertIn("final", result.stdout)
            self.assertIn("final", "".join(chunks))

    async def test_verification_commands_pass_through_policy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            context = make_context(root)
            report = await VerificationPlanner(root).run(
                context,
                [VerificationStep("Network check", "curl https://example.com", 10)],
            )
            self.assertFalse(report.ok)
            self.assertTrue(report.steps[0]["blocked"])

    async def test_docker_driver_mounts_additional_roots_and_hardens_container(self):
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as extra:
            root = Path(td)
            extra_root = Path(extra)
            subdir = extra_root / "pkg"
            subdir.mkdir()
            roots = WorkspaceRoots(root, [extra_root])
            safety = SafetyConfig(network=False)
            sandbox = SandboxConfig(driver="docker")
            fake_result = ProcessResult("docker", 0, "", "", 1)
            with patch("borealis_coder.safety.sandbox.shutil.which", return_value="/usr/bin/docker"), patch.object(
                NativeProcessDriver,
                "run",
                new=AsyncMock(return_value=fake_result),
            ) as native_run:
                driver = DockerProcessDriver(roots, safety, sandbox)
                result = await driver.run("python -V", cwd=subdir, timeout=10, shell=True)
            self.assertTrue(result.ok)
            assert native_run.await_args is not None
            argv = native_run.await_args.args[0]
            self.assertIn("--read-only", argv)
            self.assertIn("--cap-drop", argv)
            self.assertIn(f"{extra_root.resolve()}:/workspace_roots/root1:rw", argv)
            workdir_index = argv.index("-w") + 1
            self.assertEqual(argv[workdir_index], "/workspace_roots/root1/pkg")


if __name__ == "__main__":
    unittest.main()
