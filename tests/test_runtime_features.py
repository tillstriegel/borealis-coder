from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from borealis_coder.agent import build_runner, compact_messages
from borealis_coder.config import ProviderConfig, SafetyConfig, SandboxConfig
from borealis_coder.models import Message, ModelResponse, Role, ToolCall, Usage
from borealis_coder.providers.base import Provider
from borealis_coder.providers.registry import ProviderRegistry
from borealis_coder.safety import DockerProcessDriver, NativeProcessDriver, ProcessResult, WorkspaceRoots
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
        self.assertEqual([item.content for item in compacted[-6:]], [f"m{i}" for i in range(24,30)])

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
            argv = native_run.await_args.args[0]
            self.assertIn("--read-only", argv)
            self.assertIn("--cap-drop", argv)
            self.assertIn(f"{extra_root.resolve()}:/workspace_roots/root1:rw", argv)
            workdir_index = argv.index("-w") + 1
            self.assertEqual(argv[workdir_index], "/workspace_roots/root1/pkg")


if __name__ == "__main__":
    unittest.main()
