from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

from borealis_coder import cli, terminal
from borealis_coder.config import load_config
from borealis_coder.models import AgentResult, Event, StopReason, Usage
from borealis_coder.safety import ApprovalRequest, PolicyAction, PolicyDecision
from borealis_coder.safety.checkpoints import CheckpointManager
from borealis_coder.safety.paths import WorkspaceRoots


class TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


class CLITests(unittest.TestCase):
    def run_cli(
        self,
        argv: list[str],
        *,
        env: dict[str, str],
        stdin: str = "",
    ) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.dict(os.environ, env, clear=False), patch.object(
            cli.sys, "stdin", io.StringIO(stdin)
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_end_to_end_cli_surface(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            workspace = base / "workspace"
            data = base / "data"
            env = {
                "BOREALIS_DATA_DIR": str(data),
                "BOREALIS_PROVIDER": "mock",
                "BOREALIS_APPROVAL": "never",
            }

            code, out, _ = self.run_cli(["init", "--workspace", str(workspace)], env=env)
            self.assertEqual(code, 0)
            self.assertIn("created", out)
            code, out, _ = self.run_cli(["init", "--workspace", str(workspace)], env=env)
            self.assertEqual(code, 0)
            self.assertIn("kept", out)
            code, out, _ = self.run_cli(
                ["init", "--workspace", str(workspace), "--force"], env=env
            )
            self.assertIn("created", out)

            code, out, _ = self.run_cli(
                ["doctor", "--workspace", str(workspace), "--json"], env=env
            )
            self.assertEqual(code, 0)
            diagnostics = json.loads(out)
            self.assertTrue(any(item["name"] == "database" for item in diagnostics))
            code, out, _ = self.run_cli(
                ["doctor", "--workspace", str(workspace)], env=env
            )
            self.assertIn("PASS", out)

            code, out, _ = self.run_cli(
                ["config", "--workspace", str(workspace)], env=env
            )
            self.assertEqual(json.loads(out)["agent"]["provider"], "mock")

            code, out, _ = self.run_cli(
                ["tools", "--workspace", str(workspace), "--json"], env=env
            )
            self.assertEqual(code, 0)
            schemas = json.loads(out)
            self.assertTrue(any(item["name"] == "read_file" for item in schemas))
            code, out, _ = self.run_cli(
                ["tools", "--workspace", str(workspace)], env=env
            )
            self.assertIn("read_file", out)

            code, out, err = self.run_cli(
                [
                    "run",
                    "--workspace",
                    str(workspace),
                    "--provider",
                    "mock",
                    "--non-interactive",
                    "--no-verify",
                    "--json",
                    "OFFLINE_WRITE_DEMO",
                ],
                env=env,
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            session_id = payload["session_id"]
            self.assertEqual(payload["stop_reason"], "end_turn")
            self.assertTrue((workspace / "borealis-demo.txt").exists())
            self.assertIn('"event"', err)

            code, out, err = self.run_cli(
                [
                    "run",
                    "--workspace",
                    str(workspace),
                    "--provider",
                    "mock",
                    "--non-interactive",
                    "--no-verify",
                    "--resume",
                    session_id,
                ],
                env=env,
                stdin="Confirm the state",
            )
            self.assertEqual(code, 0, err)
            self.assertIn("Offline mock response", out)
            self.assertIn("session=", err)

            code, out, _ = self.run_cli(
                ["sessions", "list", "--workspace", str(workspace)], env=env
            )
            self.assertIn(session_id, out)
            code, out, _ = self.run_cli(
                [
                    "sessions",
                    "list",
                    "--workspace",
                    str(workspace),
                    "--all-workspaces",
                ],
                env=env,
            )
            self.assertIn(session_id, out)
            code, out, _ = self.run_cli(
                ["sessions", "show", session_id, "--workspace", str(workspace)], env=env
            )
            self.assertEqual(json.loads(out)["session"]["id"], session_id)
            export_path = base / "session.json"
            code, out, _ = self.run_cli(
                [
                    "sessions",
                    "export",
                    session_id,
                    "--workspace",
                    str(workspace),
                    "--output",
                    str(export_path),
                ],
                env=env,
            )
            self.assertEqual(code, 0)
            self.assertTrue(export_path.is_file())
            self.assertIn(str(export_path), out)

            config = load_config(
                workspace,
                overrides={"storage": {"directory": str(data)}},
            )
            checkpoints = CheckpointManager(
                WorkspaceRoots(workspace),
                enabled=True,
                max_bytes=config.safety.checkpoint_max_bytes,
            ).list()
            self.assertTrue(checkpoints)
            checkpoint_id = checkpoints[0].id
            (workspace / "borealis-demo.txt").write_text("changed", encoding="utf-8")
            code, out, _ = self.run_cli(
                ["rollback", "--workspace", str(workspace)], env=env
            )
            self.assertIn(checkpoint_id, out)
            code, out, _ = self.run_cli(
                ["rollback", checkpoint_id, "--workspace", str(workspace)], env=env
            )
            self.assertEqual(code, 0)
            self.assertIn("restored", out)

            code, out, _ = self.run_cli(
                ["sessions", "delete", session_id, "--workspace", str(workspace)], env=env
            )
            self.assertIn("deleted", out)

            code, out, _ = self.run_cli(["eval", "--json"], env=env)
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(out)["ok"])
            code, out, _ = self.run_cli(["eval"], env=env)
            self.assertIn("PASS", out)

    def test_chat_commands_and_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env = {
                "BOREALIS_DATA_DIR": str(root / "data"),
                "BOREALIS_PROVIDER": "mock",
                "BOREALIS_APPROVAL": "never",
            }
            stdout = io.StringIO()
            stderr = io.StringIO()
            inputs = [
                "",
                "/help",
                "/session",
                "/clear",
                "hello",
                "/history 4",
                "/status",
                "/sessions",
                "/resume does-not-exist",
                "/mode plan",
                "/mode workspace-write",
                "/network on",
                "/network off",
                "/paste",
                "first line",
                "second line",
                "/end",
                "/quit",
            ]
            with patch.dict(os.environ, env, clear=False), patch(
                "builtins.input", side_effect=inputs
            ), redirect_stdout(stdout), redirect_stderr(stderr):
                code = cli.main(
                    [
                        "chat",
                        "--workspace",
                        str(root),
                        "--provider",
                        "mock",
                        "--no-verify",
                        "--no-history",
                    ]
                )
            self.assertEqual(code, 0)
            output = stdout.getvalue()
            self.assertIn("interactive mode", output)
            self.assertIn("/session", output)
            self.assertIn("(new conversation)", output)
            self.assertIn("fresh conversation", output)
            self.assertGreaterEqual(output.count("Offline mock response"), 2)
            self.assertIn("PROVIDER    mock", output)
            self.assertIn("Runtime updated  mode=plan", output)
            self.assertIn("Paste a multiline prompt", output)
            self.assertIn("Unknown session", output)
            self.assertIn("USER\nhello", output)

    def test_bare_command_defaults_to_interactive_and_continue(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env = {
                "BOREALIS_DATA_DIR": str(root / "data"),
                "BOREALIS_PROVIDER": "mock",
                "BOREALIS_APPROVAL": "never",
            }
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch.dict(os.environ, env, clear=False), patch(
                "builtins.input", side_effect=["/quit"]
            ), redirect_stdout(stdout), redirect_stderr(stderr):
                code = cli.main(
                    [
                        "hello from the bare command",
                        "--workspace",
                        str(root),
                        "--provider",
                        "mock",
                        "--no-verify",
                        "--no-history",
                    ]
                )
            self.assertEqual(code, 0)
            first_output = stdout.getvalue()
            self.assertIn("interactive mode", first_output)
            self.assertIn("Offline mock response", first_output)

            config = load_config(
                root,
                overrides={
                    "agent": {"provider": "mock"},
                    "storage": {"directory": str(root / "data")},
                },
            )
            from borealis_coder.sessions import SessionStore

            store = SessionStore(config.database_path)
            try:
                session_id = store.list_sessions(workspace=root, limit=1)[0].id
            finally:
                store.close()

            stdout = io.StringIO()
            with patch.dict(os.environ, env, clear=False), patch(
                "builtins.input", side_effect=["/session", "/quit"]
            ), redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                code = cli.main(
                    [
                        "--workspace",
                        str(root),
                        "--provider",
                        "mock",
                        "--continue",
                        "--no-verify",
                        "--no-history",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertIn(session_id, stdout.getvalue())
            self.assertIn("(resumed)", stdout.getvalue())

            stdout = io.StringIO()
            with patch.dict(os.environ, env, clear=False), patch(
                "builtins.input", side_effect=["/quit"]
            ), redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                code = cli.main(
                    [
                        "resume",
                        "--last",
                        "--workspace",
                        str(root),
                        "--provider",
                        "mock",
                        "--no-verify",
                        "--no-history",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertIn(session_id, stdout.getvalue())
            self.assertIn("(resumed)", stdout.getvalue())

    def test_acp_dispatch_errors_and_version(self) -> None:
        server = SimpleNamespace(serve=AsyncMock())
        with patch("borealis_coder.cli.ACPServer", return_value=server):
            self.assertEqual(cli.main(["acp"]), 0)
        server.serve.assert_awaited_once()

        code, _, err = self.run_cli(["run"], env={}, stdin="")
        self.assertEqual(code, 2)
        self.assertIn("A prompt is required", err)

        with patch("borealis_coder.cli.asyncio.run", side_effect=KeyboardInterrupt):
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(cli.main(["eval"]), 130)
            self.assertIn("Cancelled", stderr.getvalue())

        with self.assertRaises(SystemExit) as version, redirect_stdout(io.StringIO()):
            cli.build_parser().parse_args(["--version"])
        self.assertEqual(version.exception.code, 0)

    def test_renderer_approval_runtime_config_and_footer(self) -> None:
        async def render() -> tuple[str, str]:
            stdout = io.StringIO()
            stderr = io.StringIO()
            renderer = cli.ConsoleRenderer()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                await renderer.handle(Event(type="run.started"))
                await renderer.handle(
                    Event(
                        type="model.started",
                        data={"provider": "chatgpt", "model": "test-model", "turn": 1},
                    )
                )
                await renderer.handle(
                    Event(type="model.tool_call_delta", data={"name": "read_file"})
                )
                await renderer.handle(Event(type="model.tool_call_delta"))
                await renderer.handle(Event(type="model.text_delta", data={"text": "a"}))
                await renderer.handle(Event(type="model.completed", data={"text": "a"}))
                await renderer.handle(Event(type="tool.started", data={"tool": "read"}))
                await renderer.handle(
                    Event(
                        type="tool.completed",
                        data={"tool": "read", "is_error": False, "metadata": {"duration_ms": 2}},
                    )
                )
                await renderer.handle(
                    Event(
                        type="tool.completed",
                        data={"tool": "write", "is_error": True, "metadata": {}},
                    )
                )
                await renderer.handle(Event(type="context.compacted"))
                await renderer.handle(
                    Event(type="model.route_failed", data={"provider": "x"})
                )
                await renderer.handle(
                    Event(
                        type="model.retrying",
                        data={"attempt": 2, "max_attempts": 5, "delay_seconds": 1.0},
                    )
                )
                await renderer.handle(Event(type="verification.started"))
                await renderer.handle(Event(type="model.started"))
                await renderer.handle(Event(type="model.completed", data={"text": "final"}))
                renderer.heartbeat(10)
                renderer.finish_turn()
                buffered = cli.ConsoleRenderer(stream_text=False)
                await buffered.handle(Event(type="model.started"))
                await buffered.handle(Event(type="model.text_delta", data={"text": "buffered"}))
                await buffered.handle(Event(type="model.completed", data={"text": "buffered"}))
                quiet = cli.ConsoleRenderer(quiet=True)
                await quiet.handle(Event(type="model.text_delta", data={"text": "hidden"}))
                json_renderer = cli.ConsoleRenderer(json_events=True)
                await json_renderer.handle(Event(type="custom", data={"x": 1}))
            return stdout.getvalue(), stderr.getvalue()

        out, err = __import__("asyncio").run(render())
        self.assertIn("buffered", out)
        self.assertIn("final", out)
        self.assertTrue(out.endswith("\n"))
        self.assertIn('"event"', err)
        self.assertIn("→ read", err)
        self.assertIn("✓ read", err)
        self.assertIn("✗ write", err)
        self.assertIn("compacted", err)
        self.assertIn("route failed", err)
        self.assertIn("preparing workspace context", err)
        self.assertIn("model working · chatgpt/test-model · turn 1", err)
        self.assertEqual(err.count("preparing tool call"), 1)
        self.assertIn("preparing tool call · read_file", err)
        self.assertIn("provider retry 2/5 in 1s", err)
        self.assertIn("running verification", err)
        self.assertIn("still working · processing model response · 10s elapsed", err)

        request = ApprovalRequest(
            tool_name="shell",
            description="run",
            decision=PolicyDecision(PolicyAction.ASK, "reason", "high"),
            arguments_preview="echo hi",
        )
        with patch("builtins.input", side_effect=["bad", "y"]), redirect_stderr(io.StringIO()):
            self.assertEqual(cli._terminal_approval(request), "allow_once")
        with patch("builtins.input", return_value="a"), redirect_stderr(io.StringIO()):
            self.assertEqual(cli._terminal_approval(request), "allow_always")
        with patch("builtins.input", return_value=""), redirect_stderr(io.StringIO()):
            self.assertEqual(cli._terminal_approval(request), "no")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = cli.build_parser().parse_args(
                [
                    "run",
                    "--workspace",
                    str(root),
                    "--provider",
                    "mock",
                    "--model",
                    "deterministic",
                    "--fallback",
                    "other",
                    "--mode",
                    "full",
                    "--approval",
                    "never",
                    "--network",
                    "--sandbox",
                    "native",
                    "--verify",
                    "--max-turns",
                    "3",
                    "--max-cost",
                    "1.5",
                    "hello",
                ]
            )
            with patch.dict(
                os.environ,
                {"BOREALIS_DATA_DIR": str(root / "data")},
                clear=False,
            ):
                config = cli._runtime_config(args, root)
            self.assertEqual(config.agent.max_turns, 3)
            self.assertEqual(config.agent.provider_fallbacks, ["other"])
            self.assertTrue(config.safety.network)
            self.assertEqual(config.safety.mode, "full")

        result = AgentResult(
            session_id="s",
            run_id="r",
            text="",
            stop_reason=StopReason.ERROR,
            usage=Usage(input_tokens=1, output_tokens=2, cost_usd=0.5),
            turns=4,
            changed_files=["a"],
            verification={"ok": False},
            error="boom",
        )
        footer = cli._result_footer(result)
        self.assertIn("changed=1", footer)
        self.assertIn("verified=False", footer)
        self.assertIn("error=boom", footer)

    def test_renderer_streams_bounded_tool_output_without_repeating_it(self) -> None:
        async def render() -> str:
            stderr = io.StringIO()
            renderer = cli.ConsoleRenderer(
                show_tool_output=True,
                tool_output_chars=200,
                status_stream=stderr,
            )
            await renderer.handle(
                Event(
                    type="tool.started",
                    data={"tool": "shell", "tool_call_id": "call_1", "arguments": {}},
                )
            )
            await renderer.handle(
                Event(
                    type="tool.output",
                    data={"tool": "shell", "tool_call_id": "call_1", "text": "x" * 250},
                )
            )
            await renderer.handle(
                Event(
                    type="tool.completed",
                    data={
                        "tool": "shell",
                        "tool_call_id": "call_1",
                        "output": "x" * 250,
                        "is_error": False,
                        "metadata": {"duration_ms": 2},
                    },
                )
            )
            return stderr.getvalue()

        output = __import__("asyncio").run(render())
        self.assertEqual(output.count("x"), 200)
        self.assertEqual(output.count("output truncated"), 1)
        self.assertLess(output.index("x"), output.index("✓ shell"))

    def test_renderer_keeps_tool_heartbeat_after_streamed_output(self) -> None:
        async def render() -> str:
            stderr = io.StringIO()
            renderer = cli.ConsoleRenderer(show_tool_output=True, status_stream=stderr)
            await renderer.handle(
                Event(
                    type="tool.started",
                    data={"tool": "shell", "tool_call_id": "call_1", "arguments": {}},
                )
            )
            await renderer.handle(
                Event(
                    type="tool.output",
                    data={
                        "tool": "shell",
                        "tool_call_id": "call_1",
                        "stream": "stdout",
                        "text": "building\n",
                    },
                )
            )
            renderer.heartbeat(10)
            return stderr.getvalue()

        output = __import__("asyncio").run(render())
        self.assertIn("still working · running shell · 10s elapsed", output)

    def test_renderer_preserves_failure_details_after_streaming_output(self) -> None:
        async def render() -> str:
            stderr = io.StringIO()
            renderer = cli.ConsoleRenderer(show_tool_output=True, status_stream=stderr)
            await renderer.handle(
                Event(
                    type="tool.output",
                    data={
                        "tool": "shell",
                        "tool_call_id": "call_1",
                        "stream": "stdout",
                        "text": "build",
                    },
                )
            )
            await renderer.handle(
                Event(
                    type="tool.completed",
                    data={
                        "tool": "shell",
                        "tool_call_id": "call_1",
                        "output": (
                            "exit_code=2\nstdout:\nbuilding\n"
                            "stderr:\nFATAL: failed"
                        ),
                        "is_error": True,
                        "metadata": {"duration_ms": 2, "exit_code": 2},
                    },
                )
            )
            return stderr.getvalue()

        output = __import__("asyncio").run(render())
        self.assertEqual(output.count("build"), 1)
        self.assertNotIn("stdout:\n  building", output)
        self.assertIn("stdout:\n    ing", output)
        self.assertIn("exit_code=2", output)
        self.assertIn("FATAL: failed", output)

    def test_renderer_preserves_unseen_tail_after_stream_truncation(self) -> None:
        async def render() -> str:
            stderr = io.StringIO()
            renderer = cli.ConsoleRenderer(
                show_tool_output=True,
                tool_output_chars=200,
                status_stream=stderr,
            )
            streamed = "H" * 250
            await renderer.handle(
                Event(
                    type="tool.output",
                    data={"tool": "shell", "tool_call_id": "call_1", "text": streamed},
                )
            )
            await renderer.handle(
                Event(
                    type="tool.completed",
                    data={
                        "tool": "shell",
                        "tool_call_id": "call_1",
                        "output": f"exit_code=0\nstdout:\n{streamed}IMPORTANT_TAIL",
                        "is_error": False,
                        "metadata": {
                            "duration_ms": 2,
                            "stream_truncated": True,
                            "stream_complete": True,
                        },
                    },
                )
            )
            return stderr.getvalue()

        output = __import__("asyncio").run(render())
        self.assertIn("output truncated", output)
        self.assertIn("final output tail", output)
        self.assertIn("IMPORTANT_TAIL", output)

    def test_renderer_ends_bare_carriage_return_before_completion(self) -> None:
        async def render() -> str:
            stderr = io.StringIO()
            renderer = cli.ConsoleRenderer(show_tool_output=True, status_stream=stderr)
            await renderer.handle(
                Event(
                    type="tool.output",
                    data={"tool": "shell", "tool_call_id": "call_1", "text": "progress 100%\r"},
                )
            )
            await renderer.handle(
                Event(
                    type="tool.completed",
                    data={
                        "tool": "shell",
                        "tool_call_id": "call_1",
                        "output": "exit_code=0",
                        "is_error": False,
                        "metadata": {"duration_ms": 1},
                    },
                )
            )
            return stderr.getvalue()

        output = __import__("asyncio").run(render())
        self.assertIn("progress 100%\r\n✓ shell", output)

    def test_aurora_ui_and_live_interactive_renderer(self) -> None:
        async def render() -> str:
            stream = TTYBuffer()
            renderer = cli.ConsoleRenderer(
                interactive=True,
                text_stream=stream,
                status_stream=stream,
            )
            renderer.ui.color = True
            await renderer.handle(Event(type="run.started"))
            renderer.pulse(0.4, 1)
            await renderer.handle(
                Event(
                    type="model.started",
                    data={"provider": "mock", "model": "aurora", "turn": 1},
                )
            )
            renderer.pulse(0.8, 2)
            await renderer.handle(Event(type="model.text_delta", data={"text": "hello"}))
            await renderer.handle(Event(type="model.completed", data={"text": "hello"}))
            await renderer.handle(
                Event(type="tool.started", data={"tool": "read_file", "arguments": {}})
            )
            await renderer.handle(
                Event(
                    type="tool.completed",
                    data={
                        "tool": "read_file",
                        "is_error": False,
                        "metadata": {"duration_ms": 4},
                    },
                )
            )
            await renderer.handle(
                Event(
                    type="plan.updated",
                    data={
                        "items": [
                            {"status": "completed", "content": "Map the interface"},
                            {"status": "in_progress", "content": "Polish the shell"},
                        ]
                    },
                )
            )
            renderer.finish_turn()
            return stream.getvalue()

        output = __import__("asyncio").run(render())
        self.assertIn("\033[", output)
        self.assertIn("\r\033[2K", output)
        self.assertIn("✦ BOREALIS", output)
        self.assertIn("Tool complete", output)
        self.assertIn("PLAN CONSTELLATION", output)

        stream = TTYBuffer()
        ui = cli.AuroraUI(stream, color=True)
        prompt = ui.prompt("abc123")
        self.assertIn("\001\033[", prompt)
        self.assertIn("session abc123", prompt)

        plain = io.StringIO()
        cli.AuroraUI(plain, color=False).banner(
            version="v1",
            workspace=Path("/tmp/example"),
            route="mock/aurora",
            safety="workspace-write · approval on-risk · network off",
            session="new conversation",
        )
        banner = plain.getvalue()
        self.assertNotIn("\033[", banner)
        self.assertIn("AURORA SHELL", banner)
        self.assertIn("interactive mode", banner)

        selector = io.StringIO()
        cli.AuroraUI(selector, color=False).command_selector(
            [("/status", "Inspect runtime health"), ("/sessions", "Browse sessions")],
            query="/s",
            hidden=3,
        )
        selector_output = selector.getvalue()
        self.assertIn("COMMAND DECK", selector_output)
        self.assertIn("/status", selector_output)
        self.assertIn("3 more", selector_output)
        self.assertIn("Tab completes", selector_output)

        themed_selector = io.StringIO()
        cli.AuroraUI(themed_selector, color=True).command_selector(
            [("/status", "Inspect runtime health")],
            query="/s",
        )
        themed_selector_output = themed_selector.getvalue()
        self.assertIn("\033[39;1m/status", themed_selector_output)
        self.assertIn("\033[90mInspect runtime health", themed_selector_output)
        self.assertNotIn("38;2;232;244;255", themed_selector_output)

        completion_output = io.StringIO()
        history = terminal.ReadlineHistory(
            Path("/tmp/history"),
            completions=("/status", "/sessions", "/help"),
            completion_descriptions={
                "/status": "Inspect runtime health",
                "/sessions": "Browse sessions",
                "/help": "Show help",
            },
        )
        history._selector_bound = True
        redisplay = Mock()
        cast(Any, history)._readline = SimpleNamespace(
            get_line_buffer=lambda: "/s",
            redisplay=redisplay,
        )
        with redirect_stdout(completion_output):
            self.assertEqual(history._complete("/s", 0), "/status")
        self.assertIn("COMMAND DECK", completion_output.getvalue())
        redisplay.assert_called_once()

        libedit = SimpleNamespace(
            parse_and_bind=Mock(),
            __doc__="libedit readline",
        )
        terminal._bind_slash_selector(libedit)
        binding = " ".join(
            call.args[0] for call in libedit.parse_and_bind.call_args_list
        )
        self.assertIn("^V/", binding)
        self.assertIn("\\t", binding)

        with patch.dict(os.environ, {"NO_COLOR": "1"}, clear=False):
            self.assertFalse(cli.AuroraUI(TTYBuffer()).color)

        narrow = TTYBuffer()
        with patch(
            "borealis_coder.terminal.shutil.get_terminal_size",
            return_value=os.terminal_size((40, 24)),
        ):
            cli.AuroraUI(narrow, color=False).banner(
                version="v1",
                workspace=Path("/a/very/long/workspace/path/that/must/remain/readable"),
                route="provider/a-very-long-model-name",
                safety="workspace-write · approval on-risk · network off",
                session="sess_abcdefghijklmnopqrstuvwxyz (resumed)",
            )
        self.assertTrue(all(len(line) <= 40 for line in narrow.getvalue().splitlines()))

    def test_interactive_turn_cancellation_is_graceful_then_forced(self) -> None:
        class FakeTask:
            def __init__(self) -> None:
                self.cancelled = False

            def done(self) -> bool:
                return False

            def cancel(self) -> None:
                self.cancelled = True

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = load_config(
                root,
                overrides={
                    "agent": {"provider": "mock"},
                    "storage": {"directory": str(root / "data")},
                },
            )
            shell = cli.InteractiveCLI(
                workspace=root,
                config=config,
                approval_callback=None,
                history_enabled=False,
            )
            cancel = Mock(return_value=True)
            shell.runner = cast(Any, SimpleNamespace(cancel=cancel))
            shell._active_session_id = "sess_test"
            task = FakeTask()
            shell._active_task = task  # type: ignore[assignment]
            output = io.StringIO()
            shell.renderer._status_stream = output

            shell._cancel_active_turn()
            cancel.assert_called_once_with("sess_test")
            self.assertFalse(task.cancelled)

            shell._cancel_active_turn()
            self.assertTrue(task.cancelled)
            self.assertEqual(output.getvalue().count("Cancelling current turn"), 2)

    def test_interactive_prompt_keyboard_interrupt_exits_130(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env = {
                "BOREALIS_DATA_DIR": str(root / "data"),
                "BOREALIS_PROVIDER": "mock",
                "BOREALIS_APPROVAL": "never",
            }
            with patch.dict(os.environ, env, clear=False), patch(
                "builtins.input", side_effect=KeyboardInterrupt
            ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = cli.main(
                    [
                        "chat",
                        "--workspace",
                        str(root),
                        "--provider",
                        "mock",
                        "--no-verify",
                        "--no-history",
                    ]
                )
            self.assertEqual(code, 130)


if __name__ == "__main__":
    unittest.main()
