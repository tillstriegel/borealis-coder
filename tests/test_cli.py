from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

from prompt_toolkit.document import Document
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from borealis_coder import cli, interactive, terminal
from borealis_coder.config import load_config
from borealis_coder.models import AgentResult, Event, StopReason, Usage
from borealis_coder.safety import ApprovalRequest, PolicyAction, PolicyDecision
from borealis_coder.safety.checkpoints import CheckpointManager
from borealis_coder.safety.paths import WorkspaceRoots
from borealis_coder.terminal_input import (
    CompatibleFileHistory,
    TerminalInput,
    TerminalInputInterrupted,
    read_history_entries,
)


class TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


class ControlledInput:
    def __init__(self) -> None:
        self.requests: asyncio.Queue[
            tuple[str, str | Document, asyncio.Future[str]]
        ] = asyncio.Queue()
        self.reading = False
        self.current_text = ""
        self.cursor_position = 0

    @property
    def current_document(self) -> Document:
        return Document(self.current_text, self.cursor_position)

    async def read(self, prompt: str, *, default: str | Document = "") -> str:
        self.reading = True
        document = default if isinstance(default, Document) else Document(default)
        self.current_text = document.text
        self.cursor_position = document.cursor_position
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self.requests.put_nowait((prompt, default, future))
        try:
            return await future
        finally:
            self.reading = False

    def close(self) -> None:
        return None


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
        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(cli.sys, "stdin", io.StringIO(stdin)),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
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
            code, out, _ = self.run_cli(["init", "--workspace", str(workspace), "--force"], env=env)
            self.assertIn("created", out)

            code, out, _ = self.run_cli(
                ["doctor", "--workspace", str(workspace), "--json"], env=env
            )
            self.assertEqual(code, 0)
            diagnostics = json.loads(out)
            self.assertTrue(any(item["name"] == "database" for item in diagnostics))
            code, out, _ = self.run_cli(["doctor", "--workspace", str(workspace)], env=env)
            self.assertIn("PASS", out)

            code, out, _ = self.run_cli(["config", "--workspace", str(workspace)], env=env)
            self.assertEqual(json.loads(out)["agent"]["provider"], "mock")

            code, out, _ = self.run_cli(["tools", "--workspace", str(workspace), "--json"], env=env)
            self.assertEqual(code, 0)
            schemas = json.loads(out)
            self.assertTrue(any(item["name"] == "read_file" for item in schemas))
            code, out, _ = self.run_cli(["tools", "--workspace", str(workspace)], env=env)
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
            code, out, _ = self.run_cli(["rollback", "--workspace", str(workspace)], env=env)
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

    def test_json_run_reports_recoverable_max_turns(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace = root / "workspace"
            workspace.mkdir()
            env = {
                "BOREALIS_DATA_DIR": str(root / "data"),
                "BOREALIS_PROVIDER": "mock",
                "BOREALIS_APPROVAL": "never",
            }

            code, out, _ = self.run_cli(
                [
                    "run",
                    "--workspace",
                    str(workspace),
                    "--provider",
                    "mock",
                    "--non-interactive",
                    "--no-verify",
                    "--max-turns",
                    "1",
                    "--json",
                    "OFFLINE_WRITE_DEMO",
                ],
                env=env,
            )

            self.assertEqual(code, 1)
            payload = json.loads(out)
            self.assertEqual(payload["stop_reason"], "max_turns")
            self.assertTrue(payload["incomplete"])
            self.assertIn("send 'continue' to resume", payload["error"])
            self.assertFalse((workspace / "borealis-demo.txt").exists())

    def test_session_administration_does_not_build_a_provider_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "data"
            env = {"BOREALIS_DATA_DIR": str(data)}
            config = load_config(root, overrides={"storage": {"directory": str(data)}})
            from borealis_coder.sessions import SessionStore

            store = SessionStore(config.database_path)
            try:
                session = store.create_session(
                    workspace=root,
                    provider="offline",
                    model="stored-model",
                    title="Stored session",
                )
            finally:
                store.close()

            with patch(
                "borealis_coder.cli.build_runner",
                side_effect=AssertionError("runtime must not start"),
            ):
                code, output, error = self.run_cli(
                    ["sessions", "list", "--workspace", str(root)],
                    env=env,
                )
                show_code, shown, show_error = self.run_cli(
                    ["sessions", "show", session.id, "--workspace", str(root)],
                    env=env,
                )
                export_path = root / "export.json"
                export_code, _, export_error = self.run_cli(
                    [
                        "sessions",
                        "export",
                        session.id,
                        "--workspace",
                        str(root),
                        "--output",
                        str(export_path),
                    ],
                    env=env,
                )
                delete_code, _, delete_error = self.run_cli(
                    ["sessions", "delete", session.id, "--workspace", str(root)],
                    env=env,
                )
            self.assertEqual(code, 0, error)
            self.assertIn(session.id, output)
            self.assertEqual(show_code, 0, show_error)
            self.assertEqual(json.loads(shown)["session"]["id"], session.id)
            self.assertEqual(export_code, 0, export_error)
            self.assertTrue(export_path.is_file())
            self.assertEqual(delete_code, 0, delete_error)

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
            with (
                patch.dict(os.environ, env, clear=False),
                patch("builtins.input", side_effect=inputs),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
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
            with (
                patch.dict(os.environ, env, clear=False),
                patch("builtins.input", side_effect=["/quit"]),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
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
            with (
                patch.dict(os.environ, env, clear=False),
                patch("builtins.input", side_effect=["/session", "/quit"]),
                redirect_stdout(stdout),
                redirect_stderr(io.StringIO()),
            ):
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
            with (
                patch.dict(os.environ, env, clear=False),
                patch("builtins.input", side_effect=["/quit"]),
                redirect_stdout(stdout),
                redirect_stderr(io.StringIO()),
            ):
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
                await renderer.handle(
                    Event(type="model.reasoning_delta", data={"text": "Checked "})
                )
                await renderer.handle(
                    Event(type="model.reasoning_delta", data={"text": "the plan."})
                )
                await renderer.handle(Event(type="model.text_delta", data={"text": "a"}))
                await renderer.handle(
                    Event(
                        type="model.completed",
                        data={"text": "a", "reasoning_summary": "Checked the plan."},
                    )
                )
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
                await renderer.handle(Event(type="model.route_failed", data={"provider": "x"}))
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
                await buffered.handle(
                    Event(type="model.reasoning_delta", data={"text": "Buffered reasoning"})
                )
                await buffered.handle(Event(type="model.text_delta", data={"text": "buffered"}))
                await buffered.handle(
                    Event(
                        type="model.completed",
                        data={
                            "text": "buffered",
                            "reasoning_summary": "Buffered reasoning",
                        },
                    )
                )
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
        self.assertEqual(err.count("◇ reasoning summary"), 2)
        self.assertEqual(err.count("Checked the plan."), 1)
        self.assertEqual(err.count("Buffered reasoning"), 1)

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
            mutation_tracking="incomplete",
            verification={"ok": False},
            error="boom",
            incomplete=True,
        )
        footer = cli._result_footer(result)
        self.assertIn("changed=1", footer)
        self.assertIn("mutation_tracking=incomplete", footer)
        self.assertIn("incomplete=true", footer)
        self.assertIn("verified=False", footer)
        self.assertIn("error=boom", footer)
        self.assertIn("incomplete · session preserved", interactive._turn_footer(result))
        self.assertIn("mutation tracking incomplete", interactive._turn_footer(result))
        self.assertTrue(result.to_dict()["incomplete"])
        self.assertEqual(result.to_dict()["mutation_tracking"], "incomplete")

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
                        "output": ("exit_code=2\nstdout:\nbuilding\nstderr:\nFATAL: failed"),
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

    def test_live_renderer_keeps_reasoning_summary_after_progress_pulse(self) -> None:
        async def render() -> str:
            stream = TTYBuffer()
            renderer = cli.ConsoleRenderer(
                interactive=True,
                text_stream=stream,
                status_stream=stream,
            )
            renderer.ui.color = True
            await renderer.handle(Event(type="model.started"))
            renderer.pulse(0.4, 1)
            await renderer.handle(
                Event(type="model.reasoning_delta", data={"text": "First summary"})
            )
            renderer.pulse(0.8, 2)
            await renderer.handle(
                Event(type="model.reasoning_delta", data={"text": "\nSecond summary"})
            )
            renderer.pulse(1.2, 3)
            await renderer.handle(Event(type="model.text_delta", data={"text": "answer"}))
            return stream.getvalue()

        output = __import__("asyncio").run(render())
        summary_start = output.index("First summary")
        self.assertNotIn("\r\033[2K", output[summary_start:])
        self.assertIn("First summary\nSecond summary\n", terminal._strip_ansi(output))

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
        binding = " ".join(call.args[0] for call in libedit.parse_and_bind.call_args_list)
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

    def test_interactive_follow_up_is_queued_until_the_session_starts(self) -> None:
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
            runner = SimpleNamespace(
                accepts_steering=Mock(side_effect=lambda session_id: session_id == "sess_test"),
                queued_prompts=Mock(return_value=1),
                steer=Mock(),
            )
            shell.runner = cast(Any, runner)
            shell.renderer._status_stream = io.StringIO()

            shell._queue_follow_up("change direction")
            self.assertEqual(shell._pending_follow_ups, ["change direction"])
            runner.steer.assert_not_called()

            __import__("asyncio").run(
                shell._observe(Event(type="run.started", session_id="sess_test"))
            )
            runner.steer.assert_called_once_with(
                "sess_test",
                "change direction",
                metadata={"interactive": True},
            )
            self.assertEqual(shell._pending_follow_ups, [])

    def test_interactive_active_turn_follow_up_steers_immediately(self) -> None:
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
            runner = SimpleNamespace(
                accepts_steering=Mock(return_value=True),
                queued_prompts=Mock(return_value=1),
                steer=Mock(),
            )
            shell.runner = cast(Any, runner)
            shell._active_session_id = "sess_test"
            output = io.StringIO()
            shell.renderer._status_stream = output

            shell._queue_follow_up("change direction")

            runner.steer.assert_called_once_with(
                "sess_test",
                "change direction",
                metadata={"interactive": True},
            )
            self.assertEqual(shell._pending_follow_ups, [])
            self.assertIn("1 pending", output.getvalue())

    def test_interactive_uses_status_terminal_for_concurrent_input(self) -> None:
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
            shell.renderer._status_stream = TTYBuffer()
            with patch("borealis_coder.interactive.sys.stdin", TTYBuffer()):
                self.assertTrue(shell._concurrent_input())
            shell.renderer._status_stream = io.StringIO()
            with patch("borealis_coder.interactive.sys.stdin", TTYBuffer()):
                self.assertFalse(shell._concurrent_input())

    def test_terminal_input_preserves_cursor_position_when_editing(self) -> None:
        async def exercise() -> str:
            with tempfile.TemporaryDirectory() as td, create_pipe_input() as pipe:
                worker = TerminalInput(
                    Path(td) / "history",
                    history_enabled=False,
                    completions=(),
                    completion_descriptions={},
                    input=pipe,
                    output=DummyOutput(),
                )
                task = __import__("asyncio").create_task(
                    worker.read("\001\033[90m\002follow-up> \001\033[0m\002", default="abc")
                )
                await __import__("asyncio").sleep(0)
                pipe.send_text("\x1b[DX\n")
                return await task

        self.assertEqual(__import__("asyncio").run(exercise()), "abXc")

    def test_terminal_input_translates_ctrl_c_for_the_coordinator(self) -> None:
        async def exercise() -> None:
            with tempfile.TemporaryDirectory() as td, create_pipe_input() as pipe:
                terminal_input = TerminalInput(
                    Path(td) / "history",
                    history_enabled=False,
                    completions=(),
                    completion_descriptions={},
                    input=pipe,
                    output=DummyOutput(),
                )
                task = asyncio.create_task(terminal_input.read("follow-up> "))
                await asyncio.sleep(0)
                pipe.send_text("\x03")
                with self.assertRaises(TerminalInputInterrupted):
                    await task

        asyncio.run(exercise())

    def test_terminal_input_reads_existing_readline_history(self) -> None:
        async def exercise() -> list[str]:
            with tempfile.TemporaryDirectory() as td, create_pipe_input() as pipe:
                history_file = Path(td) / "history"
                history_file.write_text(
                    "_HiStOrY_V2_\nold\\040first\npath\\134name\n",
                    encoding="utf-8",
                )
                terminal_input = TerminalInput(
                    history_file,
                    history_enabled=True,
                    completions=(),
                    completion_descriptions={},
                    input=pipe,
                    output=DummyOutput(),
                )
                initial = [
                    value async for value in terminal_input._session.history.load()
                ]
                self.assertEqual(initial, [r"path\name", "old first"])
                terminal_input._session.history.append_string("new prompt")
                replacement = TerminalInput(
                    history_file,
                    history_enabled=True,
                    completions=(),
                    completion_descriptions={},
                    input=pipe,
                    output=DummyOutput(),
                )
                return [value async for value in replacement._session.history.load()]

        self.assertEqual(
            asyncio.run(exercise()),
            ["new prompt", r"path\name", "old first"],
        )

    def test_readline_history_preserves_shared_prompt_toolkit_file(self) -> None:
        class FakeReadline:
            __doc__ = "GNU readline"

            def __init__(self) -> None:
                self.history: list[str] = []
                self.completer: Any = None
                self.delimiters = " \t\n"

            def add_history(self, entry: str) -> None:
                self.history.append(entry)

            def get_current_history_length(self) -> int:
                return len(self.history)

            def get_history_item(self, index: int) -> str | None:
                return self.history[index - 1] if 0 < index <= len(self.history) else None

            def set_history_length(self, _length: int) -> None:
                return None

            def get_completer(self) -> Any:
                return self.completer

            def set_completer(self, completer: Any) -> None:
                self.completer = completer

            def get_completer_delims(self) -> str:
                return self.delimiters

            def set_completer_delims(self, delimiters: str) -> None:
                self.delimiters = delimiters

            def parse_and_bind(self, _binding: str) -> None:
                return None

        with tempfile.TemporaryDirectory() as td:
            history_file = Path(td) / "history"
            CompatibleFileHistory(history_file).store_string("interactive prompt")
            fake_readline = FakeReadline()
            with (
                patch.dict(sys.modules, {"readline": fake_readline}),
                terminal.ReadlineHistory(history_file),
            ):
                fake_readline.add_history("serial prompt")

            self.assertEqual(
                read_history_entries(history_file),
                ["interactive prompt", "serial prompt"],
            )
            self.assertNotIn("_HiStOrY_V2_", history_file.read_text(encoding="utf-8"))

    def test_interactive_renders_all_stream_events_while_follow_up_prompt_is_active(
        self,
    ) -> None:
        async def exercise() -> str:
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
                stream = TTYBuffer()
                shell.renderer._text_stream = stream
                shell.renderer._status_stream = stream
                shell.renderer.ui.color = False
                shell.renderer.show_tool_output = True
                shell._input = cast(Any, SimpleNamespace(reading=True))

                await shell._render_event(
                    Event(type="model.text_delta", data={"text": "first-middle-last"})
                )
                await shell._render_event(
                    Event(
                        type="tool.output",
                        data={"tool": "shell", "tool_call_id": "call", "text": "tool-output"},
                    )
                )
                await shell._render_event(
                    Event(type="model.completed", data={"text": "first-middle-last"})
                )
                return terminal._strip_ansi(stream.getvalue())

        output = __import__("asyncio").run(exercise())
        self.assertEqual(output.count("first-middle-last"), 1)
        self.assertEqual(output.count("tool-output"), 1)

    def test_interactive_pauses_live_pulse_while_follow_up_prompt_is_active(
        self,
    ) -> None:
        class RunningTask:
            def __init__(self) -> None:
                self.done_calls = 0

            def done(self) -> bool:
                self.done_calls += 1
                return self.done_calls >= 3

        async def exercise() -> None:
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
                shell.renderer._status_stream = TTYBuffer()
                shell.renderer.ui.color = True
                shell.renderer._activity_phase = "working"
                shell._input = cast(Any, SimpleNamespace(reading=True))
                shell.renderer.pulse = Mock()  # type: ignore[method-assign]

                with patch(
                    "borealis_coder.interactive.asyncio.sleep", new=AsyncMock()
                ) as sleep:
                    await shell._report_progress(cast(Any, RunningTask()))

                sleep.assert_awaited_once_with(1.0)
                shell.renderer.pulse.assert_not_called()

        __import__("asyncio").run(exercise())

    def test_interactive_follow_up_entered_at_turn_boundary_starts_next_turn(self) -> None:
        class InputWorker:
            def __init__(self) -> None:
                self.calls = 0
                self.reading = False
                self.current_text = ""

            @property
            def current_document(self) -> Document:
                return Document("next task", cursor_position=4)

            async def read(
                self, _prompt: str, *, default: str | Document = ""
            ) -> str:
                self.calls += 1
                if self.calls == 1:
                    await asyncio.Event().wait()
                if self.calls == 2:
                    if not isinstance(default, Document):
                        raise AssertionError("Expected the saved prompt document")
                    return default.text
                raise EOFError

            def close(self) -> None:
                return None

        async def exercise() -> None:
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
                    initial_prompt="first task",
                    history_enabled=False,
                )
                input_worker = InputWorker()
                shell.renderer._status_stream = io.StringIO()
                runner = SimpleNamespace(close=AsyncMock())
                shell.runner = cast(Any, runner)
                submitted: list[str] = []

                async def submit(prompt: str) -> None:
                    submitted.append(prompt)

                shell._submit = submit  # type: ignore[method-assign]
                with (
                    patch.object(shell, "_new_runner", AsyncMock(return_value=runner)),
                    patch.object(shell, "_select_initial_session", AsyncMock()),
                    patch.object(shell, "_print_banner"),
                    patch.object(shell, "_concurrent_input", return_value=True),
                    patch(
                        "borealis_coder.interactive.TerminalInput",
                        return_value=input_worker,
                    ),
                ):
                    self.assertEqual(await shell.run(), 0)

                self.assertEqual(submitted, ["first task", "next task"])

        __import__("asyncio").run(exercise())

    def test_interactive_restores_command_prompt_after_turn_completion(self) -> None:
        async def exercise() -> None:
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
                    initial_prompt="first task",
                    history_enabled=False,
                )
                input_worker = ControlledInput()
                shell._input = cast(Any, input_worker)
                shell.runner = cast(Any, SimpleNamespace())
                release_submission = asyncio.Event()
                submitted: list[str] = []

                async def submit(prompt: str) -> None:
                    submitted.append(prompt)
                    await release_submission.wait()

                shell._submit = submit  # type: ignore[method-assign]
                with patch.object(
                    shell,
                    "_input_prompt",
                    side_effect=lambda active: "follow-up> " if active else "command> ",
                ):
                    loop_task = asyncio.create_task(shell._run_concurrent())
                    first_prompt, _, stale_input = await input_worker.requests.get()
                    self.assertEqual(first_prompt, "follow-up> ")
                    release_submission.set()
                    command_prompt, _, command_input = await input_worker.requests.get()
                    self.assertTrue(stale_input.cancelled())
                    self.assertEqual(command_prompt, "command> ")
                    command_input.set_result("/exit")
                    self.assertEqual(await loop_task, 0)

                self.assertEqual(submitted, ["first task"])

        asyncio.run(exercise())

    def test_interactive_unescapes_literal_slash_follow_up(self) -> None:
        async def exercise() -> None:
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
                    initial_prompt="first task",
                    history_enabled=False,
                )
                input_worker = ControlledInput()
                shell._input = cast(Any, input_worker)
                runner = SimpleNamespace(
                    accepts_steering=Mock(return_value=True),
                    queued_prompts=Mock(return_value=1),
                    steer=Mock(),
                )
                shell.runner = cast(Any, runner)
                release_submission = asyncio.Event()

                async def submit(_prompt: str) -> None:
                    shell._active_session_id = "sess_test"
                    await release_submission.wait()

                shell._submit = submit  # type: ignore[method-assign]
                loop_task = asyncio.create_task(shell._run_concurrent())
                _, _, follow_up = await input_worker.requests.get()
                follow_up.set_result("//explain /api/users")
                _, _, final_input = await input_worker.requests.get()
                runner.steer.assert_called_once_with(
                    "sess_test",
                    "/explain /api/users",
                    metadata={"interactive": True},
                )
                final_input.set_exception(EOFError())
                release_submission.set()
                self.assertEqual(await loop_task, 0)

        asyncio.run(exercise())

    def test_interactive_terminal_approval_uses_the_shared_input_loop(self) -> None:
        async def exercise() -> None:
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
                    terminal_approvals=True,
                )
                request = ApprovalRequest(
                    tool_name="shell",
                    description="run",
                    decision=PolicyDecision(PolicyAction.ASK, "reason", "high"),
                    arguments_preview="echo hi",
                )
                response = __import__("asyncio").create_task(
                    shell._request_terminal_approval(request)
                )
                queued_request, future = await shell._approval_requests.get()
                self.assertIs(queued_request, request)
                self.assertTrue(shell._answer_approval("a", future))
                self.assertEqual(await response, "allow_always")

        __import__("asyncio").run(exercise())

    def test_existing_follow_up_cannot_answer_a_new_approval(self) -> None:
        async def exercise() -> None:
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
                    initial_prompt="first task",
                    history_enabled=False,
                    terminal_approvals=True,
                )
                input_worker = ControlledInput()
                shell._input = cast(Any, input_worker)
                runner = SimpleNamespace(
                    accepts_steering=Mock(return_value=True),
                    queued_prompts=Mock(return_value=1),
                    steer=Mock(),
                )
                shell.runner = cast(Any, runner)
                release_submission = asyncio.Event()

                async def submit(_prompt: str) -> None:
                    shell._active_session_id = "sess_test"
                    await release_submission.wait()

                shell._submit = submit  # type: ignore[method-assign]
                loop_task = asyncio.create_task(shell._run_concurrent())
                _, _, follow_up = await input_worker.requests.get()
                request = ApprovalRequest(
                    tool_name="shell",
                    description="run",
                    decision=PolicyDecision(PolicyAction.ASK, "reason", "high"),
                    arguments_preview="echo hi",
                )
                approval: asyncio.Future[str] = asyncio.get_running_loop().create_future()
                shell._approval_requests.put_nowait((request, approval))
                follow_up.set_result("a")

                approval_prompt, default, approval_input = await input_worker.requests.get()
                self.assertIn("approval", approval_prompt)
                self.assertEqual(default, "")
                self.assertFalse(approval.done())
                runner.steer.assert_called_once_with(
                    "sess_test", "a", metadata={"interactive": True}
                )
                approval_input.set_result("n")
                self.assertEqual(await approval, "no")

                _, _, final_input = await input_worker.requests.get()
                release_submission.set()
                final_input.set_exception(EOFError())
                self.assertEqual(await loop_task, 0)

        asyncio.run(exercise())

    def test_approval_interrupt_preserves_draft_for_the_follow_up_prompt(self) -> None:
        async def exercise() -> None:
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
                    initial_prompt="first task",
                    history_enabled=False,
                    terminal_approvals=True,
                )
                input_worker = ControlledInput()
                shell._input = cast(Any, input_worker)
                shell.runner = cast(
                    Any,
                    SimpleNamespace(
                        accepts_steering=Mock(return_value=False),
                        queued_prompts=Mock(return_value=0),
                    ),
                )
                release_submission = asyncio.Event()

                async def submit(_prompt: str) -> None:
                    await release_submission.wait()

                shell._submit = submit  # type: ignore[method-assign]
                loop_task = asyncio.create_task(shell._run_concurrent())

                await input_worker.requests.get()
                input_worker.current_text = "keep this draft"
                input_worker.cursor_position = 4
                request = ApprovalRequest(
                    tool_name="write_file",
                    description="write",
                    decision=PolicyDecision(PolicyAction.ASK, "reason", "high"),
                    arguments_preview="file.txt",
                )
                approval: asyncio.Future[str] = asyncio.get_running_loop().create_future()
                shell._approval_requests.put_nowait((request, approval))

                _, approval_default, approval_input = await input_worker.requests.get()
                self.assertEqual(approval_default, "")
                approval_input.set_result("n")
                self.assertEqual(await approval, "no")
                _, restored_default, restored_input = await input_worker.requests.get()
                self.assertIsInstance(restored_default, Document)
                self.assertEqual(cast(Document, restored_default).text, "keep this draft")
                self.assertEqual(cast(Document, restored_default).cursor_position, 4)
                restored_input.set_exception(EOFError())
                release_submission.set()
                self.assertEqual(await loop_task, 0)

        asyncio.run(exercise())

    def test_eof_rejects_approval_requested_later_in_the_active_turn(self) -> None:
        async def exercise() -> None:
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
                    initial_prompt="first task",
                    history_enabled=False,
                    terminal_approvals=True,
                )
                input_worker = ControlledInput()
                shell._input = cast(Any, input_worker)
                shell.runner = cast(Any, SimpleNamespace())
                release_submission = asyncio.Event()

                async def submit(_prompt: str) -> None:
                    await release_submission.wait()

                shell._submit = submit  # type: ignore[method-assign]
                loop_task = asyncio.create_task(shell._run_concurrent())
                _, _, terminal_input = await input_worker.requests.get()
                terminal_input.set_exception(EOFError())
                await asyncio.sleep(0)

                request = ApprovalRequest(
                    tool_name="shell",
                    description="run",
                    decision=PolicyDecision(PolicyAction.ASK, "reason", "high"),
                    arguments_preview="echo hi",
                )
                approval: asyncio.Future[str] = asyncio.get_running_loop().create_future()
                shell._approval_requests.put_nowait((request, approval))
                self.assertEqual(await asyncio.wait_for(approval, timeout=1), "no")
                release_submission.set()
                self.assertEqual(await asyncio.wait_for(loop_task, timeout=1), 0)

        asyncio.run(exercise())

    def test_stale_approval_answer_is_ignored(self) -> None:
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

            async def exercise() -> None:
                future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
                future.cancel()
                self.assertTrue(shell._answer_approval("a", future))
                self.assertTrue(future.cancelled())

            asyncio.run(exercise())

    def test_unconsumed_steering_is_requeued_after_a_turn(self) -> None:
        async def exercise() -> None:
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
                shell.session_id = "sess_test"
                result = AgentResult(
                    session_id="sess_test",
                    run_id="run_test",
                    text="done",
                    stop_reason=StopReason.END_TURN,
                    usage=Usage(),
                    turns=1,
                )
                runner = SimpleNamespace(
                    run=AsyncMock(return_value=result),
                    reclaim_steering=Mock(return_value=["late direction"]),
                )
                shell.runner = cast(Any, runner)
                shell.renderer._status_stream = io.StringIO()
                shell.renderer._text_stream = io.StringIO()

                await shell._submit("first task")

                runner.reclaim_steering.assert_called_once_with("sess_test")
                self.assertEqual(shell._pending_follow_ups, ["late direction"])

        asyncio.run(exercise())

    def test_interactive_prompt_keyboard_interrupt_exits_130(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env = {
                "BOREALIS_DATA_DIR": str(root / "data"),
                "BOREALIS_PROVIDER": "mock",
                "BOREALIS_APPROVAL": "never",
            }
            with (
                patch.dict(os.environ, env, clear=False),
                patch("builtins.input", side_effect=KeyboardInterrupt),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
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
