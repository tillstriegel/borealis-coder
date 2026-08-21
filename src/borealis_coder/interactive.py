"""Persistent, dependency-free interactive shell for Borealis Coder."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import shlex
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import __version__
from .agent import AgentRunner, build_runner
from .config import Config
from .diagnostics import run_diagnostics
from .errors import BorealisError, ConfigurationError, SessionError
from .models import AgentResult, Event, Role
from .safety import ApprovalRequest
from .terminal import ConsoleRenderer, ReadlineHistory
from .util import truncate_text

ApprovalCallback = Callable[[ApprovalRequest], bool | str | Any]

_COMMANDS = (
    "/help",
    "/status",
    "/session",
    "/sessions",
    "/resume",
    "/history",
    "/new",
    "/clear",
    "/provider",
    "/model",
    "/mode",
    "/approval",
    "/network",
    "/verify",
    "/sandbox",
    "/tools",
    "/doctor",
    "/rollback",
    "/paste",
    "/exit",
    "/quit",
)


class InteractiveCLI:
    """Codex-style persistent terminal session around :class:`AgentRunner`."""

    def __init__(
        self,
        *,
        workspace: Path,
        config: Config,
        approval_callback: ApprovalCallback | None,
        resume: str | None = None,
        resume_last: bool = False,
        initial_prompt: str = "",
        stream_text: bool = True,
        show_tool_output: bool = False,
        history_enabled: bool = True,
    ) -> None:
        self.workspace = workspace.resolve()
        self.config = config
        self.approval_callback = approval_callback
        self.resume = resume
        self.resume_last = resume_last
        self.initial_prompt = initial_prompt.strip()
        self.history_enabled = history_enabled
        self.renderer = ConsoleRenderer(
            stream_text=stream_text,
            interactive=True,
            show_tool_output=show_tool_output,
            status_stream=sys.stdout,
        )
        self.runner: AgentRunner | None = None
        self.session_id: str | None = None
        self._active_session_id: str | None = None
        self._active_task: asyncio.Task[AgentResult] | None = None
        self._interrupt_count = 0

    async def run(self) -> int:
        self.runner = await self._new_runner(self.config)
        await self._select_initial_session()
        self._print_banner()
        history_file = self.config.storage_dir / "cli-history"
        try:
            with ReadlineHistory(
                history_file,
                enabled=self.history_enabled,
                completions=_COMMANDS,
            ):
                if self.initial_prompt:
                    await self._submit(self.initial_prompt)
                while True:
                    try:
                        line = _terminal_input(self._prompt_label())
                    except EOFError:
                        print()
                        break
                    except KeyboardInterrupt:
                        print()
                        return 130
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("//"):
                        await self._submit(line[1:])
                        continue
                    if line.startswith("/"):
                        try:
                            should_exit, prompt = await self._command(line)
                        except (BorealisError, ConfigurationError, OSError, ValueError) as error:
                            print(f"command error: {error}", file=self.renderer.status_stream)
                            continue
                        if prompt:
                            await self._submit(prompt)
                        if should_exit:
                            break
                        continue
                    await self._submit(line)
        finally:
            if self.runner is not None:
                await self.runner.close()
                self.runner = None
        return 0

    async def _new_runner(self, config: Config) -> AgentRunner:
        runner = await build_runner(
            self.workspace,
            config=config,
            approval_callback=self.approval_callback,
            interactive=True,
        )
        runner.events.subscribe(self.renderer.handle)
        runner.events.subscribe(self._observe)
        return runner

    async def _select_initial_session(self) -> None:
        runner = self._require_runner()
        selected = self.resume
        if self.resume_last and not selected:
            values = await asyncio.to_thread(
                runner.sessions.list_sessions,
                workspace=self.workspace,
                limit=1,
            )
            selected = values[0].id if values else None
        if selected:
            await self._resume(selected, announce=False)

    def _print_banner(self) -> None:
        route = self._primary_route()
        print(f"Borealis Coder {__version__} — interactive mode")
        print(f"workspace  {self.workspace}")
        print(f"model      {route.name}/{route.model}")
        print(
            f"safety     {self.config.safety.mode} · approval={self.config.safety.approval} "
            f"· network={'on' if self.config.safety.network else 'off'}"
        )
        if self.session_id:
            print(f"session    {self.session_id} (resumed)")
        print("Type /help for commands. Use //text to send a prompt beginning with '/'.")

    def _prompt_label(self) -> str:
        if not _is_tty(sys.stdin):
            return ""
        short = self.session_id[-8:] if self.session_id else "new"
        return f"[{short}] › "  # noqa: RUF001 - intentional prompt glyph

    async def _submit(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt:
            return
        runner = self._require_runner()
        self.renderer.reset_turn()
        self._active_session_id = self.session_id
        self._interrupt_count = 0
        task = asyncio.create_task(runner.run(prompt, session_id=self.session_id))
        progress_task = asyncio.create_task(self._report_progress(task))
        self._active_task = task
        previous_handler = self._install_turn_interrupt_handler()
        try:
            result = await task
        except asyncio.CancelledError:
            self.renderer.finish_turn()
            print("Turn cancelled.", file=self.renderer.status_stream)
            return
        except (BorealisError, ConfigurationError, OSError, ValueError) as error:
            self.renderer.finish_turn()
            print(f"error: {error}", file=self.renderer.status_stream)
            return
        finally:
            progress_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await progress_task
            self._restore_interrupt_handler(previous_handler)
            self._active_task = None
            self._active_session_id = None
            self.renderer.finish_turn()
        self.session_id = result.session_id
        if result.text and not self.renderer.has_rendered(result.text):
            print(result.text)
        print(_turn_footer(result), file=self.renderer.status_stream)
        if result.error and result.stop_reason.value != "cancelled":
            print(
                f"[{result.stop_reason.value}] {result.error}",
                file=self.renderer.status_stream,
            )

    async def _report_progress(self, task: asyncio.Task[AgentResult]) -> None:
        loop = asyncio.get_running_loop()
        started = loop.time()
        generation = self.renderer.activity_generation
        while not task.done():
            await asyncio.sleep(10)
            if task.done():
                return
            current_generation = self.renderer.activity_generation
            if current_generation == generation:
                self.renderer.heartbeat(max(1, int(loop.time() - started)))
            generation = current_generation

    async def _command(self, line: str) -> tuple[bool, str | None]:
        try:
            parts = shlex.split(line)
        except ValueError as error:
            print(f"command error: {error}", file=self.renderer.status_stream)
            return False, None
        if not parts:
            return False, None
        command = parts[0].lower()
        args = parts[1:]

        if command in {"/exit", "/quit"}:
            return True, None
        if command == "/help":
            self._print_help()
        elif command in {"/new", "/clear"}:
            self.session_id = None
            print("Started a fresh conversation. The previous session remains saved.")
        elif command in {"/status", "/session"}:
            await self._print_status(full=command == "/status")
        elif command == "/sessions":
            await self._list_sessions(_optional_positive_int(args, default=12))
        elif command == "/resume":
            await self._resume(args[0] if args else "last")
        elif command == "/history":
            await self._print_history(_optional_positive_int(args, default=12))
        elif command == "/provider":
            await self._set_provider(args)
        elif command == "/model":
            await self._set_model(args)
        elif command == "/mode":
            await self._set_choice(
                args,
                label="mode",
                choices={"plan", "workspace-write", "full"},
                getter=lambda config: config.safety.mode,
                setter=lambda config, value: setattr(config.safety, "mode", value),
            )
        elif command == "/approval":
            await self._set_choice(
                args,
                label="approval",
                choices={"never", "on-risk", "always"},
                getter=lambda config: config.safety.approval,
                setter=lambda config, value: setattr(config.safety, "approval", value),
            )
        elif command == "/sandbox":
            await self._set_choice(
                args,
                label="sandbox",
                choices={"native", "docker"},
                getter=lambda config: config.sandbox.driver,
                setter=lambda config, value: setattr(config.sandbox, "driver", value),
            )
        elif command == "/network":
            await self._set_boolean(
                args,
                label="network",
                getter=lambda config: config.safety.network,
                setter=lambda config, value: setattr(config.safety, "network", value),
            )
        elif command == "/verify":
            await self._set_boolean(
                args,
                label="verification",
                getter=lambda config: config.agent.auto_verify,
                setter=lambda config, value: setattr(config.agent, "auto_verify", value),
            )
        elif command == "/tools":
            self._print_tools(args[0] if args else "")
        elif command == "/doctor":
            self._print_diagnostics()
        elif command == "/rollback":
            self._rollback(args)
        elif command == "/paste":
            return False, await self._read_multiline()
        else:
            print(
                f"Unknown command: {command}. Type /help.",
                file=self.renderer.status_stream,
            )
        return False, None

    async def _resume(self, value: str, *, announce: bool = True) -> None:
        runner = self._require_runner()
        if value.lower() == "last":
            sessions = await asyncio.to_thread(
                runner.sessions.list_sessions,
                workspace=self.workspace,
                limit=1,
            )
            if not sessions:
                print("No saved sessions for this workspace.")
                return
            info = sessions[0]
        else:
            info = await asyncio.to_thread(runner.sessions.get_session, value)
            if Path(info.workspace).resolve() != self.workspace:
                raise SessionError(
                    f"Session workspace is {info.workspace}, not {self.workspace}"
                )
        self.session_id = info.id
        if announce:
            print(
                f"Resumed {info.id} · {info.provider}/{info.model} · "
                f"{info.updated_at} · {info.title}"
            )

    async def _list_sessions(self, limit: int) -> None:
        runner = self._require_runner()
        values = await asyncio.to_thread(
            runner.sessions.list_sessions,
            workspace=self.workspace,
            limit=limit,
        )
        if not values:
            print("No saved sessions for this workspace.")
            return
        for item in values:
            marker = "*" if item.id == self.session_id else " "
            print(
                f"{marker} {item.id}  {item.updated_at}  {item.status:<7} "
                f"{item.provider}/{item.model}  {item.title}"
            )

    async def _print_status(self, *, full: bool) -> None:
        route = self._primary_route()
        print(f"workspace: {self.workspace}")
        print(f"provider:  {route.name}")
        print(f"model:     {route.model}")
        print(f"session:   {self.session_id or '(new conversation)'}")
        if full:
            print(f"mode:      {self.config.safety.mode}")
            print(f"approval:  {self.config.safety.approval}")
            print(f"network:   {'on' if self.config.safety.network else 'off'}")
            print(f"sandbox:   {self.config.sandbox.driver}")
            print(f"verify:    {'on' if self.config.agent.auto_verify else 'off'}")
        if self.session_id:
            runner = self._require_runner()
            info = await asyncio.to_thread(runner.sessions.get_session, self.session_id)
            usage = await asyncio.to_thread(runner.sessions.usage, self.session_id)
            print(f"title:     {info.title}")
            print(
                f"usage:     {usage.total_tokens} tokens · {usage.requests} requests "
                f"· ${usage.cost_usd:.4f}"
            )

    async def _print_history(self, limit: int) -> None:
        if not self.session_id:
            print("No active session.")
            return
        runner = self._require_runner()
        messages = await asyncio.to_thread(runner.sessions.messages, self.session_id)
        messages = messages[-limit:]
        if not messages:
            print("Session history is empty.")
            return
        for message in messages:
            if message.role == Role.TOOL:
                label = f"tool:{message.tool_name or 'unknown'}"
            else:
                label = message.role.value
            content = message.content.strip() or (
                ", ".join(call.name for call in message.tool_calls)
                if message.tool_calls
                else "(empty)"
            )
            print(f"\n{label}\n{truncate_text(content, 2_000)}")

    async def _set_provider(self, args: list[str]) -> None:
        if not args:
            current = self._primary_route().name
            for name, provider in sorted(self.config.providers.items()):
                marker = "*" if name == current else " "
                model = provider.model or "(model required)"
                print(f"{marker} {name:<20} {provider.type:<18} {model}")
            return
        name = args[0]
        if name not in self.config.providers:
            print(f"Unknown provider: {name}", file=self.renderer.status_stream)
            return
        model = args[1] if len(args) > 1 else self.config.providers[name].model

        def change(config: Config) -> None:
            config.agent.provider = name
            config.agent.model = model

        await self._replace_runtime(change, f"provider={name} model={model}")

    async def _set_model(self, args: list[str]) -> None:
        if not args:
            route = self._primary_route()
            print(f"{route.name}/{route.model}")
            return
        model = args[0]
        await self._replace_runtime(
            lambda config: setattr(config.agent, "model", model),
            f"model={model}",
        )

    async def _set_choice(
        self,
        args: list[str],
        *,
        label: str,
        choices: set[str],
        getter: Callable[[Config], str],
        setter: Callable[[Config, str], None],
    ) -> None:
        if not args:
            print(getter(self.config))
            return
        value = args[0].lower()
        if value not in choices:
            print(
                f"{label} must be one of: {', '.join(sorted(choices))}",
                file=self.renderer.status_stream,
            )
            return
        await self._replace_runtime(lambda config: setter(config, value), f"{label}={value}")

    async def _set_boolean(
        self,
        args: list[str],
        *,
        label: str,
        getter: Callable[[Config], bool],
        setter: Callable[[Config, bool], None],
    ) -> None:
        if not args:
            print("on" if getter(self.config) else "off")
            return
        try:
            value = _parse_boolean(args[0])
        except ValueError as error:
            print(error, file=self.renderer.status_stream)
            return
        await self._replace_runtime(
            lambda config: setter(config, value),
            f"{label}={'on' if value else 'off'}",
        )

    async def _replace_runtime(
        self,
        mutate: Callable[[Config], None],
        description: str,
    ) -> None:
        candidate = copy.deepcopy(self.config)
        mutate(candidate)
        try:
            replacement = await self._new_runner(candidate)
        except (BorealisError, ConfigurationError, OSError, ValueError) as error:
            print(
                f"Could not apply {description}: {error}",
                file=self.renderer.status_stream,
            )
            return
        previous = self._require_runner()
        self.runner = replacement
        self.config = candidate
        await previous.close()
        print(f"Updated runtime for this CLI process: {description}")

    def _print_tools(self, filter_text: str) -> None:
        schemas = self._require_runner().tools.schemas()
        needle = filter_text.lower()
        for item in schemas:
            value = f"{item['name']} {item.get('description', '')}"
            if needle and needle not in value.lower():
                continue
            print(f"{item['name']:<28} {item.get('description', '')}")

    def _print_diagnostics(self) -> None:
        for item in run_diagnostics(self.workspace, self.config):
            print(f"{'PASS' if item.ok else 'FAIL'}  {item.name}: {item.message}")

    def _rollback(self, args: list[str]) -> None:
        manager = self._require_runner().tool_context.checkpoints
        if not args:
            values = manager.list()
            if not values:
                print("No checkpoints.")
                return
            for item in values:
                print(f"{item.id}  {item.created_at}  {item.label}  {len(item.files)} file(s)")
            return
        checkpoint = manager.restore(args[0])
        print(f"Restored {checkpoint.id}: {len(checkpoint.files)} file(s)")

    async def _read_multiline(self) -> str | None:
        print("Paste a multiline prompt. Finish with a line containing only /end.")
        lines: list[str] = []
        prompt = "... " if _is_tty(sys.stdin) else ""
        while True:
            try:
                line = _terminal_input(prompt)
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nMultiline prompt cancelled.")
                return None
            if line.strip() == "/end":
                break
            lines.append(line)
        value = "\n".join(lines).strip()
        if not value:
            print("Multiline prompt was empty.")
            return None
        return value

    def _print_help(self) -> None:
        print(
            """
Interactive commands
  /status                 Show runtime, safety, session, and usage state
  /session                Show the current workspace, provider, model, and session
  /sessions [N]           List recent sessions for this workspace
  /resume [ID|last]       Resume a durable session (defaults to the latest)
  /history [N]            Show recent persisted conversation messages
  /new, /clear            Start a fresh conversation without deleting the old one
  /provider [NAME MODEL]  Show or switch provider; optional MODEL overrides its default
  /model [MODEL]          Show or switch the model for this CLI process
  /mode [VALUE]           plan | workspace-write | full
  /approval [VALUE]       never | on-risk | always
  /network [on|off]       Toggle model-initiated network tools
  /verify [on|off]        Toggle automatic post-change verification
  /sandbox [VALUE]        native | docker
  /tools [FILTER]         List the effective built-in, plugin, and MCP tools
  /doctor                 Run diagnostics with the current runtime settings
  /rollback [ID]          List checkpoints or restore one
  /paste                  Enter a multiline prompt; finish with /end
  /exit, /quit            Leave Borealis

Press Ctrl+C while Borealis is working to cancel the current turn and keep the
session. Press Ctrl+C at the input prompt to exit. Prefix a literal slash prompt
with an extra slash, for example: //explain this route.
""".strip()
        )

    async def _observe(self, event: Event) -> None:
        if event.type == "run.started" and event.session_id:
            self._active_session_id = event.session_id
            self.session_id = event.session_id

    def _primary_route(self):  # type: ignore[no-untyped-def]
        return self._require_runner().providers[0]

    def _require_runner(self) -> AgentRunner:
        if self.runner is None:
            raise RuntimeError("Interactive runner is not initialized")
        return self.runner

    def _install_turn_interrupt_handler(self):  # type: ignore[no-untyped-def]
        try:
            previous = signal.getsignal(signal.SIGINT)
            loop = asyncio.get_running_loop()

            def handler(signum, frame) -> None:  # type: ignore[no-untyped-def]
                del signum, frame
                loop.call_soon_threadsafe(self._cancel_active_turn)

            signal.signal(signal.SIGINT, handler)
            return previous
        except (ValueError, OSError, AttributeError):
            return None

    @staticmethod
    def _restore_interrupt_handler(previous) -> None:  # type: ignore[no-untyped-def]
        if previous is None:
            return
        with contextlib.suppress(ValueError, OSError, AttributeError):
            signal.signal(signal.SIGINT, previous)

    def _cancel_active_turn(self) -> None:
        task = self._active_task
        if task is None or task.done():
            return
        self._interrupt_count += 1
        self.renderer.finish_turn()
        runner = self._require_runner()
        cancelled = bool(
            self._active_session_id and runner.cancel(self._active_session_id)
        )
        if not cancelled or self._interrupt_count > 1:
            task.cancel()
        print("^C cancelling current turn…", file=self.renderer.status_stream, flush=True)


def _turn_footer(result: AgentResult) -> str:
    parts = [
        result.stop_reason.value,
        f"{result.turns} model turn{'s' if result.turns != 1 else ''}",
        f"{result.usage.total_tokens} tokens",
    ]
    if result.usage.cost_usd:
        parts.append(f"${result.usage.cost_usd:.4f}")
    if result.changed_files:
        parts.append(f"{len(result.changed_files)} changed file(s)")
    if result.verification is not None:
        parts.append("verified" if result.verification.get("ok") else "verification failed")
    return "─ " + " · ".join(parts)


def _optional_positive_int(args: list[str], *, default: int) -> int:
    if not args:
        return default
    try:
        value = int(args[0])
    except ValueError:
        return default
    return max(1, min(value, 100))


def _parse_boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"on", "yes", "true", "1", "enable", "enabled"}:
        return True
    if normalized in {"off", "no", "false", "0", "disable", "disabled"}:
        return False
    raise ValueError("Expected on or off")


def _is_tty(stream: object) -> bool:
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except OSError:
        return False


def _terminal_input(prompt: str) -> str:
    """Read terminal input with immediate SIGINT delivery.

    ``asyncio.run`` normally converts the first SIGINT into task cancellation.
    Cancellation cannot interrupt a synchronous ``input()`` call, so a shell
    waiting at its prompt would otherwise appear hung after Ctrl+C. Temporarily
    restoring Python's default SIGINT handler makes the keystroke raise
    ``KeyboardInterrupt`` at the input boundary, where the shell can exit
    deterministically.
    """

    previous = None
    try:
        previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.default_int_handler)
    except (ValueError, OSError, AttributeError):
        previous = None
    try:
        return input(prompt)
    finally:
        if previous is not None:
            with contextlib.suppress(ValueError, OSError, AttributeError):
                signal.signal(signal.SIGINT, previous)
