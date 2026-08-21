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
from .terminal import AuroraUI, ConsoleRenderer, ReadlineHistory
from .util import truncate_text

ApprovalCallback = Callable[[ApprovalRequest], bool | str | Any]

_COMMAND_OPTIONS = (
    ("/help", "Show every command and keyboard control"),
    ("/status", "Inspect runtime, safety, usage, and cache health"),
    ("/session", "Show the active workspace, route, and session"),
    ("/sessions", "Browse recent sessions in this workspace"),
    ("/resume", "Return to a saved session"),
    ("/history", "Read recent persisted messages"),
    ("/new", "Start a fresh conversation"),
    ("/clear", "Start fresh and keep the previous session"),
    ("/provider", "Inspect or switch the provider route"),
    ("/model", "Inspect or switch the active model"),
    ("/mode", "Change the workspace safety mode"),
    ("/approval", "Change the approval policy"),
    ("/network", "Toggle model-initiated network tools"),
    ("/verify", "Toggle automatic post-change verification"),
    ("/sandbox", "Switch between native and Docker isolation"),
    ("/tools", "Browse effective built-in, plugin, and MCP tools"),
    ("/doctor", "Diagnose the current runtime"),
    ("/rollback", "List or restore recovery points"),
    ("/paste", "Enter a multiline prompt"),
    ("/exit", "Leave Borealis with the session saved"),
    ("/quit", "Leave Borealis with the session saved"),
)
_COMMANDS = tuple(command for command, _ in _COMMAND_OPTIONS)
_COMMAND_DESCRIPTIONS = dict(_COMMAND_OPTIONS)


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

    @property
    def ui(self) -> AuroraUI:
        return self.renderer.ui

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
                completion_descriptions=_COMMAND_DESCRIPTIONS,
            ):
                if self.initial_prompt:
                    await self._submit(self.initial_prompt)
                while True:
                    try:
                        line = _terminal_input(self._prompt_label())
                    except EOFError:
                        print()
                        self.ui.notice("info", "Aurora shell closed", "session saved")
                        break
                    except KeyboardInterrupt:
                        print()
                        self.ui.notice("warning", "Aurora shell interrupted", "session saved")
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
                            self.ui.notice("error", "Command failed", str(error))
                            continue
                        if prompt:
                            await self._submit(prompt)
                        if should_exit:
                            self.ui.notice("info", "Aurora shell closed", "session saved")
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
        safety = (
            f"{self.config.safety.mode} · approval {self.config.safety.approval} · "
            f"network {'on' if self.config.safety.network else 'off'}"
        )
        session = f"{self.session_id} (resumed)" if self.session_id else "new conversation"
        self.ui.banner(
            version=f"v{__version__}",
            workspace=self.workspace,
            route=f"{route.name}/{route.model}",
            safety=safety,
            session=session,
        )

    def _prompt_label(self) -> str:
        if not _is_tty(sys.stdin):
            return ""
        short = self.session_id[-8:] if self.session_id else "new"
        return self.ui.prompt(short)

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
            self.ui.notice("warning", "Turn cancelled", "session state preserved")
            return
        except (BorealisError, ConfigurationError, OSError, ValueError) as error:
            self.renderer.finish_turn()
            self.ui.notice("error", "Turn failed", str(error))
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
        self.ui.turn_footer(_turn_footer(result))
        if result.error and result.stop_reason.value != "cancelled":
            self.ui.notice("error", result.stop_reason.value, result.error)

    async def _report_progress(self, task: asyncio.Task[AgentResult]) -> None:
        loop = asyncio.get_running_loop()
        started = loop.time()
        generation = self.renderer.activity_generation
        frame = 0
        next_heartbeat_at = started + 10.0
        while not task.done():
            await asyncio.sleep(0.12 if self.renderer.live_activity else 1.0)
            if task.done():
                return
            now = loop.time()
            elapsed = now - started
            if self.renderer.live_activity:
                self.renderer.pulse(elapsed, frame)
                frame += 1
                continue
            current_generation = self.renderer.activity_generation
            if current_generation != generation:
                generation = current_generation
                next_heartbeat_at = now + 10.0
            elif now >= next_heartbeat_at:
                self.renderer.heartbeat(max(1, int(elapsed)))
                next_heartbeat_at = now + 10.0

    async def _command(self, line: str) -> tuple[bool, str | None]:
        try:
            parts = shlex.split(line)
        except ValueError as error:
            self.ui.notice("error", "Could not parse command", str(error))
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
            self.ui.notice(
                "success",
                "Started a fresh conversation",
                "the previous session remains saved",
            )
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
            self.ui.notice("error", f"Unknown command: {command}", "type /help")
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
                self.ui.notice("info", "No saved sessions", str(self.workspace))
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
            self.ui.notice(
                "success",
                "Session resumed",
                f"{info.id} · {info.provider}/{info.model} · {info.title}",
            )

    async def _list_sessions(self, limit: int) -> None:
        runner = self._require_runner()
        values = await asyncio.to_thread(
            runner.sessions.list_sessions,
            workspace=self.workspace,
            limit=limit,
        )
        if not values:
            self.ui.notice("info", "No saved sessions", str(self.workspace))
            return
        rows: list[tuple[str, object]] = []
        for item in values:
            marker = "◆" if item.id == self.session_id else "◇"
            rows.append(
                (
                    f"{marker} {item.status}",
                    f"{item.id} · {item.provider}/{item.model} · {item.title}",
                )
            )
        self.ui.panel("Session orbit", rows, tone="cyan")

    async def _print_status(self, *, full: bool) -> None:
        route = self._primary_route()
        rows: list[tuple[str, object]] = [
            ("workspace", self.workspace),
            ("provider", route.name),
            ("model", route.model),
            ("session", self.session_id or "(new conversation)"),
        ]
        if full:
            rows.extend(
                [
                    ("mode", self.config.safety.mode),
                    ("approval", self.config.safety.approval),
                    ("network", "on" if self.config.safety.network else "off"),
                    ("sandbox", self.config.sandbox.driver),
                    ("verify", "on" if self.config.agent.auto_verify else "off"),
                ]
            )
        if self.session_id:
            runner = self._require_runner()
            info = await asyncio.to_thread(runner.sessions.get_session, self.session_id)
            usage = await asyncio.to_thread(runner.sessions.usage, self.session_id)
            rows.extend(
                [
                    ("title", info.title),
                    (
                        "usage",
                        f"{usage.total_tokens} tokens · {usage.requests} requests "
                        f"· ${usage.cost_usd:.4f}",
                    ),
                    (
                        "prompt cache",
                        f"{usage.provider_cache_hit_rate:.0%} hit · "
                        f"{usage.cached_input_tokens} read · {usage.cache_write_tokens} written "
                        f"· ${usage.cache_savings_usd:.4f} net saved",
                    ),
                    (
                        "response cache",
                        f"{usage.application_cache_hits} hit · "
                        f"{usage.application_cache_misses} miss · "
                        f"{usage.application_cache_saved_tokens} tokens avoided",
                    ),
                ]
            )
        self.ui.panel("Flight status" if full else "Active session", rows, tone="mint")

    async def _print_history(self, limit: int) -> None:
        if not self.session_id:
            self.ui.notice("info", "No active session", "send a task to begin")
            return
        runner = self._require_runner()
        messages = await asyncio.to_thread(runner.sessions.messages, self.session_id)
        messages = messages[-limit:]
        if not messages:
            self.ui.notice("info", "Session history is empty")
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
            self.ui.role_header(label)
            print(truncate_text(content, 2_000))

    async def _set_provider(self, args: list[str]) -> None:
        if not args:
            current = self._primary_route().name
            rows: list[tuple[str, object]] = []
            for name, provider in sorted(self.config.providers.items()):
                marker = "◆" if name == current else "◇"
                model = provider.model or "(model required)"
                rows.append((f"{marker} {name}", f"{provider.type} · {model}"))
            self.ui.panel("Provider routes", rows, tone="violet")
            return
        name = args[0]
        if name not in self.config.providers:
            self.ui.notice("error", "Unknown provider", name)
            return
        model = args[1] if len(args) > 1 else self.config.providers[name].model

        def change(config: Config) -> None:
            config.agent.provider = name
            config.agent.model = model

        await self._replace_runtime(change, f"provider={name} model={model}")

    async def _set_model(self, args: list[str]) -> None:
        if not args:
            route = self._primary_route()
            self.ui.notice("info", "Active model", f"{route.name}/{route.model}")
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
            self.ui.notice("info", f"Current {label}", getter(self.config))
            return
        value = args[0].lower()
        if value not in choices:
            self.ui.notice(
                "error",
                f"Invalid {label}",
                f"choose {', '.join(sorted(choices))}",
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
            self.ui.notice("info", f"Current {label}", "on" if getter(self.config) else "off")
            return
        try:
            value = _parse_boolean(args[0])
        except ValueError as error:
            self.ui.notice("error", f"Invalid {label}", str(error))
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
            self.ui.notice("error", f"Could not apply {description}", str(error))
            return
        previous = self._require_runner()
        self.runner = replacement
        self.config = candidate
        await previous.close()
        self.ui.notice("success", "Runtime updated", description)

    def _print_tools(self, filter_text: str) -> None:
        schemas = self._require_runner().tools.schemas()
        needle = filter_text.lower()
        rows: list[tuple[str, object]] = []
        for item in schemas:
            value = f"{item['name']} {item.get('description', '')}"
            if needle and needle not in value.lower():
                continue
            rows.append((str(item["name"]), str(item.get("description", ""))))
        if rows:
            self.ui.panel("Tool constellation", rows, tone="cyan")
        else:
            self.ui.notice("info", "No matching tools", filter_text)

    def _print_diagnostics(self) -> None:
        for item in run_diagnostics(self.workspace, self.config):
            self.ui.notice(
                "success" if item.ok else "error",
                f"{'PASS' if item.ok else 'FAIL'} · {item.name}",
                item.message,
            )

    def _rollback(self, args: list[str]) -> None:
        manager = self._require_runner().tool_context.checkpoints
        if not args:
            values = manager.list()
            if not values:
                self.ui.notice("info", "No checkpoints")
                return
            rows = [
                (item.id, f"{item.created_at} · {item.label} · {len(item.files)} file(s)")
                for item in values
            ]
            self.ui.panel("Recovery points", rows, tone="warning")
            return
        checkpoint = manager.restore(args[0])
        self.ui.notice(
            "success", "Checkpoint restored", f"{checkpoint.id} · {len(checkpoint.files)} file(s)"
        )

    async def _read_multiline(self) -> str | None:
        self.ui.notice("info", "Paste a multiline prompt", "finish with /end")
        lines: list[str] = []
        prompt = self.ui.inline_prompt("paste") if _is_tty(sys.stdin) else ""
        while True:
            try:
                line = _terminal_input(prompt)
            except EOFError:
                break
            except KeyboardInterrupt:
                print()
                self.ui.notice("warning", "Multiline prompt cancelled")
                return None
            if line.strip() == "/end":
                break
            lines.append(line)
        value = "\n".join(lines).strip()
        if not value:
            self.ui.notice("warning", "Multiline prompt was empty")
            return None
        return value

    def _print_help(self) -> None:
        self.ui.help(
            [
                (
                    "Session",
                    [
                        ("/status", "runtime, safety, session, and usage"),
                        ("/session", "active workspace, route, and session"),
                        ("/sessions [N]", "recent sessions in this workspace"),
                        ("/resume [ID|last]", "return to a durable session"),
                        ("/history [N]", "recent persisted messages"),
                        ("/new  /clear", "begin fresh; keep the previous session"),
                    ],
                ),
                (
                    "Runtime",
                    [
                        ("/provider [NAME MODEL]", "inspect or switch provider route"),
                        ("/model [MODEL]", "inspect or switch active model"),
                        ("/mode [VALUE]", "plan | workspace-write | full"),
                        ("/approval [VALUE]", "never | on-risk | always"),
                        ("/network [on|off]", "toggle model-initiated network tools"),
                        ("/verify [on|off]", "toggle post-change verification"),
                        ("/sandbox [VALUE]", "native | docker"),
                    ],
                ),
                (
                    "Utilities",
                    [
                        ("/tools [FILTER]", "effective built-in, plugin, and MCP tools"),
                        ("/doctor", "diagnose the current runtime"),
                        ("/rollback [ID]", "list or restore recovery points"),
                        ("/paste", "multiline input; finish with /end"),
                        ("/exit  /quit", "leave Borealis with the session saved"),
                    ],
                ),
            ]
        )
        self.ui.notice(
            "info",
            "Controls",
            "Ctrl+C cancels a turn · // sends a prompt beginning with /",
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
        self.ui.notice("warning", "Cancelling current turn", "press Ctrl+C again to force")


def _turn_footer(result: AgentResult) -> str:
    parts = [
        result.stop_reason.value,
        f"{result.turns} model turn{'s' if result.turns != 1 else ''}",
        f"{result.usage.total_tokens} tokens",
    ]
    if result.usage.cost_usd:
        parts.append(f"${result.usage.cost_usd:.4f}")
    if result.usage.cached_input_tokens or result.usage.cache_write_tokens:
        parts.append(
            f"prompt cache {result.usage.provider_cache_hit_rate:.0%} hit "
            f"({result.usage.cached_input_tokens} read/{result.usage.cache_write_tokens} written)"
        )
    if result.usage.application_cache_hits:
        parts.append(
            f"response cache hit · {result.usage.application_cache_saved_tokens} tokens avoided"
        )
    if result.changed_files:
        parts.append(f"{len(result.changed_files)} changed file(s)")
    if result.verification is not None:
        parts.append("verified" if result.verification.get("ok") else "verification failed")
    return " · ".join(parts)


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
