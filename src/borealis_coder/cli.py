"""Dependency-free command-line interface for Borealis Coder."""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Any

from . import __version__
from .agent import build_runner
from .auth import (
    ChatGPTCredentialManager,
    launch_codex_login,
    logout_managed_chatgpt,
    managed_codex_home,
)
from .config import SAMPLE_CONFIG, Config, load_config
from .diagnostics import run_diagnostics
from .errors import BorealisError, ConfigurationError
from .interactive import InteractiveCLI
from .protocol import ACPServer
from .safety import ApprovalRequest, CheckpointManager, WorkspaceRoots
from .terminal import AuroraUI, ConsoleRenderer
from .util import atomic_write_text, json_dumps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="borealis",
        description=(
            "Policy-first, provider-portable, resumable AI coding harness. "
            "Run without a subcommand to open the interactive coding shell."
        ),
        epilog="""Examples:
  borealis
  borealis 'Fix the failing tests'
  borealis --continue
  borealis resume --last
  borealis run --json 'Explain this repository'""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"borealis-coder {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one autonomous coding task")
    _common_runtime_arguments(run, include_non_interactive=True)
    run.add_argument("prompt", nargs="*", help="Task prompt; reads stdin when omitted")
    run.add_argument("--resume", metavar="SESSION_ID", help="Resume a durable session")
    run.add_argument("--json", action="store_true", help="Emit JSON result and JSONL events")

    chat = sub.add_parser(
        "chat",
        aliases=["interactive"],
        help="Start the persistent interactive coding shell",
    )
    _common_runtime_arguments(chat)
    chat.add_argument("prompt", nargs="*", help="Optional first prompt")
    session = chat.add_mutually_exclusive_group()
    session.add_argument("--resume", metavar="SESSION_ID", help="Resume a durable session")
    session.add_argument(
        "-c",
        "--continue",
        dest="resume_last",
        action="store_true",
        help="Resume the most recent session in this workspace",
    )
    _interactive_output_arguments(chat)

    resume = sub.add_parser(
        "resume",
        help="Resume a saved session in the interactive coding shell",
    )
    _common_runtime_arguments(resume)
    resume.add_argument(
        "values",
        nargs="*",
        help="SESSION_ID followed by an optional first prompt; --last treats all values as prompt",
    )
    resume.add_argument(
        "--last",
        action="store_true",
        help="Resume the most recent session in this workspace",
    )
    _interactive_output_arguments(resume)

    init = sub.add_parser("init", help="Create workspace configuration and directories")
    init.add_argument("--workspace", type=Path, default=Path.cwd())
    init.add_argument("--force", action="store_true")

    auth = sub.add_parser("auth", help="Manage ChatGPT/Codex account authentication")
    auth_sub = auth.add_subparsers(dest="auth_action", required=True)
    auth_status = auth_sub.add_parser("status", help="Inspect available ChatGPT login credentials")
    auth_status.add_argument("--workspace", type=Path, default=Path.cwd())
    auth_status.add_argument("--config", type=Path)
    auth_status.add_argument("--json", action="store_true")
    auth_login = auth_sub.add_parser("login", help="Sign in with ChatGPT through the official Codex CLI")
    auth_login.add_argument("--device-code", action="store_true", help="Use Codex device-code login")
    auth_login.add_argument("--codex-home", type=Path, help="Credential directory; defaults to a Borealis-managed directory")
    auth_login.add_argument("--codex-command", default="codex")
    auth_logout = auth_sub.add_parser("logout", help="Revoke/delete a selected Codex credential store")
    auth_logout.add_argument("--codex-home", type=Path, help="Defaults to the Borealis-managed credential directory")
    auth_logout.add_argument("--codex-command", default="codex")

    doctor = sub.add_parser("doctor", help="Validate configuration and runtime dependencies")
    doctor.add_argument("--workspace", type=Path, default=Path.cwd())
    doctor.add_argument("--config", type=Path)
    doctor.add_argument("--json", action="store_true")

    tools = sub.add_parser("tools", help="List the effective tool palette")
    tools.add_argument("--workspace", type=Path, default=Path.cwd())
    tools.add_argument("--config", type=Path)
    tools.add_argument("--json", action="store_true")

    config_cmd = sub.add_parser("config", help="Print merged, redacted configuration")
    config_cmd.add_argument("--workspace", type=Path, default=Path.cwd())
    config_cmd.add_argument("--config", type=Path)

    sessions = sub.add_parser("sessions", help="List, inspect, export, or delete sessions")
    sessions.add_argument("action", choices=["list", "show", "export", "delete"])
    sessions.add_argument("session_id", nargs="?")
    sessions.add_argument("--workspace", type=Path, default=Path.cwd())
    sessions.add_argument("--config", type=Path)
    sessions.add_argument("--output", type=Path)
    sessions.add_argument("--all-workspaces", action="store_true")

    rollback = sub.add_parser("rollback", help="List or restore pre-mutation checkpoints")
    rollback.add_argument("checkpoint_id", nargs="?")
    rollback.add_argument("--workspace", type=Path, default=Path.cwd())
    rollback.add_argument("--config", type=Path)

    sub.add_parser("acp", help="Serve ACP v2 over stdio")

    evaluate = sub.add_parser("eval", help="Run deterministic offline acceptance checks")
    evaluate.add_argument("--json", action="store_true")

    return parser


def _interactive_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream assistant text as it arrives",
    )
    parser.add_argument(
        "--show-tool-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show bounded tool output in addition to tool status",
    )
    parser.add_argument(
        "--history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist local prompt history in the Borealis data directory",
    )


def _common_runtime_arguments(
    parser: argparse.ArgumentParser, *, include_non_interactive: bool = False
) -> None:
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--fallback", action="append", default=[])
    parser.add_argument("--mode", choices=["plan", "workspace-write", "full"])
    parser.add_argument("--approval", choices=["never", "on-risk", "always"])
    parser.add_argument("--network", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--sandbox", choices=["native", "docker"])
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--max-cost", type=float)
    if include_non_interactive:
        parser.add_argument("--non-interactive", action="store_true")


_TOP_LEVEL_COMMANDS = {
    "run",
    "chat",
    "interactive",
    "resume",
    "init",
    "auth",
    "doctor",
    "tools",
    "config",
    "sessions",
    "rollback",
    "acp",
    "eval",
}


def _normalize_argv(argv: list[str] | None) -> list[str]:
    """Make the interactive shell the default while preserving subcommands.

    ``borealis`` and ``borealis <prompt>`` become ``borealis chat``. Explicit
    command names, top-level help, and version requests retain their existing
    behavior.
    """

    values = list(sys.argv[1:] if argv is None else argv)
    if not values:
        return ["chat"]
    if values[0] in {"-h", "--help", "--version"}:
        return values
    if values[0] in _TOP_LEVEL_COMMANDS:
        return values
    return ["chat", *values]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(argv))
    coroutine = _main(args)
    try:
        return asyncio.run(coroutine)
    except KeyboardInterrupt:
        coroutine.close()
        print("\nCancelled.", file=sys.stderr)
        return 130
    except (BorealisError, ConfigurationError, OSError, ValueError) as error:
        coroutine.close()
        print(f"error: {error}", file=sys.stderr)
        return 2


async def _main(args: argparse.Namespace) -> int:
    if args.command == "run":
        return await _run(args)
    if args.command in {"chat", "interactive"}:
        return await _chat(args)
    if args.command == "resume":
        return await _resume_interactive(args)
    if args.command == "init":
        return _init(args)
    if args.command == "doctor":
        return _doctor(args)
    if args.command == "auth":
        return await _auth(args)
    if args.command == "tools":
        return await _tools(args)
    if args.command == "config":
        return _config(args)
    if args.command == "sessions":
        return await _sessions(args)
    if args.command == "rollback":
        return _rollback(args)
    if args.command == "acp":
        await ACPServer().serve()
        return 0
    if args.command == "eval":
        return await _eval(args)
    raise AssertionError(args.command)


async def _run(args: argparse.Namespace) -> int:
    workspace = args.workspace.expanduser().resolve()
    prompt = " ".join(args.prompt).strip()
    if not prompt:
        prompt = sys.stdin.read().strip()
    if not prompt:
        raise ValueError("A prompt is required")
    config = _runtime_config(args, workspace)
    renderer = ConsoleRenderer(json_events=args.json, quiet=args.json)
    approval = None if args.non_interactive else _terminal_approval
    runner = await build_runner(workspace, config=config, approval_callback=approval, interactive=not args.non_interactive)
    runner.events.subscribe(renderer.handle)
    try:
        result = await runner.run(prompt, session_id=args.resume)
    finally:
        renderer.finish_turn()
        await runner.close()
    if args.json:
        print(json_dumps(result.to_dict()))
    else:
        if result.text and not renderer.has_rendered(result.text):
            print(result.text)
        print(_result_footer(result), file=sys.stderr)
    return 0 if result.stop_reason.value == "end_turn" and not (result.verification and not result.verification.get("ok", True)) else 1


async def _chat(args: argparse.Namespace) -> int:
    return await _launch_interactive(
        args,
        resume=args.resume,
        resume_last=args.resume_last,
        initial_prompt=" ".join(args.prompt),
    )


async def _resume_interactive(args: argparse.Namespace) -> int:
    values = list(args.values)
    if args.last or not values:
        resume = None
        resume_last = True
        initial_prompt = " ".join(values)
    else:
        resume = values[0]
        resume_last = False
        initial_prompt = " ".join(values[1:])
    return await _launch_interactive(
        args,
        resume=resume,
        resume_last=resume_last,
        initial_prompt=initial_prompt,
    )


async def _launch_interactive(
    args: argparse.Namespace,
    *,
    resume: str | None,
    resume_last: bool,
    initial_prompt: str,
) -> int:
    workspace = args.workspace.expanduser().resolve()
    config = _runtime_config(args, workspace)
    shell = InteractiveCLI(
        workspace=workspace,
        config=config,
        approval_callback=_terminal_approval,
        resume=resume,
        resume_last=resume_last,
        initial_prompt=initial_prompt,
        stream_text=args.stream,
        show_tool_output=args.show_tool_output,
        history_enabled=args.history,
    )
    return await shell.run()


def _init(args: argparse.Namespace) -> int:
    workspace = args.workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    config_dir = workspace / ".borealis"
    paths = {
        config_dir / "config.toml": SAMPLE_CONFIG,
        workspace / ".borealisignore": ".git/\n.borealis/\nnode_modules/\n.venv/\ndist/\nbuild/\n",
    }
    for path, content in paths.items():
        if path.exists() and not args.force:
            print(f"kept {path}")
            continue
        atomic_write_text(path, content)
        print(f"created {path}")
    (config_dir / "skills").mkdir(parents=True, exist_ok=True)
    (config_dir / "plugins").mkdir(parents=True, exist_ok=True)
    return 0


async def _auth(args: argparse.Namespace) -> int:
    if args.auth_action == "login":
        home = (args.codex_home or managed_codex_home()).expanduser().resolve()
        auth_file = await asyncio.to_thread(
            launch_codex_login,
            codex_home=home,
            codex_command=args.codex_command,
            device_code=args.device_code,
        )
        print(f"ChatGPT login ready: {auth_file}")
        print('Use `borealis --provider chatgpt` or set agent.provider = "chatgpt".')
        return 0
    if args.auth_action == "logout":
        home = (args.codex_home or managed_codex_home()).expanduser().resolve()
        auth_file = await asyncio.to_thread(
            logout_managed_chatgpt,
            codex_home=home,
            codex_command=args.codex_command,
        )
        print(f"ChatGPT credentials removed: {auth_file}")
        return 0
    if args.auth_action == "status":
        workspace = args.workspace.expanduser().resolve()
        config = load_config(
            workspace,
            explicit_path=args.config,
            overrides={"agent": {"provider": "mock"}},
        )
        provider = config.providers.get("chatgpt")
        status = ChatGPTCredentialManager(provider).status()
        if args.json:
            print(json_dumps(status.to_dict(), pretty=True))
        else:
            marker = "READY" if status.available else "MISSING"
            print(f"{marker}  ChatGPT/Codex: {status.message}")
            if status.source:
                print(f"source: {status.source}")
            if status.email:
                print(f"account: {status.email}")
            if status.plan:
                print(f"plan: {status.plan}")
            if status.expires_at:
                print(f"expires_at: {status.expires_at}")
        return 0 if status.available else 1
    raise AssertionError(args.auth_action)


def _doctor(args: argparse.Namespace) -> int:
    workspace = args.workspace.expanduser().resolve()
    config = load_config(workspace, explicit_path=args.config)
    diagnostics = run_diagnostics(workspace, config)
    if args.json:
        print(json_dumps([{"name": item.name, "ok": item.ok, "message": item.message, "details": item.details} for item in diagnostics], pretty=True))
    else:
        for item in diagnostics:
            print(f"{'PASS' if item.ok else 'FAIL'}  {item.name}: {item.message}")
    return 0 if all(item.ok for item in diagnostics) else 1


async def _tools(args: argparse.Namespace) -> int:
    workspace = args.workspace.expanduser().resolve()
    config = load_config(workspace, explicit_path=args.config, overrides={"agent": {"provider": "mock"}})
    runner = await build_runner(workspace, config=config, interactive=False)
    try:
        schemas = runner.tools.schemas()
    finally:
        await runner.close()
    if args.json:
        print(json_dumps(schemas, pretty=True))
    else:
        for item in schemas:
            print(f"{item['name']:<28} {item.get('description', '')}")
    return 0


def _config(args: argparse.Namespace) -> int:
    workspace = args.workspace.expanduser().resolve()
    config = load_config(workspace, explicit_path=args.config)
    print(json_dumps(config.to_dict(), pretty=True))
    return 0


async def _sessions(args: argparse.Namespace) -> int:
    workspace = args.workspace.expanduser().resolve()
    config = load_config(workspace, explicit_path=args.config, overrides={"agent": {"provider": "mock"}})
    runner = await build_runner(workspace, config=config, interactive=False)
    try:
        if args.action == "list":
            values = await asyncio.to_thread(runner.sessions.list_sessions, workspace=None if args.all_workspaces else workspace)
            for item in values:
                print(f"{item.id}\t{item.updated_at}\t{item.status}\t{item.provider}/{item.model}\t{item.title}")
            return 0
        if not args.session_id:
            raise ValueError(f"sessions {args.action} requires SESSION_ID")
        if args.action == "show":
            print(json_dumps(await asyncio.to_thread(runner.sessions.export, args.session_id), pretty=True))
        elif args.action == "export":
            data = json_dumps(await asyncio.to_thread(runner.sessions.export, args.session_id), pretty=True) + "\n"
            output = args.output or Path(f"{args.session_id}.json")
            atomic_write_text(output, data)
            print(output.resolve())
        elif args.action == "delete":
            await asyncio.to_thread(runner.sessions.delete_session, args.session_id)
            print(f"deleted {args.session_id}")
    finally:
        await runner.close()
    return 0


def _rollback(args: argparse.Namespace) -> int:
    workspace = args.workspace.expanduser().resolve()
    config = load_config(workspace, explicit_path=args.config)
    roots = WorkspaceRoots(workspace, allow_outside=config.safety.allow_outside_workspace)
    manager = CheckpointManager(roots, enabled=True, max_bytes=config.safety.checkpoint_max_bytes)
    if not args.checkpoint_id:
        for item in manager.list():
            print(f"{item.id}\t{item.created_at}\t{item.label}\t{len(item.files)} file(s)")
        return 0
    checkpoint = manager.restore(args.checkpoint_id)
    print(f"restored {checkpoint.id}: {len(checkpoint.files)} file(s)")
    return 0


async def _eval(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="borealis-eval-") as temp:
        workspace = Path(temp)
        (workspace / "pyproject.toml").write_text('[project]\nname="fixture"\nversion="0.1.0"\n', encoding="utf-8")
        config = load_config(workspace, overrides={
            "agent": {"provider": "mock", "auto_verify": False},
            "storage": {"directory": str(workspace / ".data")},
            "safety": {"approval": "never"},
        })
        runner = await build_runner(workspace, config=config, interactive=False)
        try:
            result = await runner.run("OFFLINE_WRITE_DEMO")
            file_ok = (workspace / "borealis-demo.txt").read_text(encoding="utf-8").startswith("Created by")
            resumed = await runner.run("Confirm state", session_id=result.session_id)
            checks = {
                "agent_end_turn": result.stop_reason.value == "end_turn",
                "tool_cycle": file_ok,
                "durable_resume": resumed.session_id == result.session_id and resumed.turns >= 1,
                "events_persisted": bool(runner.sessions.events(result.session_id)),
                "usage_persisted": runner.sessions.usage(result.session_id).requests >= 3,
                "checkpoint_created": bool(runner.tool_context.checkpoints.list()),
            }
        finally:
            await runner.close()
    if args.json:
        print(json_dumps({"ok": all(checks.values()), "checks": checks}, pretty=True))
    else:
        for name, ok in checks.items():
            print(f"{'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(checks.values()) else 1


def _runtime_config(args: argparse.Namespace, workspace: Path) -> Config:
    overrides: dict[str, Any] = {"agent": {}, "safety": {}, "sandbox": {}}
    mapping = {
        ("agent", "provider"): args.provider,
        ("agent", "model"): args.model,
        ("agent", "provider_fallbacks"): args.fallback or None,
        ("agent", "max_turns"): args.max_turns,
        ("agent", "max_cost_usd"): args.max_cost,
        ("agent", "auto_verify"): args.verify,
        ("safety", "mode"): args.mode,
        ("safety", "approval"): args.approval,
        ("safety", "network"): args.network,
        ("sandbox", "driver"): args.sandbox,
    }
    for (section, key), value in mapping.items():
        if value is not None:
            overrides[section][key] = value
    return load_config(workspace, explicit_path=args.config, overrides=overrides)



def _terminal_approval(request: ApprovalRequest) -> str:
    ui = AuroraUI(sys.stderr)
    ui.panel(
        "Approval required",
        [
            ("tool", request.tool_name),
            ("risk", request.decision.risk),
            ("reason", request.decision.reason),
            ("request", request.arguments_preview),
        ],
        tone="warning",
    )
    while True:
        answer = input(ui.inline_prompt("allow [y] once · [a] session · [n] reject")).strip().lower()
        if answer in {"y", "yes"}:
            return "allow_once"
        if answer in {"a", "always"}:
            return "allow_always"
        if answer in {"n", "no", ""}:
            return "no"


def _result_footer(result) -> str:  # type: ignore[no-untyped-def]
    parts = [
        f"session={result.session_id}", f"stop={result.stop_reason.value}",
        f"turns={result.turns}", f"tokens={result.usage.total_tokens}",
        f"cost=${result.usage.cost_usd:.4f}",
    ]
    if result.changed_files:
        parts.append(f"changed={len(result.changed_files)}")
    if result.usage.cached_input_tokens or result.usage.cache_write_tokens:
        parts.append(f"cache_hit={result.usage.provider_cache_hit_rate:.1%}")
        parts.append(f"cache_read={result.usage.cached_input_tokens}")
        parts.append(f"cache_write={result.usage.cache_write_tokens}")
        parts.append(f"cache_saved=${result.usage.cache_savings_usd:.4f}")
    if result.usage.application_cache_hits:
        parts.append(f"response_cache_hits={result.usage.application_cache_hits}")
        parts.append(f"tokens_avoided={result.usage.application_cache_saved_tokens}")
    if result.verification is not None:
        parts.append(f"verified={result.verification.get('ok')}")
    if result.error:
        parts.append(f"error={result.error}")
    return " · ".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
