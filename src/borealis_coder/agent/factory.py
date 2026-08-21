"""Runtime assembly for CLI, ACP, tests, and embedding applications."""

from __future__ import annotations

from pathlib import Path

from ..config import Config, load_config
from ..context import ContextBuilder
from ..events import EventBus, JsonlTrace
from ..mcp import MCPManager
from ..plugins import load_entrypoint_tools, load_workspace_plugins
from ..providers.registry import DEFAULT_PROVIDER_REGISTRY, ProviderRegistry
from ..safety import (
    ApprovalManager,
    CheckpointManager,
    PolicyEngine,
    WorkspaceRoots,
    build_process_driver,
)
from ..sessions import SessionStore
from ..tools import ToolContext, build_builtin_registry
from .runner import AgentRunner, ProviderRoute


async def build_runner(
    workspace: Path,
    *,
    config: Config | None = None,
    approval_callback=None,  # type: ignore[no-untyped-def]
    interactive: bool = True,
    provider_registry: ProviderRegistry | None = None,
    additional_roots: list[Path] | None = None,
) -> AgentRunner:
    workspace = workspace.resolve()
    config = config or load_config(workspace)
    roots = WorkspaceRoots(workspace, additional_roots, allow_outside=config.safety.allow_outside_workspace)
    sessions = SessionStore(config.database_path)
    trace = JsonlTrace(config.storage_dir / "traces" / "events.jsonl") if config.storage.trace_jsonl else None
    events = EventBus(trace=trace)
    events.subscribe(sessions.append_event)
    context_builder = ContextBuilder(workspace, config)
    tools = build_builtin_registry()
    load_entrypoint_tools(tools)
    load_workspace_plugins(workspace, tools)
    mcp = MCPManager(workspace, config)
    await mcp.connect_all(tools)
    process_network_isolated = (
        config.sandbox.driver == "docker"
        and (not config.safety.network or not config.sandbox.docker_network)
    )
    policy = PolicyEngine(
        config.safety,
        interactive=interactive,
        process_network_isolated=process_network_isolated,
    )
    approvals = ApprovalManager(approval_callback, cache=config.safety.approval_cache == "session")
    process = build_process_driver(roots, config.safety, config.sandbox)
    checkpoints = CheckpointManager(roots, enabled=config.safety.checkpoints, max_bytes=config.safety.checkpoint_max_bytes)
    tool_context = ToolContext(
        workspace=workspace, roots=roots, config=config, events=events, policy=policy,
        approvals=approvals, process=process, checkpoints=checkpoints,
        session_id="", run_id="", metadata={"context_builder": context_builder},
    )
    registry = provider_registry or DEFAULT_PROVIDER_REGISTRY
    routes: list[ProviderRoute] = []
    primary_name, primary_model, primary = registry.create(config)
    routes.append(ProviderRoute(primary_name, primary_model, primary))
    for name in getattr(config.agent, "provider_fallbacks", []):
        fallback_name, fallback_model, fallback = registry.create(config, name)
        if fallback_name not in {item.name for item in routes}:
            routes.append(ProviderRoute(fallback_name, fallback_model, fallback))
    runner = AgentRunner(
        workspace=workspace, config=config, providers=routes, tools=tools,
        tool_context=tool_context, context_builder=context_builder,
        sessions=sessions, events=events,
    )
    runner.mcp_manager = mcp
    return runner
