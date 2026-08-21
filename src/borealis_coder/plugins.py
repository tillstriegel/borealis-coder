"""Explicit extension loading through Python entry points or opted-in workspace modules."""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path

from .tools.base import Tool, ToolRegistry


def load_entrypoint_tools(registry: ToolRegistry) -> list[str]:
    loaded: list[str] = []
    try:
        entry_points = metadata.entry_points(group="borealis.tools")
    except TypeError:  # Python 3.11 compatibility with older importlib metadata API
        entry_points = metadata.entry_points().get("borealis.tools", [])  # type: ignore[assignment]
    for entry in entry_points:
        value = entry.load()
        tools = value() if callable(value) and not isinstance(value, type) else value
        if isinstance(tools, Tool):
            candidates: Iterable[object] = (tools,)
        elif isinstance(tools, Iterable):
            candidates = tools
        else:
            raise TypeError(f"Entry point {entry.name} returned a non-iterable value")
        for tool in candidates:
            if not isinstance(tool, Tool):
                raise TypeError(f"Entry point {entry.name} returned non-Tool value")
            registry.register(tool)
            loaded.append(tool.name)
    return loaded


def load_workspace_plugins(workspace: Path, registry: ToolRegistry) -> list[str]:
    """Load trusted local plugins only after an explicit environment opt-in."""
    if os.getenv("BOREALIS_ENABLE_WORKSPACE_PLUGINS", "").lower() not in {"1", "true", "yes"}:
        return []
    directory = workspace / ".borealis" / "plugins"
    loaded: list[str] = []
    if not directory.is_dir():
        return loaded
    for path in sorted(directory.glob("*.py")):
        name = f"borealis_workspace_plugin_{path.stem}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        register = getattr(module, "register", None)
        if not callable(register):
            raise TypeError(f"Workspace plugin {path} must expose register(registry)")
        before = set(registry.names())
        register(registry)
        loaded.extend(sorted(set(registry.names()) - before))
    return loaded
