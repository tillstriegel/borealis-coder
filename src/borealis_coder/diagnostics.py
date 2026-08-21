"""Operational diagnostics used by ``borealis doctor`` and release validation."""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
from pathlib import Path

from .auth import ChatGPTCredentialManager
from .config import Config
from .errors import Diagnostic


def run_diagnostics(workspace: Path, config: Config) -> list[Diagnostic]:
    items: list[Diagnostic] = []
    items.append(Diagnostic("python", sys.version_info >= (3, 11), f"Python {sys.version.split()[0]}"))
    items.append(Diagnostic("workspace", workspace.is_dir(), str(workspace)))
    git = shutil.which("git")
    items.append(Diagnostic("git", bool(git), git or "git executable not found"))
    try:
        config.storage_dir.mkdir(parents=True, exist_ok=True)
        probe = config.storage_dir / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        items.append(Diagnostic("storage", True, str(config.storage_dir)))
    except OSError as error:
        items.append(Diagnostic("storage", False, str(error)))
    try:
        connection = sqlite3.connect(config.database_path)
        value = connection.execute("PRAGMA integrity_check").fetchone()[0]
        connection.close()
        items.append(Diagnostic("database", value == "ok", str(value)))
    except sqlite3.Error as error:
        items.append(Diagnostic("database", False, str(error)))
    try:
        name, provider = config.provider()
        if provider.type == "chatgpt":
            status = ChatGPTCredentialManager(provider).status()
            message = f"{name} ({provider.type}); {status.message}"
            if status.email:
                message += f"; {status.email}"
            if status.plan:
                message += f"; plan={status.plan}"
            if status.source:
                message += f"; source={status.source}"
            items.append(Diagnostic("provider", status.available, message, status.to_dict()))
        else:
            key_present = (
                not provider.api_key_env
                or bool(os.getenv(provider.api_key_env))
                or provider.type in {"mock", "openai_compatible"}
            )
            items.append(Diagnostic("provider", key_present, f"{name} ({provider.type})" + ("" if key_present else f"; missing {provider.api_key_env}")))
    except Exception as error:
        items.append(Diagnostic("provider", False, str(error)))
    if config.sandbox.driver == "docker":
        docker = shutil.which("docker")
        items.append(Diagnostic("docker", bool(docker), docker or "docker executable not found"))
    else:
        items.append(Diagnostic("sandbox", True, "native policy/resource boundary; Docker offers stronger isolation"))
    for name, server in config.mcp_servers.items():
        if not server.enabled:
            continue
        if server.type == "stdio":
            executable = shutil.which(server.command) if server.command else None
            items.append(Diagnostic(f"mcp:{name}", bool(executable), executable or f"command not found: {server.command}"))
        elif server.type == "http":
            items.append(Diagnostic(f"mcp:{name}", bool(server.url), server.url or "missing URL"))
        else:
            items.append(Diagnostic(f"mcp:{name}", False, f"unsupported transport {server.type}"))
    return items
