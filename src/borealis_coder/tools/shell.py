"""Policy-gated command execution."""

from __future__ import annotations

from typing import Any

from ..models import Effect, ToolResult
from ..safety.commands import assess_command
from .base import Tool, ToolContext, nullable, object_schema


class ShellTool(Tool):
    name = "shell"
    description = "Run a bounded shell command in the workspace. High-risk and network operations require explicit policy permission."
    effect = Effect.EXECUTE
    default_risk = "high"
    parameters = object_schema({
        "command": {"type": "string", "minLength": 1, "maxLength": 20_000},
        "cwd": {"type": "string"},
        "timeout_seconds": nullable("integer", minimum=1, maximum=3600),
        "description": {"type": "string", "maxLength": 500},
    })

    def risk(self, arguments: dict[str, Any]) -> str:
        risk = assess_command(str(arguments.get("command") or "")).risk
        return {0: "low", 1: "high", 2: "high", 3: "critical", 4: "critical"}.get(int(risk), "high")

    def approval_description(self, arguments: dict[str, Any]) -> str:
        return str(arguments.get("description") or arguments.get("command") or "Run shell command")

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        cwd = context.roots.resolve(arguments.get("cwd") or ".", must_exist=True, kind="dir").path
        timeout = int(arguments.get("timeout_seconds") or context.config.safety.command_timeout_seconds)
        result = await context.process.run(str(arguments["command"]), cwd=cwd, timeout=timeout, shell=True)
        return ToolResult(result.render(), is_error=not result.ok, metadata={
            "exit_code": result.exit_code, "duration_ms": result.duration_ms,
            "timed_out": result.timed_out, "cwd": context.roots.display(cwd),
        })
