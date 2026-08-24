"""Focused git tools with explicit mutation boundaries."""

from __future__ import annotations

from typing import Any

from ..models import Effect, ToolResult
from .base import MutationScope, Tool, ToolContext, nullable, object_schema


async def _git(context: ToolContext, args: list[str], timeout: int = 120) -> ToolResult:
    safety_flags = ["-c", "core.fsmonitor=false", "-c", "diff.external="]
    if not context.config.safety.allow_git_hooks:
        safety_flags += ["-c", "core.hooksPath=/dev/null"]
    result = await context.process.run(
        ["git", *safety_flags, *args],
        cwd=context.workspace,
        timeout=timeout,
        shell=False,
    )
    return ToolResult(result.render(), is_error=not result.ok, metadata={"exit_code": result.exit_code, "duration_ms": result.duration_ms})


class GitStatusTool(Tool):
    name = "git_status"
    description = "Show branch and porcelain-v2 repository status."
    effect = Effect.READ
    concurrent = True
    parameters = object_schema({})

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        return await _git(context, ["status", "--short", "--branch"])


class GitDiffTool(Tool):
    name = "git_diff"
    description = "Show the current working-tree or staged diff, optionally for one path."
    effect = Effect.READ
    concurrent = True
    parameters = object_schema({
        "path": nullable("string"),
        "staged": {"type": "boolean"},
        "stat": {"type": "boolean"},
    })

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        args = ["diff", "--no-ext-diff", "--no-textconv"]
        if arguments["staged"]:
            args.append("--cached")
        if arguments["stat"]:
            args.append("--stat")
        if arguments.get("path"):
            resolved = context.roots.resolve(arguments["path"])
            args += ["--", str(resolved.path)]
        return await _git(context, args)


class GitLogTool(Tool):
    name = "git_log"
    description = "Show a concise recent commit log."
    effect = Effect.READ
    concurrent = True
    parameters = object_schema({"limit": {"type": "integer", "minimum": 1, "maximum": 100}})

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        return await _git(context, ["log", f"-{int(arguments['limit'])}", "--date=short", "--pretty=format:%h %ad %an %s"])


class GitCommitTool(Tool):
    name = "git_commit"
    description = "Create a local git commit from already staged changes. Never stages files automatically."
    effect = Effect.WRITE
    default_risk = "high"
    parameters = object_schema({"message": {"type": "string", "minLength": 1, "maxLength": 1000}})

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        args = ["commit", "-m", str(arguments["message"])]
        if not context.config.safety.allow_git_hooks:
            args.append("--no-verify")
        return await _git(context, args, timeout=300)


class GitPushTool(Tool):
    name = "git_push"
    description = "Push the current branch to a configured remote. Requires explicit network and publish permission."
    effect = Effect.NETWORK
    mutation_scope = MutationScope.EXTERNAL
    default_risk = "critical"
    parameters = object_schema({
        "remote": {"type": "string", "minLength": 1},
        "branch": nullable("string"),
    })

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        args = ["push", str(arguments["remote"])]
        if arguments.get("branch"):
            args.append(str(arguments["branch"]))
        return await _git(context, args, timeout=600)
