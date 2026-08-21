"""Lazy repository-map, instruction, and skill tools."""

from __future__ import annotations

from typing import Any

from ..context import ContextBuilder
from ..models import Effect, ToolResult
from .base import Tool, ToolContext, object_schema


class RepoMapTool(Tool):
    name = "repo_map"
    description = "Build a token-bounded, symbol-aware repository map ranked for a query."
    effect = Effect.READ
    concurrent = True
    parameters = object_schema({
        "query": {"type": "string", "maxLength": 2000},
        "max_chars": {"type": "integer", "minimum": 1000, "maximum": 100000},
    })

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        builder: ContextBuilder = context.metadata["context_builder"]
        output = builder.repo_map.build(query=str(arguments["query"]), max_chars=int(arguments["max_chars"]))
        return ToolResult(output)


class ReadSkillTool(Tool):
    name = "read_skill"
    description = "Load the full body of one project-local SKILL.md discovered in the system skill catalog."
    effect = Effect.READ
    concurrent = True
    parameters = object_schema({"name": {"type": "string", "minLength": 1}})

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        builder: ContextBuilder = context.metadata["context_builder"]
        skill = builder.skills.get(str(arguments["name"]))
        if skill is None:
            return ToolResult(f"Unknown skill: {arguments['name']}", is_error=True)
        return ToolResult(
            f"# Skill: {skill.name}\nSource: {skill.path.relative_to(context.workspace).as_posix()}\n\n{skill.body}",
            metadata={"name": skill.name, "path": str(skill.path)},
        )


class ReadInstructionsTool(Tool):
    name = "read_instructions"
    description = "Read all hierarchical instruction files applicable to a workspace path."
    effect = Effect.READ
    concurrent = True
    parameters = object_schema({"path": {"type": "string", "minLength": 1}})

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        builder: ContextBuilder = context.metadata["context_builder"]
        target = context.roots.resolve(arguments["path"]).path
        documents = builder.instructions.for_path(target)
        if not documents:
            return ToolResult("No applicable instruction files")
        return ToolResult("\n\n".join(f"## {item.relative_path}\n\n{item.content}" for item in documents))
