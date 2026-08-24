"""Tool contracts, schema validation, and policy-aware execution."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from ..config import Config
from ..errors import BudgetExceeded, Cancelled, PolicyError, ToolError, ToolValidationError
from ..events import EventBus
from ..models import Effect, ToolCall, ToolResult
from ..safety import (
    ApprovalManager,
    ApprovalRequest,
    CheckpointManager,
    PolicyEngine,
    ProcessDriver,
    WorkspaceRoots,
)
from ..util import json_dumps, truncate_text


@dataclass(slots=True)
class ToolContext:
    workspace: Path
    roots: WorkspaceRoots
    config: Config
    events: EventBus
    policy: PolicyEngine
    approvals: ApprovalManager
    process: ProcessDriver
    checkpoints: CheckpointManager
    session_id: str
    run_id: str
    tool_call_id: str = ""
    changed_files: set[str] = field(default_factory=set)
    changed_roots: set[Path] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    mutation_tracking: str = "complete"


class MutationScope(StrEnum):
    """How the runner accounts for a tool's mutations."""

    NONE = "none"
    TRACKED = "tracked"
    WORKSPACE = "workspace"
    EXTERNAL = "external"


class Tool:
    name: str = "tool"
    description: str = ""
    parameters: dict[str, Any] = {  # noqa: RUF012 - subclasses override this schema
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    effect: Effect = Effect.READ
    concurrent: bool = False
    default_risk: str = "low"
    mutation_scope: MutationScope | None = None

    @property
    def effective_mutation_scope(self) -> MutationScope:
        if self.mutation_scope is not None:
            return self.mutation_scope
        if self.effect in {Effect.WRITE, Effect.EXECUTE}:
            return MutationScope.WORKSPACE
        if self.effect in {Effect.NETWORK, Effect.CONTROL}:
            return MutationScope.EXTERNAL
        return MutationScope.NONE

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        raise NotImplementedError

    def risk(self, arguments: dict[str, Any]) -> str:
        return self.default_risk

    def approval_description(self, arguments: dict[str, Any]) -> str:
        return self.description or self.name

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class FunctionTool(Tool):
    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        function: Callable[[dict[str, Any], ToolContext], ToolResult | Awaitable[ToolResult]],
        effect: Effect = Effect.READ,
        concurrent: bool = False,
        default_risk: str = "low",
        mutation_scope: MutationScope | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function
        self.effect = effect
        self.concurrent = concurrent
        self.default_risk = default_risk
        self.mutation_scope = mutation_scope

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        result = self.function(arguments, context)
        if inspect.isawaitable(result):
            result = await result
        return result


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool, *, replace: bool = False) -> None:
        if not tool.name or not tool.name.replace("_", "").isalnum():
            raise ValueError(f"Invalid tool name: {tool.name!r}")
        if tool.name in self._tools and not replace:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [self._tools[name].schema() for name in sorted(self._tools)]

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        context.tool_call_id = call.id
        tool = self.get(call.name)
        if tool is None:
            return ToolResult(f"Unknown tool: {call.name}", is_error=True)
        started = asyncio.get_running_loop().time()
        await context.events.emit(
            "tool.started", session_id=context.session_id, run_id=context.run_id,
            tool_call_id=call.id, tool=call.name, arguments=call.arguments,
        )
        try:
            validate_schema(call.arguments, tool.parameters, path="$arguments")
            decision = context.policy.decide(
                tool_name=tool.name,
                effect=tool.effect,
                arguments=call.arguments,
                risk_hint=tool.risk(call.arguments),
            )
            await context.events.emit(
                "tool.policy", session_id=context.session_id, run_id=context.run_id,
                tool_call_id=call.id, tool=call.name, decision=decision.action.value,
                reason=decision.reason, risk=decision.risk,
            )
            await context.approvals.enforce(
                ApprovalRequest(
                    tool_name=tool.name,
                    description=tool.approval_description(call.arguments),
                    decision=decision,
                    arguments_preview=truncate_text(json_dumps(call.arguments, pretty=True), 4_000),
                )
            )
            result = await tool.execute(call.arguments, context)
        except (ToolError, ToolValidationError, PolicyError, OSError, ValueError) as error:
            result = ToolResult(str(error), is_error=True, metadata={"error_type": type(error).__name__})
        except asyncio.CancelledError:
            await context.events.emit(
                "tool.cancelled", session_id=context.session_id, run_id=context.run_id,
                tool_call_id=call.id, tool=call.name,
            )
            raise
        except (BudgetExceeded, Cancelled):
            raise
        except Exception as error:  # defensive boundary around third-party tools
            result = ToolResult(
                f"Unexpected {type(error).__name__}: {error}",
                is_error=True,
                metadata={"error_type": type(error).__name__, "unexpected": True},
            )
        duration_ms = int((asyncio.get_running_loop().time() - started) * 1000)
        result.output = truncate_text(result.output, context.config.context.tool_output_chars)
        result.metadata.setdefault("duration_ms", duration_ms)
        await context.events.emit(
            "tool.completed", session_id=context.session_id, run_id=context.run_id,
            tool_call_id=call.id, tool=call.name, is_error=result.is_error,
            output=result.output, metadata=result.metadata,
        )
        return result


def object_schema(
    properties: dict[str, Any],
    *,
    required: list[str] | None = None,
    additional_properties: bool = False,
) -> dict[str, Any]:
    """Create a strict-friendly object schema.

    For provider strict modes, optional values should be represented as a union with
    null and still included in ``required``. When required is omitted every property
    is required, matching OpenAI strict-schema requirements.
    """
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties) if required is None else required,
        "additionalProperties": additional_properties,
    }


def nullable(schema_type: str, **kwargs: Any) -> dict[str, Any]:
    return {"type": [schema_type, "null"], **kwargs}


def validate_schema(value: Any, schema: dict[str, Any], *, path: str = "$") -> None:
    """Validate the JSON Schema subset used by Borealis tool definitions."""
    if not schema:
        return
    if "anyOf" in schema:
        errors: list[str] = []
        for variant in schema["anyOf"]:
            try:
                validate_schema(value, variant, path=path)
                return
            except ToolValidationError as error:
                errors.append(str(error))
        raise ToolValidationError(f"{path} did not match any allowed schema: {'; '.join(errors[:3])}")
    expected = schema.get("type")
    if isinstance(expected, list):
        if value is None and "null" in expected:
            return
        variants = [item for item in expected if item != "null"]
        if not any(_matches_type(value, item) for item in variants):
            raise ToolValidationError(f"{path} must be one of {expected}; got {type(value).__name__}")
    elif expected and not _matches_type(value, expected):
        raise ToolValidationError(f"{path} must be {expected}; got {type(value).__name__}")

    if "enum" in schema and value not in schema["enum"]:
        raise ToolValidationError(f"{path} must be one of {schema['enum']!r}")
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        missing = [name for name in schema.get("required", []) if name not in value]
        if missing:
            raise ToolValidationError(f"{path} is missing required keys: {', '.join(missing)}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise ToolValidationError(f"{path} has unknown keys: {', '.join(extra)}")
        for key, item in value.items():
            if key in properties:
                validate_schema(item, properties[key], path=f"{path}.{key}")
    elif isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            raise ToolValidationError(f"{path} must contain at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise ToolValidationError(f"{path} must contain at most {schema['maxItems']} items")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                validate_schema(item, item_schema, path=f"{path}[{index}]")
    elif isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            raise ToolValidationError(f"{path} is too short")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            raise ToolValidationError(f"{path} is too long")
    elif isinstance(value, int | float) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ToolValidationError(f"{path} must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ToolValidationError(f"{path} must be <= {schema['maximum']}")


def _matches_type(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, int | float) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(expected, True)
