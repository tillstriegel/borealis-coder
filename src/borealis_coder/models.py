"""Data contracts shared by providers, tools, storage, and protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .util import new_id, utc_now


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Effect(StrEnum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    CONTROL = "control"


class StopReason(StrEnum):
    END_TURN = "end_turn"
    MAX_TURNS = "max_turns"
    BUDGET = "budget"
    CANCELLED = "cancelled"
    ERROR = "error"
    STUCK = "stuck"


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=lambda: new_id("call"))
    raw_arguments: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "raw_arguments": self.raw_arguments,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ToolCall:
        return cls(
            id=str(value.get("id") or new_id("call")),
            name=str(value["name"]),
            arguments=dict(value.get("arguments") or {}),
            raw_arguments=value.get("raw_arguments"),
        )


@dataclass(slots=True)
class Message:
    role: Role
    content: str = ""
    id: str = field(default_factory=lambda: new_id("msg"))
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "tool_calls": [item.to_dict() for item in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "is_error": self.is_error,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Message:
        return cls(
            id=str(value.get("id") or new_id("msg")),
            role=Role(value["role"]),
            content=str(value.get("content") or ""),
            tool_calls=[ToolCall.from_dict(item) for item in value.get("tool_calls", [])],
            tool_call_id=value.get("tool_call_id"),
            tool_name=value.get("tool_name"),
            is_error=bool(value.get("is_error", False)),
            metadata=dict(value.get("metadata") or {}),
            created_at=str(value.get("created_at") or utc_now()),
        )


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    requests: int = 0
    cost_usd: float = 0.0
    cache_savings_usd: float = 0.0
    application_cache_hits: int = 0
    application_cache_misses: int = 0
    application_cache_saved_tokens: int = 0
    application_cache_saved_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def uncached_input_tokens(self) -> int:
        return max(0, self.input_tokens - self.cached_input_tokens - self.cache_write_tokens)

    @property
    def provider_cache_hit_rate(self) -> float:
        return self.cached_input_tokens / self.input_tokens if self.input_tokens else 0.0

    @property
    def application_cache_hit_rate(self) -> float:
        attempts = self.application_cache_hits + self.application_cache_misses
        return self.application_cache_hits / attempts if attempts else 0.0

    def add(self, other: Usage) -> Usage:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.requests += other.requests
        self.cost_usd += other.cost_usd
        self.cache_savings_usd += other.cache_savings_usd
        self.application_cache_hits += other.application_cache_hits
        self.application_cache_misses += other.application_cache_misses
        self.application_cache_saved_tokens += other.application_cache_saved_tokens
        self.application_cache_saved_cost_usd += other.application_cache_saved_cost_usd
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "requests": self.requests,
            "cost_usd": round(self.cost_usd, 8),
            "cache_savings_usd": round(self.cache_savings_usd, 8),
            "provider_cache_hit_rate": round(self.provider_cache_hit_rate, 6),
            "application_cache_hits": self.application_cache_hits,
            "application_cache_misses": self.application_cache_misses,
            "application_cache_hit_rate": round(self.application_cache_hit_rate, 6),
            "application_cache_saved_tokens": self.application_cache_saved_tokens,
            "application_cache_saved_cost_usd": round(
                self.application_cache_saved_cost_usd, 8
            ),
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> Usage:
        value = value or {}
        return cls(
            input_tokens=int(value.get("input_tokens", 0) or 0),
            output_tokens=int(value.get("output_tokens", 0) or 0),
            cached_input_tokens=int(value.get("cached_input_tokens", 0) or 0),
            cache_write_tokens=int(value.get("cache_write_tokens", 0) or 0),
            reasoning_tokens=int(value.get("reasoning_tokens", 0) or 0),
            requests=int(value.get("requests", 0) or 0),
            cost_usd=float(value.get("cost_usd", 0.0) or 0.0),
            cache_savings_usd=float(value.get("cache_savings_usd", 0.0) or 0.0),
            application_cache_hits=int(value.get("application_cache_hits", 0) or 0),
            application_cache_misses=int(value.get("application_cache_misses", 0) or 0),
            application_cache_saved_tokens=int(
                value.get("application_cache_saved_tokens", 0) or 0
            ),
            application_cache_saved_cost_usd=float(
                value.get("application_cache_saved_cost_usd", 0.0) or 0.0
            ),
        )


@dataclass(slots=True)
class ModelResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None
    response_id: str | None = None
    model: str | None = None
    raw: dict[str, Any] | None = None


@dataclass(slots=True)
class ProviderRequest:
    model: str
    system: str
    messages: list[Message]
    tools: list[dict[str, Any]] = field(default_factory=list)
    max_output_tokens: int = 16_000
    temperature: float | None = None
    reasoning_effort: str | None = None
    parallel_tool_calls: bool = True
    response_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    output: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"output": self.output, "is_error": self.is_error, "metadata": self.metadata}


@dataclass(slots=True)
class Event:
    type: str
    session_id: str | None = None
    run_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("evt"))
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "data": self.data,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class AgentResult:
    session_id: str
    run_id: str
    text: str
    stop_reason: StopReason
    usage: Usage
    turns: int
    changed_files: list[str] = field(default_factory=list)
    verification: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "text": self.text,
            "stop_reason": self.stop_reason.value,
            "usage": self.usage.to_dict(),
            "turns": self.turns,
            "changed_files": self.changed_files,
            "verification": self.verification,
            "error": self.error,
        }


@dataclass(slots=True)
class SessionInfo:
    id: str
    workspace: str
    title: str
    provider: str
    model: str
    status: str
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
