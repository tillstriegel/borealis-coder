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
    def is_empty(self) -> bool:
        return not any(
            (
                self.input_tokens,
                self.output_tokens,
                self.cached_input_tokens,
                self.cache_write_tokens,
                self.reasoning_tokens,
                self.requests,
                self.cost_usd,
                self.cache_savings_usd,
                self.application_cache_hits,
                self.application_cache_misses,
                self.application_cache_saved_tokens,
                self.application_cache_saved_cost_usd,
            )
        )

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
            "application_cache_saved_cost_usd": round(self.application_cache_saved_cost_usd, 8),
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
            application_cache_saved_tokens=int(value.get("application_cache_saved_tokens", 0) or 0),
            application_cache_saved_cost_usd=float(
                value.get("application_cache_saved_cost_usd", 0.0) or 0.0
            ),
        )


@dataclass(slots=True)
class ContinuationState:
    """Opaque provider output required to continue a stateless interaction."""

    kind: str
    items: list[dict[str, Any]] = field(default_factory=list)

    def to_metadata(self, *, provider: str, model: str) -> dict[str, Any] | None:
        if (
            not self.kind
            or not self.items
            or any(not isinstance(item, dict) for item in self.items)
        ):
            return None
        return {
            "version": 1,
            "provider": provider,
            "model": model,
            "kind": self.kind,
            "items": [dict(item) for item in self.items],
        }

    @classmethod
    def from_metadata(
        cls,
        value: Any,
        *,
        provider: str,
        model: str,
        kind: str | None = None,
    ) -> ContinuationState | None:
        if not isinstance(value, dict) or value.get("version") != 1:
            return None
        if value.get("provider") != provider or value.get("model") != model:
            return None
        value_kind = value.get("kind")
        items = value.get("items")
        if (
            not isinstance(value_kind, str)
            or not value_kind
            or (kind is not None and value_kind != kind)
        ):
            return None
        if (
            not isinstance(items, list)
            or not items
            or any(not isinstance(item, dict) for item in items)
        ):
            return None
        return cls(kind=value_kind, items=[dict(item) for item in items])


@dataclass(slots=True)
class ModelResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None
    response_id: str | None = None
    model: str | None = None
    raw: dict[str, Any] | None = None
    continuation_state: ContinuationState | None = None
    reasoning_summary: str = ""


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
    incomplete: bool = False
    mutation_tracking: str = "complete"

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "text": self.text,
            "stop_reason": self.stop_reason.value,
            "usage": self.usage.to_dict(),
            "turns": self.turns,
            "changed_files": self.changed_files,
            "mutation_tracking": self.mutation_tracking,
            "verification": self.verification,
            "error": self.error,
            "incomplete": self.incomplete,
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


@dataclass(slots=True)
class CompactionArtifact:
    """Immutable provider-context artifact derived from durable session messages."""

    session_id: str
    version: int
    strategy: str
    source_message_ids: list[str]
    source_hash: str
    summary_text: str
    config_fingerprint: str
    estimated_tokens_before: int
    estimated_tokens_after: int
    id: str = field(default_factory=lambda: new_id("cmp"))
    source_start_sequence: int | None = None
    source_end_sequence: int | None = None
    provider: str | None = None
    model: str | None = None
    usage: Usage = field(default_factory=Usage)
    created_at: str = field(default_factory=utc_now)
    parent_artifact_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "version": self.version,
            "strategy": self.strategy,
            "source_message_ids": self.source_message_ids,
            "source_hash": self.source_hash,
            "summary_text": self.summary_text,
            "provider": self.provider,
            "model": self.model,
            "config_fingerprint": self.config_fingerprint,
            "estimated_tokens_before": self.estimated_tokens_before,
            "estimated_tokens_after": self.estimated_tokens_after,
            "usage": self.usage.to_dict(),
            "created_at": self.created_at,
            "parent_artifact_id": self.parent_artifact_id,
            "source_start_sequence": self.source_start_sequence,
            "source_end_sequence": self.source_end_sequence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CompactionArtifact:
        return cls(
            id=str(value["id"]),
            session_id=str(value["session_id"]),
            version=int(value["version"]),
            strategy=str(value["strategy"]),
            source_message_ids=[str(item) for item in value.get("source_message_ids", [])],
            source_hash=str(value["source_hash"]),
            summary_text=str(value["summary_text"]),
            provider=value.get("provider"),
            model=value.get("model"),
            config_fingerprint=str(value["config_fingerprint"]),
            estimated_tokens_before=int(value.get("estimated_tokens_before", 0)),
            estimated_tokens_after=int(value.get("estimated_tokens_after", 0)),
            usage=Usage.from_dict(value.get("usage")),
            created_at=str(value.get("created_at") or utc_now()),
            parent_artifact_id=value.get("parent_artifact_id"),
            source_start_sequence=value.get("source_start_sequence"),
            source_end_sequence=value.get("source_end_sequence"),
            metadata=dict(value.get("metadata") or {}),
        )
