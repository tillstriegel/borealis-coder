"""Deterministic quality evaluation for compaction v2 fixtures."""

from __future__ import annotations

import html
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import Message, Role, ToolCall
from ..util import estimate_tokens
from .compaction import compact_messages, validate_tool_call_order


@dataclass(frozen=True, slots=True)
class CompactionEvaluation:
    name: str
    critical_fact_recall: float
    false_completion_claims: int
    boundary_escape_cases: int
    invalid_tool_sequences: int
    tokens_before: int
    tokens_after: int
    below_target: bool
    deterministic: bool
    latency_ms: float
    cost_usd: float = 0.0
    full_history_quality: float | None = None
    compacted_quality: float | None = None

    @property
    def reduction_percentage(self) -> float:
        if not self.tokens_before:
            return 0.0
        return max(0.0, (1.0 - self.tokens_after / self.tokens_before) * 100)

    @property
    def release_gate_passed(self) -> bool:
        quality_ok = bool(
            self.full_history_quality is None
            or self.compacted_quality is None
            or self.compacted_quality >= self.full_history_quality - 0.05
        )
        return bool(
            self.critical_fact_recall == 1.0
            and self.false_completion_claims == 0
            and self.boundary_escape_cases == 0
            and self.invalid_tool_sequences == 0
            and self.below_target
            and self.deterministic
            and quality_ok
        )


def load_compaction_corpus(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("Compaction corpus must be a JSON array")
    return [dict(item) for item in value if isinstance(item, dict)]


def evaluate_compaction_case(
    case: dict[str, Any],
    *,
    completion_scorer: Callable[[list[Message]], float] | None = None,
) -> CompactionEvaluation:
    messages = [_message_from_fixture(item) for item in case.get("messages", [])]
    target_tokens = int(case.get("target_tokens", 2_000))
    started = time.perf_counter()
    first = compact_messages(
        messages,
        keep_recent_bundles=int(case.get("keep_recent_bundles", 2)),
        target_tokens=target_tokens,
        summary_tokens=target_tokens,
        summary_bytes=target_tokens * 4,
        force=True,
    )
    second = compact_messages(
        messages,
        keep_recent_bundles=int(case.get("keep_recent_bundles", 2)),
        target_tokens=target_tokens,
        summary_tokens=target_tokens,
        summary_bytes=target_tokens * 4,
        force=True,
    )
    latency_ms = (time.perf_counter() - started) * 1_000
    provider_messages = [message for message in first if message.role != Role.SYSTEM]
    invalid_sequences = 0
    try:
        validate_tool_call_order(provider_messages)
    except ValueError:
        invalid_sequences = 1
    rendered = html.unescape("\n".join(message.content for message in first))
    facts = [str(item) for item in case.get("critical_facts", [])]
    recalled = sum(fact in rendered for fact in facts)
    forbidden = [str(item) for item in case.get("forbidden_completion_claims", [])]
    false_claims = sum(claim in rendered for claim in forbidden)
    summary = first[0].content if first and first[0].role == Role.SYSTEM else ""
    boundary_escapes = max(
        0,
        summary.count("</deterministic_conversation_summary>") - 1,
    )
    tokens_before = sum(estimate_tokens(message.content) + 12 for message in messages)
    tokens_after = sum(estimate_tokens(message.content) + 12 for message in first)
    if completion_scorer is None:
        full_history_quality = _fixture_completion_quality(messages, case)
        compacted_quality = _fixture_completion_quality(first, case)
    else:
        full_history_quality = completion_scorer(messages)
        compacted_quality = completion_scorer(first)
    return CompactionEvaluation(
        name=str(case.get("name") or "unnamed"),
        critical_fact_recall=recalled / len(facts) if facts else 1.0,
        false_completion_claims=false_claims,
        boundary_escape_cases=boundary_escapes,
        invalid_tool_sequences=invalid_sequences,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        below_target=tokens_after <= target_tokens,
        deterministic=[_provider_shape(message) for message in first]
        == [_provider_shape(message) for message in second],
        latency_ms=latency_ms,
        full_history_quality=full_history_quality,
        compacted_quality=compacted_quality,
    )


def _fixture_completion_quality(
    messages: list[Message], case: dict[str, Any]
) -> float:
    """Score source-backed actionable facts without requiring an online judge."""

    rendered = html.unescape("\n".join(message.content for message in messages))
    requirements = [str(item) for item in case.get("critical_facts", [])]
    if not requirements:
        recall = 1.0
    else:
        recall = sum(item in rendered for item in requirements) / len(requirements)
    forbidden = [str(item) for item in case.get("forbidden_completion_claims", [])]
    false_claims = sum(item in rendered for item in forbidden)
    return max(0.0, recall - min(1.0, false_claims * 0.25))


def _message_from_fixture(value: dict[str, Any]) -> Message:
    content = str(value.get("content") or "") * int(value.get("repeat", 1))
    calls = [ToolCall.from_dict(dict(item)) for item in value.get("tool_calls", [])]
    return Message(
        id=str(value.get("id") or f"fixture-{value.get('role')}-{id(value)}"),
        role=Role(str(value["role"])),
        content=content,
        tool_calls=calls,
        tool_call_id=value.get("tool_call_id"),
        tool_name=value.get("tool_name"),
        is_error=bool(value.get("is_error", False)),
        metadata=dict(value.get("metadata") or {}),
        created_at=str(value.get("created_at") or "2026-01-01T00:00:00+00:00"),
    )


def _provider_shape(message: Message) -> dict[str, Any]:
    return {
        "role": message.role.value,
        "content": message.content,
        "tool_calls": [call.to_dict() for call in message.tool_calls],
        "tool_call_id": message.tool_call_id,
        "tool_name": message.tool_name,
        "is_error": message.is_error,
    }
