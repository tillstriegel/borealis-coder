"""Structural and release evaluation for compaction v2 fixtures."""

from __future__ import annotations

import asyncio
import html
import json
import math
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import (
    AgentConfig,
    CacheConfig,
    Config,
    ContextConfig,
    ProviderConfig,
    SafetyConfig,
    StorageConfig,
)
from ..context import PromptContext
from ..models import Message, ModelResponse, ProviderRequest, Role, ToolCall, Usage
from ..providers.mock import MockProvider
from ..util import estimate_tokens
from .compaction import compact_messages, validate_tool_call_order
from .factory import build_runner

CompletionScorer = Callable[[list[Message], dict[str, Any]], float]
ReleaseSummarizer = Callable[[str, dict[str, Any]], ModelResponse]


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
    strategy: str = "deterministic"
    llm_compaction_evaluated: bool = False
    summarizer_calls: int = 0
    resume_artifact_reused: bool | None = None
    resume_summarizer_calls: int | None = None
    resume_cost_usd: float | None = None

    @property
    def reduction_percentage(self) -> float:
        if not self.tokens_before:
            return 0.0
        return max(0.0, (1.0 - self.tokens_after / self.tokens_before) * 100)

    @property
    def structural_gate_passed(self) -> bool:
        return bool(
            self.critical_fact_recall == 1.0
            and self.false_completion_claims == 0
            and self.boundary_escape_cases == 0
            and self.invalid_tool_sequences == 0
            and self.below_target
            and self.deterministic
        )

    @property
    def quality_gate_passed(self) -> bool | None:
        if self.full_history_quality is None or self.compacted_quality is None:
            return None
        return self.compacted_quality >= self.full_history_quality - 0.05

    @property
    def release_gate_passed(self) -> bool | None:
        quality_gate_passed = self.quality_gate_passed
        if quality_gate_passed is None or not self.llm_compaction_evaluated:
            return None
        return bool(
            self.structural_gate_passed
            and quality_gate_passed
            and self.strategy == "llm"
            and self.summarizer_calls > 0
            and self.resume_artifact_reused
            and self.resume_summarizer_calls == 0
            and self.resume_cost_usd == 0.0
        )


def load_compaction_corpus(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("Compaction corpus must be a JSON array")
    return [dict(item) for item in value if isinstance(item, dict)]


def evaluate_compaction_case(
    case: dict[str, Any],
    *,
    completion_scorer: CompletionScorer | None = None,
) -> CompactionEvaluation:
    messages = [_message_from_fixture(item) for item in case.get("messages", [])]
    target_tokens = int(case.get("target_tokens", 2_000))
    started = time.perf_counter()
    first = compact_messages(
        messages,
        keep_recent_bundles=int(case.get("keep_recent_bundles", 2)),
        target_tokens=target_tokens,
        target_bytes=target_tokens * 4,
        summary_tokens=target_tokens,
        summary_bytes=target_tokens * 4,
        force=True,
    )
    second = compact_messages(
        messages,
        keep_recent_bundles=int(case.get("keep_recent_bundles", 2)),
        target_tokens=target_tokens,
        target_bytes=target_tokens * 4,
        summary_tokens=target_tokens,
        summary_bytes=target_tokens * 4,
        force=True,
    )
    latency_ms = (time.perf_counter() - started) * 1_000
    return _build_evaluation(
        case,
        messages=messages,
        compacted=first,
        comparison=second,
        target_tokens=target_tokens,
        latency_ms=latency_ms,
        completion_scorer=completion_scorer,
    )


async def evaluate_compaction_release_case(
    case: dict[str, Any],
    *,
    summarizer: ReleaseSummarizer,
    completion_scorer: CompletionScorer,
) -> CompactionEvaluation:
    """Exercise LLM compaction and unchanged durable resume through the runner."""

    messages = [_message_from_fixture(item) for item in case.get("messages", [])]
    target_tokens = int(case.get("target_tokens", 2_000))
    started = time.perf_counter()
    prompt_context = PromptContext(stable="Compaction v2 release evaluation.")
    fixed_tokens = estimate_tokens(prompt_context.text) + 512
    effective_target_ratio = min(0.70, 0.82 * 0.85)
    desired_total_target = target_tokens + fixed_tokens
    max_input_tokens = (
        math.ceil(desired_total_target / effective_target_ratio) + 1 + 512
    )

    with tempfile.TemporaryDirectory(prefix="borealis-compaction-eval-") as directory:
        root = Path(directory)
        config = Config(
            agent=AgentConfig(
                provider="mock",
                auto_verify=False,
                deterministic_compaction=False,
                max_input_tokens=max_input_tokens,
                max_output_tokens=1,
                compaction_safety_margin_tokens=0,
            ),
            context=ContextConfig(compact_tool_output_tokens=1),
            storage=StorageConfig(
                directory=str(root / ".data"),
                trace_jsonl=False,
            ),
            safety=SafetyConfig(approval="never"),
            cache=CacheConfig(response_cache_enabled=False),
            providers={
                "mock": ProviderConfig(
                    type="mock",
                    model="deterministic",
                    api_style="mock",
                    max_retries=0,
                )
            },
        )
        runner = await build_runner(root, config=config, interactive=False)
        session = runner.sessions.create_session(
            workspace=root,
            provider="mock",
            model="deterministic",
        )
        for message in messages:
            runner.sessions.append_message(session.id, message)
        first_usage = Usage()
        first_summary_calls = 0
        provider = runner.providers[0].provider
        if not isinstance(provider, MockProvider):
            await runner.close()
            raise TypeError("Compaction release evaluation requires the mock provider")

        def first_handler(request: ProviderRequest, _call: int) -> ModelResponse:
            nonlocal first_summary_calls
            if request.metadata.get("purpose") != "compaction_summary":
                raise RuntimeError("Unexpected non-summary provider request")
            first_summary_calls += 1
            response = summarizer(request.messages[-1].content, case)
            if not isinstance(response, ModelResponse):
                raise TypeError("Release summarizer must return a ModelResponse")
            return response

        provider.handler = first_handler

        async def first_usage_sink(usage: Usage) -> None:
            first_usage.add(usage)
            await asyncio.to_thread(runner.sessions.add_usage, session.id, usage)

        async def first_settled_usage_sink(usage: Usage) -> None:
            first_usage.add(usage)

        try:
            first = await runner._prepare_provider_request(
                prompt_context=prompt_context,
                messages=runner.sessions.messages(session.id),
                schemas=[],
                final_turn=False,
                verification_finalization_pending=False,
                adaptive_cache=False,
                conversation_cache=True,
                usage_sink=first_usage_sink,
                settled_usage_sink=first_settled_usage_sink,
                cancel=asyncio.Event(),
                session_id=session.id,
                run_id="compaction-eval-first",
                last_prune_signature=None,
            )
            first_artifacts = runner.sessions.compaction_artifacts(session.id)
        finally:
            await runner.close()

        resumed = await build_runner(root, config=config, interactive=False)
        resumed_provider = resumed.providers[0].provider
        if not isinstance(resumed_provider, MockProvider):
            await resumed.close()
            raise TypeError("Compaction release evaluation requires the mock provider")
        resume_usage = Usage()
        resume_summary_calls = 0

        def resume_handler(request: ProviderRequest, _call: int) -> ModelResponse:
            nonlocal resume_summary_calls
            if request.metadata.get("purpose") != "compaction_summary":
                raise RuntimeError("Unexpected non-summary provider request")
            resume_summary_calls += 1
            response = summarizer(request.messages[-1].content, case)
            if not isinstance(response, ModelResponse):
                raise TypeError("Release summarizer must return a ModelResponse")
            return response

        resumed_provider.handler = resume_handler

        async def resume_usage_sink(usage: Usage) -> None:
            resume_usage.add(usage)
            await asyncio.to_thread(resumed.sessions.add_usage, session.id, usage)

        try:
            second = await resumed._prepare_provider_request(
                prompt_context=prompt_context,
                messages=resumed.sessions.messages(session.id),
                schemas=[],
                final_turn=False,
                verification_finalization_pending=False,
                adaptive_cache=False,
                conversation_cache=True,
                usage_sink=resume_usage_sink,
                cancel=asyncio.Event(),
                session_id=session.id,
                run_id="compaction-eval-resume",
                last_prune_signature=None,
            )
            artifacts = resumed.sessions.compaction_artifacts(session.id)
        finally:
            await resumed.close()

    artifact = first_artifacts[-1] if first_artifacts else None
    compacted = (
        [Message(role=Role.SYSTEM, content=artifact.summary_text), *first.request.messages]
        if artifact is not None
        else first.request.messages
    )
    resumed_artifact = artifacts[-1] if artifacts else None
    comparison = (
        [
            Message(role=Role.SYSTEM, content=resumed_artifact.summary_text),
            *second.request.messages,
        ]
        if resumed_artifact is not None
        else second.request.messages
    )
    metadata = first.compaction_metadata or {}
    resumed_metadata = second.compaction_metadata or {}
    latency_ms = (time.perf_counter() - started) * 1_000
    return _build_evaluation(
        case,
        messages=messages,
        compacted=compacted,
        comparison=comparison,
        target_tokens=int(metadata.get("target_tokens") or target_tokens),
        latency_ms=latency_ms,
        completion_scorer=completion_scorer,
        tokens_before=int(metadata.get("estimated_tokens_before") or 0),
        tokens_after=first.estimated_tokens,
        below_target=bool(
            first.compacted
            and first.estimated_tokens
            <= int(metadata.get("target_tokens") or target_tokens)
        ),
        deterministic=(
            first.request.system == second.request.system
            and [_provider_shape(message) for message in first.request.messages]
            == [_provider_shape(message) for message in second.request.messages]
        ),
        strategy=artifact.strategy if artifact is not None else "none",
        llm_compaction_evaluated=True,
        summarizer_calls=first_summary_calls,
        resume_artifact_reused=bool(resumed_metadata.get("artifact_reused"))
        and len(artifacts) == 1,
        resume_summarizer_calls=resume_summary_calls,
        cost_usd=first_usage.cost_usd,
        resume_cost_usd=resume_usage.cost_usd,
    )


def _build_evaluation(
    case: dict[str, Any],
    *,
    messages: list[Message],
    compacted: list[Message],
    comparison: list[Message],
    target_tokens: int,
    latency_ms: float,
    completion_scorer: CompletionScorer | None,
    tokens_before: int | None = None,
    tokens_after: int | None = None,
    below_target: bool | None = None,
    deterministic: bool | None = None,
    strategy: str = "deterministic",
    llm_compaction_evaluated: bool = False,
    summarizer_calls: int = 0,
    resume_artifact_reused: bool | None = None,
    resume_summarizer_calls: int | None = None,
    cost_usd: float = 0.0,
    resume_cost_usd: float | None = None,
) -> CompactionEvaluation:
    provider_messages = [
        message for message in compacted if message.role != Role.SYSTEM
    ]
    invalid_sequences = 0
    try:
        validate_tool_call_order(provider_messages)
    except ValueError:
        invalid_sequences = 1
    rendered = html.unescape("\n".join(message.content for message in compacted))
    facts = [str(item) for item in case.get("critical_facts", [])]
    recalled = sum(fact in rendered for fact in facts)
    forbidden = [str(item) for item in case.get("forbidden_completion_claims", [])]
    false_claims = sum(claim in rendered for claim in forbidden)
    summary = (
        compacted[0].content
        if compacted and compacted[0].role == Role.SYSTEM
        else ""
    )
    closing_boundary = (
        "</llm_conversation_summary>"
        if strategy == "llm"
        else "</deterministic_conversation_summary>"
    )
    boundary_escapes = max(
        0,
        summary.count(closing_boundary) - 1,
    )
    measured_tokens_before = (
        tokens_before
        if tokens_before is not None
        else sum(estimate_tokens(message.content) + 12 for message in messages)
    )
    measured_tokens_after = (
        tokens_after
        if tokens_after is not None
        else sum(estimate_tokens(message.content) + 12 for message in compacted)
    )
    if completion_scorer is None:
        full_history_quality = None
        compacted_quality = None
    else:
        full_history_quality = _validated_completion_score(
            completion_scorer(messages, case)
        )
        compacted_quality = _validated_completion_score(
            completion_scorer(compacted, case)
        )
    return CompactionEvaluation(
        name=str(case.get("name") or "unnamed"),
        critical_fact_recall=recalled / len(facts) if facts else 1.0,
        false_completion_claims=false_claims,
        boundary_escape_cases=boundary_escapes,
        invalid_tool_sequences=invalid_sequences,
        tokens_before=measured_tokens_before,
        tokens_after=measured_tokens_after,
        below_target=(
            measured_tokens_after <= target_tokens
            if below_target is None
            else below_target
        ),
        deterministic=(
            [_provider_shape(message) for message in compacted]
            == [_provider_shape(message) for message in comparison]
            if deterministic is None
            else deterministic
        ),
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        full_history_quality=full_history_quality,
        compacted_quality=compacted_quality,
        strategy=strategy,
        llm_compaction_evaluated=llm_compaction_evaluated,
        summarizer_calls=summarizer_calls,
        resume_artifact_reused=resume_artifact_reused,
        resume_summarizer_calls=resume_summarizer_calls,
        resume_cost_usd=resume_cost_usd,
    )


def _validated_completion_score(value: float) -> float:
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("Completion scorer must return a finite score from 0 to 1")
    return score


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
