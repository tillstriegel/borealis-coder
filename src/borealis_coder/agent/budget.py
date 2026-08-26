"""Run budget accounting and deterministic request-size estimation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..config import AgentConfig
from ..errors import BudgetExceeded
from ..models import Message, Usage
from ..util import estimate_tokens, json_dumps, monotonic_ms

_PROVIDER_FRAMING_ALLOWANCE = {
    "openai": 512,
    "chatgpt": 512,
    "anthropic": 768,
    "gemini": 1_024,
    "openrouter": 768,
    "openai_compatible": 768,
}
_RECOVERY_CONTINUATION_PROMPT = "continue"


def is_recovery_continuation_prompt(prompt: str) -> bool:
    """Return whether a user prompt is the bare documented recovery command."""

    return prompt.strip().casefold() == _RECOVERY_CONTINUATION_PROMPT


def max_turns_recovery_message(max_turns: int) -> str:
    return (
        f"Run incomplete: maximum {max_turns} model turns reached. "
        f"Session preserved; send '{_RECOVERY_CONTINUATION_PROMPT}' to resume."
    )


@dataclass(slots=True)
class Budget:
    config: AgentConfig
    started_ms: int
    turns: int = 0
    usage: Usage | None = None

    @classmethod
    def start(cls, config: AgentConfig) -> Budget:
        return cls(config=config, started_ms=monotonic_ms(), usage=Usage())

    @property
    def elapsed_seconds(self) -> float:
        return (monotonic_ms() - self.started_ms) / 1000

    def before_turn(self) -> None:
        if self.turns >= self.config.max_turns:
            raise BudgetExceeded("turns", max_turns_recovery_message(self.config.max_turns))
        if self.elapsed_seconds >= self.config.max_time_seconds:
            raise BudgetExceeded(
                "time", f"Maximum {self.config.max_time_seconds}s run time reached"
            )
        assert self.usage is not None
        if self.config.max_cost_usd > 0 and self.usage.cost_usd >= self.config.max_cost_usd:
            raise BudgetExceeded(
                "cost", f"Maximum ${self.config.max_cost_usd:.2f} model cost reached"
            )
        self.turns += 1

    def retry_current_turn(self) -> None:
        """Keep a failed provider attempt within its current logical model turn."""

        if self.turns <= 0:
            raise RuntimeError("Cannot retry a model turn before it starts")
        self.turns -= 1

    def add_usage(self, usage: Usage) -> None:
        assert self.usage is not None
        self.usage.add(usage)
        if self.config.max_cost_usd > 0 and self.usage.cost_usd > self.config.max_cost_usd:
            raise BudgetExceeded(
                "cost",
                f"Model cost ${self.usage.cost_usd:.4f} exceeded ${self.config.max_cost_usd:.2f}",
            )


def estimate_request_tokens(system: str, messages: list[Message], tools: list[dict]) -> int:  # type: ignore[type-arg]
    tokens = estimate_tokens(system) + estimate_tokens(json_dumps(tools))
    for message in messages:
        tokens += estimate_tokens(message.content)
        if message.tool_calls:
            tokens += estimate_tokens(json_dumps([call.to_dict() for call in message.tool_calls]))
    # Add framing overhead per message and tool.
    return tokens + len(messages) * 12 + len(tools) * 30


def _provider_message_view(message: Message) -> dict[str, object]:
    payload: dict[str, object] = {
        "role": message.role.value,
        "content": message.content,
    }
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "name": call.name,
                "arguments": (
                    call.raw_arguments
                    if call.raw_arguments is not None
                    else call.arguments
                ),
            }
            for call in message.tool_calls
        ]
    if message.role.value == "tool":
        payload.update(
            {
                "tool_call_id": message.tool_call_id,
                "tool_name": message.tool_name,
                "is_error": message.is_error,
            }
        )
    continuation_state = message.metadata.get("continuation_state")
    if continuation_state is not None:
        payload["continuation_state"] = continuation_state
    return payload


def estimate_request_bytes(
    system: str,
    messages: list[Message],
    tools: list[dict],  # type: ignore[type-arg]
) -> int:
    """Return deterministic UTF-8 bytes for the complete provider-facing view."""

    return len(
        json_dumps(
            {
                "system": system,
                "messages": [_provider_message_view(message) for message in messages],
                "tools": tools,
            }
        ).encode("utf-8")
    )


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Provider-facing input budget with explicit fixed and variable costs."""

    input_limit: int
    reserved_output_tokens: int
    system_tokens: int
    tool_schema_tokens: int
    provider_framing_tokens: int
    continuation_state_tokens: int
    safety_margin_tokens: int
    trigger_tokens: int
    target_tokens: int
    message_target_tokens: int
    target_bytes: int
    message_target_bytes: int

    @classmethod
    def calculate(
        cls,
        config: AgentConfig,
        *,
        system: str,
        tools: list[dict],  # type: ignore[type-arg]
        messages: list[Message],
        provider: str = "",
        providers: Iterable[str] = (),
        overflow_retry_count: int = 0,
    ) -> ContextBudget:
        system_tokens = estimate_tokens(system)
        tool_schema_tokens = estimate_tokens(json_dumps(tools)) + len(tools) * 30
        continuation_state_tokens = sum(
            estimate_tokens(json_dumps(message.metadata.get("continuation_state")))
            for message in messages
            if message.metadata.get("continuation_state")
        )
        provider_framing = max(
            [
                config.compaction_provider_framing_tokens,
                _PROVIDER_FRAMING_ALLOWANCE.get(provider, 0),
                *(
                    _PROVIDER_FRAMING_ALLOWANCE.get(name, 0)
                    for name in providers
                ),
            ]
        )
        dynamic_margin = config.compaction_safety_margin_tokens + (
            provider_framing * max(0, overflow_retry_count)
        )
        available = max(
            1,
            config.max_input_tokens
            - config.max_output_tokens
            - dynamic_margin
            - provider_framing
            - continuation_state_tokens,
        )
        trigger = max(1, int(available * config.compact_at_ratio))
        effective_target_ratio = min(
            config.compaction_target_ratio,
            max(0.1, config.compact_at_ratio * 0.85),
        )
        target = max(1, int(available * effective_target_ratio))
        fixed = (
            system_tokens
            + tool_schema_tokens
            + provider_framing
            + continuation_state_tokens
        )
        message_target = max(1, target - fixed)
        target_bytes = target * 4
        empty_request_bytes = estimate_request_bytes("", [], [])
        fixed_request_bytes = (
            estimate_request_bytes(system, [], tools) - empty_request_bytes
        )
        message_target_bytes = max(1, target_bytes - fixed_request_bytes)
        return cls(
            input_limit=config.max_input_tokens,
            reserved_output_tokens=config.max_output_tokens,
            system_tokens=system_tokens,
            tool_schema_tokens=tool_schema_tokens,
            provider_framing_tokens=provider_framing,
            continuation_state_tokens=continuation_state_tokens,
            safety_margin_tokens=dynamic_margin,
            trigger_tokens=trigger,
            target_tokens=target,
            message_target_tokens=message_target,
            target_bytes=target_bytes,
            message_target_bytes=message_target_bytes,
        )

    def estimated_total(self, messages: list[Message]) -> int:
        return (
            self.system_tokens
            + self.tool_schema_tokens
            + self.provider_framing_tokens
            + self.continuation_state_tokens
            + sum(estimate_tokens(message.content) + 12 for message in messages)
            + sum(
                estimate_tokens(json_dumps([call.to_dict() for call in message.tool_calls]))
                for message in messages
                if message.tool_calls
            )
        )

    def below_target(self, messages: list[Message]) -> bool:
        return self.estimated_total(messages) <= self.target_tokens
