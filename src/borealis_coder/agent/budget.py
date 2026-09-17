"""Run budget accounting and deterministic request-size estimation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from ..config import AgentConfig
from ..errors import BudgetExceeded, RouteContextExceeded
from ..models import Message, ProviderRequest, Usage
from ..providers.base import Provider
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


def max_model_requests_recovery_message(max_model_requests: int) -> str:
    return (
        f"Run incomplete: maximum {max_model_requests} model requests reached. "
        f"Session preserved; send '{_RECOVERY_CONTINUATION_PROMPT}' to resume."
    )


@dataclass(slots=True)
class Budget:
    config: AgentConfig
    started_ms: int
    turns: int = 0
    model_requests: int = 0
    usage: Usage | None = None
    pending_cost_usd: float = 0.0
    _held_usage: dict[int, tuple[float, bool]] = field(default_factory=dict, repr=False)

    @classmethod
    def start(cls, config: AgentConfig) -> Budget:
        return cls(config=config, started_ms=monotonic_ms(), usage=Usage())

    @property
    def elapsed_seconds(self) -> float:
        return (monotonic_ms() - self.started_ms) / 1000

    def before_turn(self) -> None:
        if self.turns >= self.config.max_turns:
            raise BudgetExceeded("turns", max_turns_recovery_message(self.config.max_turns))
        self._check_time_and_cost()
        self.turns += 1

    def _check_time_and_cost(self) -> None:
        if self.elapsed_seconds >= self.config.max_time_seconds:
            raise BudgetExceeded(
                "time", f"Maximum {self.config.max_time_seconds}s run time reached"
            )
        assert self.usage is not None
        if self.config.max_cost_usd > 0 and self.usage.cost_usd >= self.config.max_cost_usd:
            raise BudgetExceeded(
                "cost", f"Maximum ${self.config.max_cost_usd:.2f} model cost reached"
            )

    def retry_current_turn(self) -> None:
        """Keep a failed provider attempt within its current logical model turn."""

        if self.turns <= 0:
            raise RuntimeError("Cannot retry a model turn before it starts")
        self.turns -= 1

    def before_model_request(self) -> None:
        """Reserve one logical provider request across the whole run."""

        if self.model_requests >= self.config.max_model_requests:
            raise BudgetExceeded(
                "model_requests",
                max_model_requests_recovery_message(
                    self.config.max_model_requests
                ),
            )
        self._check_time_and_cost()
        self.model_requests += 1

    def reserve_cost(self, provider: Provider, request: ProviderRequest) -> float:
        self._check_time_and_cost()
        if self.config.max_cost_usd <= 0 or provider.name == "mock":
            return 0.0
        self.check_unsettled(Usage())
        assert self.usage is not None
        models = [request.model, *(provider.config.model_fallbacks if provider.name == "openrouter" else [])]
        candidates = [provider.prices(model) for model in models]
        rates = provider.prices(request.model)
        if rates is None or any(value is None for value in candidates):
            raise BudgetExceeded("cost", f"Strict dollar budget requires input/output pricing for {provider.name}/{request.model}")
        multiplier = (2.0 if request.metadata.get("prompt_cache_ttl") == "1h" else 1.25) if provider.name == "anthropic" else 1.25 if provider.name == "openai" else 1.0
        known_rates = [candidate for candidate in candidates if candidate is not None]
        input_rate = max(max(item[0], item[2] if item[2] is not None else item[0],
                             item[3] if item[3] is not None else item[0] * multiplier) for item in known_rates)
        output_rate = max(item[1] for item in known_rates)
        tools = [*request.tools, *([{"response_schema": request.response_schema}] if request.response_schema else [])]
        context = ContextBudget.calculate(self.config, system=request.system, tools=tools, messages=request.messages, provider=provider.name)
        estimate = (context.estimated_total(request.messages) * input_rate
                    + request.max_output_tokens * output_rate) / 1_000_000
        if self.usage.cost_usd + self.pending_cost_usd + estimate > self.config.max_cost_usd:
            raise BudgetExceeded("cost", "Estimated request cost plus concurrent reservations exceeds the dollar budget")
        self.pending_cost_usd += estimate
        return estimate

    def hold_usage(self, usage: Usage) -> None:
        """Expose a logical request's observed charges until its sink takes over."""
        previous, _ = self._held_usage.get(id(usage), (0.0, False))
        self.pending_cost_usd += usage.cost_usd - previous
        self._held_usage[id(usage)] = (usage.cost_usd, not usage.is_empty and usage.cost_status in {"unknown", "incomplete"})

    def release_usage(self, usage: Usage) -> None:
        held, _ = self._held_usage.pop(id(usage), (0.0, False))
        self.release_cost(held)

    def check_unsettled(self, usage: Usage) -> None:
        """Check retry/fallback usage not yet settled by the single accounting sink."""
        self._check_time_and_cost()
        if self.config.max_cost_usd <= 0:
            return
        assert self.usage is not None
        if (any(incomplete for _, incomplete in self._held_usage.values())
                or any(not item.is_empty and item.cost_status in {"unknown", "incomplete"} for item in (self.usage, usage))):
            raise BudgetExceeded("cost", "Strict dollar budget cannot retry with unreconciled usage")
        unheld_cost = 0.0 if id(usage) in self._held_usage else usage.cost_usd
        if self.usage.cost_usd + self.pending_cost_usd + unheld_cost > self.config.max_cost_usd:
            raise BudgetExceeded("cost", "Retry or fallback cost exceeds the remaining dollar budget")

    def release_cost(self, reservation: float) -> None:
        self.pending_cost_usd = max(0.0, round(self.pending_cost_usd - reservation, 12))

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
    hard_bytes: int
    trigger_bytes: int
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
            - dynamic_margin,
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
        hard_bytes = 2**63 - 1  # Unknown endpoint byte limit; checked on the actual route.
        trigger_bytes = max(1, int(hard_bytes * config.compact_at_ratio))
        target_bytes = max(1, int(hard_bytes * effective_target_ratio))
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
            hard_bytes=hard_bytes,
            trigger_bytes=trigger_bytes,
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


def resolve_route_limits(config: AgentConfig, provider: Provider, model: str, max_output_tokens: int, *, overflow_retry_count: int = 0) -> tuple[AgentConfig, dict[str, int | None]]:
    """Intersect operator, endpoint and exact-model capabilities without a catalog."""
    limits = {key: getattr(provider.config, key) for key in (
        "context_tokens", "input_token_limit", "output_token_limit", "request_byte_limit",
    )}
    for key, value in provider.config.model_limits.get(model, {}).items():
        limits[key] = min(limits[key] or value, value)
    if provider.name == "openrouter":
        for model in provider.config.model_fallbacks:
            for key, value in provider.config.model_limits.get(model, {}).items():
                limits[key] = min(limits[key] or value, value)
    output = min(max_output_tokens, limits["output_token_limit"] or max_output_tokens)
    context_limit = int(min(config.max_input_tokens, limits["context_tokens"] or config.max_input_tokens) * (0.8 ** overflow_retry_count))
    if limits["input_token_limit"]:
        context_limit = min(context_limit, limits["input_token_limit"] + output)
    return replace(config, max_input_tokens=context_limit, max_output_tokens=output), limits


def prepare_route_request(request: ProviderRequest, provider: Provider, config: AgentConfig, *, overflow_retry_count: int = 0) -> ProviderRequest:
    """Apply the actual endpoint/model limits using the existing compaction v2."""
    from .compaction import (
        CompactionError,
        compact_messages,
        prune_provider_messages,
        validate_tool_call_order,
    )

    reserved_fields = {
        "model", "models", "messages", "input", "system", "instructions",
        "tools", "functions", "tool_choice", "function_call", "parallel_tool_calls",
        "response_format", "text", "previous_response_id", "conversation",
        "max_tokens", "max_output_tokens", "max_completion_tokens",
    }
    if reserved_fields.intersection(provider.config.extra_body):
        raise BudgetExceeded("context", "extra_body overrides request fields that must be validated; use route/model settings instead")
    effective, limits = resolve_route_limits(config, provider, request.model, request.max_output_tokens, overflow_retry_count=overflow_retry_count)
    output, context_limit = effective.max_output_tokens, effective.max_input_tokens
    budget_tools = [*request.tools, *([{ "response_schema": request.response_schema}] if request.response_schema else [])]
    messages, _ = prune_provider_messages(request.messages)
    system = request.system
    byte_limit = limits["request_byte_limit"]
    capacity_error = RouteContextExceeded if context_limit < config.max_input_tokens or byte_limit else BudgetExceeded

    def fits(items: list[Message], prompt: str) -> bool:
        current = ContextBudget.calculate(effective, system=prompt, tools=budget_tools, messages=items, provider=provider.name)
        if current.estimated_total(items) + output + current.safety_margin_tokens > context_limit:
            return False
        if byte_limit:
            metadata = dict(request.metadata)
            if prompt != request.system:
                metadata.pop("system_blocks", None)
            candidate = replace(request, system=prompt, messages=items, max_output_tokens=output, metadata=metadata)
            size = provider.request_bytes(candidate)
            if size is None:
                raise BudgetExceeded("context", "This adapter cannot validate the configured request-byte limit")
            return size <= byte_limit
        return True

    if not fits([], system):
        raise capacity_error("context", "Protected task state, instructions and tool schemas exceed the route budget")
    compacted = False
    if not fits(messages, system):
        # Scoped helpers have only their own user instructions, never the parent transcript.
        protected = [m.content for m in messages if m.role.value == "user"]
        if not request.metadata.get("protected_task_state") and protected:
            system += "\n\nScoped user requirements (user authority):\n" + json_dumps(protected)
        budget = ContextBudget.calculate(effective, system=system, tools=budget_tools, messages=messages, provider=provider.name)
        available = context_limit - output - budget.safety_margin_tokens - budget.provider_framing_tokens - budget.system_tokens - budget.tool_schema_tokens
        if available <= 0 or not fits([], system):
            raise capacity_error("context", "Protected task requirements cannot fit the safe route budget")
        try:
            messages = compact_messages(messages, force=True, keep_recent=1,
                                        target_tokens=max(1, int(available * 0.8)),
                                        target_bytes=max(1, byte_limit - estimate_request_bytes(system, [], budget_tools)) if byte_limit else 0)
        except CompactionError as error:
            raise capacity_error("context", "Route context cannot be compacted within its safe budget") from error
        if messages and messages[0].metadata.get("compacted"):
            system += "\n\n" + messages[0].content
            messages = messages[1:]
        compacted = True
    validate_tool_call_order(messages)
    if not fits(messages, system):
        raise capacity_error("context", "Prepared request exceeds the actual model/endpoint limits")
    metadata = dict(request.metadata)
    if compacted:
        metadata.pop("system_blocks", None)
        metadata.pop("compaction_artifact_id", None)
        metadata.pop("compacted_context_hashes", None)
    metadata["resolved_limits"] = {**limits, "configured_context_tokens": config.max_input_tokens,
                                   "effective_context_tokens": context_limit,
                                   "token_count_method": "conservative_estimate"}
    return replace(request, system=system, messages=messages, max_output_tokens=output, metadata=metadata)
