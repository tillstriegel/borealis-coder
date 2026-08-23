"""Run budget accounting and deterministic request-size estimation."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import AgentConfig
from ..errors import BudgetExceeded
from ..models import Message, Usage
from ..util import estimate_tokens, json_dumps, monotonic_ms


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
            raise BudgetExceeded("turns", f"Maximum {self.config.max_turns} model turns reached")
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
