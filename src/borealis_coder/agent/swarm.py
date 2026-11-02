"""Run-local delegation limits and bounded, draining tool workers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TypeVar

from ..config import AgentConfig
from ..util import finish_on_cancellation

T = TypeVar("T")
R = TypeVar("R")


@dataclass(slots=True)
class SwarmState:
    max_depth: int
    max_descendants: int
    max_concurrent: int
    max_turns: int
    allow_writes: bool
    admitted: int = 0
    active_requests: int = 0
    mutations: int = 0
    provider_slots: asyncio.Semaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.provider_slots = asyncio.Semaphore(self.max_concurrent)

    @classmethod
    def start(cls, config: AgentConfig) -> SwarmState:
        return cls(config.swarm_max_depth, config.swarm_max_descendants,
                   config.swarm_max_concurrent, config.swarm_max_turns,
                   config.swarm_allow_writes)

    def admit(self, depth: int) -> str | None:
        """Reserve a descendant before the first await. Reservations never recycle."""
        if not self.max_depth or not self.max_descendants:
            return "Delegation is disabled by the operator"
        if depth > self.max_depth:
            return f"Delegation depth limit reached ({self.max_depth})"
        if self.admitted >= self.max_descendants:
            return f"Delegation descendant limit reached ({self.max_descendants})"
        self.admitted += 1
        return None

    def limits(self) -> dict[str, int]:
        return {"admitted": self.admitted, "max_descendants": self.max_descendants,
                "active_requests": self.active_requests, "max_concurrent": self.max_concurrent,
                "max_depth": self.max_depth, "max_turns": self.max_turns}


async def bounded_calls(
    items: Sequence[T], limit: int, execute: Callable[[T], Awaitable[R]],
) -> list[R]:
    """Allocate only a fixed worker set; cancel and drain it on every fatal exit."""
    pending = iter(enumerate(items))
    results: dict[int, R] = {}

    async def worker() -> None:
        for index, item in pending:
            results[index] = await execute(item)

    tasks = [asyncio.create_task(worker()) for _ in range(min(len(items), max(1, limit)))]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await finish_on_cancellation(asyncio.gather(*tasks, return_exceptions=True))
        raise
    return [results[index] for index in range(len(items))]
