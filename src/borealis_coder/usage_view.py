"""Small in-memory view of root context and persisted session accounting."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Usage


@dataclass
class UsageView:
    usage: Usage = field(default_factory=Usage)
    context_tokens: int | None = None
    context_limit: int | None = None

    def clear_context(self) -> None:
        self.context_tokens = self.context_limit = None

    def render(self, columns: int) -> str:
        if self.context_tokens is None or not self.context_limit:
            context = "Context unavailable"
        else:
            ratio = self.context_tokens / self.context_limit
            filled = min(8, max(0, int(ratio * 8)))
            context = f"Context est. [{'#' * filled}{'-' * (8 - filled)}] {ratio:.0%}"
        cache = (f"Cache read {self.usage.provider_cache_hit_rate:.0%}"
                 if self.usage.input_tokens else "Cache read n/a")
        totals = f"Tokens {self.usage.total_tokens:,}"
        if len(f"{context} | {cache} | {totals}") <= columns:
            return f"{context} | {cache} | {totals}"
        return f"{context}\n{cache} | {totals}"
